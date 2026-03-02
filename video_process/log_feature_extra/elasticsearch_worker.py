import os
import time
import logging
import threading
from queue import Queue, Empty, Full

import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

import mysql.connector
from mysql.connector import Error


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'database': 'monitor_database',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
}


def get_unembedded_logs(max_count: int = 50, db_config: dict = None):
    """
    生产者SQL：只拉取“已生成description但尚未向量化”的视频。

    核心判定：
    - videos.description 非空
    - video_logs_index 中不存在对应 video_id
    """
    cfg = db_config or DB_CONFIG
    connection = None
    records = []

    try:
        connection = mysql.connector.connect(**cfg)
        cursor = connection.cursor(dictionary=True)
        sql = """
            SELECT
                v.id AS video_id,
                v.description AS original_text
            FROM videos v
            LEFT JOIN video_logs_index vlv
                ON vlv.video_id = v.id
            WHERE v.description IS NOT NULL
              AND v.description <> ''
              AND vlv.id IS NULL
            ORDER BY v.id ASC
            LIMIT %s
        """
        try:
            cursor.execute(sql, (max_count,))
        except Error as e:
            # 兼容部分历史表结构（可能没有自增主键id）
            if "Unknown column 'vlv.id'" in str(e):
                sql_fallback = """
                    SELECT
                        v.id AS video_id,
                        v.description AS original_text
                    FROM videos v
                    LEFT JOIN video_logs_index vlv
                        ON vlv.video_id = v.id
                    WHERE v.description IS NOT NULL
                      AND v.description <> ''
                      AND vlv.video_id IS NULL
                    ORDER BY v.id ASC
                    LIMIT %s
                """
                cursor.execute(sql_fallback, (max_count,))
            else:
                raise

        rows = cursor.fetchall()
        cursor.close()

        for row in rows:
            records.append({
                'video_id': int(row['video_id']),
                'original_text': row['original_text']
            })

    except Error as e:
        logger.error(f"get_unembedded_logs 查询失败: {e}")
    finally:
        if connection and connection.is_connected():
            connection.close()

    return records


class ElasticSearchWorker(threading.Thread):
    def __init__(
        self,
        emd_model_path: str,
        video_dsp_queue: Queue,
        db_config: dict = None,
        es_url: str = "http://219.216.99.30:9200",
        index_name: str = "video_logs_index"
    ):
        super().__init__(daemon=True)
        self.emd_model_path = emd_model_path
        self.embedding_model = SentenceTransformer(self.emd_model_path, device='cuda:2')
        self.video_dsp_queue = video_dsp_queue
        self.db_config = db_config or DB_CONFIG

        self.es_url = es_url
        self.index_name = index_name
        self.running = True

        self.es = self._init_es_client()
        self._init_index_mapping()

    def _get_connection(self):
        try:
            return mysql.connector.connect(**self.db_config)
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return None

    def _init_es_client(self) -> Elasticsearch:
        """
        初始化 Elasticsearch 客户端。

        说明：
        1) 使用配置的 self.es_url 连接 ES 8.x。
        2) 启用 retry_on_timeout + max_retries，增强网络抖动时的稳健性。
        3) 先走 ping(HEAD /) 快速探活；若失败，再回退到 info(GET /) 二次确认。
        4) 若检测到客户端主版本过高导致的 media_type_header_exception（compatible-with=9），
           自动切换到 ES8 兼容请求头重试，避免必须立刻降级依赖。
        """
        def _build_client(compat_8_headers: bool = False) -> Elasticsearch:
            headers = None
            if compat_8_headers:
                headers = {
                    "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                    "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
                }
            return Elasticsearch(
                self.es_url,
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3,
                headers=headers,
            )

        def _probe_client(es_client: Elasticsearch) -> Elasticsearch:
            if es_client.ping():
                logger.info(f"Elasticsearch连接成功(ping): {self.es_url}")
                return es_client

            info = es_client.info()
            version = info.get("version", {}).get("number", "unknown")
            cluster = info.get("cluster_name", "unknown")
            logger.info(
                f"Elasticsearch连接成功(info fallback): {self.es_url}, "
                f"cluster={cluster}, version={version}"
            )
            return es_client

        try:
            return _probe_client(_build_client(compat_8_headers=False))
        except Exception as e:
            err_text = str(e)
            is_media_type_mismatch = (
                "media_type_header_exception" in err_text
                or "compatible-with=9" in err_text
                or "Accept version must be either version 8 or 7" in err_text
            )

            if is_media_type_mismatch:
                logger.warning(
                    "检测到ES客户端版本与服务端版本请求头不兼容，"
                    "将自动启用 ES8 兼容请求头重试。"
                )
                try:
                    return _probe_client(_build_client(compat_8_headers=True))
                except Exception as retry_e:
                    logger.error(f"ES8兼容头重试失败: {retry_e}")
                    raise

            logger.error(f"初始化Elasticsearch失败: {e}")
            raise

    def _init_index_mapping(self):
        """
        初始化索引及 Mapping：video_logs_index

        Mapping 要求：
        - video_id: integer
        - description: text (analyzer=standard)
        - embedding: dense_vector (dims=1024, index=true, similarity=cosine)
        """
        body = {
            "mappings": {
                "properties": {
                    "video_id": {"type": "integer"},
                    "description": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, **body)
                logger.info(f"已创建ES索引: {self.index_name}")
            else:
                logger.info(f"ES索引已存在: {self.index_name}")
        except es_exceptions.RequestError as e:
            logger.error(f"创建/检查ES索引失败: {e}")
            raise

    def _upsert_marker_to_db(self, video_id: int, original_text: str, embedding: np.ndarray) -> bool:
        """
        向 video_logs_index 写入“已处理标记”：
        - 已清理遗留的 faiss_idx 字段
        - vector_data 写入BLOB（float32二进制）
        - 对已存在 video_id 做幂等更新
        """
        connection = self._get_connection()
        if not connection:
            return False

        try:
            vector_blob = embedding.astype(np.float32).tobytes()
            cursor = connection.cursor()

            cursor.execute("SELECT id FROM video_logs_index WHERE video_id = %s LIMIT 1", (video_id,))
            row = cursor.fetchone()

            if row:
                cursor.execute(
                    """
                    UPDATE video_logs_index
                    SET original_text = %s,
                        vector_data = %s
                    WHERE video_id = %s
                    """,
                    # 已清理遗留的 faiss_idx 字段
                    (original_text, vector_blob, video_id)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO video_logs_index (video_id, original_text, vector_data)
                    VALUES (%s, %s, %s)
                    """,
                    # 已清理遗留的 faiss_idx 字段
                    (video_id, original_text, vector_blob)
                )

            connection.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"写入已处理标记失败(video_id={video_id}): {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                connection.close()

    def add_data(self, task: dict):
        """
        消费单个向量化任务。
        task: {'video_id': int, 'original_text': str}

        关键约束：
        - ES 文档 ID 强制使用 str(video_id)，实现幂等写入，防止重复数据。
        - ES写入成功后再写 MySQL 标记，确保 LEFT JOIN 轮询语义正确。
        """
        if not isinstance(task, dict):
            logger.warning(f"非法任务类型: {type(task)}")
            return

        video_id = task.get('video_id')
        original_text = task.get('original_text')

        if video_id is None or not original_text:
            logger.warning(f"任务字段缺失: {task}")
            return

        video_id = int(video_id)
        original_text = str(original_text).strip()
        if not original_text:
            logger.warning(f"任务文本为空: video_id={video_id}")
            return

        try:
            embedding = self.embedding_model.encode([original_text])[0].astype(np.float32)
            embedding_list = embedding.tolist()
        except Exception as e:
            logger.error(f"向量化失败(video_id={video_id}): {e}")
            return

        doc = {
            "video_id": video_id,
            "description": original_text,
            "embedding": embedding_list,
        }

        try:
            # 强制文档ID=str(video_id)：同一个视频重复写入会覆盖更新而非新增
            es_resp = self.es.index(
                index=self.index_name,
                id=str(video_id),
                document=doc,
                refresh=False,
            )
            result = es_resp.get("result", "unknown")
            logger.info(f"ES写入成功: video_id={video_id}, result={result}")
        except Exception as e:
            logger.error(f"ES写入失败(video_id={video_id}): {e}")
            return

        ok = self._upsert_marker_to_db(video_id=video_id, original_text=original_text, embedding=embedding)
        if not ok:
            logger.error(f"video_id={video_id} 已写入ES，但MySQL标记写入失败，请排查")
            return

        logger.info(f"任务处理完成: video_id={video_id}")

    def run(self):
        while self.running:
            try:
                task = self.video_dsp_queue.get(timeout=1)
            except Empty:
                continue

            try:
                self.add_data(task)
            except Exception as e:
                logger.error(f"消费任务失败: {e}")

    def stop(self):
        self.running = False

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        使用 ES 8.x 原生 Hybrid 检索（同一请求融合向量召回 + BM25文本匹配）。

        核心设计：
        - knn 子句：对 embedding 做语义向量检索，boost=alpha
        - query.match 子句：对 description 做 BM25 检索，boost=(1-alpha)
        - 两路分数由 ES 在同一个 search 请求内融合到统一 _score

        返回：
        [
          {"video_id": int, "description": str, "score": float},
          ...
        ]
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        size = max(1, int(k))
        alpha = max(0.0, min(1.0, float(alpha)))

        try:
            query_vec = self.embedding_model.encode([query])[0].astype(np.float32).tolist()
        except Exception as e:
            logger.error(f"查询向量化失败: {e}")
            return []

        try:
            # -----------------------------
            # ES Hybrid DSL 详细说明：
            # 1) query: match(description)
            #    - 走倒排索引 + BM25
            #    - boost=1-alpha，控制关键词匹配在总分中的权重
            #
            # 2) knn: embedding
            #    - 走 dense_vector 的近邻检索
            #    - boost=alpha，控制语义向量相似度在总分中的权重
            #
            # 3) 同一个 search 请求中同时传入 query 和 knn
            #    - 由 ES 统一融合打分并返回 _score
            #    - 无需应用层手动归一化、手动加权、手动合并候选
            # -----------------------------
            response = self.es.search(
                index=self.index_name,
                size=size,
                query={
                    "match": {
                        "description": {
                            "query": query,
                            "boost": (1.0 - alpha)
                        }
                    }
                },
                knn={
                    "field": "embedding",
                    "query_vector": query_vec,
                    "k": max(size * 2, size),
                    "num_candidates": max(100, size * 10),
                    "boost": alpha,
                },
                source=["video_id", "description"],
            )

            hits = response.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                src = hit.get("_source", {})
                results.append({
                    "video_id": int(src.get("video_id")) if src.get("video_id") is not None else None,
                    "description": src.get("description", ""),
                    "score": float(hit.get("_score", 0.0)),
                })

            return results
        except Exception as e:
            logger.error(f"Hybrid检索失败: {e}")
            return []


if __name__ == "__main__":
    # 生产者-消费者：主线程生产，ElasticSearchWorker线程消费
    task_queue = Queue(maxsize=100)

    worker = ElasticSearchWorker(
        emd_model_path="/root/data1/bge_zh_v1.5/",
        video_dsp_queue=task_queue,
        db_config=DB_CONFIG,
        es_url="http://219.216.99.30:9200",
        index_name="video_logs_index",
    )
    worker.start()

    LOW_WATERMARK = 20
    FETCH_BATCH = 50

    logger.info("ElasticSearch向量化服务已启动，进入持续SQL轮询。")

    try:
        while True:
            qsize = task_queue.qsize()

            # 水位线补货：队列低于20才去数据库拉任务
            if qsize < LOW_WATERMARK:
                new_tasks = get_unembedded_logs(max_count=FETCH_BATCH, db_config=DB_CONFIG)

                if not new_tasks:
                    # 没有新任务：10秒后再查，避免空转
                    time.sleep(10)
                    continue

                for task in new_tasks:
                    try:
                        task_queue.put(task, block=True, timeout=1)
                    except Full:
                        logger.warning("任务队列已满，本轮补货提前结束。")
                        break

                logger.info(f"本轮补货完成，当前队列长度={task_queue.qsize()}")
                time.sleep(1)
            else:
                # 队列较满：短休眠，避免CPU空转
                time.sleep(5)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，准备优雅退出...")
        worker.stop()
        worker.join(timeout=5)
        logger.info("ElasticSearch服务已退出。")
