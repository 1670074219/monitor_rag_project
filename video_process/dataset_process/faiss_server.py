import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
import threading
from queue import Queue, Empty, Full
from rank_bm25 import BM25Okapi
import jieba
import pickle
import time
import mysql.connector
from mysql.connector import Error

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    - video_log_vectors 中不存在对应 video_id
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
            LEFT JOIN video_log_vectors vlv
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
                    LEFT JOIN video_log_vectors vlv
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


class FaissServer(threading.Thread):
    def __init__(self,
                 emd_model_path: str,
                 video_dsp_queue: Queue,
                 index_path: str = None,
                 db_config: dict = None):
        super().__init__(daemon=True)
        self.emd_model_path = emd_model_path
        self.embedding_model = SentenceTransformer(self.emd_model_path, device='cuda:2')
        self.video_dsp_queue = video_dsp_queue
        self.index_path = index_path or os.path.join(base_dir, "faiss_cache/faiss_ifl2.index")
        self.db_config = db_config or DB_CONFIG

        # 线程控制
        self.running = True
        self.index_lock = threading.Lock()

        # 纯内存映射（由DB初始化）
        self.idx_to_vid = {}    # faiss_idx -> video_id
        self.vid_to_idx = {}    # video_id -> faiss_idx
        self.vid_to_text = {}   # video_id -> original_text

        # BM25内存索引
        self.bm25_corpus = []
        self.bm25_keys = []
        self.bm25_model = None

        self.index = None
        self._init_memory_from_db()
        self._load_or_build_faiss_index()
        self._build_bm25_from_memory()

    def _get_connection(self):
        try:
            return mysql.connector.connect(**self.db_config)
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return None

    def _init_memory_from_db(self):
        """
        启动时一次性从 DB 加载映射到内存，彻底根除 JSON I/O。
        """
        connection = self._get_connection()
        if not connection:
            return

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT video_id, faiss_idx, original_text
                FROM video_log_vectors
                WHERE faiss_idx IS NOT NULL
                ORDER BY faiss_idx ASC
            """)
            rows = cursor.fetchall()
            cursor.close()

            self.idx_to_vid.clear()
            self.vid_to_idx.clear()
            self.vid_to_text.clear()

            for row in rows:
                video_id = int(row['video_id'])
                faiss_idx = int(row['faiss_idx'])
                original_text = row.get('original_text') or ""

                self.idx_to_vid[faiss_idx] = video_id
                self.vid_to_idx[video_id] = faiss_idx
                self.vid_to_text[video_id] = original_text

            logger.info(f"内存映射初始化完成: idx_to_vid={len(self.idx_to_vid)}")
        except Error as e:
            logger.error(f"初始化内存映射失败: {e}")
        finally:
            if connection.is_connected():
                connection.close()

    def _load_or_build_faiss_index(self):
        """
        加载本地 FAISS 索引；若不可用则从 DB 的 vector_data 重建。
        使用 IndexFlatIP + L2 归一化向量，等价余弦相似度检索。
        """
        if os.path.exists(self.index_path):
            try:
                loaded = faiss.read_index(self.index_path)
                metric = getattr(loaded, 'metric_type', None)
                if metric == faiss.METRIC_INNER_PRODUCT:
                    self.index = loaded
                    logger.info(f"加载本地IndexFlatIP成功，ntotal={self.index.ntotal}")
                    return
                logger.warning("本地FAISS索引不是IP度量，将从DB重建。")
            except Exception as e:
                logger.warning(f"本地FAISS索引加载失败，将从DB重建: {e}")

        self._rebuild_faiss_from_db_vectors()

    def _rebuild_faiss_from_db_vectors(self):
        """从 video_log_vectors.vector_data(BLOB) 重建 FAISS 索引。"""
        connection = self._get_connection()
        if not connection:
            return

        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT video_id, faiss_idx, vector_data
                FROM video_log_vectors
                WHERE vector_data IS NOT NULL
                ORDER BY faiss_idx ASC
            """)
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                self.index = None
                logger.info("DB中暂无向量，FAISS索引保持空。")
                return

            self.index = None
            self.idx_to_vid.clear()
            self.vid_to_idx.clear()

            for expected_idx, row in enumerate(rows):
                video_id = int(row['video_id'])
                blob = row['vector_data']
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.size == 0:
                    continue
                vec = vec.reshape(1, -1).astype(np.float32)

                # 防御性归一化，确保向量符合IP检索前提
                faiss.normalize_L2(vec)

                if self.index is None:
                    self.index = faiss.IndexFlatIP(vec.shape[1])

                self.index.add(vec)
                real_idx = self.index.ntotal - 1

                self.idx_to_vid[real_idx] = video_id
                self.vid_to_idx[video_id] = real_idx

                # 若DB中faiss_idx不连续/不一致，回写修正
                if int(row['faiss_idx']) != real_idx:
                    upd = connection.cursor()
                    upd.execute("UPDATE video_log_vectors SET faiss_idx = %s WHERE video_id = %s", (real_idx, video_id))
                    upd.close()

            connection.commit()

            if self.index is not None:
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                logger.info(f"已从DB重建并保存FAISS索引，ntotal={self.index.ntotal}")

        except Error as e:
            logger.error(f"从DB重建FAISS失败: {e}")
            if connection:
                connection.rollback()
        finally:
            if connection and connection.is_connected():
                connection.close()

    def _build_bm25_from_memory(self):
        """根据内存文本缓存构建BM25模型（key统一为int型video_id）。"""
        self.bm25_corpus = []
        self.bm25_keys = []

        for vid, text in self.vid_to_text.items():
            if not text:
                continue
            tokens = list(jieba.cut(text))
            self.bm25_corpus.append(tokens)
            self.bm25_keys.append(int(vid))

        if self.bm25_corpus:
            self.bm25_model = BM25Okapi(self.bm25_corpus)
            logger.info(f"BM25模型构建完成，文档数={len(self.bm25_corpus)}")
        else:
            self.bm25_model = None
            logger.info("BM25语料为空，暂不构建模型。")

    def _upsert_bm25_doc(self, video_id: int, text: str):
        """增量更新BM25文档，并重建模型（简单稳定优先）。"""
        tokens = list(jieba.cut(text))
        if video_id in self.bm25_keys:
            idx = self.bm25_keys.index(video_id)
            self.bm25_corpus[idx] = tokens
        else:
            self.bm25_keys.append(video_id)
            self.bm25_corpus.append(tokens)

        self.bm25_model = BM25Okapi(self.bm25_corpus)

    def _upsert_vector_to_db(self, video_id: int, faiss_idx: int, original_text: str, vector: np.ndarray):
        """
        将向量写入 video_log_vectors：
        - vector_data 使用 BLOB（二进制）存储
        - 主键按 video_id 逻辑幂等更新
        """
        connection = self._get_connection()
        if not connection:
            return False

        try:
            vector_blob = vector.astype(np.float32).tobytes()
            cursor = connection.cursor()

            cursor.execute("SELECT id FROM video_log_vectors WHERE video_id = %s LIMIT 1", (video_id,))
            row = cursor.fetchone()

            if row:
                cursor.execute(
                    """
                    UPDATE video_log_vectors
                    SET faiss_idx = %s,
                        original_text = %s,
                        vector_data = %s
                    WHERE video_id = %s
                    """,
                    (faiss_idx, original_text, vector_blob, video_id)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO video_log_vectors (video_id, faiss_idx, original_text, vector_data)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (video_id, faiss_idx, original_text, vector_blob)
                )

            connection.commit()
            cursor.close()
            return True
        except Error as e:
            logger.error(f"向量入库失败(video_id={video_id}): {e}")
            connection.rollback()
            return False
        finally:
            if connection.is_connected():
                connection.close()

    def _normalize_bm25_scores(self, bm25_dict: dict):
        """
        BM25 Min-Max 归一化到 [0, 1]。
        若 max == min，则全部赋值 1.0。
        """
        if not bm25_dict:
            return {}

        vals = list(bm25_dict.values())
        v_min, v_max = min(vals), max(vals)
        if v_min == v_max:
            return {k: 1.0 for k in bm25_dict}

        denom = v_max - v_min
        return {k: (v - v_min) / denom for k, v in bm25_dict.items()}

    def add_data(self, task: dict):
        """
        消费单个向量化任务。
        task: {'video_id': int, 'original_text': str}
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

        # 已向量化直接跳过（幂等）
        if video_id in self.vid_to_idx:
            logger.info(f"video_id={video_id} 已存在向量，跳过")
            return

        # 关键修复1：归一化向量 + IndexFlatIP
        embedding = self.embedding_model.encode([original_text]).astype(np.float32)
        faiss.normalize_L2(embedding)

        with self.index_lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(embedding.shape[1])

            self.index.add(embedding)
            faiss_idx = int(self.index.ntotal - 1)

            # 先写本地索引（保证崩溃恢复）
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)

        # 写数据库（BLOB）
        ok = self._upsert_vector_to_db(video_id, faiss_idx, original_text, embedding[0])
        if not ok:
            logger.error(f"video_id={video_id} 写库失败，索引已增加，请排查一致性")

        # 更新内存映射（检索阶段O(1)）
        self.idx_to_vid[faiss_idx] = video_id
        self.vid_to_idx[video_id] = faiss_idx
        self.vid_to_text[video_id] = original_text

        # 更新BM25
        self._upsert_bm25_doc(video_id, original_text)

        logger.info(f"向量化完成: video_id={video_id}, faiss_idx={faiss_idx}")

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

    def search(self, query: str, k: int = 3):
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. Cannot perform search.")
            return []

        query_embedding = self.embedding_model.encode([query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding, k)

        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(sim)))

        return results

    def bm25_search(self, query: str, k: int = 3):
        if not self.bm25_model:
            logger.warning("BM25模型为空，无法检索。")
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.bm25_model.get_scores(query_tokens)
        ranked = sorted(zip(self.bm25_keys, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(int(vid), float(score)) for vid, score in ranked]

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        start_time = time.time()

        bm25_raw = dict(self.bm25_search(query, k * 2))
        bm25_norm = self._normalize_bm25_scores(bm25_raw)

        faiss_results = self.search(query, k * 2)
        faiss_dict = {}
        for idx, sim in faiss_results:
            vid = self.idx_to_vid.get(idx)
            if vid is None:
                continue
            faiss_dict[vid] = max(0.0, min(1.0, float(sim)))

        all_vids = set(bm25_norm.keys()).union(faiss_dict.keys())
        merged = []
        for vid in all_vids:
            bm25_score = bm25_norm.get(vid, 0.0)
            faiss_score = faiss_dict.get(vid, 0.0)
            hybrid_score = alpha * faiss_score + (1 - alpha) * bm25_score
            merged.append((vid, hybrid_score))

        merged.sort(key=lambda x: x[1], reverse=True)
        result = merged[:k]
        logger.info(f"Hybrid检索完成: k={k}, 返回={len(result)}, 耗时={time.time() - start_time:.3f}s")
        return result

    def get_analyse_kv(self, idxs: list):
        """
        根据 (faiss_idx, score) 列表返回 (video_id, original_text, score)
        全程O(1)字典查找，无磁盘I/O。
        """
        buffer = []
        for idx, score in idxs:
            vid = self.idx_to_vid.get(int(idx))
            if vid is None:
                continue
            buffer.append((vid, self.vid_to_text.get(vid), score))
        return buffer


if __name__ == "__main__":
    # 生产者-消费者：主线程生产，FaissServer线程消费
    task_queue = Queue(maxsize=100)

    faiss_server = FaissServer(
        emd_model_path="/root/data1/bge_zh_v1.5/",
        video_dsp_queue=task_queue,
        index_path="/root/data1/monitor_rag_project/video_process/faiss_cache/faiss_ifl2.index",
        db_config=DB_CONFIG
    )
    faiss_server.start()

    LOW_WATERMARK = 20
    FETCH_BATCH = 50

    logger.info("Faiss向量化服务已启动，进入持续SQL轮询。")

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
        faiss_server.stop()
        faiss_server.join(timeout=5)
        logger.info("Faiss服务已退出。")




        



