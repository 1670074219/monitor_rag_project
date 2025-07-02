import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import logging
import threading
from queue import Queue
from rank_bm25 import BM25Okapi
import jieba
import pickle
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

class FaissServer(threading.Thread):
    def __init__(self, 
                 emd_model_path: str,  
                 video_dsp_queue: Queue,
                 video_description_path: str,
                 buffer_size: int = 5,
                 index_path: str = None):
        super().__init__(daemon=True)
        self.emd_model_path = emd_model_path
        self.emd_model = SentenceTransformer(self.emd_model_path, device='cuda:2')
        self.index_path = index_path or os.path.join(base_dir, "faiss_cache/faiss_ifl2.index")
        self.index = faiss.read_index(self.index_path) if os.path.exists(self.index_path) else None
        self.buffer_size = buffer_size
        self.buffer = []
        self.video_dsp_queue = video_dsp_queue
        self.video_description_path = video_description_path
        self.json_lock = threading.Lock()

        # 初始化BM25
        self.bm25_corpus = []
        self.bm25_keys = []
        self.bm25_model = None
        self.load_bm25_corpus()

    def run(self):
        while True:
            v_k = self.video_dsp_queue.get()
            self.add_data(v_k, self.video_description_path)

    def add_data(self, v_k: str, video_description_path: str = None):
        with self.json_lock:
            with open(video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

        if video_data[v_k]['is_embedding'] is not True:
            embeddings = self.emd_model.encode([video_data[v_k]['analyse_result']]).astype('float32')

            if self.index is None:
                embedding_dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(embedding_dim)

            self.index.add(embeddings)
            logger.info(f"Adde {v_k} to index.")

            faiss.write_index(self.index, self.index_path)

            video_data[v_k]['is_embedding'] = True
            video_data[v_k]['idx'] = self.index.ntotal - 1


            with self.json_lock:
                with open(video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, ensure_ascii=False, indent=4)
                logger.info(f"已更新{v_k}json:is_embedding, idx")
            
            # 更新 BM25
            tokens = list(jieba.cut(video_data[v_k]['analyse_result']))

            if v_k not in self.bm25_keys:
                self.bm25_corpus.append(tokens)
                self.bm25_keys.append(v_k)
                self.bm25_model = BM25Okapi(self.bm25_corpus)
                self.save_bm25_corpus()
                logger.info(f"BM25 增量更新: {v_k}")
            else:
                logger.info(f"BM25 中已存在 {v_k} , 跳过更新")
        else:
            logger.info(f"{v_k} has embedding")

    def save_bm25_corpus(self, cache_dir: str = None):
        cache_dir = cache_dir or os.path.join(base_dir, "bm25_cache")
        os.makedirs(cache_dir, exist_ok=True)

        with open(os.path.join(cache_dir, "bm25_corpus.pkl"), "wb") as f:
            pickle.dump(self.bm25_corpus, f)

        with open(os.path.join(cache_dir, "bm25_keys.pkl"), "wb") as f:
            pickle.dump(self.bm25_keys, f)

        logger.info(f"BM25 语料库已保存到 {cache_dir}")
    
    def load_bm25_corpus_from_cache(self, cache_dir: str = None):
        cache_dir = cache_dir or os.path.join(base_dir, "bm25_cache")
        corpus_path = os.path.join(cache_dir, "bm25_corpus.pkl")
        keys_path = os.path.join(cache_dir, "bm25_keys.pkl")

        if not os.path.exists(corpus_path) or not os.path.exists(keys_path):
            logger.warning("BM25缓存不存在，跳过加载")
            return False

        with open(corpus_path, "rb") as f:
            self.bm25_corpus = pickle.load(f)

        with open(keys_path, "rb") as f:
            self.bm25_keys = pickle.load(f)

        if self.bm25_corpus:
            self.bm25_model = BM25Okapi(self.bm25_corpus)
            logger.info(f"BM25缓存已加载，文档数：{len(self.bm25_corpus)}")
            return True
        else:
            logger.warning("BM25缓存为空")
            return False
        
    def load_bm25_corpus(self):
        if self.load_bm25_corpus_from_cache():
            return

        # 否则重新构建
        with self.json_lock:
            with open(self.video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

        self.bm25_corpus = []
        self.bm25_keys = []

        for v_k, v_info in video_data.items():
            if v_info.get('analyse_result'):
                tokens = list(jieba.cut(v_info['analyse_result']))
                self.bm25_corpus.append(tokens)
                self.bm25_keys.append(v_k)

        if self.bm25_corpus:
            self.bm25_model = BM25Okapi(self.bm25_corpus)
            logger.info(f"BM25模型已更新，当前文档数：{len(self.bm25_corpus)}")

        # 保存缓存
        self.save_bm25_corpus()

    def search(self, query: str, k: int = 3):
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty. Cannot perform search.")
            return []

        query_embedding = self.emd_model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append((idx, distance))
        
        logger.info(f"Search for query '{query}' returned {len(results)} results.")
        return results
    
    def bm25_search(self, query: str, k: int = 3):
        if not self.bm25_model:
            logger.warning("BM25模型为空，无法检索。")
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.bm25_model.get_scores(query_tokens)

        ranked_results = sorted(zip(self.bm25_keys, scores), key=lambda x: x[1], reverse=True)[:k]
        logger.info(f"BM25 检索返回 {len(ranked_results)} 个结果。")
        return ranked_results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
        start_time = time.time()
        
        # BM25检索
        bm25_start = time.time()
        bm25_results = self.bm25_search(query, k * 2)
        bm25_dict = {v_k: score for v_k, score in bm25_results}
        bm25_time = time.time() - bm25_start
        logger.info(f"BM25检索耗时: {bm25_time:.3f}秒")

        # Faiss检索
        faiss_start = time.time()
        faiss_results = self.search(query, k * 2)
        faiss_idx_to_v_k = {}

        with self.json_lock:
            with open(self.video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

            for v_k, v_info in video_data.items():
                if v_info.get('idx') is not None:
                    faiss_idx_to_v_k[v_info['idx']] = v_k

        faiss_dict = {}
        for idx, distance in faiss_results:
            v_k = faiss_idx_to_v_k.get(idx)
            if v_k:
                sim_score = 1 / (1 + distance)
                faiss_dict[v_k] = sim_score
        faiss_time = time.time() - faiss_start
        logger.info(f"Faiss检索耗时: {faiss_time:.3f}秒")

        # 混合排序
        all_keys = set(bm25_dict.keys()).union(faiss_dict.keys())
        hybrid_scores = []
        for v_k in all_keys:
            bm25_score = bm25_dict.get(v_k, 0)
            faiss_score = faiss_dict.get(v_k, 0)
            hybrid_score = alpha * faiss_score + (1 - alpha) * bm25_score
            hybrid_scores.append((v_k, hybrid_score))

        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k]
        total_time = time.time() - start_time
        logger.info(f"Hybrid 检索返回 {len(hybrid_scores)} 个结果。总耗时: {total_time:.3f}秒")
        return hybrid_scores
    
    def get_analyse_kv(self, idxs: list, video_description_path: str):
        buffer = []
        with self.json_lock:
            with open(video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)
        
        for i, j in idxs:
            for v_k, v_info in video_data.items():
                if v_info.get('idx') == i:
                    buffer.append((v_k, v_info.get('analyse_result'), j))
        
        return buffer

if __name__ == "__main__":
    v_dsp_q = Queue()

    json_lock = threading.Lock()
    # with json_lock:
    #     with open("/root/data1/monitor_rag_project/video_process/video_description.json", "r", encoding="utf-8") as f:
    #         video_data = json.load(f)

    # for v_k, v_info in video_data.items():
    #     if v_info['analyse_result'] is not None and v_info['is_embedding'] is False:
    #         v_dsp_q.put(v_k)
    
    # logging.info(f"Init v_q put {v_dsp_q.qsize()}")

    faiss_server = FaissServer(emd_model_path="/root/data1/bge_zh_v1.5/",
                               video_dsp_queue=v_dsp_q,
                               index_path="/root/data1/monitor_rag_project/video_process/faiss_cache/faiss_ifl2.index",
                               video_description_path="/root/data1/monitor_rag_project/video_process/video_description.json")
    
    # faiss_server.start()
    # faiss_server.join()
    start_time = time.time()
    results = faiss_server.hybrid_search("深色衣服，戴着帽子，体型较瘦", k=5, alpha=0.8)
    end_time = time.time()
    print(f"Hybrid 检索时间: {end_time - start_time} 秒")
    for v_k, score in results:
        print(f"视频: {v_k}, Hybrid分数: {score}\n")




        



