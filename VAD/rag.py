import faiss
import jieba
import pickle
import numpy as np
from utils import *
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, w_faiss=0.5, w_bm25=0.5):
        self.base_w_faiss = w_faiss
        self.base_w_bm25 = w_bm25
        self.sentences = []
        self.index = None
        self.bm25 = None

    def build(self, json_load_file, index_file, bm25_keyword_file):
        # 读取数据
        docs = load_json(json_load_file)
        self.sentences = [value for _, value in docs.items()]

        # 构建FAISS索引
        embeddings = []
        for i in range(0, len(self.sentences), 10):
            results = rag_embedding(self.sentences[i:i+10])
            embeddings.extend(results)
        embeddings = np.array(embeddings)
        # 归一化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, index_file)

        # 构建BM25索引
        tokenized_corpus = [list(jieba.cut(s)) for s in self.sentences]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(bm25_keyword_file, 'wb') as f:
            pickle.dump((self.bm25, self.sentences), f)

        print(f"[HybridRetriever] 索引建立完成，文档数: {len(self.sentences)}")

    def load(self, index_file, bm25_keyword_file):
        self.index = faiss.read_index(index_file)
        with open(bm25_keyword_file, 'rb') as f:
            self.bm25, self.sentences = pickle.load(f)
        print(f"[HybridRetriever] 索引加载完成，文档数: {len(self.sentences)}")

    def minmax_normalize(self, scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def retrieve(self, query, top_k=10):
        if self.index is None or self.bm25 is None:
            raise ValueError("请先 build() 或 load() 索引！")

        w_faiss, w_bm25 = self.base_w_faiss, self.base_w_bm25

        # FAISS检索
        query_embedding = rag_embedding([query])
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        D, faiss_idx = self.index.search(query_embedding, k=min(500, len(self.sentences)))
        faiss_scores = D[0]  # 相似度
        faiss_idx = faiss_idx[0]

        # BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = np.array(bm25_scores)

        # 取 top_k*5 扩大召回范围再融合
        faiss_top_idx = faiss_idx[:top_k * 5]
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k * 5]

        # 取召回的所有候选文档
        candidate_idx = list(set(faiss_top_idx) | set(bm25_top_idx))
        
        # 提取候选得分
        faiss_candidate_scores = np.array([faiss_scores[i] for i in candidate_idx])
        bm25_candidate_scores = np.array([bm25_scores[i] for i in candidate_idx])

        # 归一化
        faiss_norm_scores = self.minmax_normalize(faiss_candidate_scores)
        bm25_norm_scores = self.minmax_normalize(bm25_candidate_scores)

        # 融合
        hybrid_scores = w_faiss * faiss_norm_scores + w_bm25 * bm25_norm_scores

        # 双塔交集的得分稍微加权提升
        common_idx = set(faiss_top_idx) & set(bm25_top_idx)
        for i, idx in enumerate(candidate_idx):
            if idx in common_idx:
                hybrid_scores[i] += 0.1  

        # 排序
        sorted_idx = np.argsort(hybrid_scores)[::-1][:top_k]
        final_results = [self.sentences[candidate_idx[i]] for i in sorted_idx]

        return final_results

if __name__ == "__main__":
    retriever = HybridRetriever(w_faiss=0.6, w_bm25=0.4)
    retriever.build(json_load_file=train_desc_file, index_file=faiss_file, bm25_keyword_file=bm25_file)
    retriever.load(index_file=faiss_file, bm25_keyword_file=bm25_file)
    query = "有一个人在骑自行车横穿马路"
    results = retriever.retrieve(query, top_k=5)

    print("\n【检索结果】")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")
