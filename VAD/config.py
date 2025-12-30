from openai import OpenAI
from ultralytics import YOLO
import os

# ========== 配置说明 ==========
# 使用本地 vLLM 服务器 (Qwen2.5-VL-7B-Instruct-AWQ)
# 运行在 http://localhost:8000
# ==============================

# 方式1：使用本地 vLLM 服务器（推荐）
USE_LOCAL_VLLM = True  # 设置为 True 使用本地 vLLM，False 使用通义千问 API

if USE_LOCAL_VLLM:
    # 本地 vLLM 配置
    client = OpenAI(
        api_key="token-abc123",  # vLLM 的 API key
        base_url="http://localhost:8000/v1",  # 本地 vLLM 地址
    )
    llm_model_name = "qwen2.5"  # 本地模型名称
    vlm_model_name = "qwen2.5"  # 同一个模型（VL 模型可以处理文本）
    emb_model_name = None  # 本地 vLLM 不支持 embedding（如需要需另外配置）
else:
    # 通义千问 API 配置
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_model_name = "qwen2.5-14b-instruct"
    vlm_model_name = "qwen2.5-vl-32b-instruct"
    emb_model_name = "text_embedding_v3"

yolo_model = YOLO('yolo11x.pt')
temperature = 0.0
emb_d = 768

train_videos_dir = "SHTech/train"
test_videos_dir = "SHTech/test"
train_desc_file = "SHTech_Train.json"
test_desc_file = "SHTech_Test.json"
rules_origin_file = "SHTech_Train_Rules.txt"
rules_file = "SHTech_Rules.txt"
test_results_file = "SHTech_Test_Results.json"

faiss_file = "doc_index.faiss"
bm25_file = "bm25.pickle"
