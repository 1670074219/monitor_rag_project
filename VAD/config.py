from openai import OpenAI
from ultralytics import YOLO
import os

client = OpenAI(
    api_key= os.getenv("OPENAI_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

yolo_model = YOLO('yolo11x.pt')
llm_model_name = "qwen2.5-14b-instruct"
vlm_model_name = "qwen2.5-vl-32b-instruct"
emb_model_name = "text_embedding_v3"
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
