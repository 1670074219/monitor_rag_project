import logging
import os
from queue import Queue, Empty
import threading
import base64
from datetime import datetime
import openai
from PIL import Image
from io import BytesIO
import numpy as np
from qwen_vl_utils import process_vision_info
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

class VideoAnalyServer(threading.Thread):
    def __init__(self,
                 video_queue: Queue,
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 video_description_path: str,
                 video_dsp_queue: Queue = None):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.openai_client = openai.OpenAI(api_key=api_key, base_url=api_url)
        self.video_dsp_queue = video_dsp_queue
        self.video_description_path = video_description_path
        self.json_lock = threading.Lock()

    def run(self):
        while True:
            try:
                v_k = self.video_queue.get()
                logger.info(f"开始分析: {v_k}")
                analyse_result = self.analyze_video(v_k, self.video_description_path)
                logging.info(f"分析完成: {v_k}")
                if analyse_result is not None:
                    self.save_to_json(v_k, self.video_description_path, analyse_result)
            except Empty:
                logger.info("视频队列为空，等待新的视频")
                continue
        
    def analyze_video(self, v_k: str, video_description_path: str = None):
        camera_id_part, timestamp_part = v_k.split("_", 1)
        template=f"""你是一个专业的监控视频分析专家，你的分析应该是以人为主体的，环境信息是你分析人的行为时的参考。
                    请分析以下视频内容，并严格按照以下格式输出：
                    要求输出的描述要详细，不要遗漏任何细节。

                    1.  视频中一共出现了几个人？
                    2.  详细描述这些人的外貌特征（例如衣着、发型、体态等）。
                    3.  分析这些人在视频中都在做什么？请描述他们的具体动作和行为。
                    4.  这些人是否与周围环境发生了互动？
                    5.  如果发生了互动，请具体描述他们是如何与环境互动的（例如触摸了什么物体、操作了什么设备、对环境变化做出了什么反应等）。
                    视频总结：根据上面的分析，总结视频内容，并给出视频的总体描述，涉及每个人的动作行为，不要遗漏任何细节。

                    输出格式:
                    人数：[此处填写人数]
                    每个人的外貌特征：
                    - 人物1：[描述人物1的外貌]
                    - 人物2：[描述人物2的外貌]
                    ... （根据实际人数增减）
                    每个人的行为动作：
                    - 人物1：[描述人物1的行为]
                    - 人物2：[描述人物2的行为]
                    ... （根据实际人数增减）
                    与环境的交互：[描述人物与环境的交互情况，如果没有交互请说明无交互]
                    视频总结：摄像头{camera_id_part}在{timestamp_part}的监控视频分析报告：一个[人物1的外貌]在[人物1的行为]，一个[人物2的外貌]在[人物2的行为]，他们与环境[此处填写环境信息]发生了[此处填写交互情况]。
                    """
        with self.json_lock:
            with open(video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)

        messages = [
            {"role": "system", "content": "你是一个专业的监控视频分析专家，你的分析应该是以人为主体的，环境信息是你分析人的行为时的参考"},
            {"role": "user", "content": [
                {"type": "text", "text": template},
                {
                    "type": "video",
                    "video": video_data[v_k]['video_path'],
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 2,
                    "fps": 3.0
                }
            ]}
        ]

        processed_messages, video_kwargs = self.prepare_message_for_vllm(messages)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=processed_messages,
                extra_body={'mm_processor_kwargs': video_kwargs}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return None
    
    def prepare_message_for_vllm(self, messages):
        vllm_messages, fps_list = [], []

        for message in messages:
            content_list = message["content"]
            if not isinstance(content_list, list):
                vllm_messages.append(message)
                continue

            new_content = []
            for part in content_list:
                if part.get("type") == "video":
                    video_message = [{"content": [part]}]
                    _, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                    frames = video_inputs.pop().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get("fps", []))

                    b64_frames = []
                    for frame in frames:
                        img = Image.fromarray(frame)
                        with BytesIO() as buf:
                            img.save(buf, format="jpeg")
                            b64_frames.append(base64.b64encode(buf.getvalue()).decode())

                    new_content.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(b64_frames)}"}
                    })
                else:
                    new_content.append(part)
            message["content"] = new_content
            vllm_messages.append(message)

        return vllm_messages, {"fps": fps_list}

    def save_to_json(self, v_k: str, video_description_path: str, analyse_result: str):
        with self.json_lock:
            with open(video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)
            
            video_data[v_k]['analyse_result'] = analyse_result
            
            with open(video_description_path, "w", encoding="utf-8") as f:
                json.dump(video_data, f, ensure_ascii=False, indent=4)
            logger.info(f"已更新{v_k}json:analyse_result")

            if self.video_dsp_queue is not None:
                self.video_dsp_queue.put(v_k)
                logger.info(f"已放入embedding队列:{v_k}")

if __name__ == "__main__":
    video_queue = Queue()
    video_queue.put("camera1_20250604_104536")
    video_queue.put("camera2_20250604_104554")
    video_queue.put("camera1_20250604_104620")

    video_analyse_server = VideoAnalyServer(video_queue, "qwen2.5", "http://localhost:8000/v1", "token-abc123", "/root/data1/monitor_rag_project/video_process/video_description.json")
    video_analyse_server.start()
    video_analyse_server.join()



        
