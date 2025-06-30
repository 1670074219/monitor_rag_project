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
import time
import traceback

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
                time.sleep(1)
                continue
        
    def _handle_problematic_video(self, v_k: str):
        """处理有问题的视频：删除文件和JSON条目"""
        with self.json_lock:
            try:
                # 1. Read the current data
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                video_info = video_data.get(v_k)
                if not video_info:
                    logger.warning(f"无法在JSON中找到视频记录: {v_k}，可能已被处理。")
                    return

                # 2. Delete the video file
                video_path = video_info.get('video_path')
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        logger.info(f"已删除有问题的视频文件: {video_path}")
                    except OSError as e:
                        logger.error(f"删除视频文件 {video_path} 失败: {e}")
                
                # 3. Delete the entry from the dictionary
                del video_data[v_k]

                # 4. Write the updated data back to the file
                with open(self.video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, ensure_ascii=False, indent=4)
                logger.info(f"已从JSON中删除视频记录: {v_k}")

            except json.JSONDecodeError:
                logger.error(f"读写JSON文件时出错 {self.video_description_path}，文件可能已损坏。")
            except Exception as e:
                logger.error(f"处理有问题的视频 {v_k} 时发生未知错误: {e}")
                logger.debug(traceback.format_exc())

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
        import gc
        vllm_messages, fps_list = [], []

        for message in messages:
            content_list = message["content"]
            if not isinstance(content_list, list):
                vllm_messages.append(message)
                continue

            new_content = []
            for part in content_list:
                if part.get("type") == "video":
                    try:
                        video_message = [{"content": [part]}]
                        _, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                        frames = video_inputs.pop().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                        fps_list.extend(video_kwargs.get("fps", []))

                        # 内存优化：限制帧数
                        max_frames = min(len(frames), 8)  # 最多8帧
                        frames = frames[:max_frames]
                        logger.info(f"处理视频帧数: {len(frames)}")

                        b64_frames = []
                        for i, frame in enumerate(frames):
                            try:
                                img = Image.fromarray(frame)
                                with BytesIO() as buf:
                                    # 降低质量减少内存
                                    img.save(buf, format="jpeg", quality=60, optimize=True)
                                    b64_frames.append(base64.b64encode(buf.getvalue()).decode())
                                
                                # 每处理2帧清理一次内存
                                if (i + 1) % 2 == 0:
                                    gc.collect()
                                    
                            except Exception as e:
                                logger.error(f"处理帧 {i} 失败: {e}")
                                continue

                        new_content.append({
                            "type": "video_url",
                            "video_url": {"url": f"data:video/jpeg;base64,{','.join(b64_frames)}"}
                        })
                        
                        # 清理大对象
                        del frames, b64_frames, video_inputs
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"处理视频时发生错误: {e}")
                        logger.error(f"跳过有问题的视频处理")
                        continue
                        
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

    json_lock = threading.Lock()
    with json_lock:
        with open(os.path.join(base_dir, "video_description.json"), "r", encoding="utf-8") as f:
            video_data = json.load(f)

    for v_k, v_info in video_data.items():
        if v_info['video_path'] is not None and v_info['analyse_result'] is None and (video_queue.qsize() < 5):
            video_queue.put(v_k)
        elif video_queue.qsize() >= 5:
            break

    logging.info(f"Init v_q put {video_queue.qsize()}")

    video_analyse_server = VideoAnalyServer(video_queue, "qwen2.5", "http://localhost:8000/v1", "token-abc123", "/root/data1/monitor_rag_project/video_process/video_description.json")
    video_analyse_server.start()
    video_analyse_server.join()



        
