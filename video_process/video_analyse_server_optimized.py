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

class OptimizedVideoAnalyServer(threading.Thread):
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
        
        # 内存中缓存，只保存当前正在处理的视频信息
        self._video_cache = {}
        self._cache_size_limit = 10  # 最多缓存10个视频信息

    def run(self):
        while True:
            try:
                v_k = self.video_queue.get()
                logger.info(f"开始分析: {v_k}")
                analyse_result = self.analyze_video(v_k)
                logging.info(f"分析完成: {v_k}")
                if analyse_result is not None:
                    self.save_analysis_result(v_k, analyse_result)
                    # 清理缓存
                    self._cleanup_cache(v_k)
            except Empty:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理视频 {v_k} 时发生错误: {e}")
                logger.debug(traceback.format_exc())
                continue

    def _cleanup_cache(self, processed_key: str):
        """清理缓存，保持内存使用最小"""
        if processed_key in self._video_cache:
            del self._video_cache[processed_key]
        
        # 如果缓存过大，清理一半
        if len(self._video_cache) > self._cache_size_limit:
            keys_to_remove = list(self._video_cache.keys())[:len(self._video_cache)//2]
            for key in keys_to_remove:
                del self._video_cache[key]

    def get_video_info(self, v_k: str):
        """高效获取单个视频信息，使用缓存避免重复读取"""
        # 检查缓存
        if v_k in self._video_cache:
            return self._video_cache[v_k]
        
        # 只读取需要的视频信息
        try:
            with self.json_lock:
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    # TODO: 对于超大JSON文件，考虑使用ijson流式读取
                    video_data = json.load(f)
                
                video_info = video_data.get(v_k)
                if video_info:
                    # 缓存视频信息，但限制缓存大小
                    if len(self._video_cache) < self._cache_size_limit:
                        self._video_cache[v_k] = video_info
                
                return video_info
        except Exception as e:
            logger.error(f"读取视频信息失败 {v_k}: {e}")
            return None

    def analyze_video(self, v_k: str):
        """分析视频，优化内存使用"""
        camera_id_part, timestamp_part = v_k.split("_", 1)
        template = f"""你是一个专业的监控视频分析专家，你的分析应该是以人为主体的，环境信息是你分析人的行为时的参考。
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
        
        video_info = self.get_video_info(v_k)
        if not video_info:
            logger.error(f"无法找到视频信息: {v_k}")
            return None

        messages = [
            {"role": "system", "content": "你是一个专业的监控视频分析专家，你的分析应该是以人为主体的，环境信息是你分析人的行为时的参考"},
            {"role": "user", "content": [
                {"type": "text", "text": template},
                {
                    "type": "video",
                    "video": video_info['video_path'],
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
            logger.error(f"模型调用错误: {e}")
            return None

    def prepare_message_for_vllm(self, messages):
        """优化视频处理，最小化内存占用"""
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

                        # 激进的内存优化
                        max_frames = min(len(frames), 4)  # 只取4帧
                        # 每隔n帧取一帧，确保代表性
                        if len(frames) > max_frames:
                            step = len(frames) // max_frames
                            frames = frames[::step][:max_frames]
                        
                        logger.info(f"处理视频帧数: {len(frames)}")

                        b64_frames = []
                        for i, frame in enumerate(frames):
                            try:
                                img = Image.fromarray(frame)
                                # 更激进的压缩
                                original_size = img.size
                                new_size = (original_size[0]//3, original_size[1]//3)  # 压缩到1/3
                                img = img.resize(new_size, Image.LANCZOS)
                                
                                with BytesIO() as buf:
                                    img.save(buf, format="jpeg", quality=40, optimize=True)
                                    b64_frames.append(base64.b64encode(buf.getvalue()).decode())
                                
                                # 立即清理
                                del img
                                gc.collect()
                                    
                            except Exception as e:
                                logger.error(f"处理帧 {i} 失败: {e}")
                                continue

                        new_content.append({
                            "type": "video_url",
                            "video_url": {"url": f"data:video/jpeg;base64,{','.join(b64_frames)}"}
                        })
                        
                        # 清理所有大对象
                        del frames, b64_frames, video_inputs
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"处理视频时发生错误: {e}")
                        continue
                        
                else:
                    new_content.append(part)
            
            message["content"] = new_content
            vllm_messages.append(message)

        return vllm_messages, {"fps": fps_list}

    def save_analysis_result(self, v_k: str, analyse_result: str):
        """高效保存分析结果，最小化JSON读写"""
        with self.json_lock:
            try:
                # 读取现有数据
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                # 更新特定条目
                if v_k in video_data:
                    video_data[v_k]['analyse_result'] = analyse_result
                    
                    # 写回文件
                    with open(self.video_description_path, "w", encoding="utf-8") as f:
                        json.dump(video_data, f, ensure_ascii=False, indent=4)
                    
                    logger.info(f"已更新{v_k}的分析结果")
                    
                    # 放入下一个处理队列
                    if self.video_dsp_queue is not None:
                        self.video_dsp_queue.put(v_k)
                        logger.info(f"已放入embedding队列:{v_k}")
                else:
                    logger.error(f"未找到视频记录: {v_k}")
                    
            except Exception as e:
                logger.error(f"保存分析结果失败 {v_k}: {e}")

def get_unprocessed_videos_efficient(json_path: str, max_count: int = 5):
    """高效获取未处理视频列表"""
    unprocessed_videos = []
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        
        count = 0
        for v_k, v_info in video_data.items():
            if (v_info.get('video_path') is not None and 
                v_info.get('analyse_result') is None and 
                count < max_count):
                unprocessed_videos.append(v_k)
                count += 1
            elif count >= max_count:
                break
        
        logger.info(f"找到 {len(unprocessed_videos)} 个未处理的视频")
        
    except Exception as e:
        logger.error(f"读取视频列表失败: {e}")
    
    return unprocessed_videos

if __name__ == "__main__":
    video_queue = Queue()
    json_path = os.path.join(base_dir, "video_description.json")
    
    # 获取未处理视频列表
    unprocessed_videos = get_unprocessed_videos_efficient(json_path, max_count=5)
    
    for v_k in unprocessed_videos:
        video_queue.put(v_k)

    logging.info(f"初始化队列，放入 {video_queue.qsize()} 个视频")

    # 使用优化版本的服务器
    video_analyse_server = OptimizedVideoAnalyServer(
        video_queue, 
        "qwen2.5", 
        "http://localhost:8000/v1", 
        "token-abc123", 
        json_path
    )
    video_analyse_server.start()
    video_analyse_server.join() 