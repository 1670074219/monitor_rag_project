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
import gc
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

class MemoryOptimizedVideoAnalyServer(threading.Thread):
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
        
        # 强制内存管理
        self.max_frames = 3  # 极限压缩：只用3帧
        self.frame_quality = 30  # 极低质量
        self.frame_resize_factor = 4  # 尺寸压缩到1/4

    def run(self):
        while True:
            try:
                v_k = self.video_queue.get()
                logger.info(f"开始分析: {v_k}")
                
                # 立即清理内存
                self._force_memory_cleanup()
                
                analyse_result = self.analyze_video(v_k)
                logging.info(f"分析完成: {v_k}")
                
                if analyse_result is not None:
                    self.save_to_json(v_k, analyse_result)
                
                # 处理完成后立即清理
                self._force_memory_cleanup()
                
            except Empty:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理视频 {v_k} 时发生错误: {e}")
                logger.debug(traceback.format_exc())
                # 出错时也要清理内存
                self._force_memory_cleanup()
                continue

    def _force_memory_cleanup(self):
        """强制清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def get_video_info(self, v_k: str):
        """只获取单个视频的信息"""
        with self.json_lock:
            with open(self.video_description_path, "r", encoding="utf-8") as f:
                video_data = json.load(f)
            return video_data.get(v_k)

    def analyze_video(self, v_k: str):
        """内存优化的视频分析"""
        camera_id_part, timestamp_part = v_k.split("_", 1)
        template = f"""你是一个专业的监控视频分析专家，请简洁分析以下视频内容：

                    1. 视频中有几个人？
                    2. 每个人的外貌特征(衣着、体态等)
                    3. 每个人在做什么？
                    4. 与环境的交互情况

                    输出格式:
                    人数：[数字]
                    外貌特征：[简要描述]  
                    行为动作：[简要描述]
                    环境交互：[简要描述]
                    视频总结：摄像头{camera_id_part}在{timestamp_part}监控到[具体情况简述]
                    """
        
        video_info = self.get_video_info(v_k)
        if not video_info:
            logger.error(f"无法找到视频信息: {v_k}")
            return None

        try:
            # 内存优化的消息处理
            processed_messages = self._create_optimized_messages(template, video_info['video_path'])
            
            if not processed_messages:
                logger.error(f"视频处理失败: {v_k}")
                return None

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                max_tokens=1024,  # 减少输出长度
                messages=processed_messages
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"模型调用错误: {e}")
            return None

    def _create_optimized_messages(self, template: str, video_path: str):
        """创建内存优化的消息，避免巨大的base64字符串"""
        try:
            # 第一步：处理视频，获取关键帧
            video_message = [{"content": [{"type": "video", "video": video_path}]}]
            
            # 使用更小的参数
            _, video_inputs, video_kwargs = process_vision_info(
                video_message, 
                return_video_kwargs=True
            )
            
            if not video_inputs:
                logger.error("视频处理失败，没有得到帧数据")
                return None
                
            # 获取tensor并立即删除video_inputs引用
            frames_tensor = video_inputs.pop()
            del video_inputs  # 立即删除
            
            # 转换为numpy（只取最少的帧）
            frames = frames_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
            del frames_tensor  # 立即删除tensor
            
            # 极限压缩：只取最多3帧
            total_frames = len(frames)
            if total_frames > self.max_frames:
                # 均匀采样
                indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
                frames = frames[indices]
            
            logger.info(f"处理视频帧数: {len(frames)} (原始: {total_frames})")
            
            # 转换为极小的base64字符串
            optimized_frames = []
            for i, frame in enumerate(frames):
                try:
                    img = Image.fromarray(frame)
                    
                    # 极限压缩尺寸
                    original_size = img.size
                    new_size = (
                        max(32, original_size[0] // self.frame_resize_factor), 
                        max(32, original_size[1] // self.frame_resize_factor)
                    )
                    img = img.resize(new_size, Image.LANCZOS)
                    
                    # 极限压缩质量
                    with BytesIO() as buf:
                        img.save(buf, format="jpeg", quality=self.frame_quality, optimize=True)
                        img_data = buf.getvalue()
                        
                    optimized_frames.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
                        }
                    })
                    
                    # 立即清理
                    del img, img_data
                    
                except Exception as e:
                    logger.error(f"处理帧 {i} 失败: {e}")
                    continue
            
            # 立即清理frames
            del frames
            self._force_memory_cleanup()
            
            if not optimized_frames:
                logger.error("没有成功处理任何帧")
                return None
            
            # 构建消息（避免巨大的data URL）
            messages = [
                {"role": "system", "content": "你是一个专业的监控视频分析专家"},
                {"role": "user", "content": [
                    {"type": "text", "text": template}
                ] + optimized_frames}
            ]
            
            logger.info(f"成功创建消息，包含 {len(optimized_frames)} 个压缩帧")
            return messages
            
        except Exception as e:
            logger.error(f"创建优化消息失败: {e}")
            logger.debug(traceback.format_exc())
            return None

    def save_to_json(self, v_k: str, analyse_result: str):
        """保存分析结果"""
        with self.json_lock:
            try:
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                if v_k in video_data:
                    video_data[v_k]['analyse_result'] = analyse_result
                    
                    with open(self.video_description_path, "w", encoding="utf-8") as f:
                        json.dump(video_data, f, ensure_ascii=False, indent=4)
                    
                    logger.info(f"已更新{v_k}的分析结果")
                    
                    if self.video_dsp_queue is not None:
                        self.video_dsp_queue.put(v_k)
                        logger.info(f"已放入embedding队列:{v_k}")
                else:
                    logger.error(f"未找到视频记录: {v_k}")
                    
            except Exception as e:
                logger.error(f"保存分析结果失败 {v_k}: {e}")

def get_unprocessed_videos_safe(json_path: str, max_count: int = 3):
    """安全获取未处理视频列表，减少批次大小"""
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
    # 设置环境变量减少内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    video_queue = Queue()
    json_path = os.path.join(base_dir, "video_description.json")
    
    # 只处理3个视频，避免内存积累
    unprocessed_videos = get_unprocessed_videos_safe(json_path, max_count=3)
    
    for v_k in unprocessed_videos:
        video_queue.put(v_k)

    logging.info(f"初始化队列，放入 {video_queue.qsize()} 个视频 (安全模式)")

    # 使用内存优化版本
    video_analyse_server = MemoryOptimizedVideoAnalyServer(
        video_queue, 
        "qwen2.5", 
        "http://localhost:8000/v1", 
        "token-abc123", 
        json_path
    )
    video_analyse_server.start()
    video_analyse_server.join()
    
    logging.info("处理完成，程序退出") 