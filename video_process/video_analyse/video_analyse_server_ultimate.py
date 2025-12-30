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

class UltimateVideoAnalyServer(threading.Thread):
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
        
        # 🔥 关键改进：只在内存中保存需要处理的视频信息
        self.processing_videos = {}  # 只保存正在处理的视频信息
        
        # 视频处理参数
        self.max_frames = 2  # 极限：只用2帧
        self.frame_quality = 25  # 极低质量
        self.frame_resize_factor = 6  # 尺寸压缩到1/6

    def load_processing_videos(self, video_keys: list):
        """只加载需要处理的视频信息到内存"""
        with self.json_lock:
            try:
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    all_video_data = json.load(f)
                
                # 只保存需要处理的视频信息
                for v_k in video_keys:
                    if v_k in all_video_data:
                        self.processing_videos[v_k] = all_video_data[v_k].copy()
                
                # 立即清理大JSON数据
                del all_video_data
                gc.collect()
                
                logger.info(f"已加载 {len(self.processing_videos)} 个视频信息到内存")
                
            except Exception as e:
                logger.error(f"加载视频信息失败: {e}")

    def run(self):
        while True:
            try:
                v_k = self.video_queue.get()
                logger.info(f"开始分析: {v_k}")
                
                # 强制清理内存
                self._force_memory_cleanup()
                
                analyse_result = self.analyze_video(v_k)
                logging.info(f"分析完成: {v_k}")
                
                if analyse_result is not None:
                    self.save_result_efficiently(v_k, analyse_result)
                
                # 清理已处理的视频信息
                if v_k in self.processing_videos:
                    del self.processing_videos[v_k]
                
                self._force_memory_cleanup()
                
            except Empty:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理视频 {v_k} 时发生错误: {e}")
                logger.debug(traceback.format_exc())
                self._force_memory_cleanup()
                continue

    def _force_memory_cleanup(self):
        """强制清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze_video(self, v_k: str):
        """内存优化的视频分析 - 不读取JSON文件"""
        camera_id_part, timestamp_part = v_k.split("_", 1)
        template = f"""监控视频分析：

                    简要分析视频内容：
                    1. 人数
                    2. 外貌特征 
                    3. 行为动作
                    4. 环境交互

                    格式:
                    人数：[数字]
                    外貌：[简述]
                    行为：[简述] 
                    交互：[简述]
                    总结：摄像头{camera_id_part}在{timestamp_part}记录到[情况]
                    """
        
        # 从内存中获取视频信息，避免读取JSON文件
        video_info = self.processing_videos.get(v_k)
        if not video_info:
            logger.error(f"内存中未找到视频信息: {v_k}")
            return None

        try:
            processed_messages = self._create_ultra_optimized_messages(template, video_info['video_path'])
            
            if not processed_messages:
                logger.error(f"视频处理失败: {v_k}")
                return None

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                max_tokens=512,  # 进一步减少输出
                messages=processed_messages
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"模型调用错误: {e}")
            return None

    def _create_ultra_optimized_messages(self, template: str, video_path: str):
        """创建超级优化的消息，最小化内存使用"""
        try:
            video_message = [{"content": [{"type": "video", "video": video_path}]}]
            
            _, video_inputs, video_kwargs = process_vision_info(
                video_message, 
                return_video_kwargs=True
            )
            
            if not video_inputs:
                logger.error("视频处理失败")
                return None
                
            # 立即获取并删除tensor
            frames_tensor = video_inputs.pop()
            del video_inputs
            
            frames = frames_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
            del frames_tensor
            
            # 超级采样：只取2帧
            total_frames = len(frames)
            if total_frames > self.max_frames:
                # 取首尾帧
                if total_frames >= 2:
                    frames = frames[[0, total_frames-1]]
                else:
                    frames = frames[:1]
            
            logger.info(f"处理帧数: {len(frames)} (原始: {total_frames})")
            
            # 超级压缩
            compressed_frames = []
            for i, frame in enumerate(frames):
                try:
                    img = Image.fromarray(frame)
                    
                    # 超级压缩
                    original_size = img.size
                    new_size = (
                        max(24, original_size[0] // self.frame_resize_factor), 
                        max(24, original_size[1] // self.frame_resize_factor)
                    )
                    img = img.resize(new_size, Image.NEAREST)  # 使用最快的插值
                    
                    with BytesIO() as buf:
                        img.save(buf, format="jpeg", quality=self.frame_quality, optimize=True)
                        img_data = buf.getvalue()
                        
                    compressed_frames.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
                        }
                    })
                    
                    del img, img_data
                    
                except Exception as e:
                    logger.error(f"处理帧 {i} 失败: {e}")
                    continue
            
            del frames
            self._force_memory_cleanup()
            
            if not compressed_frames:
                logger.error("没有成功处理任何帧")
                return None
            
            messages = [
                {"role": "system", "content": "你是监控视频分析专家"},
                {"role": "user", "content": [
                    {"type": "text", "text": template}
                ] + compressed_frames}
            ]
            
            logger.info(f"创建消息包含 {len(compressed_frames)} 个超压缩帧")
            return messages
            
        except Exception as e:
            logger.error(f"创建消息失败: {e}")
            return None

    def save_result_efficiently(self, v_k: str, analyse_result: str):
        """高效保存结果 - 只更新特定条目"""
        with self.json_lock:
            try:
                # 读取JSON
                with open(self.video_description_path, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                # 只更新这一个条目
                if v_k in video_data:
                    video_data[v_k]['analyse_result'] = analyse_result
                    
                    # 立即写回
                    with open(self.video_description_path, "w", encoding="utf-8") as f:
                        json.dump(video_data, f, ensure_ascii=False, indent=4)
                    
                    logger.info(f"已保存{v_k}的分析结果")
                    
                    if self.video_dsp_queue is not None:
                        self.video_dsp_queue.put(v_k)
                
                # 立即清理JSON数据
                del video_data
                gc.collect()
                    
            except Exception as e:
                logger.error(f"保存结果失败 {v_k}: {e}")

def create_lightweight_processor(json_path: str, max_videos: int = 2):
    """创建轻量级处理器，只处理少量视频"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        
        # 只获取少量未处理视频
        unprocessed = []
        for v_k, v_info in video_data.items():
            if (v_info.get('video_path') is not None and 
                v_info.get('analyse_result') is None and 
                len(unprocessed) < max_videos):
                unprocessed.append(v_k)
        
        # 立即清理大JSON
        del video_data
        gc.collect()
        
        logger.info(f"选择处理 {len(unprocessed)} 个视频")
        return unprocessed
        
    except Exception as e:
        logger.error(f"创建处理器失败: {e}")
        return []

if __name__ == "__main__":
    # 设置极限内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16'
    
    json_path = os.path.join(base_dir, "video_description.json")
    
    # 只处理2个视频
    unprocessed_videos = create_lightweight_processor(json_path, max_videos=2)
    
    if not unprocessed_videos:
        logger.info("没有视频需要处理")
        exit(0)
    
    # 创建队列
    video_queue = Queue()
    for v_k in unprocessed_videos:
        video_queue.put(v_k)

    logger.info(f"启动超级内存优化版本，处理 {video_queue.qsize()} 个视频")

    # 创建服务器
    server = UltimateVideoAnalyServer(
        video_queue, 
        "qwen2.5", 
        "http://localhost:8000/v1", 
        "token-abc123", 
        json_path
    )
    
    # 预加载需要处理的视频信息
    server.load_processing_videos(unprocessed_videos)
    
    # 启动处理
    server.start()
    server.join()
    
    logger.info("🎉 处理完成，内存使用最优化!") 