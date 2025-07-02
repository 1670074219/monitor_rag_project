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
import mysql.connector
from mysql.connector import Error
import cv2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.151',
    'port': 3306,
    'database': 'monitor_rag',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
    'autocommit': False,
    'pool_name': 'video_analysis_pool',
    'pool_size': 5
}

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.db_lock = threading.Lock()
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
            return self.connection
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return None
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")

def get_video_duration(video_path: str) -> float:
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频时长（秒），如果获取失败返回-1
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return -1
        
        # 获取总帧数和帧率
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            logger.info(f"视频 {os.path.basename(video_path)} 时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
            return duration
        else:
            logger.error(f"无法获取视频帧率: {video_path}")
            return -1
            
    except Exception as e:
        logger.error(f"获取视频时长失败 {video_path}: {e}")
        return -1

class VideoAnalyServerMySQL(threading.Thread):
    def __init__(self,
                 video_queue: Queue,
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 video_dsp_queue: Queue = None):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.openai_client = openai.OpenAI(api_key=api_key, base_url=api_url)
        self.video_dsp_queue = video_dsp_queue
        self.db_manager = DatabaseManager(DB_CONFIG)

    def run(self):
        while True:
            try:
                v_k = self.video_queue.get()
                logger.info(f"开始分析: {v_k}")
                analyse_result = self.analyze_video(v_k)
                logging.info(f"分析完成: {v_k}")
                if analyse_result is not None:
                    self.save_to_database(v_k, analyse_result)
            except Empty:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理视频 {v_k} 时发生错误: {e}")
                logger.debug(traceback.format_exc())
        
    def _handle_problematic_video(self, v_k: str):
        """处理有问题的视频：删除文件和数据库记录"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return

                cursor = connection.cursor()
                
                # 1. 获取视频信息
                select_sql = "SELECT video_path FROM v_dsp WHERE v_name = %s"
                cursor.execute(select_sql, (v_k,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"无法在数据库中找到视频记录: {v_k}")
                    cursor.close()
                    return

                video_path = result[0]
                
                # 2. 删除视频文件
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        logger.info(f"已删除有问题的视频文件: {video_path}")
                    except OSError as e:
                        logger.error(f"删除视频文件 {video_path} 失败: {e}")
                
                # 3. 删除数据库记录
                delete_sql = "DELETE FROM v_dsp WHERE v_name = %s"
                cursor.execute(delete_sql, (v_k,))
                connection.commit()
                cursor.close()
                
                logger.info(f"已从数据库中删除视频记录: {v_k}")

            except Error as e:
                logger.error(f"处理有问题的视频 {v_k} 时数据库操作失败: {e}")
                if connection:
                    connection.rollback()
            except Exception as e:
                logger.error(f"处理有问题的视频 {v_k} 时发生未知错误: {e}")
                logger.debug(traceback.format_exc())

    def get_video_info(self, v_k: str):
        """从数据库获取单个视频的信息"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return None

                cursor = connection.cursor(dictionary=True)
                select_sql = "SELECT * FROM v_dsp WHERE v_name = %s"
                cursor.execute(select_sql, (v_k,))
                result = cursor.fetchone()
                cursor.close()
                
                return result
                
            except Error as e:
                logger.error(f"获取视频信息失败: {e}")
                return None

    def mark_video_as_skipped(self, v_k: str, reason: str):
        """将视频标记为跳过分析"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return False

                cursor = connection.cursor()
                
                # 更新分析结果为跳过原因
                skip_message = f"跳过分析: {reason}"
                update_sql = """
                UPDATE v_dsp 
                SET analyse_result = %s 
                WHERE v_name = %s
                """
                cursor.execute(update_sql, (skip_message, v_k))
                connection.commit()
                cursor.close()
                
                logger.info(f"已标记视频 {v_k} 为跳过: {reason}")
                return True
                
            except Error as e:
                logger.error(f"标记跳过视频失败: {e}")
                if connection:
                    connection.rollback()
                return False

    def analyze_video(self, v_k: str):
        camera_id_part, timestamp_part = v_k.split("_", 1)
        
        # 从数据库获取当前视频的信息
        video_info = self.get_video_info(v_k)
        if not video_info:
            logger.error(f"无法找到视频信息: {v_k}")
            return None

        video_path = video_info['video_path']
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            self.mark_video_as_skipped(v_k, "视频文件不存在")
            return None
        
        # 检查视频时长
        duration = get_video_duration(video_path)
        if duration == -1:
            logger.error(f"无法获取视频时长: {video_path}")
            self.mark_video_as_skipped(v_k, "无法获取视频时长")
            return None
        
        # 跳过超过10分钟的视频
        MAX_DURATION = 600  # 10分钟 = 600秒
        if duration > MAX_DURATION:
            logger.warning(f"视频时长超过{MAX_DURATION/60}分钟，跳过分析: {v_k} (时长: {duration/60:.2f}分钟)")
            self.mark_video_as_skipped(v_k, f"视频时长超过{MAX_DURATION/60}分钟 (实际时长: {duration/60:.2f}分钟)")
            return None
        
        logger.info(f"视频时长检查通过，开始分析: {v_k} (时长: {duration:.2f}秒)")
        
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

        messages = [
            {"role": "system", "content": "你是一个专业的监控视频分析专家，你的分析应该是以人为主体的，环境信息是你分析人的行为时的参考"},
            {"role": "user", "content": [
                {"type": "text", "text": template},
                {
                    "type": "video",
                    "video": video_path,
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

                        # 内存优化：进一步限制帧数和质量
                        max_frames = min(len(frames), 6)  # 减少到6帧
                        frames = frames[:max_frames]
                        logger.info(f"处理视频帧数: {len(frames)}")

                        b64_frames = []
                        for i, frame in enumerate(frames):
                            try:
                                img = Image.fromarray(frame)
                                # 进一步压缩图片尺寸
                                img = img.resize((img.width//2, img.height//2), Image.LANCZOS)
                                with BytesIO() as buf:
                                    # 降低质量减少内存
                                    img.save(buf, format="jpeg", quality=50, optimize=True)
                                    b64_frames.append(base64.b64encode(buf.getvalue()).decode())
                                
                                # 每处理1帧清理一次内存
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

    def save_to_database(self, v_k: str, analyse_result: str):
        """将分析结果保存到数据库"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return False

                cursor = connection.cursor()
                
                # 更新分析结果
                update_sql = """
                UPDATE v_dsp 
                SET analyse_result = %s 
                WHERE v_name = %s
                """
                cursor.execute(update_sql, (analyse_result, v_k))
                connection.commit()
                cursor.close()
                
                logger.info(f"已更新数据库中{v_k}的分析结果")

                # 如果有embedding队列，将视频加入队列
                if self.video_dsp_queue is not None:
                    self.video_dsp_queue.put(v_k)
                    logger.info(f"已放入embedding队列:{v_k}")
                
                return True
                
            except Error as e:
                logger.error(f"保存分析结果到数据库失败: {e}")
                if connection:
                    connection.rollback()
                return False

    def __del__(self):
        """析构函数，关闭数据库连接"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close_connection()

def get_unprocessed_videos_from_db(max_count: int = 5):
    """从数据库获取未处理的视频列表"""
    db_manager = DatabaseManager(DB_CONFIG)
    unprocessed_videos = []
    
    try:
        connection = db_manager.get_connection()
        if not connection:
            logger.error("无法连接数据库")
            return unprocessed_videos

        cursor = connection.cursor()
        
        # 查询未分析的视频（analyse_result为空或NULL）
        select_sql = """
        SELECT v_name 
        FROM v_dsp 
        WHERE video_path IS NOT NULL 
        AND (analyse_result IS NULL OR analyse_result = '') 
        LIMIT %s
        """
        
        cursor.execute(select_sql, (max_count,))
        results = cursor.fetchall()
        
        unprocessed_videos = [row[0] for row in results]
        cursor.close()
        
        logger.info(f"从数据库找到 {len(unprocessed_videos)} 个未处理的视频")
        
    except Error as e:
        logger.error(f"从数据库获取未处理视频失败: {e}")
    finally:
        db_manager.close_connection()
    
    return unprocessed_videos

def get_database_stats():
    """获取数据库统计信息"""
    db_manager = DatabaseManager(DB_CONFIG)
    
    try:
        connection = db_manager.get_connection()
        if not connection:
            logger.error("无法连接数据库")
            return

        cursor = connection.cursor()
        
        # 统计总记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp")
        total_count = cursor.fetchone()[0]
        
        # 统计已分析的记录数（不包括跳过的）
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NOT NULL AND analyse_result != '' AND analyse_result NOT LIKE '跳过分析:%'")
        analyzed_count = cursor.fetchone()[0]
        
        # 统计跳过的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result LIKE '跳过分析:%'")
        skipped_count = cursor.fetchone()[0]
        
        # 统计真正未分析的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NULL OR analyse_result = ''")
        unanalyzed_count = cursor.fetchone()[0]
        
        # 统计不同跳过原因
        cursor.execute("""
            SELECT analyse_result, COUNT(*) as count 
            FROM v_dsp 
            WHERE analyse_result LIKE '跳过分析:%' 
            GROUP BY analyse_result 
            ORDER BY count DESC
        """)
        skip_reasons = cursor.fetchall()
        
        cursor.close()
        
        logger.info(f"📊 数据库统计信息:")
        logger.info(f"  总记录数: {total_count}")
        logger.info(f"  ✅ 已分析: {analyzed_count}")
        logger.info(f"  ⏸️ 跳过分析: {skipped_count}")
        logger.info(f"  ⏳ 待分析: {unanalyzed_count}")
        
        if total_count > 0:
            progress = (analyzed_count / total_count) * 100
            logger.info(f"  📈 分析进度: {progress:.2f}%")
        
        if skip_reasons:
            logger.info(f"  🚫 跳过原因统计:")
            for reason, count in skip_reasons:
                logger.info(f"    - {reason}: {count} 个视频")
        
    except Error as e:
        logger.error(f"获取数据库统计信息失败: {e}")
    finally:
        db_manager.close_connection()

if __name__ == "__main__":
    # 获取数据库统计信息
    get_database_stats()
    
    # 获取未处理的视频列表
    video_queue = Queue()
    unprocessed_videos = get_unprocessed_videos_from_db(max_count=10000)
    
    for v_k in unprocessed_videos:
        video_queue.put(v_k)

    logging.info(f"Init v_q put {video_queue.qsize()}")

    # 启动视频分析服务器
    video_analyse_server = VideoAnalyServerMySQL(
        video_queue, 
        "qwen2.5", 
        "http://localhost:8000/v1", 
        "token-abc123"
    )
    video_analyse_server.start()
    video_analyse_server.join() 