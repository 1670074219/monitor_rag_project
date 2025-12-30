import logging
import os
import json
from ultralytics import YOLO
import cv2
import threading
from datetime import datetime
import subprocess
import numpy as np
import ffmpeg
import time
from queue import Queue
import socket
import urllib.parse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

class VideoCaptureServer:
    def __init__(self, 
                 camera_config: list[dict],
                 yolo_path: str,
                 saved_video_path: str,
                 people_detected_frames_threshold: int = 10,
                 no_person_frames_threshold: int = 30,
                 video_queue: Queue = None,
                 video_description_path: str = None):
        self.camera_config = camera_config
        self.yolo_path = yolo_path
        self.saved_video_path = saved_video_path
        self.yolo_model = YOLO(self.yolo_path).to(device="cuda:2")
        self.people_detected_frames_threshold = people_detected_frames_threshold
        self.no_person_frames_threshold = no_person_frames_threshold
        self.video_queue = video_queue
        self.video_description_path = video_description_path
        self.json_lock = threading.Lock()

    def get_video_resolution(self, camear_id: str, camera_url: str):
        info = ffmpeg.probe(camera_url)
        video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        logging.info(f"{camear_id}_width:{width}_height:{height}")
        return width, height 

    def ffmpeg_frame_reader(self, camera_id: str, camera_url: str, width: int, height: int):
        consecutive_failures = 0
        max_consecutive_failures = 3
        base_retry_delay = 2
        
        while True:
            try:
                # 简化的ffmpeg命令参数
                command = [
                    'ffmpeg',
                    '-rtsp_transport', 'tcp',           # 使用TCP传输，更稳定
                    '-i', camera_url,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-'
                ]

                logger.info(f"{camera_id} 正在连接RTSP流...")
                pipe = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,  # 捕获错误信息而不是丢弃
                    bufsize=width * height * 3 * 5  # 增大缓冲区，可存储5帧
                )
                
                consecutive_failures = 0  # 成功连接后重置失败计数
                logger.info(f"{camera_id} RTSP流连接成功")
                
                frame_count = 0
                last_success_time = time.time()
                
                while True:
                    try:
                        # 设置读取超时
                        start_time = time.time()
                        raw_image = pipe.stdout.read(width * height * 3)
                        
                        if not raw_image:
                            # 检查进程状态
                            if pipe.poll() is not None:
                                # 进程已结束，读取错误信息
                                stderr_output = pipe.stderr.read().decode('utf-8', errors='ignore')
                                logger.error(f"{camera_id} ffmpeg进程异常结束: {stderr_output[-200:]}")  # 只显示最后200字符
                            else:
                                logger.error(f"{camera_id} 读取到空帧")
                            break
                        
                        # 检查数据完整性
                        if len(raw_image) != width * height * 3:
                            logger.warning(f"{camera_id} 帧数据不完整: 期望{width * height * 3}字节，实际{len(raw_image)}字节")
                            continue
                        
                        try:
                            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
                            frame_count += 1
                            last_success_time = time.time()
                            
                            # 每1000帧打印一次状态
                            if frame_count % 1000 == 0:
                                logger.info(f"{camera_id} 已成功处理 {frame_count} 帧")
                            
                            yield frame
                            
                        except ValueError as e:
                            logger.error(f"{camera_id} 帧重塑失败: {e}")
                            continue
                        
                        # 检查是否长时间没有收到帧
                        if time.time() - last_success_time > 30:  # 30秒无帧则重连
                            logger.warning(f"{camera_id} 30秒内未收到有效帧，主动重连")
                            break
                            
                    except Exception as e:
                        logger.error(f"{camera_id} 读取帧时发生异常: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"{camera_id} 启动ffmpeg时发生异常: {e}")
            
            finally:
                # 清理进程
                if 'pipe' in locals():
                    try:
                        pipe.stdout.close()
                        pipe.stderr.close()
                        pipe.terminate()
                        pipe.wait(timeout=5)
                    except:
                        try:
                            pipe.kill()
                        except:
                            pass
            
            # 增加连续失败计数
            consecutive_failures += 1
            
            # 计算重试延迟（指数退避）
            retry_delay = min(base_retry_delay * (2 ** min(consecutive_failures - 1, 4)), 60)
            
            if consecutive_failures <= max_consecutive_failures:
                logger.warning(f"{camera_id} 连接失败 ({consecutive_failures}/{max_consecutive_failures})，{retry_delay}秒后重试...")
            else:
                logger.error(f"{camera_id} 连续失败超过{max_consecutive_failures}次，{retry_delay}秒后继续尝试...")
            
            time.sleep(retry_delay)

    def record_video(self, camera_id: str, camera_url: str, video_queue: Queue = None, video_description_path: str = None):
        # 首先测试连接
        logger.info(f"{camera_id} 开始初始化，先进行连接测试...")
        if not self.test_rtsp_connection(camera_id, camera_url):
            logger.error(f"{camera_id} 初始连接测试失败，将在循环中继续尝试")
        
        # --- 状态变量 ---
        people_detected_frames = 0
        no_person_frames = 0
        is_recording = False
        video_writer = None
        width, height = self.get_video_resolution(camera_id, camera_url)
        
        for frame in self.ffmpeg_frame_reader(camera_id, camera_url, width, height):
            results = self.yolo_model(frame, verbose=False)
            if results:
                classes = results[0].boxes.cls.cpu().numpy()
                has_person = 0 in classes  # YOLO 类别 0 是人
            else:
                logger.error(f"Error: Failed to detect person from {camera_id}")
                continue

            # --- 连续帧判断 ---
            if has_person:
                people_detected_frames += 1
                no_person_frames = 0
            else:
                people_detected_frames = 0
                no_person_frames += 1

            # --- 开始录制 ---
            if people_detected_frames >= self.people_detected_frames_threshold and not is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.saved_video_path, f"{camera_id}_{timestamp}.mp4")
                # 尝试多种编码格式，优先使用浏览器兼容的H.264
                video_writer = None
                codecs_to_try = ['H264', 'X264', 'h264', 'avc1', 'mp4v']
                
                for codec in codecs_to_try:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        video_writer = cv2.VideoWriter(filename, fourcc, 25, (width, height))
                        if video_writer.isOpened():
                            logger.info(f"成功使用编码: {codec}")
                            break
                        else:
                            video_writer.release()
                    except Exception as e:
                        logger.warning(f"编码 {codec} 失败: {e}")
                        continue
                
                if video_writer is None or not video_writer.isOpened():
                    logger.error(f"所有编码方式都失败，跳过录制")
                    continue
                is_recording = True
                logger.info(f"开始录制: {filename}")

            # --- 正在录制中 ---
            if is_recording:
                video_writer.write(frame)

            # --- 停止录制 ---
            if no_person_frames >= self.no_person_frames_threshold and is_recording:
                is_recording = False
                video_writer.release()
                logger.info(f"停止录制: {filename}")

                with self.json_lock:
                    if video_description_path is not None:
                        with open(video_description_path, "r", encoding="utf-8") as f:
                            video_data = json.load(f)

                        v_k = f"{camera_id}_{timestamp}"
                        video_data[v_k] = {
                            "video_path": filename,
                            "analyse_result": None,
                            "is_embedding": False,
                            "idx": None
                        }

                        with open(video_description_path, "w", encoding="utf-8") as f:
                            json.dump(video_data, f, indent=4, ensure_ascii=False)
                        logger.info(f"已更新{v_k}json:video_path")
                
                        if video_queue is not None:
                            video_queue.put(v_k)
                            logging.info(f"已放入视频分析队列:{v_k}")
    
    def run_all_cameras(self):
        threads = []

        for cam in self.camera_config:
            camera_id = cam['camera_id']
            camera_url = cam['camera_url']

            # 对每个摄像头启动一个线程
            thread = threading.Thread(
                target=self.record_video,
                args=(camera_id, camera_url, self.video_queue, self.video_description_path),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            logger.info(f"线程已启动: 摄像头 {camera_id}")
        
        for thread in threads:
            thread.join()
        
    def test_rtsp_connection(self, camera_id: str, camera_url: str):
        """测试RTSP连接的可达性和响应性"""
        try:
            # 解析RTSP URL
            parsed = urllib.parse.urlparse(camera_url)
            host = parsed.hostname
            port = parsed.port or 554  # RTSP默认端口
            
            logger.info(f"{camera_id} 测试网络连接到 {host}:{port}")
            
            # 测试TCP连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            start_time = time.time()
            result = sock.connect_ex((host, port))
            connection_time = time.time() - start_time
            sock.close()
            
            if result == 0:
                logger.info(f"{camera_id} TCP连接成功，耗时 {connection_time:.2f}秒")
                
                # 使用ffprobe测试RTSP流
                return self.test_rtsp_stream(camera_id, camera_url)
            else:
                logger.error(f"{camera_id} TCP连接失败，错误码: {result}")
                return False
                
        except Exception as e:
            logger.error(f"{camera_id} 连接测试异常: {e}")
            return False
    
    def test_rtsp_stream(self, camera_id: str, camera_url: str):
        """使用ffprobe测试RTSP流"""
        try:
            logger.info(f"{camera_id} 测试RTSP流...")
            
            command = [
                'ffprobe',
                '-rtsp_transport', 'tcp',
                '-rw_timeout', '5000000',  # 5秒超时，单位微秒
                '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'csv=p=0',
                camera_url
            ]
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"{camera_id} RTSP流测试成功: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"{camera_id} RTSP流测试失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"{camera_id} RTSP流测试超时")
            return False
        except Exception as e:
            logger.error(f"{camera_id} RTSP流测试异常: {e}")
            return False

if __name__ == "__main__":
    with open(os.path.join(base_dir, "camera_config.json"), "r") as f:
        camera_config = json.load(f)

    video_queue = Queue()
    video_capture_server = VideoCaptureServer(camera_config["camera_config"],
                                             os.path.join(base_dir, "yolo/yolo11s.pt"),
                                             os.path.join(base_dir, "saved_video"),
                                             video_queue=video_queue,
                                             video_description_path=os.path.join(base_dir, "video_description.json"))
    
    video_capture_server.get_video_resolution(video_capture_server.camera_config[1]['camera_url'])
    video_capture_server.record_video(video_capture_server.camera_config[0]['camera_id'], video_capture_server.camera_config[0]['camera_url'])
    video_capture_server.run_all_cameras()
    
    