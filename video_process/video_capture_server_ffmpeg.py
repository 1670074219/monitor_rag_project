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
import signal
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

class VideoCaptureServerFFmpeg:
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
        self.recording_processes = {}  # 存储录制进程

    def get_video_resolution(self, camera_id: str, camera_url: str):
        try:
            info = ffmpeg.probe(camera_url)
            video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            logger.info(f"{camera_id}_width:{width}_height:{height}")
            return width, height
        except Exception as e:
            logger.error(f"获取{camera_id}分辨率失败: {e}")
            return 640, 480  # 默认分辨率

    def start_ffmpeg_recording(self, camera_id: str, camera_url: str, output_file: str):
        """启动ffmpeg录制进程，转换为H.264编码"""
        try:
            cmd = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', camera_url,
                '-c:v', 'libx264',       # 强制转换为H.264编码
                '-preset', 'fast',       # 快速编码，适合实时
                '-crf', '23',            # 质量设置
                '-c:a', 'aac',           # 如果有音频，使用aac编码
                '-movflags', '+faststart', # 优化网络播放
                '-f', 'mp4',
                '-y',                    # 覆盖输出文件
                output_file
            ]
            
            logger.info(f"启动ffmpeg录制: {output_file}")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            self.recording_processes[camera_id] = {
                'process': process,
                'output_file': output_file,
                'start_time': time.time()
            }
            
            return True
        except Exception as e:
            logger.error(f"启动ffmpeg录制失败: {e}")
            return False

    def stop_ffmpeg_recording(self, camera_id: str):
        """停止ffmpeg录制"""
        if camera_id not in self.recording_processes:
            return False
        
        try:
            record_info = self.recording_processes[camera_id]
            process = record_info['process']
            output_file = record_info['output_file']
            
            # 发送'q'命令给ffmpeg优雅退出
            try:
                process.stdin.write(b'q')
                process.stdin.flush()
            except:
                pass
            
            # 等待进程结束，最多等待5秒
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # 检查输出文件
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"录制完成: {output_file}")
                
                # 更新JSON文件
                self.update_video_description(camera_id, output_file)
                return True
            else:
                logger.warning(f"录制文件无效: {output_file}")
                return False
        
        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            return False
        finally:
            # 清理记录
            if camera_id in self.recording_processes:
                del self.recording_processes[camera_id]

    def update_video_description(self, camera_id: str, output_file: str):
        """更新视频描述JSON文件"""
        try:
            # 从文件名提取时间戳
            filename = os.path.basename(output_file)
            timestamp_part = filename.replace(f"{camera_id}_", "").replace(".mp4", "")
            
            with self.json_lock:
                if self.video_description_path and os.path.exists(self.video_description_path):
                    with open(self.video_description_path, "r", encoding="utf-8") as f:
                        video_data = json.load(f)
                else:
                    video_data = {}

                v_k = f"{camera_id}_{timestamp_part}"
                video_data[v_k] = {
                    "video_path": output_file,
                    "analyse_result": None,
                    "is_embedding": False,
                    "idx": None
                }

                with open(self.video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, indent=4, ensure_ascii=False)
                logger.info(f"已更新{v_k}json:video_path")
        
                if self.video_queue is not None:
                    self.video_queue.put(v_k)
                    logger.info(f"已放入视频分析队列:{v_k}")
        except Exception as e:
            logger.error(f"更新视频描述失败: {e}")

    def ffmpeg_frame_reader(self, camera_id: str, camera_url: str, width: int, height: int):
        """读取RTSP流用于人员检测"""
        while True:
            command = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', camera_url,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-vcodec', 'rawvideo',
                '-'
            ]

            pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
            while True:
                raw_image = pipe.stdout.read(width * height * 3)
                if not raw_image:
                    logger.error(f"{camera_id} 帧丢失, 尝试重连...")
                    pipe.kill()
                    time.sleep(5)
                    break
                try:
                    frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
                except:
                    logger.error(f"{camera_id} reshape失败")
                    continue
                yield frame

    def monitor_camera(self, camera_id: str, camera_url: str):
        """监控摄像头并控制录制"""
        # --- 状态变量 ---
        people_detected_frames = 0
        no_person_frames = 0
        is_recording = False
        width, height = self.get_video_resolution(camera_id, camera_url)
        
        logger.info(f"开始监控摄像头: {camera_id}")
        
        for frame in self.ffmpeg_frame_reader(camera_id, camera_url, width, height):
            try:
                # YOLO人员检测
                results = self.yolo_model(frame, verbose=False)
                if results:
                    classes = results[0].boxes.cls.cpu().numpy()
                    has_person = 0 in classes  # YOLO 类别 0 是人
                else:
                    has_person = False

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
                    
                    if self.start_ffmpeg_recording(camera_id, camera_url, filename):
                        is_recording = True
                        logger.info(f"开始录制: {filename}")

                # --- 停止录制 ---
                if no_person_frames >= self.no_person_frames_threshold and is_recording:
                    if self.stop_ffmpeg_recording(camera_id):
                        logger.info(f"停止录制完成")
                    is_recording = False

            except Exception as e:
                logger.error(f"处理帧时出错: {e}")
                continue

    def run_all_cameras(self):
        """启动所有摄像头监控"""
        threads = []

        for cam in self.camera_config:
            camera_id = cam['camera_id']
            camera_url = cam['camera_url']

            # 对每个摄像头启动一个线程
            thread = threading.Thread(
                target=self.monitor_camera,
                args=(camera_id, camera_url),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            logger.info(f"线程已启动: 摄像头 {camera_id}")
        
        # 注意：信号处理器只能在主线程中设置，这里不设置
        # 如果需要信号处理，应该在main.py的主线程中设置
        
        # 等待所有线程
        for thread in threads:
            thread.join()
            
    def cleanup(self):
        """清理所有录制进程"""
        logger.info("正在停止所有录制...")
        for camera_id in list(self.recording_processes.keys()):
            self.stop_ffmpeg_recording(camera_id)

if __name__ == "__main__":
    with open(os.path.join(base_dir, "camera_config.json"), "r") as f:
        camera_config = json.load(f)

    video_queue = Queue()
    video_capture_server = VideoCaptureServerFFmpeg(
        camera_config["camera_config"],
        os.path.join(base_dir, "yolo/yolo11s.pt"),
        os.path.join(base_dir, "saved_video"),
        video_queue=video_queue,
        video_description_path=os.path.join(base_dir, "video_description.json")
    )
    
    video_capture_server.run_all_cameras() 