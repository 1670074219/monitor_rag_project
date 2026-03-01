"""使用YOLO检测并保存视频的监控系统 (优化防卡死) - 重构版: 旁路流复制录制"""
import cv2
import time
import queue
import ffmpeg
import threading
import numpy as np
import os
import subprocess
import json
import logging
import traceback 
from typing import Dict, Optional, IO, List
from collections import deque
from datetime import datetime
from ultralytics import YOLO
import torch

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_capture.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__) 

base_path = os.path.dirname(os.path.abspath(__file__))

# 配置参数
if not os.path.exists(os.path.join(base_path, "save_video")):
    os.makedirs(os.path.join(base_path, "save_video"))

SAVE_PATH = os.path.join(base_path, "save_video")
VIDEO_INFO_FILE = r"/root/data1/demo/Video_processing/pending_videos.json"
JSON_LOCK_FILE = r"/root/data1/demo/Video_processing/pending_videos.json.lock"
FFMPEG_LOG_FILE = "ffmpeg_errors.log"
ENABLE_VIDEO_SAVING = True

# --- RTSPStreamCapture 类定义 (负责拉流给 YOLO) ---
class RTSPStreamCapture:
    def __init__(self,
                 camera_config: Dict,
                 frame_queue: Optional[queue.Queue] = None,
                 ffmpeg_params: Optional[Dict] = None,
                 max_retries: int = 3,
                 color_convert: bool = True):
        self.camera_config = camera_config
        self.frame_queue = frame_queue or queue.Queue(maxsize=10)
        # 针对 YOLO 分析的拉流参数，可以降低分辨率或帧率以减少负载，这里保持原画质但只做解码
        self.ffmpeg_params = ffmpeg_params or {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "allowed_media_types": "video",
            "timeout": 5000000
        }
        self.max_retries = max_retries
        self.color_convert = color_convert
        self._running = False
        self._thread = None
        self._capture_process = None
        self.camera_name = self.camera_config.get('name', self.camera_config.get('ip', 'UnknownCam'))

        self.rtsp_url = (
            f"rtsp://{self.camera_config['user']}:{self.camera_config['password']}"
            f"@{self.camera_config['ip']}/cam/realmonitor?"
            f"channel={self.camera_config['channel']}&subtype=0"
        )

        self.video_info = self._get_video_info()

    def _get_video_info(self):
        """获取视频流信息"""
        logger.info(f"[{self.camera_name}] 正在获取摄像头视频信息...")
        try:
            probe_params = {
                "rtsp_transport": "tcp",
                "stimeout": "5000000"
            }
            probe = ffmpeg.probe(self.rtsp_url, **probe_params)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            fps_str = video_info.get('r_frame_rate', video_info.get('avg_frame_rate', '0/1'))
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)

            if fps <= 0:
                logger.warning(f"[{self.camera_name}] 无法获取有效帧率, 使用默认值 25")
                fps = 25.0
            
            width = int(video_info['width'])
            height = int(video_info['height'])
            logger.info(f"[{self.camera_name}] 视频信息: {width}x{height} @ {fps:.2f} FPS")
            return {"fps": float(fps), "width": width, "height": height}
        except Exception as e:
            logger.error(f"[{self.camera_name}] 无法解析视频流信息: {str(e)}")
            raise RuntimeError(f"无法解析视频流信息: {str(e)}")

    def _capture_loop(self):
        """拉流主循环 - 仅负责获取画面供 YOLO 分析"""
        current_retry_attempt = 0
        logger.info(f"[{self.camera_name}] 分析流捕获线程启动")
        
        while self._running:
            try:
                # 启动 FFmpeg 解码流到 pipe
                self._capture_process = (
                    ffmpeg
                    .input(self.rtsp_url, **self.ffmpeg_params)
                    .output('pipe:', format='rawvideo',
                           pix_fmt='rgb24' if self.color_convert else 'bgr24',
                           vsync=0, loglevel="quiet")
                    .run_async(pipe_stdout=True, pipe_stderr=True)
                )

                # 异步处理 stderr 避免 buffer 满阻塞
                stderr_thread = threading.Thread(target=self._log_ffmpeg_stderr, args=(self._capture_process.stderr,))
                stderr_thread.daemon = True
                stderr_thread.start()

                logger.info(f"[{self.camera_name}] 分析流 FFmpeg 已启动 (PID: {self._capture_process.pid})")
                frame_size = self.video_info['width'] * self.video_info['height'] * 3
                current_retry_attempt = 0 

                while self._running:
                    in_bytes = self._capture_process.stdout.read(frame_size)
                    if not in_bytes:
                        break

                    frame = np.frombuffer(in_bytes, np.uint8).reshape(
                        (self.video_info['height'], self.video_info['width'], 3)
                    )

                    if self.color_convert:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 放入队列，满则丢弃旧帧 (保证实时性)
                    if self.frame_queue.full():
                        try: self.frame_queue.get_nowait()
                        except queue.Empty: pass
                    
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass

                self._cleanup_capture_process()
                if not self._running: break

                current_retry_attempt += 1
                wait_time = min(2 ** current_retry_attempt, 30)
                logger.warning(f"[{self.camera_name}] 分析流中断，{wait_time}秒后重试...")
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"[{self.camera_name}] 分析流错误: {e}")
                self._cleanup_capture_process()
                time.sleep(5)

        self._running = False
        logger.info(f"[{self.camera_name}] 分析流捕获线程结束")

    def _log_ffmpeg_stderr(self, stderr_pipe: IO):
        """记录 FFmpeg stderr"""
        try:
            for line in iter(lambda: stderr_pipe.readline(), b''):
                pass # 暂时忽略分析流的详细日志，避免刷屏，需要调试时打开
        except: pass
        finally:
            try: stderr_pipe.close()
            except: pass

    def _cleanup_capture_process(self):
        if self._capture_process:
            try:
                self._capture_process.terminate()
                self._capture_process.wait(timeout=2)
            except:
                try: self._capture_process.kill()
                except: pass
            self._capture_process = None

    def start(self):
        if not self._running:
            if not self.video_info: return False
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name=f"Cap-{self.camera_name}", daemon=True)
            self._thread.start()
            return True
        return False

    def stop(self):
        self._running = False
        self._cleanup_capture_process()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_frame(self):
        try: return self.frame_queue.get_nowait()
        except queue.Empty: return None


class RTSPMonitor:
    def __init__(self,
                 camera_config: dict,
                 model_path: str,
                 save_path: str = SAVE_PATH,
                 frame_buffer_size: int = 30, # 此 buffer 仅用于 YOLO 逻辑，不再用于录制
                 detection_confidence: float = 0.5,
                 consecutive_detections: int = 3,
                 consecutive_non_detections: int = 10,
                 process_every_n_frames: int = 2,
                 camera_id: str = None,
                 **kwargs):

        self.camera_config = camera_config
        self.camera_id = camera_id or camera_config.get('name', 'Cam')
        logger.info(f"[{self.camera_id}] 初始化 Monitor (方案B: 双进程旁路录制)...")

        self.stream = RTSPStreamCapture(camera_config, color_convert=True)
        
        logger.info(f"[{self.camera_id}] 加载 YOLO: {model_path}")
        self.model = YOLO(model_path)
        
        self.detection_confidence = detection_confidence
        self.running = False
        self.is_recording = False # 录制状态标志
        self.last_detection_time = 0
        
        # 状态控制
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.required_consecutive_detections = consecutive_detections
        self.required_consecutive_non_detections = consecutive_non_detections
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.frame_counter = 0

        self.save_path = save_path
        self.video_queue = kwargs.get('video_queue')
        self.video_description_path = kwargs.get('video_description_path')
        self.json_lock = kwargs.get('json_lock') or threading.Lock()

        # 录制控制
        self.record_stop_event = threading.Event()
        self.recorder_thread = None

        self._yolo_thread = None

    def _update_video_description(self, video_path: str):
        """录制完成后更新 JSON"""
        if not self.video_description_path or not os.path.exists(video_path): return
        try:
            filename = os.path.basename(video_path)
            # 文件名格式: {camera_id}_{timestamp}.mp4
            # 需要提取 timestamp 部分作为 key 的一部分
            if filename.startswith(f"{self.camera_id}_"):
                 timestamp_part = filename[len(self.camera_id)+1:].replace(".mp4", "")
            else:
                 timestamp_part = str(int(time.time()))

            v_k = f"{self.camera_id}_{timestamp_part}"
            
            with self.json_lock:
                video_data = {}
                if os.path.exists(self.video_description_path):
                    try:
                        with open(self.video_description_path, "r", encoding="utf-8") as f:
                            video_data = json.loads(f.read() or "{}")
                    except: pass
                
                video_data[v_k] = {
                    "video_path": os.path.abspath(video_path),
                    "analyse_result": None,
                    "is_embedding": False,
                    "idx": None
                }
                
                with open(self.video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, indent=4, ensure_ascii=False)
                
            if self.video_queue:
                self.video_queue.put(v_k)
                logger.info(f"[{self.camera_id}] 录制完成并入队: {v_k}")

        except Exception as e:
            logger.error(f"[{self.camera_id}] 更新描述文件失败: {e}")

    def _record_process(self, output_path, stop_event):
        """
        独立的录制进程函数
        直接使用 FFmpeg 从 RTSP 拉流并保存 (-c copy)
        """
        rtsp_url = (
            f"rtsp://{self.camera_config['user']}:{self.camera_config['password']}"
            f"@{self.camera_config['ip']}/cam/realmonitor?"
            f"channel={self.camera_config['channel']}&subtype=0"
        )
        
        # 关键命令解释：
        # -rtsp_transport tcp: 强制 TCP 保证图像完整
        # -i rtsp_url: 直接从网络拉流，不经过 Python
        # -c copy: 核心参数！直接复制流数据，不解码不编码，零 CPU 消耗
        # -f mp4: 封装格式
        cmd = [
            'ffmpeg', '-y',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-c', 'copy',
            '-f', 'mp4',
            '-movflags', '+faststart', # 优化 MP4 头部，便于网络播放
            output_path
        ]
        
        logger.info(f"[{self.camera_id}] 启动录制进程: {' '.join(cmd)}")
        
        process = None
        try:
            # 启动 FFmpeg 进程
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,  #以此保留向 ffmpeg 发送 'q' 停止指令的能力
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE # 捕获错误日志
            )
            
            # 循环等待停止信号
            while not stop_event.is_set():
                if process.poll() is not None:
                    logger.error(f"[{self.camera_id}] 录制进程意外退出 (Code: {process.returncode})")
                    break
                time.sleep(0.5)
            
            # 停止录制
            if process.poll() is None:
                logger.info(f"[{self.camera_id}] 发送停止信号给录制进程...")
                # 尝试优雅停止：向 stdin 发送 'q'
                try:
                    process.communicate(input=b'q', timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"[{self.camera_id}] 录制进程停止超时，强制终止")
                    process.kill()
                    process.wait()
            
            logger.info(f"[{self.camera_id}] 录制结束: {output_path}")
            
        except Exception as e:
            logger.error(f"[{self.camera_id}] 录制流程异常: {e}")
            if process: 
                try: process.kill() 
                except: pass
        
        # 检查文件是否有效生成
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            return True
        return False

    def start_recording(self):
        """启动后台录制线程"""
        if self.is_recording: return
        
        self.is_recording = True
        self.record_stop_event.clear()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.camera_id}_{timestamp}.mp4"
        filepath = os.path.join(self.save_path, filename)
        
        def run_recorder():
            success = self._record_process(filepath, self.record_stop_event)
            if success:
                self._update_video_description(filepath)
            self.is_recording = False

        self.recorder_thread = threading.Thread(
            target=run_recorder, 
            name=f"Rec-{self.camera_id}", 
            daemon=True
        )
        self.recorder_thread.start()

    def stop_recording(self):
        """通知后台线程停止录制"""
        if self.is_recording:
            logger.info(f"[{self.camera_id}] 触发停止录制...")
            self.record_stop_event.set()
            # 线程会自动退出并设置 is_recording = False

    def yolo_loop(self):
        """YOLO 分析循环"""
        logger.info(f"[{self.camera_id}] YOLO 线程启动")
        while self.running:
            frame = self.stream.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                continue

            # 推理
            try:
                results = self.model(frame, verbose=False, classes=[0]) # 0=person
                detected = False
                for r in results:
                    if r.boxes and len(r.boxes.conf) > 0:
                        if torch.any(r.boxes.conf > self.detection_confidence):
                            detected = True
                            break
                
                # 状态机逻辑
                if detected:
                    self.consecutive_detections += 1
                    self.consecutive_non_detections = 0
                else:
                    self.consecutive_detections = 0
                    self.consecutive_non_detections += 1

                # 触发逻辑
                if ENABLE_VIDEO_SAVING:
                    if not self.is_recording and self.consecutive_detections >= self.required_consecutive_detections:
                        logger.info(f"[{self.camera_id}] 检测到人 ({self.consecutive_detections} 帧)，启动录制")
                        self.start_recording()
                    
                    elif self.is_recording and self.consecutive_non_detections >= self.required_consecutive_non_detections:
                        logger.info(f"[{self.camera_id}] 人员消失 ({self.consecutive_non_detections} 帧)，停止录制")
                        self.stop_recording()

            except Exception as e:
                logger.error(f"[{self.camera_id}] YOLO 出错: {e}")
                time.sleep(1)

    def run(self):
        if self.running: return
        self.running = True
        self.stream.start()
        
        self._yolo_thread = threading.Thread(target=self.yolo_loop, name=f"Yolo-{self.camera_id}", daemon=True)
        self._yolo_thread.start()
        return True

    def stop(self):
        self.running = False
        self.stop_recording()
        self.stream.stop()
        if self._yolo_thread: self._yolo_thread.join(timeout=2)
        logger.info(f"[{self.camera_id}] 已停止")



class VideoCaptureManager:
    """
    用于管理多个摄像头监控器的主类，提供与 video_capture_server_ffmpeg.py 兼容的接口。
    """
    def __init__(self,
                 camera_config: List[Dict],
                 yolo_path: str,
                 saved_video_path: str,
                 people_detected_frames_threshold: int = 3,
                 no_person_frames_threshold: int = 10,
                 video_queue: Optional[queue.Queue] = None,
                 video_description_path: str = None):
        
        self.camera_configs = self._parse_camera_configs(camera_config)
        self.yolo_path = yolo_path
        self.saved_video_path = saved_video_path
        self.consecutive_detections = people_detected_frames_threshold
        self.consecutive_non_detections = no_person_frames_threshold
        self.video_queue = video_queue
        self.video_description_path = video_description_path
        
        self.monitors: List[RTSPMonitor] = []
        self.monitor_threads: List[threading.Thread] = []
        self.running = False
        self.json_lock = threading.Lock()

        try:
             os.makedirs(self.saved_video_path, exist_ok=True)
             logger.info(f"确保视频保存目录存在: {self.saved_video_path}")
        except Exception as e:
             logger.error(f"创建主保存目录 {self.saved_video_path} 失败: {e}")

    def _parse_camera_configs(self, configs_from_json: List[Dict]) -> List[Dict]:
        """解析来自JSON的配置，提取所需字段以适配RTSPMonitor"""
        import re
        parsed_configs = []
        for cam in configs_from_json:
            url = cam.get('camera_url')
            cam_id = cam.get('camera_id')
            if not url or not cam_id:
                logger.warning(f"跳过无效的摄像头配置: {cam}")
                continue

            match = re.match(r"rtsp://(.*?):(.*?)@([^/]+)/.*?channel=(\d+)", url)
            if match:
                user, password, ip, channel = match.groups()
                parsed_configs.append({
                    "ip": ip,
                    "user": user,
                    "password": password,
                    "channel": int(channel),
                    "name": cam.get('name', cam_id),
                    "custom_id": cam_id
                })
            else:
                logger.warning(f"无法从URL解析摄像头配置: {url}, 跳过 {cam_id}")
        return parsed_configs

    def run_all_cameras(self):
        """为每个摄像头启动监控线程"""
        if self.running:
            logger.warning("监控系统已在运行。")
            return
            
        logger.info("多摄像头监控应用程序启动 (VideoCaptureManager)")
        self.running = True

        for config in self.camera_configs:
            try:
                cam_id = config.get('custom_id')
                logger.info(f"正在初始化摄像头: {cam_id}")

                monitor_instance = RTSPMonitor(
                    camera_config=config,
                    model_path=self.yolo_path,
                    save_path=self.saved_video_path,
                    consecutive_detections=self.consecutive_detections,
                    consecutive_non_detections=self.consecutive_non_detections,
                    camera_id=cam_id,
                    video_queue=self.video_queue,
                    video_description_path=self.video_description_path,
                    json_lock=self.json_lock
                )

                thread = threading.Thread(target=monitor_instance.run, name=f"Monitor-{cam_id}")
                thread.daemon = True
                self.monitor_threads.append(thread)
                self.monitors.append(monitor_instance)
                thread.start()
                logger.info(f"摄像头 {cam_id} 的监控线程已启动")
                time.sleep(1)

            except Exception as e:
                cam_name = config.get('name', '未知')
                logger.critical(f"初始化或启动摄像头 {cam_name} 时发生严重错误: {e}")
                logger.debug(traceback.format_exc())
        
        if not self.monitor_threads:
             logger.critical("没有成功启动任何摄像头监控线程，程序退出。")
             self.running = False
             return

        logger.info(f"成功启动 {len(self.monitor_threads)} 个摄像头监控线程。按 Ctrl+C 停止。")
        try:
            while self.running:
                if not all(t.is_alive() for t in self.monitor_threads):
                    logger.warning("检测到有监控线程已停止，将退出程序。")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("捕获到Ctrl+C，开始清理...")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        if not self.running:
            return
            
        logger.info("正在停止所有摄像头监控...")
        self.running = False

        for monitor in self.monitors:
            if monitor.running:
                monitor.stop()

        for thread in self.monitor_threads:
            if thread.is_alive():
                thread.join(timeout=20)
        
        logger.info("所有监控已停止。")


# --- 主程序入口 ---
if __name__ == "__main__":
    
    camera_config_path = os.path.join(base_path, "camera_config.json")
    try:
        with open(camera_config_path, "r", encoding='utf-8') as f:
            config_data = json.load(f)
        all_camera_configs = config_data.get("camera_config", [])
        if not all_camera_configs:
            logger.error(f"摄像头配置文件 {camera_config_path} 为空或格式不正确。")
            exit(1)
    except FileNotFoundError:
        logger.error(f"找不到摄像头配置文件: {camera_config_path}")
        exit(1)
    except Exception as e:
        logger.error(f"读取或解析摄像头配置文件失败: {e}")
        exit(1)

    yolo_model_path = os.path.join(base_path, "yolo/yolo11s.pt")
    save_video_path = os.path.join(base_path, "saved_video")
    video_desc_path = os.path.join(base_path, "video_description.json")

    shared_video_queue = queue.Queue()

    video_manager = VideoCaptureManager(
        camera_config=all_camera_configs,
        yolo_path=yolo_model_path,
        saved_video_path=save_video_path,
        video_queue=shared_video_queue,
        video_description_path=video_desc_path
    )
    
    import signal
    def signal_handler(sig, frame):
        logger.info("接收到终止信号，正在清理...")
        video_manager.cleanup()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    video_manager.run_all_cameras()

    logger.info("应用程序主线程退出。")
