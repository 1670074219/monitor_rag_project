"""使用YOLO检测并保存视频的监控系统 (优化防卡死)"""
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
        logging.FileHandler("video_capture.log", encoding='utf-8') # 指定UTF-8编码
    ]
)
logger = logging.getLogger(__name__) 

base_path = os.path.dirname(os.path.abspath(__file__))

# 配置参数 (保持不变，所有摄像头共享)
if not os.path.exists(os.path.join(base_path, "save_video")):
    os.makedirs(os.path.join(base_path, "save_video"))

SAVE_PATH = os.path.join(base_path, "save_video")
VIDEO_INFO_FILE = r"/root/data1/demo/Video_processing/pending_videos.json" # 共享的 JSON 文件 G:/4.shixi/chufaqi/saveV1/pending_videos.json
JSON_LOCK_FILE = r"/root/data1/demo/Video_processing/pending_videos.json.lock"  # 文件锁路径
FFMPEG_LOG_FILE = "ffmpeg_errors.log" # FFmpeg 错误日志
ENABLE_VIDEO_SAVING = True # <<< 新增：全局控制是否保存视频

# --- RTSPStreamCapture 类定义 (保持不变) ---
class RTSPStreamCapture:
    def __init__(self,
                 camera_config: Dict,
                 frame_queue: Optional[queue.Queue] = None,
                 ffmpeg_params: Optional[Dict] = None,
                 max_retries: int = 3,
                 color_convert: bool = True):
        self.camera_config = camera_config
        self.frame_queue = frame_queue or queue.Queue(maxsize=10) # 限制队列大小
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
        self._capture_process = None # 持有ffmpeg拉流进程引用
        self.camera_name = self.camera_config.get('name', self.camera_config.get('ip', 'UnknownCam')) # 获取摄像头名称用于日志

        # 构建RTSP URL
        self.rtsp_url = (
            f"rtsp://{self.camera_config['user']}:{self.camera_config['password']}"
            f"@{self.camera_config['ip']}/cam/realmonitor?"
            f"channel={self.camera_config['channel']}&subtype=0"
        )

        print(self.rtsp_url)

        # 获取视频信息
        self.video_info = self._get_video_info()

    def _get_video_info(self):
        """获取视频流信息"""
        logger.info(f"[{self.camera_name}] 正在获取摄像头视频信息...")
        try:
            # 只传递 probe 支持的参数（不使用 timeout，避免被误解为 listen 模式）
            probe_params = {
                "rtsp_transport": "tcp",
                "stimeout": "5000000"  # 使用 stimeout (socket timeout) 代替 timeout，单位：微秒
            }
            print(self.rtsp_url)
            probe = ffmpeg.probe(self.rtsp_url, **probe_params) # 增加超时
            video_info = next(
                s for s in probe['streams'] if s['codec_type'] == 'video'
            )
            # 尝试不同方式获取fps
            fps_str = video_info.get('r_frame_rate', video_info.get('avg_frame_rate', '0/1'))
            if '/' in fps_str:
                num, den = map(float, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)

            if fps <= 0: # 处理无效fps
                logger.warning(f"[{self.camera_name}] 无法获取有效帧率, 使用默认值 25")
                fps = 25.0
            logger.info(f"[{self.camera_name}] 视频信息获取成功: {video_info['width']}x{video_info['height']} @ {fps:.2f} FPS")
            return {
                "fps": float(fps),
                "width": int(video_info['width']),
                "height": int(video_info['height'])
            }
        except ffmpeg.Error as e:
            logger.error(f"[{self.camera_name}] FFprobe 错误: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"无法解析视频流信息: {e.stderr.decode() if e.stderr else str(e)}") # 抛出异常以便上层捕获
        except Exception as e:
            logger.error(f"[{self.camera_name}] 无法解析视频流信息: {str(e)}")
            logger.error(f"[{self.camera_name}] 详细错误: {traceback.format_exc()}")
            raise RuntimeError(f"无法解析视频流信息: {str(e)}") # 抛出异常以便上层捕获

    def _capture_loop(self):
        """拉流主循环"""
        # retries = 0 # 旧的重试计数器，不再用于控制循环退出
        current_retry_attempt = 0 # 新的计数器，用于指数退避
        logger.info(f"[{self.camera_name}] 捕获线程启动")
        # while self._running and retries < self.max_retries: # 旧的循环条件
        while self._running: # 只要 _running 为 True 就持续尝试
            try:
                logger.info(f"[{self.camera_name}] 尝试连接并拉流 (尝试次数: {current_retry_attempt + 1})...")
                # 创建 FFMpeg 进程来读取流
                self._capture_process = (
                    ffmpeg
                    .input(self.rtsp_url, **self.ffmpeg_params)
                    .output('pipe:', format='rawvideo',
                           pix_fmt='rgb24' if self.color_convert else 'bgr24',
                           vsync=0, loglevel="quiet") # 减少ffmpeg自身日志
                    .run_async(pipe_stdout=True, pipe_stderr=True) # 捕获stderr
                )

                # 启动一个线程来处理stderr，避免阻塞
                stderr_thread = threading.Thread(target=self._log_ffmpeg_stderr, args=(self._capture_process.stderr,))
                stderr_thread.daemon = True
                stderr_thread.start()

                logger.info(f"[{self.camera_name}] 拉流进程已启动 (PID: {self._capture_process.pid})")
                frame_size = self.video_info['width'] * self.video_info['height'] * 3
                
                # 连接成功，重置重试尝试计数器
                current_retry_attempt = 0 

                while self._running:
                    # 从stdout读取帧数据
                    in_bytes = self._capture_process.stdout.read(frame_size)
                    if not in_bytes:
                        # 检查进程是否已退出，区分正常结束和错误
                        poll_result = self._capture_process.poll()
                        if poll_result is not None:
                             logger.warning(f"[{self.camera_name}] 拉流进程已退出 (返回码: {poll_result}), 流结束。")
                        else:
                             logger.warning(f"[{self.camera_name}] 读取视频流返回空数据，可能流中断。")
                        break # 退出内层循环，将触发重连或停止

                    # 检查进程是否意外退出 (虽然上面已检查，再加一层保险)
                    if self._capture_process.poll() is not None:
                         logger.warning(f"[{self.camera_name}] 拉流进程意外退出，返回码: {self._capture_process.poll()}")
                         break

                    # 处理帧
                    frame = np.frombuffer(in_bytes, np.uint8).reshape(
                        (self.video_info['height'], self.video_info['width'], 3)
                    )

                    if self.color_convert:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # 将帧放入队列
                    try:
                        # 如果队列满，移除旧帧再放入新帧 (非阻塞)
                        if self.frame_queue.full():
                           try:
                                self.frame_queue.get_nowait()
                                logger.debug(f"[{self.camera_name}] 帧队列满，丢弃旧帧")
                           except queue.Empty:
                                pass # 可能在检查和获取之间变空
                        self.frame_queue.put(frame, block=False) # 非阻塞放入
                    except queue.Full:
                        # 理论上因为前面清除了，不应该到这里，但也处理一下
                        logger.warning(f"[{self.camera_name}] 帧队列放入时已满，丢弃当前帧")
                    except Exception as q_err:
                         logger.error(f"[{self.camera_name}] 帧队列操作失败: {q_err}")


                # 内层循环退出，说明流中断或进程结束
                logger.warning(f"[{self.camera_name}] 拉流中断，尝试关闭进程...")
                self._cleanup_capture_process()
                # 不需要等待stderr_thread，它是daemon，进程退出会自动结束
                # stderr_thread.join(timeout=2) # 等待stderr处理线程结束

                # 如果是正常停止，则退出外层循环
                if not self._running:
                    break

                # 如果是意外中断，增加重试次数并等待
                # retries += 1 # 不再使用 retries 控制外层循环退出
                current_retry_attempt += 1
                # if retries < self.max_retries: # 旧的判断
                # 指数退避，但有上限
                wait_time = min(2 ** current_retry_attempt, 60) # 例如，最大等待60秒
                logger.info(f"[{self.camera_name}] 等待 {wait_time} 秒后重试 (尝试次数: {current_retry_attempt})...")
                time.sleep(wait_time)
                # else: # 不再有最大重试次数的限制来放弃拉流
                #      logger.error(f"[{self.camera_name}] 已达到最大重试次数 ({self.max_retries})，放弃拉流。")


            except Exception as e:
                logger.error(f"[{self.camera_name}] 捕获循环发生严重错误: {str(e)}")
                logger.debug(traceback.format_exc()) # 记录详细错误
                self._cleanup_capture_process() # 确保清理进程
                # retries += 1 # 不再使用 retries 控制外层循环退出
                current_retry_attempt += 1
                # if retries < self.max_retries: # 旧的判断
                # 发生错误后等待，也采用指数退避
                wait_time = min(2 ** current_retry_attempt, 60) # 例如，最大等待60秒
                logger.info(f"[{self.camera_name}] 发生错误，等待 {wait_time} 秒后重试 (尝试次数: {current_retry_attempt})...")
                time.sleep(wait_time)
                # else: # 不再有最大重试次数的限制来放弃拉流
                #      logger.error(f"[{self.camera_name}] 发生错误且已达到最大重试次数 ({self.max_retries})，放弃拉流。")


        self._running = False # 确保退出循环时状态为停止
        logger.info(f"[{self.camera_name}] 捕获线程结束 (运行状态: {self._running})")

    def _log_ffmpeg_stderr(self, stderr_pipe: IO):
        """读取并记录FFmpeg的stderr输出"""
        try:
            # 将每个摄像头的ffmpeg日志分开记录可能更好，但暂时合并
            with open(FFMPEG_LOG_FILE, 'a', encoding='utf-8') as f_err:
                 f_err.write(f"\n--- FFmpeg拉流日志 ({self.camera_name} - {datetime.now()}) ---\n")
                 # 使用iter读取避免readline阻塞
                 for line in iter(lambda: stderr_pipe.readline(), b''):
                     try:
                         log_line = line.decode('utf-8', errors='ignore').strip()
                         if log_line:
                             logger.debug(f"FFmpeg 拉流 [{self.camera_name}]: {log_line}")
                             f_err.write(log_line + '\n')
                     except Exception as log_e:
                          logger.error(f"[{self.camera_name}] 记录FFmpeg stderr时出错: {log_e}")
        except Exception as pipe_e:
             logger.error(f"[{self.camera_name}] 处理FFmpeg stderr管道时出错: {pipe_e}")
        finally:
            try:
                 if stderr_pipe: stderr_pipe.close()
            except Exception:
                 pass # 忽略关闭错误
            logger.debug(f"[{self.camera_name}] FFmpeg stderr 处理结束")


    def _cleanup_capture_process(self):
        """安全地清理和终止捕获进程"""
        if self._capture_process and self._capture_process.poll() is None:
            pid = self._capture_process.pid
            logger.info(f"[{self.camera_name}] 正在终止拉流进程 (PID: {pid})...")
            try:
                # 先尝试 SIGTERM
                self._capture_process.terminate()
                try:
                    self._capture_process.wait(timeout=5) # 等待5秒
                    logger.info(f"[{self.camera_name}] 拉流进程 (PID: {pid}) 已终止")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[{self.camera_name}] 拉流进程 (PID: {pid}) 终止超时，强制结束 (SIGKILL)...")
                    self._capture_process.kill() # 强制结束
                    try:
                       self._capture_process.wait(timeout=2) # 等待kill完成
                       logger.info(f"[{self.camera_name}] 拉流进程 (PID: {pid}) 已强制结束")
                    except subprocess.TimeoutExpired:
                       logger.error(f"[{self.camera_name}] 拉流进程 (PID: {pid}) 强制结束也超时了?")
                    except Exception as kill_wait_e:
                       logger.error(f"[{self.camera_name}] 等待kill完成时出错: {kill_wait_e}")

            except Exception as e:
                logger.error(f"[{self.camera_name}] 清理拉流进程 (PID: {pid}) 时出错: {str(e)}")
                # 确保进程被清理掉
                if self._capture_process and self._capture_process.poll() is None:
                    try: self._capture_process.kill()
                    except: pass
        elif self._capture_process and self._capture_process.poll() is not None:
            # 进程已经退出，记录一下
             logger.info(f"[{self.camera_name}] 拉流进程 (PID: {self._capture_process.pid}) 在清理前已退出 (返回码: {self._capture_process.poll()})")


        self._capture_process = None # 清理引用


    def start(self):
        """启动拉流线程"""
        if not self._running:
            # 检查是否能获取视频信息，如果不能则不启动
            if not self.video_info:
                logger.error(f"[{self.camera_name}] 无法获取视频信息，无法启动拉流线程。")
                return False

            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop,
                name=f"Capture-{self.camera_name}", # 给线程命名
                daemon=True # 守护线程
            )
            self._thread.start()
            logger.info(f"[{self.camera_name}] 拉流线程已启动")
            return True
        return False # 如果已经在运行

    def stop(self):
        """停止拉流"""
        if self._running:
             logger.info(f"[{self.camera_name}] 正在停止拉流...")
             self._running = False # 设置标志让循环退出
             self._cleanup_capture_process() # 尝试终止ffmpeg进程
             if self._thread and self._thread.is_alive():
                 logger.debug(f"[{self.camera_name}] 等待拉流线程 {self._thread.name} 结束...")
                 self._thread.join(timeout=5) # 等待捕获线程结束
                 if self._thread.is_alive():
                     logger.warning(f"[{self.camera_name}] 捕获线程 {self._thread.name} 停止超时")
                 else:
                      logger.info(f"[{self.camera_name}] 拉流线程 {self._thread.name} 已结束")
             else:
                  logger.info(f"[{self.camera_name}] 拉流线程未运行或已结束")
             self._running = False # 再次确保状态
             logger.info(f"[{self.camera_name}] 拉流已停止")

    def get_frame(self, timeout: float = 1.0):
        """获取最新帧"""
        try:
            # 使用非阻塞获取来检查，如果为空则等待一小段时间
            # return self.frame_queue.get(timeout=timeout)
             return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_queue(self):
        """清空帧队列"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

class RTSPMonitor:
    def __init__(self,
                 camera_config: dict,
                 model_path: str,
                 save_path: str = SAVE_PATH,
                 frame_buffer_size: int = 30,
                 detection_confidence: float = 0.5,
                 consecutive_detections: int = 3,
                 consecutive_non_detections: int = 10,
                 process_every_n_frames: int = 2,
                 camera_id: str = None,
                 max_retries: int = 5,
                 **kwargs):

        self.camera_config = camera_config
        self.camera_id = camera_id or camera_config.get('name', camera_config.get('ip', f'cam-{time.time()}'))
        logger.info(f"[{self.camera_id}] 初始化 RTSPMonitor...")

        try:
            self.stream = RTSPStreamCapture(
                camera_config=camera_config,
                frame_queue=queue.Queue(maxsize=30),
                color_convert=True,
                max_retries=max_retries
            )
            if not self.stream.video_info:
                 raise RuntimeError("RTSPStreamCapture 未能成功获取视频信息。")

        except Exception as e:
             logger.error(f"[{self.camera_id}] 初始化 RTSPStreamCapture 失败: {e}")
             raise

        self.frame_width = self.stream.video_info["width"]
        self.frame_height = self.stream.video_info["height"]
        self.input_fps = self.stream.video_info["fps"]
        self.frame_interval = 1.0 / self.input_fps if self.input_fps > 0 else 0.04
        logger.info(f"[{self.camera_id}] 视频参数: {self.frame_width}x{self.frame_height} @ {self.input_fps:.2f} FPS")

        logger.info(f"[{self.camera_id}] 正在加载YOLO模型: {model_path}")
        try:
            self.model = YOLO(model_path)
            logger.info(f"[{self.camera_id}] YOLO模型加载成功。")
        except Exception as e:
             logger.error(f"[{self.camera_id}] 加载 YOLO 模型失败: {e}")
             raise

        self.detection_confidence = detection_confidence
        self.running = False
        self.person_detected = False
        self.save_video = False
        self.frame_buffer = deque(maxlen=frame_buffer_size)
        self.current_frame = None
        self.last_write_time = 0
        self.lock = threading.Lock()

        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.required_consecutive_detections = consecutive_detections
        self.required_consecutive_non_detections = consecutive_non_detections
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.frame_counter = 0

        self.save_path = save_path
        self.video_count = 0
        self.video_writer = None
        self.current_video_path = None

        self.video_queue = kwargs.get('video_queue')
        self.video_description_path = kwargs.get('video_description_path')
        self.json_lock = kwargs.get('json_lock') or threading.Lock()

        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            logger.error(f"[{self.camera_id}] 创建保存目录 {self.save_path} 失败: {e}")

        self._yolo_thread = None
        self._save_thread = None
        self._main_loop_thread = None

    def _update_video_description(self, video_path: str):
        """更新视频描述JSON文件（线程安全），并放入队列"""
        if not self.video_description_path:
            logger.warning(f"[{self.camera_id}] 未提供 video_description_path，跳过更新。")
            return
            
        try:
            filename = os.path.basename(video_path)
            timestamp_part = "_".join(filename.replace(f"{self.camera_id}_", "").split('.')[0:-1])
            
            with self.json_lock:
                video_data = {}
                if os.path.exists(self.video_description_path):
                    try:
                        with open(self.video_description_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():
                                video_data = json.loads(content)
                            else:
                                logger.warning(f"JSON文件为空: {self.video_description_path}，将创建新数据。")
                    except json.JSONDecodeError:
                        logger.warning(f"JSON文件损坏: {self.video_description_path}，将创建新文件。")

                v_k = f"{self.camera_id}_{timestamp_part}"
                video_data[v_k] = {
                    "video_path": os.path.abspath(video_path),
                    "analyse_result": None,
                    "is_embedding": False,
                    "idx": None
                }

                with open(self.video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"[{self.camera_id}] 已更新JSON描述: {v_k}")
        
                if self.video_queue is not None:
                    self.video_queue.put(v_k)
                    logger.info(f"[{self.camera_id}] 已将 {v_k} 放入视频分析队列。")
        except Exception as e:
            logger.error(f"[{self.camera_id}] 更新视频描述失败: {e}")
            logger.debug(traceback.format_exc())

    def detect_people(self, frame):
        """使用YOLO检测人员"""
        try:
            if not hasattr(self, 'model') or self.model is None:
                 logger.error(f"[{self.camera_id}] YOLO 模型未加载，无法进行检测。")
                 return False
            results = self.model(frame, verbose=False, classes=[0])
            for result in results:
                if result.boxes and len(result.boxes.conf) > 0:
                     if torch.any(result.boxes.conf > self.detection_confidence):
                         return True
            return False
        except Exception as e:
            logger.error(f"[{self.camera_id}] YOLO 检测时发生错误: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def yolo_detection_thread(self):
        """目标检测线程"""
        logger.info(f"[{self.camera_id}] YOLO检测线程启动")
        while self.running:
            frame_to_process = None
            try:
                with self.lock:
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame.copy()

                if frame_to_process is not None:
                    self.frame_counter += 1
                    if self.frame_counter % self.process_every_n_frames == 0:
                        detected = self.detect_people(frame_to_process)
                        with self.lock:
                            if detected:
                                self.consecutive_detections += 1
                                self.consecutive_non_detections = 0
                            else:
                                self.consecutive_detections = 0
                                self.consecutive_non_detections += 1

                            if ENABLE_VIDEO_SAVING and self.consecutive_detections >= self.required_consecutive_detections and not self.save_video:
                                self.person_detected = True
                                self.save_video = True
                                self.last_write_time = time.time()
                                logger.info(f"[{self.camera_id}] 连续检测到人员 {self.consecutive_detections} 次，开始保存视频...")
                            elif self.consecutive_non_detections >= self.required_consecutive_non_detections and self.save_video:
                                self.person_detected = False
                                self.save_video = False
                                logger.info(f"[{self.camera_id}] 连续 {self.consecutive_non_detections} 次未检测到人员，准备停止保存")
                else:
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"[{self.camera_id}] YOLO检测线程发生未处理错误: {str(e)}")
                logger.debug(traceback.format_exc())
                time.sleep(1)
            
            time.sleep(0.01)

        logger.info(f"[{self.camera_id}] YOLO检测线程结束")

    def video_saving_thread(self):
        """视频保存线程"""
        logger.info(f"[{self.camera_id}] 视频保存线程启动")
        ffmpeg_process = None
        buffer_written = False
        last_frame_time = 0

        while self.running:
            save_should_be_active = False
            frame_to_write = None

            try:
                with self.lock:
                    save_should_be_active = self.save_video
                    if self.current_frame is not None:
                         frame_to_write = self.current_frame.copy()

                if save_should_be_active and frame_to_write is not None:
                    if ffmpeg_process is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"{self.camera_id}_{timestamp}.mp4"
                        output_path = os.path.join(self.save_path, output_filename)
                        with self.lock:
                             self.current_video_path = output_path
                        self.video_count += 1

                        args = [
                            'ffmpeg', '-y',
                            '-f', 'rawvideo', '-vcodec', 'rawvideo',
                            '-s', f'{self.frame_width}x{self.frame_height}',
                            '-pix_fmt', 'bgr24',
                            '-r', str(self.input_fps),
                            '-i', '-',
                            '-c:v', 'libx264',
                            '-crf', '28',
                            '-preset', 'veryfast',
                            '-movflags', '+faststart',
                            '-pix_fmt', 'yuv420p',
                            '-metadata', f'comment=Camera {self.camera_id}',
                            output_path
                        ]

                        logger.info(f"[{self.camera_id}] 启动 FFmpeg 进程保存: {output_filename}")
                        try:
                            ffmpeg_process = subprocess.Popen(
                                args,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            logger.info(f"[{self.camera_id}] FFmpeg 进程已启动 (PID: {ffmpeg_process.pid})")
                            buffer_written = False
                            last_frame_time = time.time()
                        except FileNotFoundError:
                             logger.critical(f"[{self.camera_id}] 无法找到 'ffmpeg' 命令。")
                             self.running = False
                             break
                        except Exception as popen_e:
                             logger.error(f"[{self.camera_id}] 启动 FFmpeg 进程失败: {popen_e}")
                             ffmpeg_process = None

                    if ffmpeg_process and ffmpeg_process.poll() is None and not buffer_written:
                        buffer_copy = []
                        with self.lock:
                            buffer_copy = list(self.frame_buffer)

                        if buffer_copy:
                            logger.info(f"[{self.camera_id}] 写入 {len(buffer_copy)} 帧缓冲数据")
                            frames_written = 0
                            for buffered_frame in buffer_copy:
                                try:
                                     if ffmpeg_process and ffmpeg_process.stdin and ffmpeg_process.poll() is None:
                                          ffmpeg_process.stdin.write(buffered_frame.tobytes())
                                          frames_written += 1
                                     else:
                                          logger.warning(f"[{self.camera_id}] 写入缓冲帧时 FFmpeg 进程已结束。")
                                          break
                                except (OSError, BrokenPipeError) as pipe_err:
                                     logger.error(f"[{self.camera_id}] 写入缓冲帧到 FFmpeg stdin 失败: {pipe_err}")
                                     if ffmpeg_process:
                                         try: ffmpeg_process.kill()
                                         except: pass
                                     ffmpeg_process = None
                                     break
                            logger.info(f"[{self.camera_id}] 完成写入 {frames_written}/{len(buffer_copy)} 缓冲帧")
                            buffer_written = True
                            last_frame_time = time.time()

                    if ffmpeg_process and ffmpeg_process.stdin and ffmpeg_process.poll() is None:
                        current_time = time.time()
                        if current_time - last_frame_time >= self.frame_interval * 0.8:
                            try:
                                ffmpeg_process.stdin.write(frame_to_write.tobytes())
                                last_frame_time = current_time
                            except (OSError, BrokenPipeError) as pipe_err:
                                logger.error(f"[{self.camera_id}] 写入实时帧到 FFmpeg stdin 失败: {pipe_err}")
                                if ffmpeg_process:
                                     try: ffmpeg_process.kill()
                                     except: pass
                                ffmpeg_process = None

                elif ffmpeg_process is not None:
                    saved_video_path = ""
                    with self.lock:
                         saved_video_path = self.current_video_path
                    logger.info(f"[{self.camera_id}] 停止保存视频: {os.path.basename(saved_video_path) if saved_video_path else '未知路径'}")

                    pid = ffmpeg_process.pid
                    return_code = -1
                    try:
                        if ffmpeg_process.stdin:
                            ffmpeg_process.stdin.close()
                    except (OSError, BrokenPipeError):
                         pass

                    try:
                        return_code = ffmpeg_process.wait(timeout=15)
                        logger.info(f"[{self.camera_id}] FFmpeg 进程 {pid} 正常结束, 返回码: {return_code}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[{self.camera_id}] FFmpeg 进程 {pid} 等待超时，强制终止")
                        try:
                            ffmpeg_process.kill()
                            ffmpeg_process.wait(timeout=2)
                        except: pass

                    if saved_video_path and os.path.exists(saved_video_path) and return_code == 0:
                        time.sleep(0.5)
                        self._update_video_description(saved_video_path)
                    elif return_code != 0:
                         logger.error(f"[{self.camera_id}] FFmpeg 进程 {pid} 异常退出 (返回码: {return_code})，可能视频文件无效: {saved_video_path}")
                    else:
                         logger.warning(f"[{self.camera_id}] 保存的视频文件不存在或路径无效: {saved_video_path}")

                    ffmpeg_process = None
                    buffer_written = False
                    with self.lock:
                        self.current_video_path = None
                else:
                    time.sleep(0.05)

            except Exception as e:
                logger.error(f"[{self.camera_id}] 视频保存线程发生未处理错误: {str(e)}")
                logger.debug(traceback.format_exc())
                if ffmpeg_process and ffmpeg_process.poll() is None:
                    logger.warning(f"[{self.camera_id}] 发生错误，尝试终止 FFmpeg 进程")
                    try: ffmpeg_process.kill()
                    except: pass
                ffmpeg_process = None
                with self.lock: self.current_video_path = None
                time.sleep(1)

            time.sleep(0.01)

        if ffmpeg_process is not None:
            logger.info(f"[{self.camera_id}] 程序退出，正在清理视频保存进程...")
            try:
                if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
                ffmpeg_process.wait(timeout=10)
            except:
                if ffmpeg_process.poll() is None:
                    try:
                         ffmpeg_process.kill()
                         ffmpeg_process.wait(timeout=2)
                    except: pass
        logger.info(f"[{self.camera_id}] 视频保存线程结束")


    def main_loop(self):
         """主循环，负责从流获取帧并放入内部变量"""
         logger.info(f"[{self.camera_id}] 主处理循环启动")
         last_log_time = time.time()
         log_interval = 60

         while self.running:
              frame = None
              try:
                  frame = self.stream.get_frame(timeout=0.1)

                  if frame is not None:
                      with self.lock:
                          self.current_frame = frame
                          if not self.save_video:
                              self.frame_buffer.append(frame.copy())

                  elif not self.stream._running and self.stream._thread and not self.stream._thread.is_alive():
                       logger.error(f"[{self.camera_id}] 检测到视频流捕获线程已停止，将停止此监控器。")
                       self.running = False
                       break

                  current_time = time.time()
                  if current_time - last_log_time >= log_interval:
                       q_size = self.stream.frame_queue.qsize()
                       buffer_len = len(self.frame_buffer)
                       logger.info(f"[{self.camera_id}] 状态: {'记录中' if self.save_video else '监控中'}, 帧队列: {q_size}, 缓冲: {buffer_len}")

                       if self._yolo_thread and not self._yolo_thread.is_alive():
                            logger.error(f"[{self.camera_id}] YOLO检测线程已停止运行!")
                       if self._save_thread and not self._save_thread.is_alive():
                            logger.error(f"[{self.camera_id}] 视频保存线程已停止运行!")
                       if self.stream._thread and not self.stream._thread.is_alive():
                           logger.error(f"[{self.camera_id}] 视频流捕获线程已停止运行!")
                           self.running = False
                           break
                       last_log_time = current_time

              except Exception as e:
                  logger.error(f"[{self.camera_id}] 主循环发生错误: {str(e)}")
                  logger.debug(traceback.format_exc())
                  time.sleep(0.5)

              time.sleep(self.frame_interval)

         logger.info(f"[{self.camera_id}] 主处理循环结束")
         self.stop_internal_threads()


    def start_internal_threads(self):
         """启动内部工作线程 (YOLO, Save)"""
         if not self.running:
              logger.warning(f"[{self.camera_id}] 尝试启动内部线程，但监控器未运行。")
              return

         logger.info(f"[{self.camera_id}] 正在启动内部工作线程...")
         self._yolo_thread = threading.Thread(target=self.yolo_detection_thread, name=f"YOLO-{self.camera_id}", daemon=True)
         self._save_thread = threading.Thread(target=self.video_saving_thread, name=f"Saving-{self.camera_id}", daemon=True)

         self._yolo_thread.start()
         self._save_thread.start()
         logger.info(f"[{self.camera_id}] 内部工作线程已启动。")


    def stop_internal_threads(self):
         """停止内部工作线程"""
         logger.info(f"[{self.camera_id}] 正在停止内部工作线程...")
         if self._yolo_thread and self._yolo_thread.is_alive():
              self._yolo_thread.join(timeout=5)
         if self._save_thread and self._save_thread.is_alive():
              self._save_thread.join(timeout=15)


    def run(self):
        """启动监控系统"""
        if self.running:
            logger.warning(f"[{self.camera_id}] 监控器已在运行中。")
            return

        logger.info(f"[{self.camera_id}] 正在启动监控...")
        if not self.stream.start():
             logger.error(f"[{self.camera_id}] 无法启动视频流，监控启动失败。")
             return False

        self.running = True
        self.start_internal_threads()
        self.main_loop()

        logger.info(f"[{self.camera_id}] 监控器 run 方法完成。")
        return True


    def stop(self):
        """停止监控系统"""
        if not self.running:
             logger.info(f"[{self.camera_id}] 监控器未运行。")
             return

        logger.info(f"[{self.camera_id}] 接收到停止信号，开始关闭流程...")
        self.running = False
        self.stream.stop()
        logger.info(f"[{self.camera_id}] 监控器已安全关闭")

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
