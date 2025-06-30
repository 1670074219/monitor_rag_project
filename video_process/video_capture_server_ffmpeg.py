import logging
import os
import json
from ultralytics import YOLO
import cv2
import threading
from datetime import datetime
import subprocess
import numpy as np
import time
from queue import Queue, Empty
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
        os.makedirs(self.saved_video_path, exist_ok=True)
        
        self.yolo_model = YOLO(self.yolo_path).to(device="cuda:2")
        self.people_detected_frames_threshold = people_detected_frames_threshold
        self.no_person_frames_threshold = no_person_frames_threshold
        self.video_queue = video_queue
        self.video_description_path = video_description_path
        self.json_lock = threading.Lock()
        
        self.running = True
        self.camera_states = {} # 存储每个摄像头的状态、线程和队列

    def get_video_resolution(self, camera_url: str):
        """使用ffprobe获取视频分辨率和帧率，增加超时和重试"""
        for attempt in range(3):
            try:
                logger.info(f"正在获取视频信息: {camera_url} (尝试 {attempt + 1})")
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-rtsp_transport', 'tcp',
                    '-timeout', '5000000', # 5秒超时
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate',
                    '-of', 'json',
                    camera_url
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info = json.loads(result.stdout)['streams'][0]
                    width = int(info['width'])
                    height = int(info['height'])
                    
                    # 解析帧率
                    fps_str = info.get('r_frame_rate', '25/1')
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 25.0
                    
                    logger.info(f"视频信息获取成功: {width}x{height} @ {fps:.2f} FPS")
                    return width, height, fps
                else:
                    logger.error(f"ffprobe执行失败: {result.stderr}")
            except Exception as e:
                logger.error(f"获取视频信息失败: {e}")
            
            time.sleep(2) # 等待2秒后重试
        
        logger.error(f"无法获取视频信息，使用默认值 640x480 @ 25 FPS")
        return 640, 480, 25.0

    def _frame_grabber_loop(self, camera_id: str, camera_url: str, frame_queue: Queue):
        """
        单一职责：持续从摄像头拉取视频帧并放入队列。
        包含自动重连和指数退避逻辑。
        """
        width, height, _ = self.camera_states[camera_id]['resolution']
        retry_delay = 1

        while self.running:
            try:
                logger.info(f"[{camera_id}] 启动拉流进程...")
                command = [
                    'ffmpeg',
                    '-rtsp_transport', 'tcp',
                    '-timeout', '5000000', # 5秒连接超时
                    '-i', camera_url,
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-r', '25', # 指定拉流帧率
                    '-'
                ]
                pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
                self.camera_states[camera_id]['grabber_process'] = pipe
                
                logger.info(f"[{camera_id}] 拉流进程已启动 (PID: {pipe.pid})")
                retry_delay = 1 # 连接成功后重置退避延迟

                while self.running:
                    raw_image = pipe.stdout.read(width * height * 3)
                    if not raw_image:
                        # 检查进程是否已退出
                        if pipe.poll() is not None:
                            stderr_output = pipe.stderr.read().decode(errors='ignore')
                            logger.error(f"[{camera_id}] 拉流进程意外退出。返回码: {pipe.poll()}. FFmpeg输出:\n{stderr_output}")
                        else:
                            logger.warning(f"[{camera_id}] 读取到空帧，可能流已中断。")
                        break # 退出内层循环，触发重连
                    
                    try:
                        # 非阻塞方式放入队列，如果队列满则丢弃旧帧
                        if frame_queue.full():
                            frame_queue.get_nowait()
                        frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
                        frame_queue.put(frame, block=False)
                    except Empty:
                        pass # 队列在检查后变空，忽略
                    except Exception as qe:
                        logger.error(f"[{camera_id}] 帧队列操作失败: {qe}")

                # 清理旧进程
                if pipe.poll() is None:
                    pipe.terminate()
                    pipe.wait(timeout=5)
                
            except Exception as e:
                logger.error(f"[{camera_id}] 拉流循环发生严重错误: {e}")
            
            if not self.running:
                break

            logger.info(f"[{camera_id}] 拉流中断，将在 {retry_delay} 秒后尝试重连...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60) # 指数退避，最大60秒

        logger.info(f"[{camera_id}] 拉流线程结束。")

    def _frame_processor_loop(self, camera_id: str, camera_url: str, frame_queue: Queue):
        """
        单一职责：从队列获取帧，进行AI分析，并控制录制启停。
        """
        people_detected_frames = 0
        no_person_frames = 0
        is_recording = False
        frame_counter = 0
        
        width, height, fps = self.camera_states[camera_id]['resolution']

        while self.running:
            try:
                frame = frame_queue.get(timeout=1.0)
                frame_counter += 1
                
                # 每隔几帧处理一次，降低CPU负载
                if frame_counter % 2 != 0:
                    continue
                
                # YOLO人员检测
                results = self.yolo_model(frame, verbose=False, classes=[0])
                has_person = len(results[0].boxes) > 0

                if has_person:
                    people_detected_frames += 1
                    no_person_frames = 0
                else:
                    people_detected_frames = 0
                    no_person_frames += 1

                # 开始录制
                if people_detected_frames >= self.people_detected_frames_threshold and not is_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.saved_video_path, f"{camera_id}_{timestamp}.mp4")
                    
                    if self._start_ffmpeg_recording(camera_id, filename, width, height, fps):
                        is_recording = True
                        logger.info(f"[{camera_id}] 开始录制: {filename}")
                
                # 停止录制
                if no_person_frames >= self.no_person_frames_threshold and is_recording:
                    logger.info(f"[{camera_id}] 连续未检测到人员，准备停止录制。")
                    self._stop_ffmpeg_recording(camera_id)
                    is_recording = False
                    # 重置计数器，避免立即再次触发
                    people_detected_frames = 0
                    no_person_frames = 0

                # 如果正在录制，将当前帧写入管道
                if is_recording:
                    recording_proc = self.camera_states[camera_id].get('recording_process')
                    if recording_proc and recording_proc.poll() is None:
                        try:
                            recording_proc.stdin.write(frame.tobytes())
                        except (BrokenPipeError, OSError):
                            logger.error(f"[{camera_id}] 写入录制进程管道失败，可能已关闭。")
                            self._stop_ffmpeg_recording(camera_id)
                            is_recording = False
                    else:
                        logger.warning(f"[{camera_id}] 录制进程丢失，停止录制状态。")
                        is_recording = False

            except Empty:
                continue # 队列空，继续等待
            except Exception as e:
                logger.error(f"[{camera_id}] 处理帧时出错: {e}", exc_info=True)
                continue

        # 确保循环退出时，如果仍在录制则停止
        if is_recording:
            self._stop_ffmpeg_recording(camera_id)

        logger.info(f"[{camera_id}] 处理线程结束。")

    def _start_ffmpeg_recording(self, camera_id: str, output_file: str, width: int, height: int, fps: float):
        """启动ffmpeg录制进程，从管道读取原始帧并编码为H264 MP4"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(int(fps)),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '25',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_file
            ]
            
            logger.info(f"[{camera_id}] 启动FFmpeg录制进程: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            self.camera_states[camera_id]['recording_process'] = process
            self.camera_states[camera_id]['output_file'] = output_file
            return True
        except Exception as e:
            logger.error(f"[{camera_id}] 启动ffmpeg录制失败: {e}")
            return False

    def _stop_ffmpeg_recording(self, camera_id: str):
        """停止指定摄像头的ffmpeg录制进程"""
        state = self.camera_states.get(camera_id)
        if not state or 'recording_process' not in state:
            return False
        
        process = state['recording_process']
        output_file = state['output_file']
        
        if process is None or process.poll() is not None:
            return False

        logger.info(f"[{camera_id}] 正在停止录制: {output_file}")
        try:
            # 关闭stdin，ffmpeg会因此结束
            process.stdin.close()
            # 等待进程结束
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"[{camera_id}] FFmpeg录制进程关闭超时，强制终止。")
                process.kill()
                process.wait(timeout=2)

            # 检查输出文件
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1024: # 确保文件有效
                logger.info(f"[{camera_id}] 录制完成: {output_file}")
                self.update_video_description(camera_id, output_file)
                return True
            else:
                stderr_output = process.stderr.read().decode(errors='ignore')
                logger.warning(f"[{camera_id}] 录制文件无效或大小为0: {output_file}. FFmpeg输出:\n{stderr_output}")
                return False
        
        except Exception as e:
            logger.error(f"[{camera_id}] 停止录制时发生错误: {e}")
            return False
        finally:
            state['recording_process'] = None
            state['output_file'] = None

    def update_video_description(self, camera_id: str, output_file: str):
        """更新视频描述JSON文件（线程安全）"""
        try:
            filename = os.path.basename(output_file)
            timestamp_part = filename.replace(f"{camera_id}_", "").replace(".mp4", "")
            
            with self.json_lock:
                video_data = {}
                if self.video_description_path and os.path.exists(self.video_description_path):
                    try:
                        with open(self.video_description_path, "r", encoding="utf-8") as f:
                            video_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"JSON文件损坏: {self.video_description_path}，将创建新文件。")

                v_k = f"{camera_id}_{timestamp_part}"
                video_data[v_k] = {
                    "video_path": output_file,
                    "analyse_result": None,
                    "is_embedding": False,
                    "idx": None
                }

                with open(self.video_description_path, "w", encoding="utf-8") as f:
                    json.dump(video_data, f, indent=4, ensure_ascii=False)
                
                logger.info(f"已更新JSON描述: {v_k}")
        
                if self.video_queue is not None:
                    self.video_queue.put(v_k)
                    logger.info(f"已将 {v_k} 放入视频分析队列。")
        except Exception as e:
            logger.error(f"更新视频描述失败: {e}")

    def run_all_cameras(self):
        """为每个摄像头启动拉流和处理线程"""
        threads = []
        for cam in self.camera_config:
            camera_id = cam['camera_id']
            camera_url = cam['camera_url']

            width, height, fps = self.get_video_resolution(camera_url)

            self.camera_states[camera_id] = {
                'frame_queue': Queue(maxsize=50), # 增加队列大小
                'resolution': (width, height, fps),
                'threads': [],
                'grabber_process': None,
                'recording_process': None,
                'output_file': None
            }
            
            frame_queue = self.camera_states[camera_id]['frame_queue']

            # 启动拉流线程
            grabber_thread = threading.Thread(
                target=self._frame_grabber_loop,
                args=(camera_id, camera_url, frame_queue),
                daemon=True,
                name=f"Grabber-{camera_id}"
            )
            
            # 启动处理线程
            processor_thread = threading.Thread(
                target=self._frame_processor_loop,
                args=(camera_id, camera_url, frame_queue),
                daemon=True,
                name=f"Processor-{camera_id}"
            )
            
            self.camera_states[camera_id]['threads'].extend([grabber_thread, processor_thread])
            threads.extend([grabber_thread, processor_thread])

            grabber_thread.start()
            processor_thread.start()
            
            logger.info(f"摄像头 {camera_id} 的监控线程已启动。")
            time.sleep(1) # 错开启动，避免瞬时资源竞争

        # 主线程等待，直到外部发出停止信号（例如通过调用cleanup）
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("捕获到Ctrl+C，开始清理...")
            self.cleanup()
        finally:
            for thread in threads:
                if thread.is_alive():
                    thread.join()

    def cleanup(self):
        """清理所有资源"""
        logger.info("正在停止所有摄像头监控...")
        self.running = False
        
        for camera_id in self.camera_states:
            logger.info(f"正在关闭 {camera_id}...")
            # 停止录制进程
            self._stop_ffmpeg_recording(camera_id)
            
            # 停止拉流进程
            grabber_proc = self.camera_states[camera_id].get('grabber_process')
            if grabber_proc and grabber_proc.poll() is None:
                try:
                    grabber_proc.terminate()
                    grabber_proc.wait(timeout=5)
                except:
                    grabber_proc.kill()
            
            # 等待线程结束
            for thread in self.camera_states[camera_id]['threads']:
                if thread.is_alive():
                    thread.join(timeout=5)
        
        logger.info("所有监控已停止。")

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

    # 设置信号处理器以优雅地关闭
    def signal_handler(sig, frame):
        logger.info("接收到终止信号，正在清理...")
        video_capture_server.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    video_capture_server.run_all_cameras() 