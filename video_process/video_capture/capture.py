import logging
import os
import json
from ultralytics import YOLO
import threading
from datetime import datetime
import subprocess
import numpy as np
import time
from queue import Queue, Empty, Full
import signal
import sys
import socket
import math
import re
import pymysql

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'database': 'monitor_database',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
}

def check_network_connectivity(ip, port):
    """检查网络连通性"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5秒超时
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"网络连通性检查失败: {e}")
        return False

class VideoCaptureServerFFmpeg:
    def __init__(self,
                 camera_config: list[dict],
                 yolo_path: str,
                 saved_video_path: str,
                 people_detected_frames_threshold: int = 10,
                 no_person_frames_threshold: int = 30,
                 pre_record_seconds: float = 2.0,
                 post_record_seconds: float = 2.0,
                 video_queue: Queue = None):
        self.camera_config = camera_config
        self.yolo_path = os.path.abspath(yolo_path)
        self.saved_video_path = os.path.abspath(saved_video_path)
        os.makedirs(self.saved_video_path, exist_ok=True)

        if not os.path.exists(self.yolo_path):
            raise FileNotFoundError(f"YOLO模型不存在: {self.yolo_path}")

        self.yolo_model = YOLO(self.yolo_path).to(device="cuda:2")
        self.people_detected_frames_threshold = people_detected_frames_threshold
        self.no_person_frames_threshold = no_person_frames_threshold
        self.pre_record_seconds = pre_record_seconds
        self.post_record_seconds = post_record_seconds
        self.video_queue = video_queue
        # 关键参数1：切片时长提升到5秒，尽量对齐GOP关键帧，降低坏片概率
        self.segment_seconds = 5.0
        # 关键参数2：保留窗口适当增加，避免事件拼接时清理过早
        self.segment_keep_seconds = max(60.0, self.pre_record_seconds + self.post_record_seconds + 15.0)
        self.segment_stable_seconds = 1.5
        # 关键参数3：降低最小文件阈值，只过滤空片或极小异常片
        self.segment_min_size_bytes = 1024
        
        self.running = True
        self.camera_states = {} # 存储每个摄像头的状态、线程和队列

    def _parse_created_time_from_video_name(self, video_name: str):
        """从 video_name 提取创建时间，格式: camera1_20260301_130245 -> 2026-03-01 13:02:45"""
        match = re.search(r'(\d{8})_(\d{6})', video_name)
        if not match:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            dt = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _save_video_to_database(self, video_name: str, video_path: str):
        """将视频元数据写入 monitor_database.videos 表。"""
        conn = None
        try:
            conn = pymysql.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                charset=DB_CONFIG['charset'],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False
            )

            created_time = self._parse_created_time_from_video_name(video_name)
            insert_sql = """
                INSERT INTO videos (video_name, video_path, person_count, description, created_time)
                VALUES (%s, %s, %s, %s, %s)
            """

            with conn.cursor() as cursor:
                cursor.execute(insert_sql, (video_name, video_path, 0, None, created_time))

            conn.commit()
            logger.info(f"已写入数据库 videos 表: {video_name}")
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"写入数据库失败({video_name}): {e}")
            return False
        finally:
            if conn:
                conn.close()

    def get_video_resolution(self, camera_url: str):
        """使用ffprobe获取视频分辨率和帧率，增加超时和重试"""
        host = None
        port = 554

        # 解析URL获取IP和端口
        try:
            from urllib.parse import urlparse
            parsed = urlparse(camera_url)
            host_port = parsed.netloc.split('@')[-1]  # 获取主机:端口部分
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host = host_port
                port = 554  # 默认RTSP端口
        except Exception as e:
            logger.error(f"URL解析失败: {e}")
            return 1280, 720, 25.0

        # 检查网络连通性
        if not check_network_connectivity(host, port):
            logger.error(f"无法连接到 {host}:{port}，请检查网络配置")
            return 1280, 720, 25.0
        
        for attempt in range(3):
            try:
                logger.info(f"正在获取视频信息: {camera_url} (尝试 {attempt + 1})")
                
                # 使用正确的 ffprobe 参数（类似命令行成功版本）
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-rtsp_transport', 'tcp',
                    '-stimeout', '5000000',  # 使用 stimeout 而不是 timeout
                    '-i', camera_url,
                    '-show_streams',
                    '-select_streams', 'v:0',
                    '-print_format', 'json'
                ]
                
                logger.info(f"执行命令: {' '.join(probe_cmd)}")
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    try:
                        # 从 ffprobe 输出中提取信息
                        info = json.loads(result.stdout)
                        if 'streams' in info and len(info['streams']) > 0:
                            stream = info['streams'][0]
                            width = int(stream['width'])
                            height = int(stream['height'])
                            # 解析帧率
                            avg_frame_rate = stream.get('avg_frame_rate', '25/1')
                            if avg_frame_rate and avg_frame_rate != '0/0':
                                try:
                                    num, den = map(int, avg_frame_rate.split('/'))
                                    fps = num / den if den != 0 else 25.0
                                except:
                                    fps = 25.0
                            else:
                                fps = 25.0
                            
                            logger.info(f"视频信息获取成功: {width}x{height} @ {fps:.2f} FPS")
                            return width, height, fps
                        else:
                            logger.warning("ffprobe未找到视频流信息")
                    except json.JSONDecodeError:
                        logger.error(f"ffprobe输出不是有效JSON: {result.stdout[:200]}...")
                else:
                    logger.error(f"ffprobe执行失败: {result.stderr}")
                    logger.error(f"ffprobe输出: {result.stdout}")
            except subprocess.TimeoutExpired:
                logger.error("ffprobe命令超时")
            except Exception as e:
                logger.error(f"获取视频信息失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
            
            time.sleep(2) # 等待2秒后重试
        
        logger.error(f"无法获取视频信息，使用默认值 1280x720 @ 25 FPS")
        return 1280, 720, 25.0

    def _frame_grabber_loop(self, camera_id: str, camera_url: str, frame_queue: Queue):
        """
        单一职责：持续从摄像头拉取视频帧并放入队列。
        包含自动重连和指数退避逻辑。

        说明：
        - FFmpeg单进程双路输出：
          1) rawvideo 到 stdout，供 Python/YOLO 分析。
          2) 原始码流 -c:v copy + segment 持续切片到本地缓存。
        - Python 不再把裸流写回 ffmpeg，不做二次编码。
        """
        width, height, fps = self.camera_states[camera_id]['resolution']
        retry_delay = 1
        segment_dir = self.camera_states[camera_id]['segment_dir']
        segment_pattern = os.path.join(segment_dir, '%Y%m%d_%H%M%S.mp4')

        while self.running:
            try:
                logger.info(f"[{camera_id}] 启动拉流进程...")
                
                # 方案A核心命令说明：
                # - 第一路输出: 解码后 rawvideo -> stdout (给YOLO)
                # - 第二路输出: 同一输入直接 -c:v copy -> segment mp4 (原始画面缓存)
                # 这样录制文件来自摄像头原码流，不经过Python与libx264重编码。
                command = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-rtsp_transport', 'tcp',
                    '-stimeout', '5000000',
                    '-i', camera_url,
                    '-an',
                    '-sn',
                    '-fflags', 'nobuffer',
                    '-flags', 'low_delay',
                    '-map', '0:v:0',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-r', str(fps),
                    'pipe:1',
                    '-map', '0:v:0',
                    '-c:v', 'copy',
                    '-f', 'segment',
                    '-segment_time', str(int(self.segment_seconds)),
                    '-strftime', '1',
                    '-reset_timestamps', '1',
                    '-segment_format', 'mp4',
                    segment_pattern
                ]
                
                logger.info(f"[{camera_id}] 执行拉流命令: {' '.join(command)}")
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
                    except Full:
                        pass
                    except Exception as qe:
                        logger.error(f"[{camera_id}] 帧队列操作失败: {qe}")

                    self._cleanup_old_segments(camera_id)

                # 清理旧进程
                if pipe.poll() is None:
                    pipe.terminate()
                    try:
                        pipe.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pipe.kill()
                        pipe.wait(timeout=2)
                
            except Exception as e:
                logger.error(f"[{camera_id}] 拉流循环发生严重错误: {e}")
                import traceback
                logger.error(f"[{camera_id}] 详细错误: {traceback.format_exc()}")
            
            if not self.running:
                break

            logger.info(f"[{camera_id}] 拉流中断，将在 {retry_delay} 秒后尝试重连...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60) # 指数退避，最大60秒

        logger.info(f"[{camera_id}] 拉流线程结束。")

    def _parse_segment_ts(self, segment_name: str):
        base = os.path.splitext(segment_name)[0]
        return datetime.strptime(base, "%Y%m%d_%H%M%S").timestamp()

    def _is_segment_usable(self, segment_path: str):
        if not os.path.isfile(segment_path):
            return False

        file_size = os.path.getsize(segment_path)
        if file_size < self.segment_min_size_bytes:
            return False

        file_age = time.time() - os.path.getmtime(segment_path)
        if file_age < self.segment_stable_seconds:
            return False

        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            segment_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and bool(result.stdout.strip())

    def _collect_segments_for_window(self, camera_id: str, start_ts: float, end_ts: float):
        segment_dir = self.camera_states[camera_id]['segment_dir']
        if not os.path.isdir(segment_dir):
            return []

        selected = []
        for name in os.listdir(segment_dir):
            if not name.endswith('.mp4'):
                continue
            segment_path = os.path.join(segment_dir, name)
            if not os.path.isfile(segment_path):
                continue
            try:
                seg_ts = self._parse_segment_ts(name)
            except Exception:
                continue
            # 关键修复：使用区间相交而不是“切片起点落窗”判断，避免丢失事件起始跨界切片
            # 切片区间: [seg_ts, seg_end_ts]
            # 事件区间: [start_ts, end_ts]
            # 只要有交集就保留
            seg_end_ts = seg_ts + self.segment_seconds
            if seg_end_ts >= start_ts and seg_ts <= end_ts:
                if self._is_segment_usable(segment_path):
                    selected.append((seg_ts, segment_path))
                else:
                    logger.warning(f"[{camera_id}] 跳过不可用切片: {segment_path}")

        selected.sort(key=lambda item: item[0])
        return [path for _, path in selected]

    def _cleanup_old_segments(self, camera_id: str):
        state = self.camera_states[camera_id]
        now_ts = time.time()
        if now_ts - state.get('last_segment_cleanup_ts', 0) < 5.0:
            return

        state['last_segment_cleanup_ts'] = now_ts
        segment_dir = state['segment_dir']
        if not os.path.isdir(segment_dir):
            return

        keep_from = now_ts - self.segment_keep_seconds
        for name in os.listdir(segment_dir):
            if not name.endswith('.mp4'):
                continue
            file_path = os.path.join(segment_dir, name)
            if not os.path.isfile(file_path):
                continue
            try:
                seg_ts = self._parse_segment_ts(name)
            except Exception:
                continue
            if seg_ts < keep_from:
                # 避免误删当前仍在写入的新切片
                if now_ts - os.path.getmtime(file_path) < self.segment_stable_seconds:
                    continue
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"[{camera_id}] 删除旧切片失败: {file_path}, err={e}")

    def _compose_clip_from_segments(self, camera_id: str, window_start_ts: float, window_end_ts: float):
        time.sleep(1.2)
        segment_files = self._collect_segments_for_window(camera_id, window_start_ts, window_end_ts)
        if not segment_files:
            logger.warning(f"[{camera_id}] 事件窗口内无可用切片，start={window_start_ts}, end={window_end_ts}")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.saved_video_path, f"{camera_id}_{timestamp}.mp4")
        segment_dir = self.camera_states[camera_id]['segment_dir']
        concat_list_file = os.path.join(segment_dir, f"concat_{camera_id}_{timestamp}.txt")

        current_segments = list(segment_files)
        try:
            for _ in range(3):
                if not current_segments:
                    logger.warning(f"[{camera_id}] 拼接失败，已无可用切片。")
                    return False

                with open(concat_list_file, 'w', encoding='utf-8') as f:
                    for seg in current_segments:
                        safe_seg = seg.replace("'", "'\\\\''")
                        f.write(f"file '{safe_seg}'\n")

                cmd = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list_file,
                    '-c', 'copy',
                    '-movflags', '+faststart',
                    output_file
                ]

                logger.info(f"[{camera_id}] 拼接事件视频: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    break

                stderr_text = result.stderr or ''
                logger.error(f"[{camera_id}] 拼接失败: {stderr_text}")

                bad_path = None
                for line in stderr_text.splitlines():
                    marker = "Impossible to open '"
                    if marker in line and line.endswith("'"):
                        bad_path = line.split(marker, 1)[1][:-1]
                        break

                if bad_path is None:
                    return False

                current_segments = [p for p in current_segments if p != bad_path]
                logger.warning(f"[{camera_id}] 已剔除坏切片并重试: {bad_path}")
            else:
                return False

            if os.path.exists(output_file) and os.path.getsize(output_file) > 1024:
                logger.info(f"[{camera_id}] 事件视频生成成功: {output_file}")
                self.persist_video_metadata(camera_id, output_file)
                return True

            logger.warning(f"[{camera_id}] 拼接输出无效: {output_file}")
            return False
        except Exception as e:
            logger.error(f"[{camera_id}] 拼接事件视频异常: {e}")
            return False
        finally:
            if os.path.exists(concat_list_file):
                try:
                    os.remove(concat_list_file)
                except Exception:
                    pass

    def _clip_assembler_loop(self, camera_id: str):
        """
        单一职责：处理事件窗口并异步拼接视频。

        该线程与YOLO线程完全解耦：
        - YOLO线程只投递事件窗口(start_ts/end_ts)
        - 本线程执行文件枚举与ffmpeg concat，避免阻塞检测循环
        """
        clip_queue = self.camera_states[camera_id]['clip_task_queue']

        while self.running:
            try:
                task = clip_queue.get(timeout=1.0)
            except Empty:
                continue

            window_start_ts = task.get('window_start_ts')
            window_end_ts = task.get('window_end_ts')
            if window_start_ts is None or window_end_ts is None:
                continue
            self._compose_clip_from_segments(camera_id, window_start_ts, window_end_ts)

        logger.info(f"[{camera_id}] 片段拼接线程结束。")

    def _frame_processor_loop(self, camera_id: str, camera_url: str, frame_queue: Queue):
        """
        单一职责：从队列获取帧，进行AI分析，并控制录制启停。

        注意：
        - 这里不进行任何磁盘I/O和子进程启停，避免阻塞YOLO循环。
        - 仅通过非阻塞队列发命令给 _recording_controller_loop。
        """
        people_detected_frames = 0
        is_recording = False
        last_person_detected_ts = None
        record_window_start_ts = None

        while self.running:
            try:
                frame = frame_queue.get(timeout=1.0)
                now_ts = time.time()

                # YOLO人员检测
                results = self.yolo_model(frame, verbose=False, classes=[0])
                has_person = len(results[0].boxes) > 0

                if has_person:
                    last_person_detected_ts = now_ts
                    people_detected_frames += 1
                else:
                    people_detected_frames = 0

                # 开始录制
                if people_detected_frames >= self.people_detected_frames_threshold and not is_recording:
                    is_recording = True
                    record_window_start_ts = now_ts - self.pre_record_seconds
                    logger.info(
                        f"[{camera_id}] 检测到人员，开始事件窗口。"
                        f"window_start={record_window_start_ts:.3f}, pre={self.pre_record_seconds}s"
                    )
                
                # 停止录制
                # 使用真实时间判断“人员消失后空白保留时长”，避免YOLO实际处理FPS变化导致空白录制过长。
                # 例如 YOLO 实际仅 3 FPS 时，按30帧会变成约10秒；改为按秒后固定为 post_record_seconds。
                if is_recording and last_person_detected_ts is not None and (now_ts - last_person_detected_ts) >= self.post_record_seconds:
                    window_end_ts = last_person_detected_ts + self.post_record_seconds
                    task = {
                        'window_start_ts': record_window_start_ts if record_window_start_ts is not None else now_ts - self.pre_record_seconds,
                        'window_end_ts': window_end_ts
                    }
                    clip_task_queue = self.camera_states[camera_id]['clip_task_queue']
                    try:
                        clip_task_queue.put_nowait(task)
                        logger.info(
                            f"[{camera_id}] 人员消失达到{self.post_record_seconds}s，已提交拼接任务。"
                            f"window=[{task['window_start_ts']:.3f}, {task['window_end_ts']:.3f}]"
                        )
                    except Full:
                        logger.warning(f"[{camera_id}] 拼接任务队列已满，丢弃本次事件。")

                    is_recording = False
                    record_window_start_ts = None
                    people_detected_frames = 0
                    last_person_detected_ts = None

            except Empty:
                continue # 队列空，继续等待
            except Exception as e:
                logger.error(f"[{camera_id}] 处理帧时出错: {e}", exc_info=True)
                continue

        logger.info(f"[{camera_id}] 处理线程结束。")

    def persist_video_metadata(self, camera_id: str, output_file: str):
        """仅写入数据库并投递队列，不再写入JSON文件"""
        try:
            filename = os.path.basename(output_file)
            timestamp_part = filename.replace(f"{camera_id}_", "").replace(".mp4", "")
            v_k = f"{camera_id}_{timestamp_part}"

            # 仅写数据库
            self._save_video_to_database(v_k, output_file)

            # 保留队列投递（供后续分析流程消费）
            if self.video_queue is not None:
                self.video_queue.put(v_k)
                logger.info(f"已将 {v_k} 放入视频分析队列。")
        except Exception as e:
            logger.error(f"持久化视频元数据失败: {e}")

    def run_all_cameras(self):
        """为每个摄像头启动拉流和处理线程"""
        threads = []
        for cam in self.camera_config:
            camera_id = cam['camera_id']
            camera_url = cam['camera_url']

            logger.info(f"开始初始化摄像头 {camera_id}，URL: {camera_url}")
            width, height, fps = self.get_video_resolution(camera_url)

            self.camera_states[camera_id] = {
                'frame_queue': Queue(maxsize=50), # 增加队列大小
                'clip_task_queue': Queue(maxsize=20),
                'resolution': (width, height, fps),
                'threads': [],
                'grabber_process': None,
                'segment_dir': os.path.join(self.saved_video_path, '_segments', camera_id),
                'last_segment_cleanup_ts': 0.0
            }

            os.makedirs(self.camera_states[camera_id]['segment_dir'], exist_ok=True)
            
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

            assembler_thread = threading.Thread(
                target=self._clip_assembler_loop,
                args=(camera_id,),
                daemon=True,
                name=f"ClipAssembler-{camera_id}"
            )
            
            self.camera_states[camera_id]['threads'].extend([grabber_thread, processor_thread, assembler_thread])
            threads.extend([grabber_thread, processor_thread, assembler_thread])

            grabber_thread.start()
            processor_thread.start()
            assembler_thread.start()
            
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
    camera_config_path = os.path.abspath(os.path.join(base_dir, '..', 'camera_config.json'))
    yolo_model_path = os.path.abspath(os.path.join(base_dir, '..', 'models/yolo', 'yolo11s.pt'))
    saved_video_dir = os.path.abspath(os.path.join(base_dir, '..', 'saved_video'))

    if not os.path.exists(camera_config_path):
        raise FileNotFoundError(f"camera_config.json不存在: {camera_config_path}")
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO模型不存在: {yolo_model_path}")

    with open(camera_config_path, "r", encoding="utf-8") as f:
        camera_config = json.load(f)

    video_queue = Queue()
    video_capture_server = VideoCaptureServerFFmpeg(
        camera_config["camera_config"],
        yolo_model_path,
        saved_video_dir,
        video_queue=video_queue
    )

    # 设置信号处理器以优雅地关闭
    def signal_handler(sig, frame):
        logger.info("接收到终止信号，正在清理...")
        video_capture_server.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    video_capture_server.run_all_cameras()



