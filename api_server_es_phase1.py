import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from queue import Queue
import threading
import glob
import re
from datetime import datetime
import math
import random
import subprocess
import signal
import sys
import atexit
import mysql.connector
from mysql.connector import Error
import cv2
import time

from video_process.log_feature_extra.elasticsearch_worker import ElasticSearchWorker
from video_process.person_search.person_search_engine import PersonSearchEngine

# Force FFmpeg to use TCP for RTSP to improve stability and avoid UDP packet loss/timeouts
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# 摄像头配置路径
CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_process', 'camera_config.json')
TRACKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_process', 'realtime_stream_tracker.py')
TRACKING_PUSH_URL = 'http://127.0.0.1:5000/api/tracking/push'
STREAM_IDLE_TIMEOUT_SECONDS = 30

# 摄像头类
class Camera:
    def __init__(self, camera_id, rtsp_url):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame = None
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.cap = None
        self.active_clients = 0
        self.last_access_time = time.time()
        self.idle_timeout = STREAM_IDLE_TIMEOUT_SECONDS

    def _release_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def ensure_started(self):
        with self.state_lock:
            self.last_access_time = time.time()
            if self.thread and self.thread.is_alive():
                return
            self.running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            logger.info(f"Lazy start stream thread for {self.camera_id}")

    def acquire_viewer(self):
        with self.state_lock:
            self.active_clients += 1
            self.last_access_time = time.time()
        self.ensure_started()

    def release_viewer(self):
        with self.state_lock:
            if self.active_clients > 0:
                self.active_clients -= 1
            self.last_access_time = time.time()

    def touch(self):
        with self.state_lock:
            self.last_access_time = time.time()

    def update(self):
        try:
            while True:
                with self.state_lock:
                    if not self.running:
                        break

                    idle_seconds = time.time() - self.last_access_time
                    if self.active_clients == 0 and idle_seconds > self.idle_timeout:
                        logger.info(f"Camera {self.camera_id} idle for {idle_seconds:.1f}s, stopping stream thread")
                        self.running = False
                        break

                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened():
                        logger.warning(f"Warning: Could not open video source for {self.camera_id}. Retrying in 5 seconds...")
                        self._release_capture()
                        time.sleep(5)
                        continue

                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    logger.info(f"Successfully connected to {self.camera_id}")

                ret, frame = self.cap.read()
                if ret:
                    try:
                        height, width = frame.shape[:2]
                        target_width = 640
                        if width > target_width:
                            scale = target_width / width
                            target_height = int(height * scale)
                            frame = cv2.resize(frame, (target_width, target_height))

                        with self.lock:
                            self.frame = frame
                    except Exception as e:
                        logger.error(f"Error processing frame for {self.camera_id}: {e}")
                else:
                    logger.warning(f"Lost connection to {self.camera_id}. Reconnecting...")
                    self._release_capture()
                    time.sleep(1)
        finally:
            self._release_capture()
            with self.state_lock:
                self.running = False
                self.thread = None

    def get_frame(self):
        self.touch()
        with self.lock:
            if self.frame is None:
                return None
            
            try:
                ret, jpeg = cv2.imencode('.jpg', self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    return None
                return jpeg.tobytes()
            except Exception as e:
                logger.error(f"Error encoding frame for {self.camera_id}: {e}")
                return None

    def stop(self):
        with self.state_lock:
            self.running = False
            worker = self.thread
        if worker and worker.is_alive():
            worker.join(timeout=2)
        self._release_capture()

# 全局摄像头字典
cameras = {}
camera_urls = {}
camera_registry_lock = threading.Lock()

# AI 子进程管理
ai_processes = {}
ai_process_lock = threading.Lock()

def load_cameras():
    global camera_urls
    if not os.path.exists(CAMERA_CONFIG_PATH):
        logger.error(f"Config file not found at {CAMERA_CONFIG_PATH}")
        return

    try:
        with open(CAMERA_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        loaded_urls = {}
        for cam_conf in config.get('camera_config', []):
            cam_id = cam_conf['camera_id']
            url = cam_conf['camera_url']
            loaded_urls[cam_id] = url

        with camera_registry_lock:
            camera_urls = loaded_urls

        logger.info(f"Loaded {len(camera_urls)} camera configs (lazy mode)")
    except Exception as e:
        logger.error(f"Error loading cameras: {e}")

def get_or_create_camera(camera_id):
    with camera_registry_lock:
        if camera_id in cameras:
            return cameras[camera_id]

        rtsp_url = camera_urls.get(camera_id)
        if not rtsp_url:
            return None

        camera = Camera(camera_id, rtsp_url)
        cameras[camera_id] = camera
        return camera

def _reap_ai_processes_locked():
    dead_ids = []
    for camera_id, proc in ai_processes.items():
        if proc.poll() is not None:
            try:
                proc.wait(timeout=0)
            except Exception:
                pass
            dead_ids.append(camera_id)
    for camera_id in dead_ids:
        ai_processes.pop(camera_id, None)

def _stop_ai_process_locked(camera_id, wait_timeout=8):
    proc = ai_processes.get(camera_id)
    if not proc:
        return False, 'not_running'

    if proc.poll() is not None:
        try:
            proc.wait(timeout=0)
        except Exception:
            pass
        ai_processes.pop(camera_id, None)
        return False, 'already_exited'

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=3)

    ai_processes.pop(camera_id, None)
    return True, 'stopped'

def cleanup_runtime_resources():
    with ai_process_lock:
        for camera_id in list(ai_processes.keys()):
            _stop_ai_process_locked(camera_id)

    with camera_registry_lock:
        for camera in list(cameras.values()):
            camera.stop()
        cameras.clear()

atexit.register(cleanup_runtime_resources)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'database': 'monitor_database',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
    'autocommit': False,
    'pool_name': 'api_server_pool',
    'pool_size': 5
}

def get_db_connection():
    """获取数据库连接"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        logger.error(f"数据库连接失败: {e}")
        return None

# 全局变量
es_worker = None
person_search_engine = None
video_description_path = None
saved_video_path = None
person_crops_path = None

# 全局布局配置
LAYOUT_CONFIG = {
    'mode': 'grid',  # 'grid', 'spiral', 'random'
    'grid_spacing': 45,
    'spiral_radius_step': 35,
    'random_offset_range': 80
}

# 精确的摄像头区域映射配置（基于实际地图尺寸 3156x1380 像素）
# 大红色框范围：X(374-2728) Y(345-1030)，只在蓝色走廊区域生成事件点
CAMERA_REGIONS = {
    'camera1': {
        # 上方横向主走廊（大框顶部，房间上方）
        'min_pixel_x': 374, 'max_pixel_x': 2728, 
        'min_pixel_y': 345, 'max_pixel_y': 445
    },
    'camera2': {
        # 左侧纵向走廊（左侧房间的左边）
        'min_pixel_x': 374, 'max_pixel_x': 574, 
        'min_pixel_y': 445, 'max_pixel_y': 1030
    },
    'camera3': {
        # 右侧纵向走廊（右侧房间的右边） 
        'min_pixel_x': 2528, 'max_pixel_x': 2728, 
        'min_pixel_y': 445, 'max_pixel_y': 1030
    },
    'camera4': {
        # 下方横向走廊（大框底部，房间下方）
        'min_pixel_x': 374, 'max_pixel_x': 2728, 
        'min_pixel_y': 965, 'max_pixel_y': 1030
    },
    'camera5': {
        # 房间之间的走廊（左侧房间和中间设备之间）
        'min_pixel_x': 1342, 'max_pixel_x': 1522, 
        'min_pixel_y': 565, 'max_pixel_y': 965
    },
    'camera6': {
        # 房间之间的走廊（中间设备和右侧房间之间）
        'min_pixel_x': 1622, 'max_pixel_x': 1802, 
        'min_pixel_y': 565, 'max_pixel_y': 965
    },
    'default': {
        # 默认走廊区域（上方主走廊）
        'min_pixel_x': 374, 'max_pixel_x': 2728, 
        'min_pixel_y': 345, 'max_pixel_y': 445
    }
}



@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """获取摄像头列表"""
    try:
        if not os.path.exists(CAMERA_CONFIG_PATH):
            return jsonify({"error": "Config file not found"}), 404
            
        with open(CAMERA_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return jsonify(config.get('camera_config', []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def gen(camera):
    """视频流生成器"""
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/video_feed/<camera_id>', methods=['GET', 'OPTIONS'])
def video_feed(camera_id):
    """视频流路由"""
    if request.method == 'OPTIONS':
        response = Response(status=204)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = '*'
        return response

    if not camera_urls:
        load_cameras()

    camera = get_or_create_camera(camera_id)
    if not camera:
        return jsonify({'error': f'Camera {camera_id} not found'}), 404

    def stream_generator():
        camera.acquire_viewer()
        try:
            yield from gen(camera)
        finally:
            camera.release_viewer()

    response = Response(stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/api/ai/start/<camera_id>', methods=['POST'])
def start_ai_inference(camera_id):
    """按需启动指定摄像头 AI 推理进程"""
    if not camera_urls:
        load_cameras()
    if camera_id not in camera_urls:
        return jsonify({'ok': False, 'error': f'camera_id {camera_id} not found in config'}), 404

    with ai_process_lock:
        _reap_ai_processes_locked()
        existing = ai_processes.get(camera_id)
        if existing and existing.poll() is None:
            return jsonify({'ok': True, 'status': 'already_running', 'camera_id': camera_id, 'pid': existing.pid})

        cmd = [
            sys.executable,
            TRACKER_SCRIPT_PATH,
            '--camera',
            camera_id,
            '--api-url',
            TRACKING_PUSH_URL,
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(TRACKER_SCRIPT_PATH),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        ai_processes[camera_id] = proc

    logger.info(f"Started AI process for {camera_id}, pid={proc.pid}")
    return jsonify({'ok': True, 'status': 'started', 'camera_id': camera_id, 'pid': proc.pid})

@app.route('/api/ai/stop/<camera_id>', methods=['POST'])
def stop_ai_inference(camera_id):
    """停止指定摄像头 AI 推理进程（优雅终止 + 防僵尸回收）"""
    with ai_process_lock:
        _reap_ai_processes_locked()
        stopped, status = _stop_ai_process_locked(camera_id)

    if status == 'not_running':
        return jsonify({'ok': True, 'status': 'not_running', 'camera_id': camera_id})

    logger.info(f"Stopped AI process for {camera_id}, status={status}")
    return jsonify({'ok': True, 'status': status, 'camera_id': camera_id, 'stopped': stopped})

def init_servers():
    """初始化服务器"""
    global es_worker, person_search_engine, video_description_path, saved_video_path, person_crops_path
    
    # 初始化摄像头
    logger.info("Loading cameras...")
    load_cameras()
    
    base_dir = os.path.join(os.path.dirname(__file__), "video_process")
    video_description_path = os.path.join(base_dir, "video_description.json")
    saved_video_path = os.path.join(base_dir, "saved_video")
    person_crops_path = os.path.join(base_dir, "person_crops")
    
    # =========================
    # 【阶段一-仅替换搜索大动脉】
    # 将原 FAISS 检索实例替换为 ElasticSearchWorker。
    # 注意：轨迹/事件/视频信息相关逻辑保持不变。
    # =========================
    logger.info("Initializing Elasticsearch worker (phase-1 query migration)...")
    es_worker = ElasticSearchWorker(
        emd_model_path="/root/data1/bge_zh_v1.5/",
        video_dsp_queue=Queue(),
        db_config=DB_CONFIG,
        es_url="http://219.216.99.30:9200",
        index_name="video_logs_index"
    )
    logger.info("Elasticsearch worker initialized.")

    # 初始化行人搜索引擎
    logger.info("Initializing Person Search Engine...")
    try:
        person_search_engine = PersonSearchEngine()
        logger.info("PersonSearchEngine 初始化成功")
    except Exception as e:
        logger.error(f"PersonSearchEngine 初始化失败: {e}")
    
    logger.info("所有服务器初始化完成")

@app.route('/api/query', methods=['POST'])
def query():
    """处理前端的查询请求"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'error_message': '请提供查询内容'
            }), 400
        
        query_text = data['query'].strip()
        if not query_text:
            return jsonify({
                'status': 'error',
                'error_message': '查询内容不能为空'
            }), 400
        
        logger.info(f"收到查询请求: {query_text}")

        # =========================
        # 【阶段一-仅替换搜索链路】
        # 只替换 /api/query 的检索后端为 ES Hybrid Search。
        # 其余轨迹/事件接口一律保持原状。
        # =========================
        if not es_worker:
            return jsonify({
                'status': 'error',
                'error_message': 'ES检索服务未初始化'
            }), 500
        
        # 保持原有参数语义
        k = data.get('k', 5)  # 默认返回5个结果
        alpha = data.get('alpha', 0.7)  # 语义检索权重，默认0.7
        
        # ES 返回结构: [{"video_id": int, "description": str, "score": float}, ...]
        search_results = es_worker.hybrid_search(query_text, k=k, alpha=alpha)
        
        if not search_results:
            return jsonify({
                'status': 'success',
                'data': {
                    '相关日志': [],
                    '总结报告': '没有找到相关的监控记录。'
                }
            })
        
        # =========================
        # 将 ES 结果中的 video_id 映射回前端沿用的 video_name (v_k)
        # =========================
        result_video_ids = [item.get('video_id') for item in search_results if item.get('video_id') is not None]
        if not result_video_ids:
            return jsonify({
                'status': 'success',
                'data': {
                    '相关日志': [],
                    '总结报告': '没有找到相关的监控记录。'
                }
            })

        conn = None
        cursor = None
        id_to_name = {}
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                placeholders = ','.join(['%s'] * len(result_video_ids))
                sql = f"SELECT id, video_name FROM videos WHERE id IN ({placeholders})"
                cursor.execute(sql, tuple(result_video_ids))
                rows = cursor.fetchall()
                id_to_name = {int(row['id']): row['video_name'] for row in rows if row.get('video_name')}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        
        # 构建相关日志数据
        related_logs = []
        summaries = []

        # 直接使用 ES 返回的 description，保持原返回结构不变
        for item in search_results:
            video_id = item.get('video_id')
            description = (item.get('description') or '').strip()
            if video_id is None:
                continue

            v_k = id_to_name.get(int(video_id), str(video_id))

            log_entry = {
                f"{v_k}日志": v_k,
                f"{v_k}概述": description[:100] + ('...' if len(description) > 100 else '')
            }
            related_logs.append(log_entry)

            if description:
                summaries.append(f"时间{v_k}: {description}")
        
        # 生成总结报告
        summary_base = f"基于查询{query_text}，找到{len(search_results)}条相关监控记录。"
        if summaries:
            summary_detail = "主要内容包括：" + "; ".join(summaries[:3])
        else:
            summary_detail = ""
        summary_report = summary_base + summary_detail
        
        response_data = {
            '相关日志': related_logs,
            '总结报告': summary_report
        }
        
        return jsonify({
            'status': 'success',
            'data': response_data
        })
        
    except Exception as e:
        logger.error(f"查询处理错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_message': f'服务器内部错误: {str(e)}'
        }), 500

@app.route('/api/video_persons/<video_id>', methods=['GET'])
def get_video_persons(video_id):
    """获取视频中出现的所有人物（含可选图片路径）"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 首先检查 video_id 是否存在 (可能是 id 或 video_name)
        # 这里假设传入的是 video_id (int)
        # 如果传入的是 video_name (str)，需要先查 id
        
        real_video_id = video_id
        if not str(video_id).isdigit():
             # 尝试通过名称查找 ID
            cursor.execute("SELECT id FROM videos WHERE video_name = %s", (video_id,))
            res = cursor.fetchone()
            if res:
                real_video_id = res['id']
            else:
                return jsonify({'error': '视频不存在'}), 404

        if not person_search_engine:
            return jsonify({'error': '搜索引擎未初始化'}), 500

        results = person_search_engine.get_persons_in_video(real_video_id)
        unique_persons_dict = {}
        for row in results:
            person_index = row.get('person_index')
            if person_index is None:
                continue
            if person_index not in unique_persons_dict:
                unique_persons_dict[person_index] = {
                    'person_index': person_index,
                    'person_image_path': row.get('person_image_path')
                }

        persons = [
            unique_persons_dict[idx]
            for idx in sorted(unique_persons_dict.keys())
        ]
        
        return jsonify({
            'video_id': real_video_id,
            'persons': persons
        })
        
    except Exception as e:
        logger.error(f"获取视频人物列表失败: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route('/api/person_crops/<path:filename>', methods=['GET'])
def get_person_crop(filename):
    """提供人物裁剪图静态访问"""
    try:
        if not person_crops_path or not os.path.exists(person_crops_path):
            return jsonify({'error': '人物图片目录不存在'}), 404

        normalized_filename = os.path.normpath(filename).replace('\\', '/')
        if normalized_filename.startswith('..') or os.path.isabs(normalized_filename):
            return jsonify({'error': '非法路径'}), 400

        return send_from_directory(person_crops_path, normalized_filename)
    except Exception as e:
        logger.error(f"获取人物图片失败: {e}")
        return jsonify({'error': '获取人物图片失败'}), 500

@app.route('/api/global_trajectory', methods=['POST'])
def global_trajectory():
    """全局轨迹搜索"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        person_index = data.get('person_index')
        time_window = data.get('time_window', 10)
        
        if not video_id or person_index is None:
            return jsonify({'error': '缺少必要参数 video_id 或 person_index'}), 400
            
        # 确保 video_id 是 int
        real_video_id = video_id
        if not str(video_id).isdigit():
             # 如果前端传的是 video_name，需要转换
             conn = get_db_connection()
             if conn:
                 cursor = conn.cursor(dictionary=True)
                 cursor.execute("SELECT id FROM videos WHERE video_name = %s", (video_id,))
                 res = cursor.fetchone()
                 cursor.close()
                 conn.close()
                 if res:
                     real_video_id = res['id']
                 else:
                     return jsonify({'error': '视频不存在'}), 404
        
        if not person_search_engine:
            return jsonify({'error': '搜索引擎未初始化'}), 500
            
        results = person_search_engine.search_target(real_video_id, person_index, time_window)
        
        # 转换结果格式以适配前端显示
        # 前端需要 scene_coords (x, y, z)
        formatted_results = []
        
        for res in results:
            traj_data = res['trajectory']
            coordinates = traj_data.get('points', [])
            scene_coords = []
            
            for coord in coordinates:
                if len(coord) >= 2:
                    real_x, real_y = coord[0], coord[1]
                    scene_x, scene_z = convert_pixel_to_scene_coords(real_x, real_y)
                    scene_coords.append({
                        'x': round(scene_x, 3),
                        'y': 0,
                        'z': round(scene_z, 3)
                    })
            
            if scene_coords:
                formatted_results.append({
                    'video_id': res['video_id'],
                    'video_name': res['video_name'],
                    'camera_id': res['camera_id'],
                    'person_index': res['person_index'],
                    'similarity': res['similarity'],
                    'time': res['time'],
                    'coordinates': scene_coords
                })
                
        return jsonify({
            'status': 'success',
            'count': len(formatted_results),
            'trajectories': formatted_results
        })
        
    except Exception as e:
        logger.error(f"全局轨迹搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<video_name>')
def get_video(video_name):
    """提供视频文件访问"""
    try:
        # 安全检查文件名
        if '..' in video_name or '/' in video_name:
            return jsonify({'error': '无效的文件名'}), 400
        
        # 检查文件是否存在
        video_path = os.path.join(saved_video_path, video_name)
        if not os.path.exists(video_path):
            return jsonify({'error': '视频文件不存在'}), 404
        
        # 返回视频文件，并设置正确的MIME类型和头信息
        response = send_from_directory(saved_video_path, video_name)
        response.headers['Content-Type'] = 'video/mp4'
        response.headers['Accept-Ranges'] = 'bytes'  # 支持范围请求，用于视频播放控制
        response.headers['Cache-Control'] = 'public, max-age=3600'  # 缓存1小时
        return response
        
    except Exception as e:
        logger.error(f"视频访问错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

@app.route('/api/video_info/<video_key>')
def get_video_info(video_key):
    """获取特定视频的详细信息"""
    try:
        with open(video_description_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        if video_key not in video_data:
            return jsonify({'error': '视频信息不存在'}), 404
        
        video_info = video_data[video_key]
        
        # 查找对应的视频文件
        video_files = []
        if saved_video_path and os.path.exists(saved_video_path):
            pattern = os.path.join(saved_video_path, f"{video_key}*")
            video_files = [os.path.basename(f) for f in glob.glob(pattern)]
        
        result = {
            'video_key': video_key,
            'analyse_result': video_info.get('analyse_result'),
            'timestamp': video_info.get('timestamp'),
            'camera_id': video_info.get('camera_id'),
            'video_files': video_files,
            'is_embedding': video_info.get('is_embedding', False)
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取视频信息错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

@app.route('/api/health')
def health_check():
    """健康检查接口"""
    # =========================
    # 【阶段一-健康检查切换到ES】
    # =========================
    es_alive = False
    index_total = 0
    if es_worker:
        try:
            es_alive = bool(es_worker.es.ping())
            if es_alive:
                index_total = int(es_worker.es.count(index=es_worker.index_name).get('count', 0))
        except Exception as e:
            logger.warning(f"健康检查读取ES状态失败: {e}")

    return jsonify({
        'status': 'healthy',
        'service': 'api_server_es_phase1',
        'query_backend': 'elasticsearch',
        'es_ready': es_worker is not None,
        'es_alive': es_alive,
        'index_total': index_total
    })

@app.route('/api/stats')
def get_stats():
    """获取系统统计信息"""
    try:
        with open(video_description_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        total_videos = len(video_data)
        analyzed_videos = sum(1 for v in video_data.values() if v.get('analyse_result'))
        embedded_videos = sum(1 for v in video_data.values() if v.get('is_embedding'))

        # =========================
        # 【阶段一-统计中的索引规模切换到ES】
        # =========================
        index_size = 0
        if es_worker:
            try:
                index_size = int(es_worker.es.count(index=es_worker.index_name).get('count', 0))
            except Exception as e:
                logger.warning(f"读取ES索引数量失败: {e}")
        
        return jsonify({
            'total_videos': total_videos,
            'analyzed_videos': analyzed_videos,
            'embedded_videos': embedded_videos,
            'index_size': index_size
        })
        
    except Exception as e:
        logger.error(f"获取统计信息错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

def parse_video_key_to_timestamp(video_key):
    """解析视频key中的时间戳，格式如camera1_20250604_150510"""
    try:
        parts = video_key.split('_')
        if len(parts) >= 3:
            date_str = parts[1]  # 20250604
            time_str = parts[2]  # 150510
            
            # 解析日期 YYYYMMDD
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            # 解析时间 HHMMSS
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            # 创建datetime对象并转换为时间戳
            dt = datetime(year, month, day, hour, minute, second)
            return int(dt.timestamp())
    except:
        pass
    return None

def get_camera_position(camera_id):
    """根据摄像头ID返回在地图上的位置坐标"""
    config_path = os.path.join(os.path.dirname(__file__), "video_process", "camera_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                camera_configs = config.get('camera_config', [])
                
                # 在摄像头配置列表中查找对应的摄像头
                for camera_config in camera_configs:
                    if camera_config.get('camera_id') == camera_id:
                        position = camera_config.get('position')
                        if position and 'pixel_x' in position and 'pixel_y' in position:
                            return position
                
                # 如果找不到特定摄像头配置，使用默认位置
                default_pos = config.get('default_position', {'pixel_x': 400, 'pixel_y': 250})
                logger.warning(f"摄像头 {camera_id} 未在配置文件中找到位置信息，使用默认位置")
                return default_pos
        else:
            logger.warning(f"摄像头配置文件 {config_path} 不存在，使用硬编码位置")
    except Exception as e:
        logger.error(f"读取摄像头配置文件错误: {str(e)}")
    
    # 备用硬编码位置（如果配置文件不存在或读取失败）
    camera_positions = {
        'camera1': {'pixel_x': 180, 'pixel_y': 120},
        'camera2': {'pixel_x': 520, 'pixel_y': 280},
    }
    return camera_positions.get(camera_id, {'pixel_x': 400, 'pixel_y': 250})

@app.route('/api/events')
def get_events():
    """获取所有事件数据，转换为前端需要的格式"""
    try:
        import random
        with open(video_description_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        # 先按摄像头分组，计算每个摄像头的事件数量
        camera_events = {}
        for video_key, video_info in video_data.items():
            camera_id = video_key.split('_')[0]
            timestamp = parse_video_key_to_timestamp(video_key)
            if not timestamp:
                continue
                
            if camera_id not in camera_events:
                camera_events[camera_id] = []
            
            camera_events[camera_id].append({
                'video_key': video_key,
                'video_info': video_info,
                'timestamp': timestamp
            })
        
        # 按时间戳排序每个摄像头的事件
        for camera_id in camera_events:
            camera_events[camera_id].sort(key=lambda x: x['timestamp'])
        
        events = []
        layout_mode = LAYOUT_CONFIG['mode']  # 使用全局配置的布局模式
        
        for camera_id, camera_event_list in camera_events.items():
            base_position = get_camera_position(camera_id)
            total_events = len(camera_event_list)
            
            for event_index, event_data in enumerate(camera_event_list):
                video_key = event_data['video_key']
                video_info = event_data['video_info']
                timestamp = event_data['timestamp']
                
                # 根据布局模式计算位置
                if layout_mode == 'grid':
                    position = calculate_grid_position(camera_id, event_index, base_position, total_events)
                elif layout_mode == 'spiral':
                    position = calculate_spiral_position(camera_id, event_index, base_position)
                else:  # 随机布局（原始方式）
                    random.seed(hash(video_key) % 1000000)
                    offset_range = LAYOUT_CONFIG['random_offset_range']
                    offset_x = random.randint(-offset_range, offset_range)
                    offset_y = random.randint(-offset_range, offset_range)
                    position = {
                        'pixel_x': base_position['pixel_x'] + offset_x,
                        'pixel_y': base_position['pixel_y'] + offset_y
                    }
                
                # 获取视频文件名
                video_file = None
                if video_info.get('video_path'):
                    video_file = os.path.basename(video_info['video_path'])
                
                # 检查是否为异常事件
                is_abnormal = False
                analyse_result = video_info.get('analyse_result', '')
                if analyse_result:
                    abnormal_keywords = ['异常', '打架', '闯入', '跌倒', '争执', '暴力', '紧急']
                    is_abnormal = any(keyword in analyse_result for keyword in abnormal_keywords)
                
                # 构建事件对象
                event = {
                    'id': video_key,
                    'timestamp': timestamp,
                    'position': position,
                    'content': analyse_result or '监控记录',
                    'videoFile': video_file,
                    'cam_id': camera_id,
                    'date_str': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                    'time_str': datetime.fromtimestamp(timestamp).strftime('%H:%M:%S'),
                    'index': video_info.get('idx', 0),
                    'sourceFile': video_key,
                    'is_abnormal': is_abnormal,
                    'camera_event_index': event_index,  # 在该摄像头中的序号
                    'total_camera_events': total_events  # 该摄像头总事件数
                }
                
                events.append(event)
        
        # 按时间戳排序所有事件
        events.sort(key=lambda x: x['timestamp'])
        
        # 统计摄像头分布
        camera_event_counts = {}
        for camera_id, camera_event_list in camera_events.items():
            camera_event_counts[camera_id] = len(camera_event_list)
        
        logger.info(f"返回 {len(events)} 个事件，使用 {layout_mode} 布局，摄像头分布: {camera_event_counts}")
        return jsonify(events)
        
    except Exception as e:
        logger.error(f"获取事件数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

def calculate_grid_position(camera_id, event_index, base_position, total_events_for_camera):
    """
    为事件计算网格布局位置，避免重叠
    """
    if total_events_for_camera <= 1:
        return base_position
    
    # 网格配置
    grid_spacing = LAYOUT_CONFIG['grid_spacing']  # 使用全局配置
    max_cols = 8       # 最大列数
    
    # 计算网格行列
    cols = min(max_cols, int(total_events_for_camera**0.5) + 1)
    row = event_index // cols
    col = event_index % cols
    
    # 计算偏移，使网格以摄像头位置为中心
    offset_x = (col - (cols - 1) / 2) * grid_spacing
    offset_y = (row - 1) * grid_spacing  # 向下排列
    
    return {
        'pixel_x': base_position['pixel_x'] + offset_x,
        'pixel_y': base_position['pixel_y'] + offset_y
    }

def calculate_spiral_position(camera_id, event_index, base_position):
    """
    螺旋布局：以摄像头为中心，按螺旋形排列事件点
    """
    if event_index == 0:
        return base_position
    
    # 螺旋参数
    radius_step = LAYOUT_CONFIG['spiral_radius_step']   # 使用全局配置
    angle_step = 60    # 角度步进（度）
    
    # 计算当前点在第几圈
    circle = int((event_index - 1) // 6) + 1  # 每圈6个点
    position_in_circle = (event_index - 1) % 6
    
    # 计算半径和角度
    radius = circle * radius_step
    angle_degrees = position_in_circle * angle_step + (circle % 2) * 30  # 每圈错开30度
    angle_radians = math.radians(angle_degrees)
    
    # 计算坐标
    offset_x = radius * math.cos(angle_radians)
    offset_y = radius * math.sin(angle_radians)
    
    return {
        'pixel_x': base_position['pixel_x'] + offset_x,
        'pixel_y': base_position['pixel_y'] + offset_y
    }

@app.route('/api/layout_config', methods=['GET', 'POST'])
def layout_config():
    """布局配置管理接口"""
    global LAYOUT_CONFIG
    
    if request.method == 'GET':
        return jsonify(LAYOUT_CONFIG)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # 更新配置
            if 'mode' in data and data['mode'] in ['grid', 'spiral', 'random']:
                LAYOUT_CONFIG['mode'] = data['mode']
            
            if 'grid_spacing' in data and isinstance(data['grid_spacing'], (int, float)):
                LAYOUT_CONFIG['grid_spacing'] = max(20, min(100, data['grid_spacing']))
            
            if 'spiral_radius_step' in data and isinstance(data['spiral_radius_step'], (int, float)):
                LAYOUT_CONFIG['spiral_radius_step'] = max(20, min(100, data['spiral_radius_step']))
            
            if 'random_offset_range' in data and isinstance(data['random_offset_range'], (int, float)):
                LAYOUT_CONFIG['random_offset_range'] = max(30, min(150, data['random_offset_range']))
            
            logger.info(f"布局配置已更新: {LAYOUT_CONFIG}")
            return jsonify({'status': 'success', 'config': LAYOUT_CONFIG})
            
        except Exception as e:
            logger.error(f"更新布局配置错误: {str(e)}")
            return jsonify({'error': '配置更新失败'}), 400

@app.route('/layout_control')
def layout_control():
    """布局控制面板页面"""
    try:
        with open('layout_control.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return "布局控制面板文件未找到", 404

def get_camera_coverage_area(camera_id):
    """根据摄像头ID返回覆盖区域范围，优先使用精确映射配置"""
    
    # 优先使用精确的区域映射配置
    if camera_id in CAMERA_REGIONS:
        logger.debug(f"使用精确映射配置为摄像头 {camera_id}")
        return CAMERA_REGIONS[camera_id]
    
    # 备用：尝试从配置文件读取
    config_path = os.path.join(os.path.dirname(__file__), "video_process", "camera_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                camera_configs = config.get('camera_config', [])
                
                # 在摄像头配置列表中查找对应的摄像头
                for camera_config in camera_configs:
                    if camera_config.get('camera_id') == camera_id:
                        coverage_area = camera_config.get('coverage_area')
                        if coverage_area and all(key in coverage_area for key in ['min_pixel_x', 'max_pixel_x', 'min_pixel_y', 'max_pixel_y']):
                            logger.debug(f"从配置文件获取摄像头 {camera_id} 的区域配置")
                            return coverage_area
                
                # 如果找不到特定摄像头配置，使用默认区域
                default_area = config.get('default_coverage_area')
                if default_area and all(key in default_area for key in ['min_pixel_x', 'max_pixel_x', 'min_pixel_y', 'max_pixel_y']):
                    logger.warning(f"摄像头 {camera_id} 未在配置文件中找到覆盖区域，使用默认区域")
                    return default_area
        else:
            logger.warning(f"摄像头配置文件 {config_path} 不存在，使用精确映射区域")
    except Exception as e:
        logger.error(f"读取摄像头配置文件错误: {str(e)}")
    
    # 最终备用：使用精确映射中的默认区域
    logger.warning(f"为摄像头 {camera_id} 使用默认精确映射区域")
    return CAMERA_REGIONS.get('default', {
        'min_pixel_x': 300, 'max_pixel_x': 500, 
        'min_pixel_y': 150, 'max_pixel_y': 350
    })

def generate_random_position_in_area(camera_id, event_key):
    """在摄像头覆盖区域内生成随机位置，采用精确区域定位"""
    try:
        coverage_area = get_camera_coverage_area(camera_id)
        
        # 验证区域配置的完整性
        required_keys = ['min_pixel_x', 'max_pixel_x', 'min_pixel_y', 'max_pixel_y']
        if not all(key in coverage_area for key in required_keys):
            logger.error(f"摄像头 {camera_id} 的覆盖区域配置不完整")
            return None
        
        # 使用事件key作为随机种子，确保同一事件的位置一致但分散
        random.seed(hash(event_key) % 1000000)
        
        # 计算区域尺寸，用于控制分散程度
        area_width = coverage_area['max_pixel_x'] - coverage_area['min_pixel_x']
        area_height = coverage_area['max_pixel_y'] - coverage_area['min_pixel_y']
        
        # 在区域内生成随机坐标，增加边界缓冲区避免贴边
        margin_x = max(5, area_width * 0.05)  # 5% 边界缓冲或最少5像素
        margin_y = max(5, area_height * 0.05)
        
        pixel_x = random.uniform(
            coverage_area['min_pixel_x'] + margin_x, 
            coverage_area['max_pixel_x'] - margin_x
        )
        pixel_y = random.uniform(
            coverage_area['min_pixel_y'] + margin_y, 
            coverage_area['max_pixel_y'] - margin_y
        )
        
        logger.debug(f"摄像头 {camera_id} 生成位置: ({pixel_x:.2f}, {pixel_y:.2f}) 在区域 {coverage_area}")
        
        return {
            'pixel_x': round(pixel_x, 2),
            'pixel_y': round(pixel_y, 2)
        }
        
    except Exception as e:
        logger.error(f"生成随机位置失败: {str(e)}")
        return None

def parse_event_filename(filename):
    """解析事件文件名格式：CAM[ID]_[日期]_[时间]_[序号].txt 或 video_key格式"""
    try:
        # 去除文件扩展名
        basename = os.path.splitext(filename)[0]
        
        # 尝试解析新格式：CAM1_20250609_143020_001.txt
        cam_pattern = r'CAM(\d+)_(\d{8})_(\d{6})_(\d+)'
        match = re.match(cam_pattern, basename)
        if match:
            cam_id, date_str, time_str, seq = match.groups()
            camera_id = f"camera{cam_id}"
            
            # 解析日期时间
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            dt = datetime(year, month, day, hour, minute, second)
            timestamp = int(dt.timestamp())
            
            return {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'sequence': int(seq),
                'original_filename': filename
            }
        
        # 尝试解析原有格式：camera1_20250604_150510
        old_pattern = r'(camera\d+)_(\d{8})_(\d{6})'
        match = re.match(old_pattern, basename)
        if match:
            camera_id, date_str, time_str = match.groups()
            
            # 解析日期时间
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])
            
            dt = datetime(year, month, day, hour, minute, second)
            timestamp = int(dt.timestamp())
            
            return {
                'camera_id': camera_id,
                'timestamp': timestamp,
                'sequence': 0,
                'original_filename': filename
            }
            
    except Exception as e:
        logger.error(f"解析文件名 {filename} 失败: {str(e)}")
    
    return None

@app.route('/api/3d_config')
def get_3d_config():
    """获取3D显示配置"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "video_process", "camera_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # 提取3D相关配置
                result = {
                    'map_info': config.get('map_info', {}),
                    '3d_display_config': config.get('3d_display_config', {}),
                    'camera_coverage_areas': {}
                }
                
                # 添加所有摄像头的覆盖区域
                for camera_config in config.get('camera_config', []):
                    camera_id = camera_config.get('camera_id')
                    if camera_id and 'coverage_area' in camera_config:
                        result['camera_coverage_areas'][camera_id] = camera_config['coverage_area']
                
                # 添加默认覆盖区域
                if 'default_coverage_area' in config:
                    result['default_coverage_area'] = config['default_coverage_area']
                
                return jsonify(result)
        else:
            # 返回默认配置
            return jsonify({
                'map_info': {
                    'map_size': {'width': 800, 'height': 600}
                },
                '3d_display_config': {
                    'plane_size': {'width': 20.0, 'height': 15.0},
                    'marker_config': {
                        'normal_color': '#00ffff',
                        'abnormal_color': '#ff0000',
                        'height': 0.8,
                        'radius': 0.15
                    }
                }
            })
            
    except Exception as e:
        logger.error(f"获取3D配置错误: {str(e)}")
        return jsonify({'error': '获取配置失败'}), 500

@app.route('/api/events_3d')
def get_events_3d():
    """获取3D显示用的事件数据"""
    conn = None
    cursor = None
    try:
        # 读取 JSON 作为分析文本/兼容回退
        video_data = {}
        if video_description_path and os.path.exists(video_description_path):
            try:
                with open(video_description_path, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取 video_description.json 失败，将仅使用DB: {e}")

        conn = get_db_connection()
        if not conn:
            logger.warning("events_3d 数据源回退到 JSON：数据库连接失败")
            if not video_data:
                return jsonify([])

            events = []
            for video_key, video_info in video_data.items():
                event_info = parse_event_filename(video_key)
                if not event_info:
                    camera_id = video_key.split('_')[0] if '_' in video_key else 'camera1'
                    timestamp = parse_video_key_to_timestamp(video_key)
                    if not timestamp:
                        continue
                    event_info = {
                        'camera_id': camera_id,
                        'timestamp': timestamp,
                        'sequence': 0,
                        'original_filename': video_key
                    }

                has_trajectory = 'trajectory_data' in video_info and 'trajectories' in video_info['trajectory_data']
                if has_trajectory and video_info['trajectory_data']['trajectories']:
                    first_trajectory = video_info['trajectory_data']['trajectories'][0]
                    if first_trajectory.get('coordinates') and len(first_trajectory['coordinates'][0]) >= 2:
                        start_coord = first_trajectory['coordinates'][0]
                        position = {'pixel_x': start_coord[0], 'pixel_y': start_coord[1]}
                    else:
                        position = generate_random_position_in_area(event_info['camera_id'], video_key)
                else:
                    position = generate_random_position_in_area(event_info['camera_id'], video_key)

                if not position:
                    continue

                analyse_result = video_info.get('analyse_result', '')
                abnormal_keywords = ['异常', '打架', '闯入', '跌倒', '争执', '暴力', '紧急', '危险']
                is_abnormal = any(keyword in analyse_result for keyword in abnormal_keywords) if analyse_result else False

                video_file = os.path.basename(video_info['video_path']) if video_info.get('video_path') else None
                events.append({
                    'id': video_key,
                    'timestamp': event_info['timestamp'],
                    'position': position,
                    'content': analyse_result or '监控记录',
                    'videoFile': video_file,
                    'camera_id': event_info['camera_id'],
                    'sequence': event_info['sequence'],
                    'date_str': datetime.fromtimestamp(event_info['timestamp']).strftime('%Y-%m-%d'),
                    'time_str': datetime.fromtimestamp(event_info['timestamp']).strftime('%H:%M:%S'),
                    'is_abnormal': is_abnormal,
                    'type': 'abnormal' if is_abnormal else 'normal',
                    'has_trajectory': has_trajectory
                })

            events.sort(key=lambda x: x['timestamp'])
            logger.info(f"返回 {len(events)} 个3D事件 (数据源: JSON回退)")
            return jsonify(events)

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, video_name, video_path, description FROM videos ORDER BY id DESC")
        db_videos = cursor.fetchall()
        if not db_videos:
            logger.info("events_3d: videos 表为空，返回空列表")
            return jsonify([])

        cursor.execute("SELECT video_id, person_trajectory FROM video_vectors ORDER BY video_id, person_index")
        vector_rows = cursor.fetchall()
        first_traj_by_video = {}
        for row in vector_rows:
            vid = row.get('video_id')
            if vid in first_traj_by_video:
                continue
            traj_json = row.get('person_trajectory')
            if not traj_json:
                continue
            try:
                traj_data = json.loads(traj_json) if isinstance(traj_json, str) else traj_json
                points = traj_data.get('points', []) if isinstance(traj_data, dict) else []
                if points and len(points[0]) >= 2:
                    first_traj_by_video[vid] = points[0]
            except Exception:
                continue

        events = []
        for video in db_videos:
            video_id = video.get('id')
            video_name = video.get('video_name') or ''
            video_path = video.get('video_path') or ''
            video_key = os.path.splitext(video_name)[0] if video_name else str(video_id)

            event_info = parse_event_filename(video_key)
            if not event_info:
                camera_match = re.search(r'(camera\d+)', video_name, re.IGNORECASE)
                camera_id = camera_match.group(1).lower() if camera_match else 'camera1'
                timestamp = parse_video_key_to_timestamp(video_key)
                if not timestamp:
                    timestamp = int(time.time())
                event_info = {
                    'camera_id': camera_id,
                    'timestamp': timestamp,
                    'sequence': 0,
                    'original_filename': video_key
                }

            start_point = first_traj_by_video.get(video_id)
            has_trajectory = bool(start_point)
            if has_trajectory and len(start_point) >= 2:
                position = {'pixel_x': start_point[0], 'pixel_y': start_point[1]}
            else:
                position = generate_random_position_in_area(event_info['camera_id'], video_key)

            if not position:
                continue

            video_info = video_data.get(video_key, {}) if isinstance(video_data, dict) else {}
            analyse_result = video.get('description') or video_info.get('analyse_result', '') or '监控记录'
            abnormal_keywords = ['异常', '打架', '闯入', '跌倒', '争执', '暴力', '紧急', '危险']
            is_abnormal = any(keyword in analyse_result for keyword in abnormal_keywords)

            events.append({
                'id': video_key,
                'timestamp': event_info['timestamp'],
                'position': position,
                'content': analyse_result,
                'videoFile': os.path.basename(video_path) if video_path else video_name,
                'camera_id': event_info['camera_id'],
                'sequence': event_info['sequence'],
                'date_str': datetime.fromtimestamp(event_info['timestamp']).strftime('%Y-%m-%d'),
                'time_str': datetime.fromtimestamp(event_info['timestamp']).strftime('%H:%M:%S'),
                'is_abnormal': is_abnormal,
                'type': 'abnormal' if is_abnormal else 'normal',
                'has_trajectory': has_trajectory
            })

        events.sort(key=lambda x: x['timestamp'])
        track_count = sum(1 for item in events if item.get('has_trajectory'))
        logger.info(f"返回 {len(events)} 个3D事件 (数据源: DB优先, 含轨迹事件: {track_count})")
        return jsonify(events)
        
    except Exception as e:
        logger.error(f"获取3D事件数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/api/trajectory/<event_id>')
def get_event_trajectory(event_id):
    """获取指定事件的轨迹数据（从数据库读取）"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 1. 查询视频信息
        # 尝试匹配 id、event_id 或 event_id.mp4
        if str(event_id).isdigit():
            query_video = "SELECT id, person_count FROM videos WHERE id = %s OR video_name = %s OR video_name = %s"
            cursor.execute(query_video, (int(event_id), str(event_id), f"{event_id}.mp4"))
        else:
            query_video = "SELECT id, person_count FROM videos WHERE video_name = %s OR video_name = %s"
            cursor.execute(query_video, (str(event_id), f"{event_id}.mp4"))
        video = cursor.fetchone()
        
        if not video:
            # 如果数据库没找到，尝试回退到 JSON 文件（为了兼容性）
            logger.warning(f"数据库中未找到视频 {event_id}，尝试读取 JSON 文件")
            try:
                with open(video_description_path, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
                if event_id in video_data:
                    video_info = video_data[event_id]
                    trajectory_data = video_info.get('trajectory_data')
                    if trajectory_data and 'trajectories' in trajectory_data:
                        return jsonify({
                            'event_id': event_id,
                            'person_count': trajectory_data.get('person_count', 0),
                            'coordinate_system': trajectory_data.get('coordinate_system', 'real_world'),
                            'unit': trajectory_data.get('unit', 'centimeters'),
                            'trajectories': trajectory_data.get('trajectories', [])
                        })
            except Exception as e:
                logger.error(f"回退读取 JSON 失败: {e}")
            
            return jsonify({'error': '事件不存在'}), 404
            
        video_id = video['id']
        person_count = video.get('person_count', 0)
        
        # 2. 查询轨迹数据（优先包含 person_image_path）
        try:
            query_vectors = "SELECT person_index, person_trajectory, person_image_path FROM video_vectors WHERE video_id = %s"
            cursor.execute(query_vectors, (video_id,))
            vectors = cursor.fetchall()
        except Exception:
            # 向后兼容旧库结构
            query_vectors = "SELECT person_index, person_trajectory FROM video_vectors WHERE video_id = %s"
            cursor.execute(query_vectors, (video_id,))
            vectors = cursor.fetchall()
            for row in vectors:
                row['person_image_path'] = None
        
        if not vectors:
            return jsonify({'error': '该事件没有轨迹数据'}), 404
            
        trajectories = []
        for vec in vectors:
            try:
                traj_json = vec['person_trajectory']
                if not traj_json:
                    continue
                    
                # 兼容多层字符串化的 JSON
                traj_data = traj_json
                while isinstance(traj_data, str):
                    try:
                        traj_data = json.loads(traj_data)
                    except Exception:
                        break

                if not isinstance(traj_data, dict):
                    continue
                    
                # 适配数据格式
                # 数据库格式: {"unit": "cm", "length": 14, "points": [[x, y], ...]}
                # API返回格式: {"track_id": 1, "coordinates": [[x, y], ...]}
                
                trajectories.append({
                    'track_id': vec['person_index'],
                    'coordinates': traj_data.get('points', []),
                    'length': traj_data.get('length', 0)
                })
            except Exception as e:
                logger.error(f"解析轨迹数据失败: {e}")
                continue
                
        return jsonify({
            'event_id': event_id,
            'person_count': person_count,
            'coordinate_system': 'real_world',
            'unit': 'centimeters', # 假设数据库单位是 cm
            'trajectories': trajectories
        })
        
    except Exception as e:
        logger.error(f"获取轨迹数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def convert_pixel_to_scene_coords(pixel_x, pixel_y):
    """将像素坐标转换为3D场景坐标"""
    # 地图像素尺寸: 3156x1380 (宽x高)
    # 前端3D场景平面大小: width=25, height=25/(3156/1380)≈10.93 单位
    
    map_width = 3156
    map_height = 1380
    scene_width = 25
    # 前端根据图片宽高比计算高度：25 / (3156/1380)
    scene_height = scene_width / (map_width / map_height)
    
    # 将像素坐标转换为3D场景坐标
    # 像素坐标系：左上角(0,0)，向右X增加，向下Y增加
    # 3D场景坐标系：中心为(0,0)，X轴左右，Z轴前后
    
    # 转换为场景坐标，中心为原点
    scene_x = (pixel_x - map_width/2) * scene_width / map_width
    scene_z = (pixel_y - map_height/2) * scene_height / map_height
    
    # 添加调试信息
    logger.debug(f"坐标转换: 像素({pixel_x:.2f}, {pixel_y:.2f}) -> 场景({scene_x:.3f}, {scene_z:.3f})")
    logger.debug(f"场景尺寸: width={scene_width}, height={scene_height:.3f}")
    
    return scene_x, scene_z

@app.route('/api/trajectory/<event_id>/scene_coords')
def get_event_trajectory_scene_coords(event_id):
    """获取转换为3D场景坐标的轨迹数据（从数据库读取）"""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '数据库连接失败'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 1. 查询视频信息
        if str(event_id).isdigit():
            query_video = "SELECT id, person_count FROM videos WHERE id = %s OR video_name = %s OR video_name = %s"
            cursor.execute(query_video, (int(event_id), str(event_id), f"{event_id}.mp4"))
        else:
            query_video = "SELECT id, person_count FROM videos WHERE video_name = %s OR video_name = %s"
            cursor.execute(query_video, (str(event_id), f"{event_id}.mp4"))
        video = cursor.fetchone()
        
        # 轨迹颜色列表
        trajectory_colors = [
            '#ff4444',  # 红色
            '#44ff44',  # 绿色  
            '#4444ff',  # 蓝色
            '#ffff44',  # 黄色
            '#ff44ff',  # 紫色
            '#44ffff',  # 青色
            '#ff8844',  # 橙色
            '#8844ff',  # 紫罗兰
            '#44ff88',  # 青绿
            '#ff4488'   # 洋红
        ]
        
        if not video:
            # 回退到 JSON 文件
            logger.warning(f"数据库中未找到视频 {event_id}，尝试读取 JSON 文件")
            try:
                with open(video_description_path, 'r', encoding='utf-8') as f:
                    video_data = json.load(f)
                if event_id in video_data:
                    video_info = video_data[event_id]
                    trajectory_data = video_info.get('trajectory_data')
                    if trajectory_data and 'trajectories' in trajectory_data:
                        scene_trajectories = []
                        for i, traj in enumerate(trajectory_data.get('trajectories', [])):
                            track_id = traj.get('track_id', i)
                            coordinates = traj.get('coordinates', [])
                            scene_coords = []
                            for coord in coordinates:
                                if len(coord) >= 2:
                                    real_x, real_y = coord[0], coord[1]
                                    scene_x, scene_z = convert_pixel_to_scene_coords(real_x, real_y)
                                    scene_coords.append({'x': round(scene_x, 3), 'y': 0, 'z': round(scene_z, 3)})
                            if scene_coords:
                                scene_trajectories.append({
                                    'track_id': track_id,
                                    'trajectory_length': len(scene_coords),
                                    'coordinates': scene_coords,
                                    'color': trajectory_colors[track_id % len(trajectory_colors)]
                                })
                        return jsonify({
                            'event_id': event_id,
                            'person_count': trajectory_data.get('person_count', 0),
                            'trajectories': scene_trajectories
                        })
            except Exception as e:
                logger.error(f"回退读取 JSON 失败: {e}")
            
            return jsonify({'error': '事件不存在'}), 404
            
        video_id = video['id']
        person_count = video.get('person_count', 0)
        
        # 2. 查询轨迹数据（优先包含 person_image_path）
        try:
            query_vectors = "SELECT person_index, person_trajectory, person_image_path FROM video_vectors WHERE video_id = %s"
            cursor.execute(query_vectors, (video_id,))
            vectors = cursor.fetchall()
        except Exception:
            # 兼容旧表结构
            query_vectors = "SELECT person_index, person_trajectory FROM video_vectors WHERE video_id = %s"
            cursor.execute(query_vectors, (video_id,))
            vectors = cursor.fetchall()
            for row in vectors:
                row['person_image_path'] = None
        
        if not vectors:
            return jsonify({'error': '该事件没有轨迹数据'}), 404
            
        scene_trajectories = []
        for vec in vectors:
            try:
                traj_json = vec['person_trajectory']
                if not traj_json:
                    continue
                    
                traj_data = traj_json
                while isinstance(traj_data, str):
                    try:
                        traj_data = json.loads(traj_data)
                    except Exception:
                        break

                if not isinstance(traj_data, dict):
                    continue
                
                track_id = vec['person_index']
                # 数据库中的 points 对应 coordinates
                coordinates = traj_data.get('points', [])
                
                scene_coords = []
                for coord in coordinates:
                    if len(coord) >= 2:
                        real_x, real_y = coord[0], coord[1]
                        scene_x, scene_z = convert_pixel_to_scene_coords(real_x, real_y)
                        
                        scene_coords.append({
                            'x': round(scene_x, 3),
                            'y': 0,  # 地面高度
                            'z': round(scene_z, 3)
                        })
                
                if scene_coords:
                    scene_trajectories.append({
                        'track_id': track_id,
                        'trajectory_length': len(scene_coords),
                        'coordinates': scene_coords,
                        'color': trajectory_colors[track_id % len(trajectory_colors)],
                        'person_image_path': vec.get('person_image_path')
                    })
                    
            except Exception as e:
                logger.error(f"解析轨迹数据失败: {e}")
                continue
        
        return jsonify({
            'event_id': event_id,
            'person_count': person_count,
            'trajectories': scene_trajectories
        })
        
    except Exception as e:
        logger.error(f"获取场景坐标轨迹数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/api/related_events/<event_id>')
def get_related_events(event_id):
    """获取与指定事件相关的其他事件"""
    try:
        # 获取查询参数
        date = request.args.get('date')
        threshold = float(request.args.get('threshold', 0.6))  # 相似度阈值
        keyword_weight = float(request.args.get('keyword_weight', 0.3))  # 关键词权重
        semantic_weight = float(request.args.get('semantic_weight', 0.7))  # 语义权重
        max_results = int(request.args.get('max_results', 10))
        
        # 读取事件数据
        with open(video_description_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        if event_id not in video_data:
            return jsonify({'error': '事件不存在'}), 404
        
        target_event = video_data[event_id]
        target_content = target_event.get('analyse_result', '')
        
        if not target_content:
            return jsonify([])  # 如果目标事件没有描述，返回空列表
        
        related_events = []
        
        # 提取关键词（简单的关键词提取）
        target_keywords = extract_keywords(target_content)
        
        for video_key, video_info in video_data.items():
            if video_key == event_id:
                continue  # 跳过自己
                
            # 按日期过滤（如果提供了日期参数）
            if date:
                try:
                    event_timestamp = parse_video_key_to_timestamp(video_key)
                    event_date = datetime.fromtimestamp(event_timestamp).strftime('%Y-%m-%d')
                    if event_date != date:
                        continue
                except:
                    continue
            
            content = video_info.get('analyse_result', '')
            if not content:
                continue
            
            # 计算相似度得分
            keyword_score = calculate_keyword_similarity(target_keywords, content)
            semantic_score = calculate_semantic_similarity(target_content, content)
            
            # 加权求和
            total_score = keyword_weight * keyword_score + semantic_weight * semantic_score
            
            if total_score >= threshold:
                related_events.append({
                    'event_id': video_key,
                    'content': content,
                    'keyword_score': round(keyword_score, 3),
                    'semantic_score': round(semantic_score, 3),
                    'total_score': round(total_score, 3),
                    'timestamp': parse_video_key_to_timestamp(video_key),
                    'camera_id': get_camera_id_from_key(video_key)
                })
        
        # 按得分排序并限制结果数量
        related_events.sort(key=lambda x: x['total_score'], reverse=True)
        related_events = related_events[:max_results]
        
        logger.info(f"找到 {len(related_events)} 个与事件 {event_id} 相关的事件")
        return jsonify(related_events)
        
    except Exception as e:
        logger.error(f"获取相关事件错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500


@app.route('/api/tracking/push', methods=['POST'])
def push_tracking_point():
    """接收算法端实时坐标并广播给前端 Socket.IO 客户端"""
    try:
        data = request.get_json(silent=True) or {}
        track_id = data.get('track_id')
        x = data.get('x')
        y = data.get('y')

        if track_id is None or x is None or y is None:
            return jsonify({'ok': False, 'error': '缺少 track_id/x/y 字段'}), 400

        payload = {
            'track_id': track_id,
            'x': x,
            'y': y
        }

        socketio.emit('tracking_point', payload, namespace='/ws/tracking')
        return jsonify({'ok': True, 'data': payload})
    except Exception as e:
        logger.error(f"处理轨迹推送失败: {str(e)}")
        return jsonify({'ok': False, 'error': '服务器错误'}), 500


@socketio.on('connect', namespace='/ws/tracking')
def tracking_connect():
    logger.info('客户端已连接到 /ws/tracking')


@socketio.on('disconnect', namespace='/ws/tracking')
def tracking_disconnect():
    logger.info('客户端已断开 /ws/tracking')

def extract_keywords(text):
    """提取关键词"""
    try:
        import jieba
        
        # 定义停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤关键词
        keywords = []
        for word in words:
            word = word.strip()
            if len(word) > 1 and word not in stop_words and word.isalpha():
                keywords.append(word)
        
        return keywords
        
    except ImportError:
        logger.warning("jieba库未安装，使用简单分词替代")
        # 简单的fallback分词：基于标点符号和空格
        import re
        
        # 定义停用词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 使用正则表达式进行简单分词
        words = re.findall(r'[\u4e00-\u9fff]+', text)  # 提取中文字符
        
        # 过滤关键词
        keywords = []
        for word in words:
            if len(word) > 1 and word not in stop_words:
                keywords.append(word)
        
        return keywords

def calculate_keyword_similarity(target_keywords, content):
    """计算关键词相似度"""
    if not target_keywords:
        return 0.0
    
    content_keywords = extract_keywords(content)
    if not content_keywords:
        return 0.0
    
    # 计算交集
    common_keywords = set(target_keywords) & set(content_keywords)
    
    # Jaccard相似度
    union_keywords = set(target_keywords) | set(content_keywords)
    similarity = len(common_keywords) / len(union_keywords) if union_keywords else 0.0
    
    return similarity

def calculate_semantic_similarity(text1, text2):
    """计算语义相似度"""
    try:
        # =========================
        # 【阶段一-语义相似度模型来源切换】
        # =========================
        if not es_worker:
            return 0.0
        
        # 使用 ES Worker 持有的嵌入模型计算相似度
        embedding1 = es_worker.embedding_model.encode([text1])
        embedding2 = es_worker.embedding_model.encode([text2])
        
        # 计算余弦相似度
        import numpy as np
        
        # 归一化向量
        embedding1 = embedding1 / np.linalg.norm(embedding1, axis=1, keepdims=True)
        embedding2 = embedding2 / np.linalg.norm(embedding2, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2.T)[0][0]
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"计算语义相似度错误: {str(e)}")
        return 0.0

def get_camera_id_from_key(video_key):
    """从视频键中提取摄像头ID"""
    try:
        # video_key格式: camera1_20250623_094842
        parts = video_key.split('_')
        if len(parts) >= 1:
            return parts[0]
        return 'unknown'
    except:
        return 'unknown'

if __name__ == '__main__':
    logger.info("正在启动API服务器...")
    
    # 初始化所有服务器
    init_servers()
    
    # 启动Flask-SocketIO应用
    logger.info("Starting Flask-SocketIO app on port 5000...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    # from gevent.pywsgi import WSGIServer
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever() 