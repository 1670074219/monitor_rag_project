import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from queue import Queue
import threading
import glob
import re
from datetime import datetime
import math
import random

from video_process.faiss_server import FaissServer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
faiss_server = None
video_description_path = None
saved_video_path = None

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



def init_servers():
    """初始化服务器"""
    global faiss_server, video_description_path, saved_video_path
    
    base_dir = os.path.join(os.path.dirname(__file__), "video_process")
    video_description_path = os.path.join(base_dir, "video_description.json")
    saved_video_path = os.path.join(base_dir, "saved_video")
    
    # 初始化Faiss服务器
    video_dsp_queue = Queue()
    faiss_server = FaissServer(
        emd_model_path="/root/data1/bge_zh_v1.5/",
        video_dsp_queue=video_dsp_queue,
        index_path=os.path.join(base_dir, "faiss_ifl2.index"),
        video_description_path=video_description_path
    )
    
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
        
        # 使用混合搜索
        k = data.get('k', 5)  # 默认返回5个结果
        alpha = data.get('alpha', 0.7)  # Faiss权重，默认0.7
        
        search_results = faiss_server.hybrid_search(query_text, k=k, alpha=alpha)
        
        if not search_results:
            return jsonify({
                'status': 'success',
                'data': {
                    '相关日志': [],
                    '总结报告': '没有找到相关的监控记录。'
                }
            })
        
        # 读取视频描述数据
        with open(video_description_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        # 构建相关日志数据
        related_logs = []
        summaries = []
        
        for v_k, score in search_results:
            if v_k in video_data:
                v_info = video_data[v_k]
                
                # 构建日志条目
                log_entry = {
                    f"{v_k}日志": v_k,
                    f"{v_k}概述": v_info.get('analyse_result', '无描述')[:100] + ('...' if len(v_info.get('analyse_result', '')) > 100 else '')
                }
                related_logs.append(log_entry)
                
                # 收集完整描述用于总结
                if v_info.get('analyse_result'):
                    summaries.append(f"时间{v_k}: {v_info['analyse_result']}")
        
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
    return jsonify({
        'status': 'healthy',
        'faiss_ready': faiss_server is not None,
        'index_total': faiss_server.index.ntotal if faiss_server and faiss_server.index else 0
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
        
        return jsonify({
            'total_videos': total_videos,
            'analyzed_videos': analyzed_videos,
            'embedded_videos': embedded_videos,
            'index_size': faiss_server.index.ntotal if faiss_server and faiss_server.index else 0
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
    try:
        # 尝试读取包含轨迹的JSON文件
        trajectory_file_path = os.path.join(os.path.dirname(__file__), "video_process", "video_description_with_trajectory.json")
        
        if os.path.exists(trajectory_file_path):
            with open(trajectory_file_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        else:
            # 备用：使用原始文件
            with open(video_description_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        
        events = []
        
        for video_key, video_info in video_data.items():
            # 解析事件信息
            event_info = parse_event_filename(video_key)
            if not event_info:
                # 如果新格式解析失败，尝试原有格式
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
            
            # 在摄像头覆盖区域内生成随机位置
            position = generate_random_position_in_area(event_info['camera_id'], video_key)
            if not position:
                continue
            
            # 检查是否为异常事件
            is_abnormal = False
            analyse_result = video_info.get('analyse_result', '')
            if analyse_result:
                abnormal_keywords = ['异常', '打架', '闯入', '跌倒', '争执', '暴力', '紧急', '危险']
                is_abnormal = any(keyword in analyse_result for keyword in abnormal_keywords)
            
            # 获取视频文件名
            video_file = None
            if video_info.get('video_path'):
                video_file = os.path.basename(video_info['video_path'])
            
            # 检查是否有轨迹数据
            has_trajectory = 'trajectory_data' in video_info and 'trajectories' in video_info['trajectory_data']
            
            # 构建事件对象
            event = {
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
            }
            
            events.append(event)
        
        # 按时间戳排序
        events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"返回 {len(events)} 个3D事件")
        return jsonify(events)
        
    except Exception as e:
        logger.error(f"获取3D事件数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

@app.route('/api/trajectory/<event_id>')
def get_event_trajectory(event_id):
    """获取指定事件的轨迹数据"""
    try:
        # 尝试读取包含轨迹的JSON文件
        trajectory_file_path = os.path.join(os.path.dirname(__file__), "video_process", "video_description_with_trajectory.json")
        
        if os.path.exists(trajectory_file_path):
            with open(trajectory_file_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        else:
            # 备用：使用原始文件
            with open(video_description_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        
        if event_id not in video_data:
            return jsonify({'error': '事件不存在'}), 404
        
        video_info = video_data[event_id]
        trajectory_data = video_info.get('trajectory_data')
        
        if not trajectory_data or 'trajectories' not in trajectory_data:
            return jsonify({'error': '该事件没有轨迹数据'}), 404
        
        # 返回原始轨迹数据
        return jsonify({
            'event_id': event_id,
            'person_count': trajectory_data.get('person_count', 0),
            'coordinate_system': trajectory_data.get('coordinate_system', 'real_world'),
            'unit': trajectory_data.get('unit', 'centimeters'),
            'trajectories': trajectory_data.get('trajectories', [])
        })
        
    except Exception as e:
        logger.error(f"获取轨迹数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500

def convert_real_world_to_scene_coords(real_x, real_y):
    """将真实世界坐标转换为3D场景坐标"""
    # 根据您的3D场景设置调整这些转换参数
    # 这些参数需要与前端的平面图设置匹配
    scene_x = (real_x - 462.5) / 100  # 调整偏移和缩放
    scene_z = (real_y - 685.5) / 100  # Y轴对应3D的Z轴
    
    return scene_x, scene_z

@app.route('/api/trajectory/<event_id>/scene_coords')
def get_event_trajectory_scene_coords(event_id):
    """获取转换为3D场景坐标的轨迹数据"""
    try:
        # 尝试读取包含轨迹的JSON文件
        trajectory_file_path = os.path.join(os.path.dirname(__file__), "video_process", "video_description_with_trajectory.json")
        
        if os.path.exists(trajectory_file_path):
            with open(trajectory_file_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        else:
            # 备用：使用原始文件
            with open(video_description_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
        
        if event_id not in video_data:
            return jsonify({'error': '事件不存在'}), 404
        
        video_info = video_data[event_id]
        trajectory_data = video_info.get('trajectory_data')
        
        if not trajectory_data or 'trajectories' not in trajectory_data:
            return jsonify({'error': '该事件没有轨迹数据'}), 404
        
        # 转换为3D场景坐标格式
        scene_trajectories = []
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
        
        for i, traj in enumerate(trajectory_data.get('trajectories', [])):
            track_id = traj.get('track_id', i)
            coordinates = traj.get('coordinates', [])
            
            # 转换坐标格式：[real_x, real_y] -> {x: scene_x, y: 0, z: scene_z}
            scene_coords = []
            for coord in coordinates:
                if len(coord) >= 2:
                    real_x, real_y = coord[0], coord[1]
                    scene_x, scene_z = convert_real_world_to_scene_coords(real_x, real_y)
                    
                    scene_coords.append({
                        'x': round(scene_x, 3),
                        'y': 0,  # 地面高度
                        'z': round(scene_z, 3)
                    })
            
            if scene_coords:  # 只添加有坐标的轨迹
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
        logger.error(f"获取场景坐标轨迹数据错误: {str(e)}")
        return jsonify({'error': '服务器错误'}), 500









if __name__ == '__main__':
    logger.info("正在启动API服务器...")
    
    # 初始化所有服务器
    init_servers()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True) 