import json
import os
import numpy as np
import mysql.connector
from mysql.connector import Error
import networkx as nx
from datetime import datetime, timedelta
from collections import deque

# ================= 配置 =================
DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'database': 'monitor_database',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4'
}

# 获取当前脚本所在目录，确保能找到 topology 文件
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOPOLOGY_PATH = os.path.join(CURRENT_DIR, 'camera_topology.json')

TIME_WINDOW_MINUTES = 2  # 默认时间窗口
SIMILARITY_THRESHOLD = 0.70 # 向量相似度阈值

class PersonSearchEngine:
    def __init__(self):
        self.topology = self.load_topology()

    def get_connection(self):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except Error as e:
            print(f"数据库连接失败: {e}")
            return None

    def load_topology(self):
        """加载摄像头拓扑关系图"""
        if not os.path.exists(TOPOLOGY_PATH):
            print(f"⚠️ 未找到拓扑文件 {TOPOLOGY_PATH}，将使用全量搜索模式")
            return None
        
        with open(TOPOLOGY_PATH, 'r') as f:
            adj_list = json.load(f)
        
        # 构建 NetworkX 图
        G = nx.Graph()
        for cam, neighbors in adj_list.items():
            for neighbor in neighbors:
                G.add_edge(cam, neighbor)
        return G

    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def extract_camera_id(self, video_name):
        """从文件名解析 camera_id (例如 camera1_xxx.mp4 -> camera1)"""
        import re
        match = re.match(r'(camera\d+)', video_name)
        return match.group(1) if match else None

    def search_target(self, target_video_id, target_person_index, time_window=10):
        """
        核心搜索函数
        :param target_video_id: 目标所在的视频ID
        :param target_person_index: 目标在该视频中的 person_index
        :param time_window: 时间窗口 (分钟)
        """
        conn = self.get_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor(dictionary=True)
            
            # 1. 获取目标人物的特征向量和元数据
            sql_target = """
                SELECT v.video_name, v.created_time, vv.vector_data, vv.person_trajectory
                FROM video_vectors vv
                JOIN videos v ON vv.video_id = v.id
                WHERE vv.video_id = %s AND vv.person_index = %s
            """
            cursor.execute(sql_target, (target_video_id, target_person_index))
            target_info = cursor.fetchone()
            
            if not target_info:
                print("❌ 未找到目标人物信息")
                return []

            target_vec = json.loads(target_info['vector_data'])
            start_time = target_info['created_time']
            if isinstance(start_time, str):
                start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                
            start_cam = self.extract_camera_id(target_info['video_name'])
            
            print(f"🎯 目标锁定: {start_cam} ({start_time})")
            
            # 2. 基于拓扑图的 BFS 搜索队列
            # 队列元素: (camera_id, depth)
            search_queue = deque([(start_cam, 0)])
            visited_cameras = {start_cam}
            found_traces = []

            print(f"🔍 开始搜索 (时间窗口: ±{time_window}分钟)...")

            while search_queue:
                current_cam, depth = search_queue.popleft()
                if not current_cam: continue

                print(f"  Processing Camera: {current_cam} (Depth: {depth})")
                
                # --- 步骤 A: 在当前摄像头中搜索 ---
                # 查找该摄像头在时间窗口内的所有视频
                time_min = start_time - timedelta(minutes=time_window)
                time_max = start_time + timedelta(minutes=time_window)
                
                # 模糊匹配摄像头名称 (假设 video_name 包含 camera_id)
                # 使用 %camera% 匹配，但在 Python 中进行精确过滤以避免 camera1 匹配 camera10
                sql_candidates = """
                    SELECT v.id as video_id, v.video_name, v.created_time, 
                           vv.person_index, vv.vector_data, vv.person_trajectory
                    FROM videos v
                    JOIN video_vectors vv ON v.id = vv.video_id
                    WHERE v.video_name LIKE %s 
                      AND v.created_time BETWEEN %s AND %s
                      AND NOT (v.id = %s AND vv.person_index = %s) -- 排除目标自己
                """
                cursor.execute(sql_candidates, (f"%{current_cam}%", time_min, time_max, target_video_id, target_person_index))
                candidates = cursor.fetchall()
                
                hit_in_this_camera = False
                
                for cand in candidates:
                    # 精确验证摄像头ID
                    cand_cam = self.extract_camera_id(cand['video_name'])
                    if cand_cam != current_cam:
                        continue

                    cand_vec = json.loads(cand['vector_data'])
                    
                    # --- Level 1: 向量相似度初筛 ---
                    score = self.cosine_similarity(target_vec, cand_vec)
                    
                    if score > SIMILARITY_THRESHOLD:
                        print(f"  ✅ 发现踪迹! 摄像头: {current_cam} | 视频: {cand['video_name']} | 相似度: {score:.4f}")
                        
                        found_traces.append({
                            "camera_id": current_cam,
                            "video_id": cand['video_id'],
                            "video_name": cand['video_name'],
                            "person_index": cand['person_index'],
                            "similarity": score,
                            "time": cand['created_time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(cand['created_time'], datetime) else str(cand['created_time']),
                            "trajectory": json.loads(cand['person_trajectory'])
                        })
                        hit_in_this_camera = True

                # --- 步骤 B: 拓扑扩展 ---
                # 只有在当前摄像头找到了人，或者这是起始点，才继续往相邻摄像头找
                if hit_in_this_camera or current_cam == start_cam:
                    if self.topology and current_cam in self.topology:
                        neighbors = self.topology[current_cam]
                        for neighbor in neighbors:
                            if neighbor not in visited_cameras:
                                visited_cameras.add(neighbor)
                                search_queue.append((neighbor, depth + 1))
                                print(f"  ➡️ 扩展搜索范围 -> {neighbor}")
                else:
                    print(f"  ⛔ {current_cam} 未发现目标，停止该分支搜索")

            return found_traces

        except Exception as e:
            print(f"❌ 搜索出错: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if conn and conn.is_connected():
                cursor.close()
                conn.close()

# ================= 测试运行 =================
if __name__ == "__main__":
    engine = PersonSearchEngine()
    
    # 假设我们要找 video_id=10 里的第 1 个人
    # 你需要先去数据库里查一个真实存在的 video_id 和 person_index
    TARGET_VIDEO_ID = 4273  
    TARGET_PERSON_INDEX = 1
    
    results = engine.search_target(TARGET_VIDEO_ID, TARGET_PERSON_INDEX, time_window=15)
    
    print(f"\n📊 最终结果: 找到 {len(results)} 条轨迹")
    print(json.dumps(results, indent=2, ensure_ascii=False))
