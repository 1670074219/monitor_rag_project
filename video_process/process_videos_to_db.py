import cv2
import numpy as np
import torch
import json
import os
import re
import pymysql
from ultralytics import YOLO
import gc  # 引入垃圾回收模块

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

# ================= 数据库配置 =================
DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'user': 'root',
    'password': 'q1w2e3az',
    'database': 'monitor_database',
    'charset': 'utf8mb4'
}

# ================= 路径与批处理配置 =================
CAMERA_CONFIG_PATH = './camera_config.json'
YOLO_MODEL_PATH = './yolo/yolo11s.pt'
DEEPSORT_CHECKPOINT = './deepsort/deep/checkpoint/ckpt.t7'

# 🔥 核心修改：批处理大小 (每次只查10个视频)
BATCH_SIZE = 10  

class VideoProcessor:
    def __init__(self):
        self.db_config = DB_CONFIG # 保存配置以便重连
        self.camera_configs = self.load_camera_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"正在加载模型 (Device: {self.device})...")
        self.yolo = YOLO(YOLO_MODEL_PATH).to(self.device)
        self.feature_extractor = Extractor(DEEPSORT_CHECKPOINT, use_cuda=torch.cuda.is_available())
        print("模型加载完成")

    def get_connection(self):
        """获取数据库连接（短连接模式，防止长事务超时）"""
        return pymysql.connect(**self.db_config, cursorclass=pymysql.cursors.DictCursor)

    def load_camera_config(self):
        try:
            with open(CAMERA_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)['camera_config']
        except Exception as e:
            print(f"错误：无法加载摄像头配置文件: {e}")
            return []

    def get_homography(self, camera_id):
        config = next((c for c in self.camera_configs if c['camera_id'] == camera_id), None)
        if not config: return None
        src_pts = np.array(config['pixel_coordinate'], dtype=np.float32)
        dst_pts = np.array(config['real_coordinate'], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def pixel_to_world(self, pixel_point, H):
        if H is None: return None
        px, py = pixel_point
        vec = np.array([[px], [py], [1]], dtype=np.float32)
        real_vec = np.dot(H, vec)
        if real_vec[2, 0] == 0: return None
        return (real_vec[0, 0] / real_vec[2, 0], real_vec[1, 0] / real_vec[2, 0])

    def process_db_videos(self):
        """🔥 主循环：分批次读取视频并处理"""
        
        current_id_pointer = 0  # 游标指针，记录上一次处理到的 ID
        batch_count = 0
        total_processed = 0

        while True:
            batch_count += 1
            print(f"\n========== 开始读取第 {batch_count} 批次 (Batch Size: {BATCH_SIZE}) ==========")
            
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    # 1. 优化 SQL：使用 WHERE id > last_id 方式分页，性能比 OFFSET 更好，且节省内存
                    sql = """
                        SELECT id, video_name, video_path 
                        FROM videos 
                        WHERE id > %s 
                        ORDER BY id ASC 
                        LIMIT %s
                    """
                    cursor.execute(sql, (current_id_pointer, BATCH_SIZE))
                    videos = cursor.fetchall()
            finally:
                conn.close() # 查完立刻关闭连接，防止长时间占用

            # 如果没有取到数据，说明所有视频都遍历完了
            if not videos:
                print("✅ 所有批次处理完毕！")
                break

            # 更新指针，下一次从这一批的最后一个 ID 后面开始查
            current_id_pointer = videos[-1]['id']

            # 开始处理这一批
            for video in videos:
                self.process_one_video_safely(video)
                total_processed += 1

            # 🔥 关键：每处理完一批，手动强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"--- 第 {batch_count} 批次完成，当前指针 ID: {current_id_pointer} ---")

    def process_one_video_safely(self, video):
        """单独处理一个视频的逻辑封装"""
        video_id = video['id']
        video_name = video['video_name']
        video_path = video['video_path']

        # 1. 检查是否已存在 (复用数据库连接)
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM video_vectors WHERE video_id = %s", (video_id,))
                if cursor.fetchone():
                    print(f"⏭️ 跳过 (已存在): {video_name}")
                    return
        finally:
            conn.close()

        if not os.path.exists(video_path):
            print(f"⚠️ 文件不存在: {video_path}")
            return

        # 提取摄像头ID
        camera_match = re.match(r'(camera\d+)', video_name)
        camera_id = camera_match.group(1) if camera_match else None
        
        if not camera_id:
            print(f"⚠️ 无法识别CameraID: {video_name}")
            return

        print(f"正在处理: {video_name} ...")
        
        # 2. 算法分析
        results = self.analyze_single_video(video_path, camera_id)
        
        # 3. 保存结果
        self.save_results_to_db(video_id, results)

    def analyze_single_video(self, video_path, camera_id):
        """处理单个视频，返回轨迹和特征"""
        H = self.get_homography(camera_id)
        deepsort = DeepSort(
            max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
            max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
        )

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        temp_tracks = {} 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % 5 != 0: continue 

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo(img_rgb, verbose=False)
            
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    if int(box.cls) == 0 and box.conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, box.conf.item()])

            if detections:
                dets = np.array(detections)
                xywh = np.zeros((len(dets), 4))
                xywh[:, 0] = (dets[:, 0] + dets[:, 2]) / 2
                xywh[:, 1] = (dets[:, 1] + dets[:, 3]) / 2
                xywh[:, 2] = dets[:, 2] - dets[:, 0]
                xywh[:, 3] = dets[:, 3] - dets[:, 1]
                confs = dets[:, 4]
                deepsort.update(xywh, confs, frame, self.feature_extractor)
            else:
                deepsort.update([], [], frame, self.feature_extractor)

            for track in deepsort.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                t_id = track.track_id
                if t_id not in temp_tracks:
                    temp_tracks[t_id] = {'traj': [], 'feats': []}

                bbox = track.to_tlbr()
                foot_x = int((bbox[0] + bbox[2]) / 2)
                foot_y = int(bbox[3])
                real_pt = self.pixel_to_world((foot_x, foot_y), H)
                
                if real_pt:
                    temp_tracks[t_id]['traj'].append([round(real_pt[0], 2), round(real_pt[1], 2)])

                if hasattr(track, 'features') and track.features:
                    temp_tracks[t_id]['feats'].append(track.features[-1])
                elif hasattr(track, 'curr_feature') and track.curr_feature is not None:
                    temp_tracks[t_id]['feats'].append(track.curr_feature)

        cap.release()
        return temp_tracks

    def save_results_to_db(self, video_id, results):
        if not results: return

        insert_data = []
        sorted_ids = sorted(results.keys()) 
        
        for idx, t_id in enumerate(sorted_ids):
            data = results[t_id]
            traj_list = data['traj']
            feat_list = data['feats']

            if not traj_list: continue

            if feat_list:
                feats_matrix = np.array(feat_list)
                mean_feat = np.mean(feats_matrix, axis=0)
                norm = np.linalg.norm(mean_feat)
                final_vector = (mean_feat / norm).tolist() if norm > 0 else mean_feat.tolist()
            else:
                final_vector = []

            trajectory_json = {
                "track_id": int(t_id),
                "points": traj_list,
                "length": len(traj_list),
                "unit": "cm"
            }

            insert_data.append((
                video_id,
                json.dumps(final_vector),
                idx + 1,
                json.dumps(trajectory_json)
            ))

        if insert_data:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO video_vectors 
                        (video_id, vector_data, person_index, person_trajectory) 
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.executemany(sql, insert_data)
                    conn.commit()
                print(f"  ✓ [VideoID: {video_id}] 存入 {len(insert_data)} 人数据")
            except Exception as e:
                print(f"  ❌ 写入失败: {e}")
                conn.rollback()
            finally:
                conn.close()

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_db_videos()