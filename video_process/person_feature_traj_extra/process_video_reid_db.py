import cv2
import numpy as np
import torch
import requests
import json
import os
import re
import pymysql
import gc
import sys
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
from datetime import datetime

# -----------------------------------------------------------
# 📌 运行路径兼容：确保可导入 video_process 下的本地包（如 deepsort）
# -----------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PROCESS_ROOT = os.path.dirname(CURRENT_DIR)
if VIDEO_PROCESS_ROOT not in sys.path:
    sys.path.insert(0, VIDEO_PROCESS_ROOT)

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

# -----------------------------------------------------------
# 📦 引入 Torchreid (你的代码部分)
# -----------------------------------------------------------
from torchreid import models

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

# ================= 路径配置 =================
CAMERA_CONFIG_PATH = './camera_config.json'
# 优先使用更强的 YOLOv11x 模型以减少误检
YOLO_MODEL_PATH = './models/yolo/yolo11x.pt' 
DEEPSORT_CHECKPOINT = './deepsort/deep/checkpoint/ckpt.t7' # 仅用于追踪

# ReID 模型配置 (你提供的)
REID_MODEL_PATH = './models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
REID_MODEL_NAME = 'osnet_ain_x1_0'
REID_NUM_CLASSES = 4101

# ================= Elasticsearch 配置 =================
ES_CONFIG = {
    'host': os.getenv('ES_HOST', 'http://219.216.99.30:9200'),
    'index': os.getenv('REID_ES_INDEX', 'video_reid_index')
}

# ===========================================================
# 🧠 类 1: 专门负责提取高质量特征 (封装你的 Torchreid 代码)
# ===========================================================
class ReIDExtractor:
    def __init__(self, model_name, num_classes, model_path, device):
        self.device = device
        print(f"🏗️ [ReID] 正在构建模型: {model_name}...")
        
        # 构建模型
        self.model = models.build_model(name=model_name, num_classes=num_classes, pretrained=False)
        self.model.to(device).eval()

        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到权重文件: {model_path}")
            
        print(f"📂 [ReID] 加载权重: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        print("✅ [ReID] 模型加载完成")

        # 预处理
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_numpy):
        """输入 OpenCV 的 BGR 图片 (numpy), 输出归一化的特征向量"""
        if img_numpy is None or img_numpy.size == 0:
            return None
            
        try:
            # OpenCV (BGR) -> PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB))
            
            # 预处理 [1, C, H, W]
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(tensor)
                feat = feat.cpu().numpy().flatten()
                
                # L2 归一化 (关键步骤)
                norm_val = np.linalg.norm(feat)
                if norm_val > 0:
                    feat = feat / norm_val
                
                return feat.tolist() # 转成 list 方便存 JSON
        except Exception as e:
            print(f"⚠️ 特征提取失败: {e}")
            return None

# ===========================================================
# 🧠 类 2: 视频处理器 (YOLO + DeepSORT + ReID + Database)
# ===========================================================
class VideoProcessor:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.camera_configs = self.load_camera_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ================= 图片落盘目录初始化 =================
        # ⭐【新增】用于保存每个行人的“最佳质量原图”
        self.person_crops_root = os.path.normpath('./person_crops')
        os.makedirs(self.person_crops_root, exist_ok=True)
        print(f"🖼️ 行人裁剪图保存目录: {self.person_crops_root}")
        
        # 1. 加载 YOLO (用于检测人)
        print(f"🚀 加载 YOLO ({self.device})...")
        # 检查路径
        yolo_path = YOLO_MODEL_PATH
        # 如果配置的路径不存在，尝试在当前目录找 yolo11s.pt 作为备选
        if not os.path.exists(yolo_path):
            print(f"⚠️ 未找到 {yolo_path}，尝试查找默认模型...")
            if os.path.exists('./yolo/yolo11s.pt'):
                yolo_path = './yolo/yolo11s.pt'
            elif os.path.exists('../yolo/yolo11s.pt'):
                yolo_path = '../yolo/yolo11s.pt'
            else:
                # 如果都找不到，尝试自动下载或报错
                yolo_path = 'yolo11s.pt' 
        
        print(f"👉 使用模型: {yolo_path}")
        self.yolo = YOLO(yolo_path).to(self.device)
        
        # 2. 加载 DeepSORT (仅用于把人跟住，不用它的特征存数据库)
        # 注意：DeepSORT 内部也需要一个小模型来跑匹配，这里保留它原有的逻辑
        self.ds_extractor = Extractor(DEEPSORT_CHECKPOINT, use_cuda=torch.cuda.is_available())
        
        # 3. 加载 ReID (你的高质量模型，用于存数据库)
        self.reid_extractor = ReIDExtractor(
            REID_MODEL_NAME, REID_NUM_CLASSES, REID_MODEL_PATH, self.device
        )

        # 4. 初始化 Elasticsearch（用于 ReID 向量检索）
        self.es_client = self.init_es_client()
        if self.es_client:
            self.ensure_reid_index()

    def get_connection(self):
        return pymysql.connect(**self.db_config, cursorclass=pymysql.cursors.DictCursor)

    def init_es_client(self):
        if Elasticsearch is None:
            print("⚠️ 未安装 elasticsearch 包，跳过 ES 双写")
            return None

        try:
            client = Elasticsearch(
                [ES_CONFIG['host']],
                request_timeout=30
            )
            client.info()
            print(f"✅ Elasticsearch 连接成功: {ES_CONFIG['host']}")
            return client
        except Exception:
            try:
                client = Elasticsearch(
                    [ES_CONFIG['host']],
                    request_timeout=30,
                    headers={
                        'Accept': 'application/vnd.elasticsearch+json; compatible-with=8',
                        'Content-Type': 'application/vnd.elasticsearch+json; compatible-with=8'
                    }
                )
                client.info()
                print(f"✅ Elasticsearch 连接成功(ES8兼容模式): {ES_CONFIG['host']}")
                return client
            except Exception as e:
                print(f"⚠️ Elasticsearch 不可用，继续仅写 MySQL: {e}")
                return None

    def ensure_reid_index(self):
        if not self.es_client:
            return

        index_name = ES_CONFIG['index']
        try:
            if self.es_client.indices.exists(index=index_name):
                print(f"ℹ️ ES 索引已存在: {index_name}")
                return

            mapping = {
                'mappings': {
                    'properties': {
                        'video_id': {'type': 'integer'},
                        'person_index': {'type': 'integer'},
                        'track_id': {'type': 'integer'},
                        'embedding': {
                            'type': 'dense_vector',
                            'dims': 512,
                            'index': True,
                            'similarity': 'cosine'
                        },
                        'created_at': {'type': 'date'}
                    }
                }
            }
            self.es_client.indices.create(index=index_name, body=mapping)
            print(f"✅ 已创建 ES 索引: {index_name}")
        except Exception as e:
            print(f"⚠️ 创建 ES 索引失败: {e}")

    def save_to_es(self, docs):
        if not self.es_client or not docs:
            return

        ok_count = 0
        for doc in docs:
            try:
                doc_id = f"{doc['video_id']}_{doc['person_index']}"
                self.es_client.index(index=ES_CONFIG['index'], id=doc_id, document=doc)
                ok_count += 1
            except Exception as e:
                print(f"  ⚠️ ES 写入失败 (video_id={doc.get('video_id')}, person_index={doc.get('person_index')}): {e}")

        if ok_count > 0:
            print(f"  ✅ ES 写入成功: {ok_count} 条")

    def load_camera_config(self):
        try:
            with open(CAMERA_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)['camera_config']
        except:
            return []

    def get_homography(self, camera_id):
        config = next((c for c in self.camera_configs if c['camera_id'] == camera_id), None)
        if not config: return None
        src = np.array(config['pixel_coordinate'], dtype=np.float32)
        dst = np.array(config['real_coordinate'], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return H

    def pixel_to_world(self, pixel_point, H):
        if H is None: return None
        vec = np.array([[pixel_point[0]], [pixel_point[1]], [1]], dtype=np.float32)
        real = np.dot(H, vec)
        if real[2, 0] == 0: return None
        return (real[0, 0] / real[2, 0], real[1, 0] / real[2, 0])

    def process_db_videos(self):
        """处理所有视频的主循环"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # 获取所有视频
                cursor.execute("SELECT id, video_name, video_path FROM videos")
                videos = cursor.fetchall()
        finally:
            conn.close()

        print(f"📋 共有 {len(videos)} 个视频待检查...")

        for video in videos:
            self.process_one_video(video)
            gc.collect() # 清理内存

    def process_one_video(self, video):
        video_id, name, path = video['id'], video['video_name'], video['video_path']
        
        # 检查是否处理过
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM video_vectors WHERE video_id = %s", (video_id,))
                if cursor.fetchone():
                    print(f"⏭️ 跳过: {name} (已存在)")
                    return
        finally:
            conn.close()

        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return

        # 尝试从文件名解析 camera_id
        camera_match = re.search(r'(camera\d+)', name)
        camera_id = camera_match.group(1) if camera_match else None
        
        print(f"▶️ 正在处理: {name} (Camera: {camera_id}) ...")
        results = self.analyze_video(path, camera_id)
        self.save_results(video_id, results)

    def analyze_video(self, video_path, camera_id):
        H = self.get_homography(camera_id)
        
        # 初始化 DeepSORT (参数调优版)
        deepsort = DeepSort(
            max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
            max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
        )

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
    
        # 数据结构: { track_id: {'traj': [], 'feats': [], 'all_crops': [], 'best_crop': None} }
        tracks_data = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # 每 5 帧处理一次
            if frame_count % 5 != 0:
                continue

            # 1. YOLO 检测
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo(img_rgb, verbose=False)
            
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    # 类别 0 是人，置信度 > 0.65（进一步收紧门槛）
                    if int(box.cls) == 0 and box.conf > 0.65:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 几何形状过滤 (宽高比)
                        w_box = x2 - x1
                        h_box = y2 - y1
                        # ⭐【新增-形态学拦截】过滤极小目标与方正目标（如巡检机器人）
                        # 1) 高度 < 80 直接过滤；2) 人体宽度通常不超过高度 65%
                        if h_box < 80 or w_box > h_box * 0.65:
                            continue
                            
                        detections.append([x1, y1, x2, y2, box.conf.item()])

            # 2. DeepSORT 更新
            if detections:
                dets = np.array(detections)
                # xyxy -> xywh
                xywh = np.zeros((len(dets), 4))
                xywh[:, 0] = (dets[:, 0] + dets[:, 2]) / 2
                xywh[:, 1] = (dets[:, 1] + dets[:, 3]) / 2
                xywh[:, 2] = dets[:, 2] - dets[:, 0]
                xywh[:, 3] = dets[:, 3] - dets[:, 1]
                confs = dets[:, 4]
                deepsort.update(xywh, confs, frame, self.ds_extractor)
            else:
                deepsort.update([], [], frame, self.ds_extractor)

            # 3. Tracking & ReID
            for track in deepsort.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                tid = track.track_id
                if tid not in tracks_data:
                    tracks_data[tid] = {'traj': [], 'feats': [], 'all_crops': [], 'best_crop': None}

                # --- A. 提取轨迹 ---
                bbox = track.to_tlbr() # [x1, y1, x2, y2]
                # 确保不越界
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))

                foot_x, foot_y = int((x1 + x2) / 2), y2

                # ⭐【关键修复1】H 缺失时不再静默跳过：降级写入像素坐标，确保轨迹不为空
                if H is not None:
                    real_pt = self.pixel_to_world((foot_x, foot_y), H)
                    if real_pt:
                        tracks_data[tid]['traj'].append([round(real_pt[0], 2), round(real_pt[1], 2)])
                        # ⭐ 实时推送：将当前追踪点立即发送给 FastAPI（用于前端 WebSocket 实时渲染）
                        payload = {
                            "track_id": int(tid),
                            "x": float(real_pt[0]),
                            "y": float(real_pt[1])
                        }
                        try:
                            # ⭐ 极短超时，确保网络异常不阻塞视频推理主线程
                            requests.post("http://127.0.0.1:8000/api/tracking/push", json=payload, timeout=0.05)
                        except Exception:
                            # ⭐ 后端未启动/超时/网络异常时直接忽略，不影响主流程
                            pass
                    else:
                        # 这里保持原算法语义：H 存在但该点投影失败时仍不写入（可避免错误映射）
                        pass
                else:
                    tracks_data[tid]['traj'].append([float(foot_x), float(foot_y)])

                # --- B. 提取高质量 ReID 特征 (你的核心需求) ---
                # 只在当前帧成功匹配了检测框时 (time_since_update == 0) 才进行特征提取
                if track.time_since_update == 0 and x2 > x1 and y2 > y1:
                    # ⭐【重构】循环内仅收集轨迹帧截图，不做任何评分与特征提取
                    tracks_data[tid]['all_crops'].append(frame[y1:y2, x1:x2].copy())

        cap.release()

        # ⭐【核心逻辑】视频处理结束后，统一提取每条轨迹的 1/3 帧特征
        for tid, data in tracks_data.items():
            all_crops = data.get('all_crops', [])
            if not all_crops:
                continue

            one_third_idx = len(all_crops) // 3
            selected_crop = all_crops[one_third_idx]

            feature_vec = self.reid_extractor.extract(selected_crop)
            if feature_vec:
                data['feats'] = [feature_vec]
                data['best_crop'] = selected_crop

            # ⭐【内存释放】中间帧提取完成后清空缓存列表
            data['all_crops'] = []

        return tracks_data

    def save_results(self, video_id, results):
        if not results:
            print(f"⚠️ Video ID {video_id}: analyze_video 返回空结果，未写入数据库")
            return

        print(f"💾 正在保存结果到数据库 (Video ID: {video_id})...")
        
        sorted_ids = sorted(results.keys())
        insert_list = []
        es_docs = []

        # ⭐【新增】为每个视频创建独立子目录，保存最佳质量行人原图
        video_crop_dir = os.path.join(self.person_crops_root, f"video_{video_id}")
        os.makedirs(video_crop_dir, exist_ok=True)

        # ⭐【关键新增】先打印总追踪数，便于定位筛选损耗
        print(f"📊 分析完成，共追踪到 {len(sorted_ids)} 个临时ID，正在筛选...")

        for idx, tid in enumerate(sorted_ids):
            data = results[tid]
            traj = data['traj']
            feats = data['feats']

            # ⭐【关键新增】打破静默失败：轨迹为空
            if not traj:
                print(f"  ⚠️ ID {tid} 被过滤: 轨迹为空")
                data['best_crop'] = None
                continue

            # ⭐【门槛提升】轨迹至少 6 点（每5帧抽帧约 1.2 秒）
            if len(traj) < 6:
                print(f"  ⚠️ ID {tid} 被过滤: 轨迹太短 (len={len(traj)} < 6)")
                data['best_crop'] = None
                continue

            # ⭐【核心新增】静止目标过滤：基于最大活动半径判断“绝对静止”背景误检
            start_pt = np.array(traj[0], dtype=np.float32)
            max_dist = 0.0
            for point in traj:
                point_arr = np.array(point, dtype=np.float32)
                dist = float(np.linalg.norm(point_arr - start_pt))
                if dist > max_dist:
                    max_dist = dist

            if max_dist < 5.0:
                print(f"  ⚠️ ID {tid} 被过滤: 目标绝对静止 (max_dist={max_dist:.2f} < 5.0)，判定为背景误检")
                data['best_crop'] = None
                continue

            # 获取最佳特征向量
            final_vector = []
            if feats:
                # 因为现在 feats 只包含一个最佳特征，直接取第一个即可
                final_vector = feats[0]
            else:
                # ⭐【关键新增】打破静默失败：未提取到特征
                print(f"  ⚠️ ID {tid} 被过滤: 未提取到特征")
                data['best_crop'] = None
                continue

            # ⭐【新增】获取“最高分特征对应”的最佳裁剪图
            best_crop = data.get('best_crop')
            if best_crop is None:
                print(f"  ⚠️ ID {tid} 被过滤: 缺少最佳原图裁剪")
                continue

            # ⭐【新增】写入图片到磁盘，并记录相对路径到数据库
            img_filename = f"person_{idx + 1}_track_{int(tid)}.jpg"
            abs_img_path = os.path.join(video_crop_dir, img_filename)
            rel_img_path = os.path.join(f"video_{video_id}", img_filename).replace('\\\\', '/')

            ok = cv2.imwrite(abs_img_path, best_crop)
            if not ok:
                print(f"  ⚠️ ID {tid} 被过滤: 图片保存失败 -> {abs_img_path}")
                data['best_crop'] = None
                continue

            traj_json = {
                "track_id": int(tid),
                "points": traj,
                "length": len(traj),
                "unit": "cm"
            }

            insert_list.append((
                video_id,
                json.dumps(final_vector),
                idx + 1,
                json.dumps(traj_json),
                rel_img_path
            ))

            # ⭐【新增】图片落盘后及时释放内存，避免长视频/多目标导致内存膨胀
            data['best_crop'] = None

            if len(final_vector) == 512:
                es_docs.append({
                    'video_id': int(video_id),
                    'person_index': int(idx + 1),
                    'track_id': int(tid),
                    'embedding': final_vector,
                    'created_at': datetime.utcnow().isoformat()
                })
            else:
                # ⭐【关键新增】补充维度异常诊断
                print(f"  ⚠️ ID {tid} 被过滤: 向量维度异常 {len(final_vector)} (期望 512)")

        if insert_list:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    sql = "INSERT INTO video_vectors (video_id, vector_data, person_index, person_trajectory, person_image_path) VALUES (%s, %s, %s, %s, %s)"
                    cursor.executemany(sql, insert_list)
                    conn.commit()
                print(f"  ✅ 保存成功: {len(insert_list)} 个人")
                self.save_to_es(es_docs)
            except Exception as e:
                print(f"  ❌ 数据库写入错误: {e}")
                conn.rollback()
            finally:
                conn.close()
        else:
            # ⭐【关键新增】本轮全部被过滤时给出明确提示
            print(f"  ⚠️ Video ID {video_id}: 所有临时ID均被过滤，未产生可写入记录")

# -----------------------------------------------------------
# ▶️ 运行
# -----------------------------------------------------------
if __name__ == "__main__":
    # 确保清理表以便重新测试
    # answer = input("是否清空 video_vectors 表重新开始? (y/n): ")
    # if answer.lower() == 'y':
    #     # ... (执行 truncate 代码) ...
    #     pass
    
    processor = VideoProcessor()
    processor.process_db_videos()