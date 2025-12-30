import cv2
import numpy as np
import torch
import json
import os
import re
import sys
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

# -----------------------------------------------------------
# 📦 引入 Torchreid
# -----------------------------------------------------------
from torchreid import models

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

# ================= 路径配置 =================
# 假设在 video_process 目录下运行
CAMERA_CONFIG_PATH = './camera_config.json'
# 优先使用更强的 YOLOv11x 模型以减少误检
YOLO_MODEL_PATH = '../yolo11x.pt' 
DEEPSORT_CHECKPOINT = './deepsort/deep/checkpoint/ckpt.t7'

# ReID 模型配置
REID_MODEL_PATH = './osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'
REID_MODEL_NAME = 'osnet_ain_x1_0'
REID_NUM_CLASSES = 4101

# ===========================================================
# 🧠 类 1: ReID 特征提取器
# ===========================================================
class ReIDExtractor:
    def __init__(self, model_name, num_classes, model_path, device):
        self.device = device
        print(f"🏗️ [ReID] 正在构建模型: {model_name}...")
        
        self.model = models.build_model(name=model_name, num_classes=num_classes, pretrained=False)
        self.model.to(device).eval()

        if not os.path.exists(model_path):
            # 尝试在上级目录找一下
            if os.path.exists(os.path.join('..', model_path)):
                 model_path = os.path.join('..', model_path)
            else:
                # 尝试在当前目录找
                if not os.path.exists(model_path):
                     raise FileNotFoundError(f"❌ 找不到权重文件: {model_path}")
            
        print(f"📂 [ReID] 加载权重: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        print("✅ [ReID] 模型加载完成")

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_numpy):
        if img_numpy is None or img_numpy.size == 0:
            return None
            
        try:
            img = Image.fromarray(cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB))
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(tensor)
                feat = feat.cpu().numpy().flatten()
                norm_val = np.linalg.norm(feat)
                if norm_val > 0:
                    feat = feat / norm_val
                return feat.tolist()
        except Exception as e:
            print(f"⚠️ 特征提取失败: {e}")
            return None

# ===========================================================
# 🧠 类 2: 快速验证处理器
# ===========================================================
class QuickVideoProcessor:
    def __init__(self):
        self.camera_configs = self.load_camera_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        
        print("🚀 加载 DeepSORT...")
        ds_path = DEEPSORT_CHECKPOINT
        if not os.path.exists(ds_path) and os.path.exists(os.path.join('..', ds_path)):
             ds_path = os.path.join('..', ds_path)
        self.ds_extractor = Extractor(ds_path, use_cuda=torch.cuda.is_available())
        
        self.reid_extractor = ReIDExtractor(
            REID_MODEL_NAME, REID_NUM_CLASSES, REID_MODEL_PATH, self.device
        )

    def load_camera_config(self):
        path = CAMERA_CONFIG_PATH
        if not os.path.exists(path) and os.path.exists(os.path.join('..', path)):
            path = os.path.join('..', path)
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)['camera_config']
        except:
            print("⚠️ 未找到相机配置文件，将无法计算真实坐标")
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

    def process_video(self, video_path, output_json_path=None):
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return

        # 尝试从文件名解析 camera_id
        filename = os.path.basename(video_path)
        camera_match = re.match(r'(camera\d+)', filename)
        camera_id = camera_match.group(1) if camera_match else None
        print(f"▶️ 正在处理: {filename} (Camera ID: {camera_id})")

        results = self.analyze_video(video_path, camera_id)
        self.export_results(results, output_json_path)

    def analyze_video(self, video_path, camera_id):
        H = self.get_homography(camera_id)
        
        deepsort = DeepSort(
            max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
            max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
        )

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        tracks_data = {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📽️ 视频总帧数: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"  ... 处理帧 {frame_count}/{total_frames}")

            if frame_count % 5 != 0: continue

            # 1. YOLO
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo(img_rgb, verbose=False)
            
            detections = []
            if results[0].boxes:
                for box in results[0].boxes:
                    if int(box.cls) == 0 and box.conf > 0.45:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 【修改 3】增加几何形状过滤 (宽高比)
                        # w_box = x2 - x1
                        # h_box = y2 - y1
                        # # 过滤掉太小的物体 (例如高度小于 50 像素) 
                        # # 或 宽高比异常的物体 (人应该是瘦高的，宽度不应超过高度的 80%)
                        # if h_box < 50 or w_box > h_box * 1.2: 
                        #     continue
                        detections.append([x1, y1, x2, y2, box.conf.item()])

            # 2. DeepSORT
            if detections:
                dets = np.array(detections)
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
                    tracks_data[tid] = {'traj': [], 'feats': [], 'max_score': 0, 'best_img': None}

                bbox = track.to_tlbr()
                h, w = frame.shape[:2]
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))

                foot_x, foot_y = int((x1 + x2) / 2), y2
                real_pt = self.pixel_to_world((foot_x, foot_y), H)
                if real_pt:
                    tracks_data[tid]['traj'].append([round(real_pt[0], 2), round(real_pt[1], 2)])

                # ReID (基于 面积 x 清晰度 的综合评分策略)
                # 【关键修改】只在当前帧成功匹配了检测框时 (time_since_update == 0) 才进行特征提取和截图
                # 否则使用的是卡尔曼滤波的预测框，可能会截取到背景
                if track.time_since_update == 0 and x2 > x1 and y2 > y1:
                    crop_img = frame[y1:y2, x1:x2].copy() # 使用 .copy() 防止内存引用问题
                    
                    # 1. 计算面积
                    area = (x2 - x1) * (y2 - y1)
                    
                    # 2. 计算清晰度 (拉普拉斯方差)
                    # 转换灰度图计算，效率更高
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # 3. 综合评分
                    # 面积保证了分辨率，清晰度保证了质量
                    current_score = area * clarity

                    if current_score > tracks_data[tid]['max_score']:
                        feature_vec = self.reid_extractor.extract(crop_img)
                        if feature_vec:
                            tracks_data[tid]['max_score'] = current_score
                            tracks_data[tid]['feats'] = [feature_vec]
                            tracks_data[tid]['best_img'] = crop_img

        cap.release()
        return tracks_data
    def export_results(self, results, output_path):
        if not results:
            print("⚠️ 未检测到任何轨迹或特征")
            return

        final_output = []
        sorted_ids = sorted(results.keys())
        
        # 准备图片保存目录
        img_save_dir = "extracted_persons"
        if output_path:
            base_dir = os.path.dirname(os.path.abspath(output_path))
            img_save_dir = os.path.join(base_dir, "extracted_persons")
        
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
            print(f"📂 创建图片保存目录: {img_save_dir}")
        
        print(f"\n✅ 分析完成，共发现 {len(sorted_ids)} 个目标:")
        
        for tid in sorted_ids:
            data = results[tid]
            traj = data['traj']

             # 【修改 4】过滤掉轨迹过短的“幽灵” ID
            if len(traj) < 10:
                continue


            feats = data['feats']
            best_img = data.get('best_img')
            
            final_vector = feats[0] if feats else []
            
            print(f"  👤 Track ID: {tid}")
            print(f"     - 轨迹点数: {len(traj)}")
            print(f"     - 特征向量维度: {len(final_vector)}")
            
            img_rel_path = ""
            if best_img is not None:
                img_filename = f"person_{tid}.jpg"
                img_full_path = os.path.join(img_save_dir, img_filename)
                cv2.imwrite(img_full_path, best_img)
                img_rel_path = os.path.join("extracted_persons", img_filename)
                print(f"     - 📸 最佳截图已保存: {img_rel_path}")

            if final_vector:
                print(f"     - 特征向量前5位: {final_vector[:5]} ...")
            
            person_data = {
                "track_id": int(tid),
                "trajectory": traj,
                "feature_vector": final_vector,
                "image_path": img_rel_path
            }
            final_output.append(person_data)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, indent=2)
            print(f"\n💾 结果已保存至: {output_path}")
            print(f"\n💾 结果已保存至: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python quick_test_video_reid.py <video_path> [output_json_path]")
    else:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "test_result.json"
        
        processor = QuickVideoProcessor()
        processor.process_video(video_path, output_path)
