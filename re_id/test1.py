import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from torchreid import models  # 确保你已安装 torchreid（源码版）

# ----------------------------
# 配置
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 加载YOLOv8模型
yolo_model = YOLO('yolov8n.pt')

# 2. 手动构建 ReID 特征提取器（OSNet）
model = models.build_model(name='osnet_x1_0', num_classes=1000)
model.to(device).eval()

# 👉 下载预训练权重（Market1501 微调版，效果远好于 ImageNet）
# 权重链接: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
# 示例：osnet_x1_0 on Market1501
model_path = './osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ 成功加载 ReID 模型权重: {model_path}")
except Exception as e:
    print(f"⚠️ 未找到 ReID 权重文件，请先下载并放在当前目录！\n错误: {e}")
    print("下载地址: https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
    exit(1)

# 图像预处理（与训练时一致）
transform = T.Compose([
    T.Resize((256, 128)),  # H=256, W=128 是 ReID 标准输入
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_reid_feature(img_cv2):
    """
    输入: OpenCV BGR 图像 (H, W, C)
    输出: 归一化的 ReID 特征向量 (numpy, shape=[512])
    """
    if img_cv2.size == 0:
        return None
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor)
        feat = feat.cpu().numpy().flatten()
        # L2 归一化（重要！用于余弦相似度）
        feat = feat / np.linalg.norm(feat)
    return feat

def extract_person_features(video_path):
    """从视频中提取所有检测到行人的 ReID 特征"""
    cap = cv2.VideoCapture(video_path)
    features_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 可选：跳帧加速
        if frame_count % 10 != 0:
            frame_count += 1
            continue

        # YOLO 检测行人（class 0 = person）
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes
        person_boxes = boxes[boxes.cls == 0]

        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_img = frame[y1:y2, x1:x2]
            feat = extract_reid_feature(person_img)
            if feat is not None:
                features_list.append(feat)

        frame_count += 1

    cap.release()
    return np.array(features_list) if features_list else np.empty((0, 512))

def cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2)  # 因为已 L2 归一化，点积 = 余弦相似度

def is_same_person(features1, features2, threshold=0.35):
    if features1.size == 0 or features2.size == 0:
        return False

    max_sim = -1.0
    for f1 in features1:
        sims = cosine_similarity(f1, features2.T)  # 向量化计算
        local_max = np.max(sims)
        if local_max > max_sim:
            max_sim = local_max

    print(f"最大相似度: {max_sim:.4f}")
    return max_sim >= threshold

# === 主程序 ===
if __name__ == "__main__":
    video1_path = "./camera3_20250623_180611.mp4"
    video2_path = "./camera6_20250623_180643.mp4"

    # 先确保 torchreid 已正确安装（至少能 import models）
    try:
        from torchreid import models
    except ImportError:
        print("❌ 请先安装 torchreid（源码版）:")
        print("   pip uninstall torchreid -y")
        print("   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
        exit(1)

    print("正在提取视频1的行人特征...")
    feats1 = extract_person_features(video1_path)

    print("正在提取视频2的行人特征...")
    feats2 = extract_person_features(video2_path)

    print(f"视频1检测到 {len(feats1)} 个行人特征")
    print(f"视频2检测到 {len(feats2)} 个行人特征")

    result = is_same_person(feats1, feats2, threshold=0.35)  # ReID 常用阈值 0.3~0.4
    if result:
        print("✅ 两个视频中存在相同的人！")
    else:
        print("❌ 未检测到相同的人。")