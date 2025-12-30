import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import models, transforms
import torch.nn.functional as F
from scipy.spatial.distance import cosine

# 初始化 DeepSORT 跟踪器
deepsort = DeepSort()

# 加载预训练的 ResNet50 模型用于特征提取
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()  # 设置为评估模式
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 函数：提取行人特征
def extract_person_feature(frame, bbox):
    x1, y1, x2, y2 = bbox
    cropped_image = frame[y1:y2, x1:x2]
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        return None
    image = transform(cropped_image).unsqueeze(0)  # 转换为 PyTorch 张量
    with torch.no_grad():
        features = resnet_model(image)
    return features.squeeze().cpu().numpy()  # 转换为 numpy 数组

# 函数：计算特征向量的相似度
def calculate_similarity(feat1, feat2):
    return 1 - cosine(feat1, feat2)

# 视频加载和行人检测
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:  # 每10帧处理一次，减少计算量
            continue

        # 深度学习检测行人（此处简化为使用 OpenCV 的HOG+SVM行人检测器）
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

        # 对检测到的每个行人进行跟踪和特征提取
        for bbox in boxes:
            feature = extract_person_feature(frame, bbox)
            if feature is not None:
                all_features.append(feature)
                # 你可以在这里使用 DeepSORT 来进行更精确的跟踪

    cap.release()
    return all_features

# 比较两个视频中的特征
def compare_videos(video1_path, video2_path):
    features_video1 = process_video(video1_path)
    features_video2 = process_video(video2_path)

    for feat1 in features_video1:
        for feat2 in features_video2:
            similarity = calculate_similarity(feat1, feat2)
            if similarity > 0.6:  # 相似度阈值，可以根据需要调整
                print("检测到相同的人!")
                return True
    print("视频中没有相同的人")
    return False

# 主程序
if __name__ == "__main__":
    video1_path = './camera1_20250623_101339.mp4'  # 第一个视频路径
    video2_path = './camera1_20250623_101339.mp4'  # 第二个视频路径
    compare_videos(video1_path, video2_path)
