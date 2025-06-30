import cv2
import numpy as np
import torch
from ultralytics import YOLO  # 导入YOLO类

# DeepSORT aimports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

def calculate_homography(pixel_points, real_world_points):
    """
    根据给定的像素点和真实世界点计算单应性矩阵。

    :param pixel_points: 像素坐标点列表，格式为 [(x1, y1), (x2, y2), ...]
    :param real_world_points: 对应的真实世界坐标点列表，格式为 [(X1, Y1), (X2, Y2), ...]
    :return: 单应性矩阵 H
    """
    if len(pixel_points) != 4 or len(real_world_points) != 4:
        raise ValueError("需要提供4个点来计算单应性矩阵")

    # 将点转换为 numpy 数组
    src_pts = np.array(pixel_points, dtype=np.float32)
    dst_pts = np.array(real_world_points, dtype=np.float32)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def transform_point(point, H):
    """
    使用单应性矩阵将单个点从像素坐标转换为真实世界坐标。

    :param point: 要转换的像素点 (x, y)
    :param H: 单应性矩阵
    :return: 转换后的真实世界坐标 (X, Y)
    """
    px, py = point
    # 转换为齐次坐标
    pixel_homogeneous = np.array([[px], [py], [1]], dtype=np.float32)
    
    # 应用单应性变换
    real_world_homogeneous = np.dot(H, pixel_homogeneous)
    
    # 转换为非齐次坐标
    if real_world_homogeneous[2, 0] != 0:
        real_X = real_world_homogeneous[0, 0] / real_world_homogeneous[2, 0]
        real_Y = real_world_homogeneous[1, 0] / real_world_homogeneous[2, 0]
        return real_X, real_Y
    else:
        # 避免除以零
        return None, None

def main():
    # --- 1. 定义标定点 ---
    # 您提供的像素坐标点 (来自 frame.jpg)
    # 点1, 点2, 点3, 点4
    pixel_points_calib = [
        (479, 117),
        (629, 122),
        (1033, 717),
        (206, 716)
    ]
    
    # 对应的真实世界坐标点 (单位：米)
    real_world_points_calib = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0)
    ]

    # --- 2. 计算单应性矩阵 ---
    try:
        homography_matrix = calculate_homography(pixel_points_calib, real_world_points_calib)
        print("单应性矩阵 (Homography Matrix) 计算成功:")
        print(homography_matrix)
    except ValueError as e:
        print(f"错误: {e}")
        exit()

    # --- 3. 加载模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    try:
        # 使用 ultralytics 包加载本地的 YOLOv11 模型
        yolo_model_path = 'video_process/yolo/yolo11s.pt'
        yolo_model = YOLO(yolo_model_path)
        yolo_model.to(device)
        print(f"YOLOv11 本地模型 '{yolo_model_path}' 加载成功。")
    except Exception as e:
        print(f"加载 YOLO 模型失败: {e}")
        exit()

    # 初始化 DeepSORT
    deepsort_model_path = 'video_process/deepsort/deep/checkpoint/ckpt.t7'
    try:
        feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
        print("DeepSORT 特征提取器加载成功。")
    except Exception as e:
        print(f"加载 DeepSORT 特征提取器失败: {e}")
        print("请确保 'video_process/deepsort/deep/checkpoint/ckpt.t7' 权重文件存在。")
        exit()
        
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.7, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )

    # --- 4. 视频处理 ---
    video_path = 'video_process/saved_video/camera6_20250617_153316.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        exit()

    frame_count = 0
    trajectories = {} # 用于存储每个ID的轨迹

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 每处理5帧进行一次检测，以提高性能
        if frame_count % 5 != 0:
            continue

        # 将图像从 BGR 转换为 RGB (YOLOv5需要)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 YOLO 模型进行目标检测
        results = yolo_model(img_rgb)

        # 获取检测结果
        # ultralytics 的 results 对象可以直接访问 xyxy, conf, cls
        person_detections = []
        for res in results:
            for box in res.boxes:
                if box.cls == 0: # 类别0是 'person'
                    person_detections.append(box.xyxy[0].tolist() + [box.conf[0].item()] + [box.cls[0].item()])
        
        person_detections = torch.tensor(person_detections)

        print(f"\n--- 帧 {frame_count} (检测到 {len(person_detections)} 个目标) ---")

        if len(person_detections) > 0:
            try:
                bbox_xywh = person_detections[:, :4].clone() # YOLO 格式是 xyxy, 需要转换
                # (center_x, center_y, width, height)
                bbox_xywh[:, 0] = (person_detections[:, 0] + person_detections[:, 2]) / 2
                bbox_xywh[:, 1] = (person_detections[:, 1] + person_detections[:, 3]) / 2
                bbox_xywh[:, 2] = person_detections[:, 2] - person_detections[:, 0]
                bbox_xywh[:, 3] = person_detections[:, 3] - person_detections[:, 1]

                confidences = person_detections[:, 4]
                
                # 更新 DeepSORT 追踪器
                outputs = deepsort.update(bbox_xywh.cpu(), confidences.cpu(), frame, feature_extractor)
            except Exception as e:
                print(f"  处理检测结果时出错: {e}")
                continue
            
            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id = output
                    
                    foot_point_pixel = (int((x1 + x2) / 2), int(y2))
                    real_x, real_y = transform_point(foot_point_pixel, homography_matrix)

                    if real_x is not None:
                        # 记录轨迹
                        if track_id not in trajectories:
                            trajectories[track_id] = []
                        trajectories[track_id].append((real_x, real_y))

                        print(f"  追踪ID {track_id}: 像素坐标 {foot_point_pixel} -> 真实世界坐标 ({real_x:.2f}, {real_y:.2f})")
                        
                        # (可选) 在帧上绘制信息
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # cv2.circle(frame, foot_point_pixel, 5, (0, 0, 255), -1)

        # 如果需要显示处理后的视频，取消下面的注释
        # cv2.imshow('YOLO Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()
    print("\n视频处理完成。")

    # 打印最终轨迹
    print("\n--- 最终轨迹记录 ---")
    for track_id, path in trajectories.items():
        print(f"  ID {track_id}: 共记录 {len(path)} 个点")

if __name__ == '__main__':
    main() 