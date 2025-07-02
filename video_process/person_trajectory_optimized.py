import cv2
import numpy as np
import torch
import json
import os
import re
from ultralytics import YOLO

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

def load_camera_config(config_path='video_process/camera_config.json'):
    """
    加载摄像头配置文件
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config['camera_config']
    except Exception as e:
        print(f"加载摄像头配置失败: {e}")
        return None

def load_video_descriptions(desc_path='video_process/video_description.json'):
    """
    加载视频描述文件
    """
    try:
        with open(desc_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        return descriptions
    except Exception as e:
        print(f"加载视频描述失败: {e}")
        return None

def save_trajectory_results(results, output_path='video_process/video_description_with_trajectory.json'):
    """
    保存轨迹结果到JSON文件
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"轨迹结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存轨迹结果失败: {e}")

def update_single_video_trajectory(video_name, trajectory_data, desc_path='video_process/video_description.json'):
    """
    更新单个视频的轨迹数据到原始JSON文件
    """
    try:
        # 读取当前的视频描述文件
        with open(desc_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        
        # 更新对应视频的轨迹数据
        if video_name in descriptions:
            descriptions[video_name]['trajectory_data'] = trajectory_data
            
            # 立即写回文件
            with open(desc_path, 'w', encoding='utf-8') as f:
                json.dump(descriptions, f, indent=4, ensure_ascii=False)
            
            print(f"✓ 视频 {video_name} 的轨迹数据已更新到 {desc_path}")
            return True
        else:
            print(f"❌ 在描述文件中未找到视频 {video_name}")
            return False
            
    except Exception as e:
        print(f"❌ 更新视频 {video_name} 轨迹数据失败: {e}")
        return False

def extract_camera_id_from_video_name(video_name):
    """
    从视频名称中提取摄像头ID
    例如: "camera1_20250623_095558" -> "camera1"
    """
    match = re.match(r'(camera\d+)', video_name)
    if match:
        return match.group(1)
    return None

def get_camera_config_by_id(camera_configs, camera_id):
    """
    根据摄像头ID获取配置
    """
    for config in camera_configs:
        if config['camera_id'] == camera_id:
            return config
    return None

def calculate_homography(pixel_points, real_world_points):
    """
    根据给定的像素点和真实世界点计算单应性矩阵。
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
        return None, None

def process_single_video(video_path, camera_config, yolo_model, feature_extractor):
    """
    处理单个视频文件，返回轨迹数据
    """
    print(f"\n开始处理视频: {video_path}")
    
    # 计算单应性矩阵
    pixel_points = camera_config['pixel_coordinate']
    real_world_points = camera_config['real_coordinate']
    
    try:
        homography_matrix = calculate_homography(pixel_points, real_world_points)
        print(f"摄像头 {camera_config['camera_id']} 单应性矩阵计算成功")
    except ValueError as e:
        print(f"计算单应性矩阵失败: {e}")
        return None

    # 初始化 DeepSORT
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.7, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None

    frame_count = 0
    trajectories = {}  # 用于存储每个ID的轨迹

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 每处理5帧进行一次检测，以提高性能
        if frame_count % 5 != 0:
            continue

        # 将图像从 BGR 转换为 RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 YOLO 模型进行目标检测
        results = yolo_model(img_rgb)

        # 获取检测结果 - 只检测人
        person_detections = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # 检查是否是人 (class 0)
                if int(box.cls.item()) == 0:
                    # 获取边界框坐标 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    
                    # 只保留置信度高的检测
                    if confidence > 0.5:
                        person_detections.append([x1, y1, x2, y2, confidence])

        if len(person_detections) > 0:
            try:
                # 转换为numpy数组
                person_detections = np.array(person_detections)
                
                # 验证和清理边界框
                valid_detections = []
                for det in person_detections:
                    x1, y1, x2, y2, conf = det
                    
                    # 确保坐标在图像范围内且有效
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # 确保边界框有效 (宽度和高度>0)
                    if x2 > x1 and y2 > y1:
                        valid_detections.append([x1, y1, x2, y2, conf])
                
                if len(valid_detections) == 0:
                    continue
                    
                valid_detections = np.array(valid_detections)
                
                # 转换xyxy到xywh格式 (DeepSORT需要)
                bbox_xywh = np.zeros_like(valid_detections[:, :4])
                bbox_xywh[:, 0] = (valid_detections[:, 0] + valid_detections[:, 2]) / 2  # center_x
                bbox_xywh[:, 1] = (valid_detections[:, 1] + valid_detections[:, 3]) / 2  # center_y  
                bbox_xywh[:, 2] = valid_detections[:, 2] - valid_detections[:, 0]        # width
                bbox_xywh[:, 3] = valid_detections[:, 3] - valid_detections[:, 1]        # height

                confidences = valid_detections[:, 4]
                
                # 更新 DeepSORT 追踪器
                outputs = deepsort.update(bbox_xywh, confidences, frame, feature_extractor)
            except Exception as e:
                print(f"  处理检测结果时出错: {e}")
                continue
            
            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id = output
                    
                    # 计算脚部中心点 (人的底部中心)
                    foot_point_pixel = (int((x1 + x2) / 2), int(y2))
                    real_x, real_y = transform_point(foot_point_pixel, homography_matrix)

                    if real_x is not None:
                        # 记录轨迹
                        if track_id not in trajectories:
                            trajectories[track_id] = []
                        trajectories[track_id].append([round(real_x, 2), round(real_y, 2)])

    cap.release()
    
    # 格式化轨迹数据
    trajectory_data = {
        "person_count": len(trajectories),
        "trajectories": [],
        "coordinate_system": "real_world",
        "unit": "centimeters"
    }
    
    for track_id, coordinates in trajectories.items():
        trajectory_data["trajectories"].append({
            "track_id": int(track_id),
            "trajectory_length": len(coordinates),
            "coordinates": coordinates
        })
    
    print(f"视频处理完成，检测到 {len(trajectories)} 个人的轨迹")
    return trajectory_data

def main():
    print("开始批量处理视频轨迹...")
    
    # 1. 加载配置和描述文件
    camera_configs = load_camera_config()
    video_descriptions = load_video_descriptions()
    
    if not camera_configs or not video_descriptions:
        print("无法加载必要的配置文件，程序退出")
        return
    
    # 2. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")

    try:
        yolo_model_path = os.path.abspath('video_process/yolo/yolo11s.pt')
        yolo_model = YOLO(yolo_model_path)
        yolo_model.to(device)
        print(f"YOLOv11 模型加载成功")
    except Exception as e:
        print(f"加载 YOLO 模型失败: {e}")
        return

    # 初始化 DeepSORT 特征提取器
    deepsort_model_path = 'video_process/deepsort/deep/checkpoint/ckpt.t7'
    try:
        feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
        print("DeepSORT 特征提取器加载成功")
    except Exception as e:
        print(f"加载 DeepSORT 特征提取器失败: {e}")
        return

    # 3. 逐个处理每个视频并立即保存
    processed_count = 0
    total_videos = len(video_descriptions)
    
    print(f"\n总共需要处理 {total_videos} 个视频")
    print("=" * 60)
    
    for video_name, video_info in video_descriptions.items():
        processed_count += 1
        print(f"\n[{processed_count}/{total_videos}] 正在处理: {video_name}")
        
        # 提取摄像头ID
        camera_id = extract_camera_id_from_video_name(video_name)
        if not camera_id:
            print(f"❌ 无法从视频名称 {video_name} 中提取摄像头ID，跳过")
            continue
        
        # 获取摄像头配置
        camera_config = get_camera_config_by_id(camera_configs, camera_id)
        if not camera_config:
            print(f"❌ 未找到摄像头 {camera_id} 的配置，跳过")
            continue
        
        # 检查视频文件是否存在
        video_path = video_info['video_path']
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}，跳过")
            continue
        
        # 处理视频
        trajectory_data = process_single_video(video_path, camera_config, yolo_model, feature_extractor)
        
        # 准备轨迹数据
        if trajectory_data:
            final_trajectory_data = trajectory_data
        else:
            final_trajectory_data = {
                "person_count": 0,
                "trajectories": [],
                "coordinate_system": "real_world",
                "unit": "centimeters"
            }
        
        # 立即更新JSON文件
        success = update_single_video_trajectory(video_name, final_trajectory_data)
        if success:
            print(f"✓ 进度: {processed_count}/{total_videos} 完成")
        else:
            print(f"❌ 更新失败，但继续处理下一个视频")
    
    print("\n" + "=" * 60)
    print(f"🎉 全部处理完成！共处理了 {processed_count} 个视频")
    print(f"📁 结果已直接保存到: video_process/video_description.json")

if __name__ == '__main__':
    main() 