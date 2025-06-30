#!/usr/bin/env python3
"""
单个视频轨迹分析测试脚本
"""

import cv2
import numpy as np
import torch
import json
import os
from ultralytics import YOLO

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

def calculate_homography():
    """计算单应性矩阵"""
    # 定义标定点 (根据您的摄像头配置)
    pixel_points_calib = [
        (517, 95),
        (667, 95),
        (1182, 720),
        (276, 720)
    ]
    
    real_world_points_calib = [
        (400, 351),
        (525, 351),
        (525, 1020),
        (400, 1020)
    ]

    src_pts = np.array(pixel_points_calib, dtype=np.float32)
    dst_pts = np.array(real_world_points_calib, dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def transform_point(point, H):
    """使用单应性矩阵转换点到真实世界坐标"""
    px, py = point
    pixel_homogeneous = np.array([[px], [py], [1]], dtype=np.float32)
    
    real_world_homogeneous = np.dot(H, pixel_homogeneous)
    
    if real_world_homogeneous[2, 0] != 0:
        real_X = real_world_homogeneous[0, 0] / real_world_homogeneous[2, 0]
        real_Y = real_world_homogeneous[1, 0] / real_world_homogeneous[2, 0]
        return real_X, real_Y
    else:
        return None, None

def test_video_trajectory(video_path, target_person_count=2):
    """测试单个视频的轨迹分析"""
    
    print(f"🎬 测试视频: {os.path.basename(video_path)}")
    print(f"   目标人数: {target_person_count}")
    
    # 计算单应性矩阵
    homography_matrix = calculate_homography()

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    # YOLO模型
    yolo_model_path = os.path.abspath('video_process/yolo/yolo11s.pt')
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device)

    # DeepSORT
    deepsort_model_path = os.path.abspath('video_process/deepsort/deep/checkpoint/ckpt.t7')
    feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )

    # 视频处理
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return None

    trajectories = {}  # {track_id: [(real_x, real_y), ...]}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"   总帧数: {total_frames}")
    print(f"   帧率: {fps:.1f} FPS")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 每处理3帧进行一次检测（更密集采样）
        if frame_count % 3 != 0:
            continue

        # 显示进度
        if frame_count % 150 == 0:
            progress = (frame_count / total_frames) * 100
            current_time = frame_count / fps
            print(f"   进度: {progress:.1f}% - 时间: {current_time:.1f}s - 已检测轨迹: {len(trajectories)}")

        # YOLO检测
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)

        # 提取人的检测结果
        person_detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls.item()) == 0:  # 人
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    
                    if confidence > 0.4:  # 降低阈值，检测更多候选
                        person_detections.append([x1, y1, x2, y2, confidence])

        if len(person_detections) == 0:
            continue

        # 验证边界框
        valid_detections = []
        h, w = frame.shape[:2]
        
        for det in person_detections:
            x1, y1, x2, y2, conf = det
            
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 > x1 and y2 > y1:
                valid_detections.append([x1, y1, x2, y2, conf])
        
        if len(valid_detections) == 0:
            continue
            
        valid_detections = np.array(valid_detections)
        
        # 转换为xywh格式
        bbox_xywh = np.zeros_like(valid_detections[:, :4])
        bbox_xywh[:, 0] = (valid_detections[:, 0] + valid_detections[:, 2]) / 2
        bbox_xywh[:, 1] = (valid_detections[:, 1] + valid_detections[:, 3]) / 2  
        bbox_xywh[:, 2] = valid_detections[:, 2] - valid_detections[:, 0]
        bbox_xywh[:, 3] = valid_detections[:, 3] - valid_detections[:, 1]

        confidences = valid_detections[:, 4]
        
        # DeepSORT跟踪
        try:
            outputs = deepsort.update(bbox_xywh, confidences, frame, feature_extractor)
            
            if len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_id = output
                    
                    # 计算脚部位置（真实世界坐标）
                    foot_point_pixel = (int((x1 + x2) / 2), int(y2))
                    real_x, real_y = transform_point(foot_point_pixel, homography_matrix)

                    if real_x is not None:
                        if track_id not in trajectories:
                            trajectories[track_id] = []
                        trajectories[track_id].append([round(real_x, 2), round(real_y, 2)])
                        
        except Exception as e:
            print(f"   ⚠️  帧{frame_count}处理错误: {e}")
            continue

    cap.release()
    
    # 分析结果
    if not trajectories:
        print("   ❌ 未检测到任何轨迹")
        return None
    
    print(f"\n📊 轨迹分析结果:")
    print(f"   检测到轨迹数: {len(trajectories)}")
    
    # 按轨迹长度排序
    sorted_trajectories = sorted(trajectories.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (track_id, trajectory) in enumerate(sorted_trajectories):
        print(f"   轨迹ID {track_id}: {len(trajectory)} 个点")
        if len(trajectory) > 0:
            print(f"      起点: {trajectory[0]}")
            print(f"      终点: {trajectory[-1]}")
    
    # 选择最长的N条轨迹
    selected_trajectories = []
    for i in range(min(target_person_count, len(sorted_trajectories))):
        track_id, trajectory = sorted_trajectories[i]
        selected_trajectories.append({
            "track_id": int(track_id),
            "trajectory_length": len(trajectory),
            "coordinates": trajectory
        })
    
    print(f"\n🎯 选择轨迹:")
    for traj in selected_trajectories:
        print(f"   ID {traj['track_id']}: {traj['trajectory_length']} 个坐标点")
    
    return selected_trajectories

def main():
    """主函数"""
    print("🚀 单视频轨迹分析测试")
    print("=" * 50)
    
    # 测试视频文件
    test_video_path = 'video_process/saved_video/camera1_20250623_094900.mp4'
    
    if not os.path.exists(test_video_path):
        print(f"❌ 测试视频文件不存在: {test_video_path}")
        print("请修改 test_video_path 变量为实际存在的视频文件路径")
        return
    
    # 运行测试
    result = test_video_trajectory(test_video_path, target_person_count=2)
    
    if result:
        print("\n✅ 测试成功!")
        
        # 保存测试结果
        output_file = 'video_process/test_trajectory_result.json'
        test_result = {
            "video_path": test_video_path,
            "trajectory_data": {
                "person_count": 2,
                "trajectories": result,
                "coordinate_system": "real_world",
                "unit": "centimeters"
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)
        
        print(f"   测试结果已保存: {output_file}")
        
        # 显示坐标样例
        if result and len(result) > 0 and len(result[0]['coordinates']) > 0:
            print(f"\n💡 坐标样例 (前5个点):")
            for i, coord in enumerate(result[0]['coordinates'][:5]):
                print(f"   点{i+1}: [{coord[0]}, {coord[1]}]")
    else:
        print("\n❌ 测试失败")

if __name__ == "__main__":
    main() 