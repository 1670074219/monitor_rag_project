#!/usr/bin/env python3
"""
行人轨迹分析并添加到JSON文件的脚本
"""

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

def calculate_homography(pixel_points, real_world_points):
    """计算单应性矩阵"""
    if len(pixel_points) != 4 or len(real_world_points) != 4:
        raise ValueError("需要提供4个点来计算单应性矩阵")

    src_pts = np.array(pixel_points, dtype=np.float32)
    dst_pts = np.array(real_world_points, dtype=np.float32)

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

def extract_person_count_from_description(analyse_result):
    """从分析结果中提取人数"""
    # 使用正则表达式匹配 "人数：X"
    match = re.search(r'人数[：:]\s*(\d+)', analyse_result)
    if match:
        return int(match.group(1))
    return 1  # 默认为1人

def analyze_video_trajectory(video_path, target_person_count):
    """分析视频轨迹并返回指定数量的最长轨迹"""
    
    print(f"🎬 分析视频: {os.path.basename(video_path)}")
    print(f"   目标人数: {target_person_count}")
    
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

    # 计算单应性矩阵
    try:
        homography_matrix = calculate_homography(pixel_points_calib, real_world_points_calib)
    except ValueError as e:
        print(f"❌ 单应性矩阵计算失败: {e}")
        return []

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        yolo_model_path = os.path.abspath('video_process/yolo/yolo11s.pt')
        yolo_model = YOLO(yolo_model_path)
        yolo_model.to(device)
    except Exception as e:
        print(f"❌ YOLO模型加载失败: {e}")
        return []

    try:
        deepsort_model_path = os.path.abspath('video_process/deepsort/deep/checkpoint/ckpt.t7')
        feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
        deepsort = DeepSort(
            max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, 
            max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
        )
    except Exception as e:
        print(f"❌ DeepSORT初始化失败: {e}")
        return []

    # 视频处理
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return []

    trajectories = {}  # {track_id: [(real_x, real_y), ...]}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   总帧数: {total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # 每处理25帧进行一次检测
        if frame_count % 25 != 0:
            continue

        # 显示进度
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames})")

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
                    
                    if confidence > 0.5:
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
    
    # 选择轨迹最长的N个ID
    if not trajectories:
        print("   ❌ 未检测到任何轨迹")
        return []
    
    # 按轨迹长度排序
    sorted_trajectories = sorted(trajectories.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 取前N个最长轨迹
    selected_trajectories = []
    for i in range(min(target_person_count, len(sorted_trajectories))):
        track_id, trajectory = sorted_trajectories[i]
        selected_trajectories.append({
            "track_id": int(track_id),
            "trajectory_length": len(trajectory),
            "coordinates": trajectory
        })
        print(f"   ✅ 轨迹ID {track_id}: {len(trajectory)} 个坐标点")
    
    print(f"   🎯 选择了 {len(selected_trajectories)} 条轨迹")
    return selected_trajectories

def process_video_descriptions_with_trajectory(json_file_path, output_file_path=None):
    """处理JSON文件，为每个视频添加轨迹数据"""
    
    if output_file_path is None:
        output_file_path = json_file_path.replace('.json', '_with_trajectory.json')
    
    print(f"📁 读取JSON文件: {json_file_path}")
    
    # 读取现有JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取JSON文件失败: {e}")
        return False
    
    print(f"📊 找到 {len(video_data)} 个视频条目")
    
    # 处理每个视频
    processed_count = 0
    for video_id, video_info in video_data.items():
        print(f"\n🎬 处理视频: {video_id}")
        
        # 检查是否已有轨迹数据
        if 'trajectory_data' in video_info:
            print(f"   ⏭️  轨迹数据已存在，跳过")
            continue
        
        video_path = video_info.get('video_path')
        analyse_result = video_info.get('analyse_result', '')
        
        if not video_path or not os.path.exists(video_path):
            print(f"   ❌ 视频文件不存在: {video_path}")
            continue
        
        # 提取人数
        person_count = extract_person_count_from_description(analyse_result)
        
        # 分析轨迹
        trajectories = analyze_video_trajectory(video_path, person_count)
        
        # 添加轨迹数据到JSON
        video_info['trajectory_data'] = {
            "person_count": person_count,
            "trajectories": trajectories,
            "coordinate_system": "real_world",  # 标明是真实世界坐标
            "unit": "centimeters"  # 坐标单位
        }
        
        processed_count += 1
        print(f"   ✅ 轨迹数据已添加")
    
    # 保存更新后的JSON文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 处理完成!")
        print(f"   处理视频数: {processed_count}")
        print(f"   输出文件: {output_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 保存JSON文件失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 行人轨迹分析脚本")
    print("=" * 60)
    
    # 配置文件路径
    json_file_path = 'video_process/video_description.json'
    
    if not os.path.exists(json_file_path):
        print(f"❌ JSON文件不存在: {json_file_path}")
        return
    
    # 处理视频并添加轨迹数据
    success = process_video_descriptions_with_trajectory(json_file_path)
    
    if success:
        print("\n✅ 所有视频轨迹分析完成！")
        print("💡 轨迹数据格式说明:")
        print("   - track_id: 轨迹ID")
        print("   - trajectory_length: 轨迹点数量")
        print("   - coordinates: [[x1,y1], [x2,y2], ...] (真实世界坐标)")
        print("   - coordinate_system: real_world (真实世界坐标系)")
        print("   - unit: centimeters (坐标单位：厘米)")
    else:
        print("\n❌ 处理失败，请检查错误信息")

if __name__ == "__main__":
    main() 