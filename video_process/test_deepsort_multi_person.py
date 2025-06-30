#!/usr/bin/env python3
"""
简化的DeepSORT多人检测测试脚本
用于调试"检测到两个人就报错"的问题
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

def test_deepsort_with_multi_person():
    """测试DeepSORT处理多人检测的能力"""
    
    print("🔧 初始化YOLO和DeepSORT...")
    
    # 加载YOLO模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = YOLO('video_process/yolo/yolo11s.pt')
    yolo_model.to(device)
    
    # 初始化DeepSORT
    deepsort_model_path = 'video_process/deepsort/deep/checkpoint/ckpt.t7'
    feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
    
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )
    
    print("✅ 模型加载成功!")
    
    # 测试用的视频或图像
    test_video = '/root/data1/monitor_rag_project/camera1_20250609_164924.mp4'
    cap = cv2.VideoCapture(test_video)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {test_video}")
        return
    
    frame_count = 0
    test_frames = 10  # 只测试前10帧
    
    while cap.isOpened() and frame_count < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        print(f"\n🎬 测试帧 {frame_count}")
        print("=" * 50)
        
        # YOLO检测
        results = yolo_model(frame)
        
        # 提取人的检测结果
        person_detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls.item()) == 0:  # 人
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    
                    if confidence > 0.5:
                        person_detections.append([x1, y1, x2, y2, confidence])
        
        print(f"📊 YOLO检测到 {len(person_detections)} 个人")
        
        if len(person_detections) == 0:
            print("⏭️  跳过：没有检测到人")
            continue
        
        # 验证边界框
        valid_detections = []
        h, w = frame.shape[:2]
        
        for det in person_detections:
            x1, y1, x2, y2, conf = det
            
            # 边界框验证
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 > x1 and y2 > y1:
                valid_detections.append([x1, y1, x2, y2, conf])
        
        if len(valid_detections) == 0:
            print("⏭️  跳过：没有有效的边界框")
            continue
        
        valid_detections = np.array(valid_detections)
        
        # 转换为xywh格式
        bbox_xywh = np.zeros_like(valid_detections[:, :4])
        bbox_xywh[:, 0] = (valid_detections[:, 0] + valid_detections[:, 2]) / 2  # center_x
        bbox_xywh[:, 1] = (valid_detections[:, 1] + valid_detections[:, 3]) / 2  # center_y  
        bbox_xywh[:, 2] = valid_detections[:, 2] - valid_detections[:, 0]        # width
        bbox_xywh[:, 3] = valid_detections[:, 3] - valid_detections[:, 1]        # height
        
        confidences = valid_detections[:, 4]
        
        print(f"✅ 有效检测: {len(valid_detections)} 个")
        print(f"📏 边界框形状: {bbox_xywh.shape}")
        print(f"🎯 置信度: {confidences}")
        
        # 测试DeepSORT
        try:
            print("🚀 开始DeepSORT跟踪...")
            outputs = deepsort.update(bbox_xywh, confidences, frame, feature_extractor)
            
            print(f"✅ DeepSORT成功! 输出轨迹数: {len(outputs)}")
            
            for output in outputs:
                x1, y1, x2, y2, track_id = output
                print(f"   轨迹ID {track_id}: [{x1}, {y1}, {x2}, {y2}]")
                
        except Exception as e:
            print(f"❌ DeepSORT失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    cap.release()
    print(f"\n🎉 测试完成! 处理了 {frame_count} 帧")

def test_specific_multi_person_case():
    """测试特定的多人情况"""
    
    print("\n🧪 测试特定多人情况...")
    
    # 模拟两个人的边界框数据
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 两个人的边界框 (xywh格式)
    bbox_xywh = np.array([
        [300, 400, 80, 160],   # 人1: 中心(300,400), 宽80, 高160
        [600, 420, 70, 150]    # 人2: 中心(600,420), 宽70, 高150
    ], dtype=np.float32)
    
    confidences = np.array([0.85, 0.92])
    
    print(f"📊 模拟数据:")
    print(f"   边界框形状: {bbox_xywh.shape}")
    print(f"   置信度: {confidences}")
    
    # 初始化DeepSORT
    deepsort_model_path = 'video_process/deepsort/deep/checkpoint/ckpt.t7'
    feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
    
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )
    
    try:
        print("🚀 测试DeepSORT...")
        outputs = deepsort.update(bbox_xywh, confidences, fake_frame, feature_extractor)
        print(f"✅ 成功! 输出数量: {len(outputs)}")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 DeepSORT多人检测测试")
    print("=" * 60)
    
    # 测试1: 真实视频帧
    test_deepsort_with_multi_person()
    
    # 测试2: 模拟数据
    test_specific_multi_person_case() 