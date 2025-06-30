#!/usr/bin/env python3
"""
快速两人检测测试脚本 - 精确复现问题
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

def test_two_person_deepsort():
    """快速测试两人情况下的DeepSORT错误"""
    
    print("🔧 快速两人检测测试")
    print("=" * 50)
    
    # 初始化DeepSORT
    import os
    deepsort_model_path = os.path.abspath('video_process/deepsort/deep/checkpoint/ckpt.t7')
    print(f"DeepSORT模型路径: {deepsort_model_path}")
    print(f"文件是否存在: {os.path.exists(deepsort_model_path)}")
    
    feature_extractor = Extractor(deepsort_model_path, use_cuda=torch.cuda.is_available())
    
    deepsort = DeepSort(
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5, 
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
    )
    
    # 创建测试图像
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("📊 测试1: 单人数据")
    # 单人数据 - 应该成功
    bbox_single = np.array([[640, 360, 80, 160]], dtype=np.float32)
    conf_single = np.array([0.9])
    
    try:
        outputs = deepsort.update(bbox_single, conf_single, test_frame, feature_extractor)
        print(f"✅ 单人测试成功: {len(outputs)} 个输出")
    except Exception as e:
        print(f"❌ 单人测试失败: {e}")
    
    print("\n📊 测试2: 两人数据")
    # 两人数据 - 可能失败
    bbox_two = np.array([
        [300, 400, 80, 160],   # 人1
        [600, 420, 70, 150]    # 人2
    ], dtype=np.float32)
    conf_two = np.array([0.85, 0.92])
    
    try:
        print("🚀 开始两人DeepSORT测试...")
        outputs = deepsort.update(bbox_two, conf_two, test_frame, feature_extractor)
        print(f"✅ 两人测试成功: {len(outputs)} 个输出")
    except Exception as e:
        print(f"❌ 两人测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_two_person_deepsort() 