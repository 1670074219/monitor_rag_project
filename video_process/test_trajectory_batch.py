#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 添加当前目录到路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from person_trajectory_optimized import main

def test_batch_trajectory_processing():
    """
    测试批量轨迹处理功能
    """
    print("=" * 60)
    print("开始测试批量视频轨迹处理功能")
    print("=" * 60)
    
    # 检查必要文件是否存在
    required_files = [
        'camera_config.json',
        'video_description.json',
        'yolo/yolo11s.pt',
        'deepsort/deep/checkpoint/ckpt.t7'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误：以下必要文件缺失:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保所有必要文件都存在后再运行测试。")
        return False
    
    print("✓ 所有必要文件都存在")
    
    try:
        # 运行主处理函数
        main()
        print("\n" + "=" * 60)
        print("测试完成！请检查生成的 video_description_with_trajectory.json 文件")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_batch_trajectory_processing()
    if success:
        print("\n🎉 批量轨迹处理测试成功完成！")
    else:
        print("\n❌ 批量轨迹处理测试失败！") 