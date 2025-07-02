#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import time

# 添加当前目录到路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from person_trajectory_optimized import main

def backup_original_file():
    """
    备份原始的video_description.json文件
    """
    original_file = os.path.join(current_dir, 'video_description.json')
    backup_file = os.path.join(current_dir, 'video_description_backup.json')
    
    if os.path.exists(original_file):
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ 已备份原始文件到: {backup_file}")
            return original_data
        except Exception as e:
            print(f"❌ 备份文件失败: {e}")
            return None
    else:
        print(f"❌ 原始文件不存在: {original_file}")
        return None

def check_trajectory_updates():
    """
    检查轨迹更新结果
    """
    original_file = os.path.join(current_dir, 'video_description.json')
    
    if os.path.exists(original_file):
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("\n" + "=" * 60)
            print("轨迹处理结果检查:")
            print("=" * 60)
            
            for video_name, video_info in data.items():
                if 'trajectory_data' in video_info:
                    trajectory = video_info['trajectory_data']
                    person_count = trajectory.get('person_count', 0)
                    trajectories = trajectory.get('trajectories', [])
                    
                    print(f"✓ {video_name}:")
                    print(f"  - 检测到人数: {person_count}")
                    print(f"  - 轨迹数量: {len(trajectories)}")
                    
                    for traj in trajectories:
                        track_id = traj.get('track_id', 'N/A')
                        length = traj.get('trajectory_length', 0)
                        print(f"    └─ 轨迹ID {track_id}: {length} 个坐标点")
                else:
                    print(f"❌ {video_name}: 未找到轨迹数据")
            
            return True
            
        except Exception as e:
            print(f"❌ 检查结果失败: {e}")
            return False
    else:
        print(f"❌ 文件不存在: {original_file}")
        return False

def test_batch_trajectory_processing():
    """
    测试批量轨迹处理功能（优化版）
    """
    print("=" * 60)
    print("开始测试优化后的批量视频轨迹处理功能")
    print("📝 新功能：逐个处理视频并立即更新JSON文件")
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
    
    # 备份原始文件
    original_data = backup_original_file()
    if not original_data:
        print("❌ 无法备份原始文件，测试终止")
        return False
    
    print(f"✓ 发现 {len(original_data)} 个视频待处理")
    
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 运行主处理函数
        print("\n🚀 开始处理...")
        main()
        
        # 记录结束时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 检查处理结果
        success = check_trajectory_updates()
        
        print("\n" + "=" * 60)
        print(f"⏱️  总处理时间: {processing_time:.2f} 秒")
        print("✅ 优化后的功能特点:")
        print("  1. 逐个处理视频，不需要等待全部完成")
        print("  2. 直接更新到 video_description.json 文件")
        print("  3. 实时显示处理进度")
        print("  4. 出错时不影响其他视频的处理")
        print("=" * 60)
        
        return success
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_batch_trajectory_processing()
    if success:
        print("\n🎉 优化后的批量轨迹处理测试成功完成！")
        print("💡 提示：原始文件已备份，如需恢复可使用 video_description_backup.json")
    else:
        print("\n❌ 优化后的批量轨迹处理测试失败！") 