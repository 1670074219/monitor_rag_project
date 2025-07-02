#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
from datetime import datetime

def backup_file(file_path):
    """
    备份原始文件
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup_{timestamp}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"✓ 已备份原始文件到: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ 备份文件失败: {e}")
        return None

def should_remove_video(video_info):
    """
    判断是否应该删除该视频记录
    删除条件：
    1. 分析结果显示人数为0
    2. 没有轨迹数据
    3. 轨迹数据为空（trajectories为空数组或person_count为0）
    """
    analyse_result = video_info.get('analyse_result', '')
    trajectory_data = video_info.get('trajectory_data', None)
    
    # 条件1：分析结果显示人数为0
    if '人数：0' in analyse_result:
        return True, "分析结果人数为0"
    
    # 条件2：没有轨迹数据
    if trajectory_data is None:
        return True, "无轨迹数据"
    
    # 条件3：轨迹数据为空
    if isinstance(trajectory_data, dict):
        # 检查person_count
        if trajectory_data.get('person_count', 0) == 0:
            return True, "轨迹人数为0"
        
        # 检查trajectories数组
        trajectories = trajectory_data.get('trajectories', [])
        if not trajectories or len(trajectories) == 0:
            return True, "轨迹数组为空"
    
    # 如果轨迹数据是列表格式（简化版本）
    elif isinstance(trajectory_data, list):
        if len(trajectory_data) == 0:
            return True, "轨迹列表为空"
    
    return False, ""

def remove_invalid_videos(file_path='video_process/video_description.json'):
    """
    去除无效的视频记录（人数为0或无轨迹数据）
    """
    print("=" * 60)
    print("开始去除无效视频记录")
    print("📝 删除条件：")
    print("  1. 分析结果人数为0")
    print("  2. 没有轨迹数据")
    print("  3. 轨迹数据为空")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 读取原始数据
        print("📖 正在读取原始数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        original_count = len(original_data)
        print(f"📊 原始数据包含 {original_count} 个视频记录")
        
        # 备份原始文件
        backup_path = backup_file(file_path)
        if not backup_path:
            print("❌ 无法备份文件，操作终止")
            return False
        
        # 过滤数据
        print("\n🔍 正在过滤无效记录...")
        filtered_data = {}
        removed_videos = {
            "分析结果人数为0": [],
            "无轨迹数据": [],
            "轨迹人数为0": [],
            "轨迹数组为空": [],
            "轨迹列表为空": []
        }
        
        for video_name, video_info in original_data.items():
            should_remove, reason = should_remove_video(video_info)
            
            if should_remove:
                removed_videos[reason].append(video_name)
                print(f"❌ 删除: {video_name} ({reason})")
            else:
                filtered_data[video_name] = video_info
        
        # 统计结果
        filtered_count = len(filtered_data)
        total_removed = sum(len(videos) for videos in removed_videos.values())
        
        print(f"\n📊 过滤结果:")
        print(f"  - 原始记录数: {original_count}")
        print(f"  - 删除记录数: {total_removed}")
        print(f"  - 保留记录数: {filtered_count}")
        print(f"  - 删除比例: {total_removed/original_count*100:.2f}%")
        
        print(f"\n📋 删除详情:")
        for reason, videos in removed_videos.items():
            if videos:
                print(f"  - {reason}: {len(videos)} 个")
        
        # 保存过滤后的数据
        print(f"\n💾 正在保存过滤后的数据到: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
        print("✅ 操作完成！")
        
        # 显示被删除的视频列表（如果不太多的话）
        if total_removed <= 30:
            print(f"\n📝 被删除的视频列表:")
            for reason, videos in removed_videos.items():
                if videos:
                    print(f"\n  【{reason}】:")
                    for video in videos:
                        print(f"    - {video}")
        elif total_removed > 30:
            print(f"\n📝 被删除的视频列表 (前20个):")
            count = 0
            for reason, videos in removed_videos.items():
                if videos and count < 20:
                    print(f"\n  【{reason}】:")
                    for video in videos[:min(10, 20-count)]:
                        print(f"    - {video}")
                        count += 1
                        if count >= 20:
                            break
            if total_removed > 20:
                print(f"  ... 还有 {total_removed - 20} 个视频被删除")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    start_time = time.time()
    
    success = remove_invalid_videos()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 无效视频记录删除成功！")
        print(f"⏱️  处理时间: {processing_time:.2f} 秒")
        print("💡 提示：原始文件已备份，如需恢复可使用备份文件")
    else:
        print("❌ 删除操作失败！")
    print("=" * 60)

if __name__ == '__main__':
    main() 