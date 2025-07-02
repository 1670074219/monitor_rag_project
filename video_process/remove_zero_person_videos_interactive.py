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

def preview_invalid_videos(file_path='video_process/video_description.json'):
    """
    预览无效的视频记录（人数为0或无轨迹数据）
    """
    print("=" * 60)
    print("预览无效视频记录")
    print("📝 删除条件：")
    print("  1. 分析结果人数为0")
    print("  2. 没有轨迹数据")
    print("  3. 轨迹数据为空")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None, None
    
    try:
        # 读取原始数据
        print("📖 正在读取数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        original_count = len(original_data)
        print(f"📊 总共 {original_count} 个视频记录")
        
        # 查找无效记录
        print("\n🔍 正在查找无效记录...")
        invalid_videos = {
            "分析结果人数为0": [],
            "无轨迹数据": [],
            "轨迹人数为0": [],
            "轨迹数组为空": [],
            "轨迹列表为空": []
        }
        
        for video_name, video_info in original_data.items():
            should_remove, reason = should_remove_video(video_info)
            
            if should_remove:
                invalid_videos[reason].append(video_name)
        
        # 统计结果
        total_invalid = sum(len(videos) for videos in invalid_videos.values())
        keep_count = original_count - total_invalid
        
        print(f"\n📊 预览结果:")
        print(f"  - 总记录数: {original_count}")
        print(f"  - 无效记录数: {total_invalid}")
        print(f"  - 将保留的记录: {keep_count}")
        print(f"  - 删除比例: {total_invalid/original_count*100:.2f}%")
        
        print(f"\n📋 无效记录详情:")
        for reason, videos in invalid_videos.items():
            if videos:
                print(f"  - {reason}: {len(videos)} 个")
        
        # 显示无效视频列表
        if total_invalid > 0:
            print(f"\n📝 无效视频列表:")
            if total_invalid <= 50:
                for reason, videos in invalid_videos.items():
                    if videos:
                        print(f"\n  【{reason}】:")
                        for i, video in enumerate(videos, 1):
                            print(f"    {i:2d}. {video}")
            else:
                print(f"  前30个:")
                count = 0
                for reason, videos in invalid_videos.items():
                    if videos and count < 30:
                        print(f"\n  【{reason}】:")
                        for video in videos[:min(15, 30-count)]:
                            count += 1
                            print(f"    {count:2d}. {video}")
                            if count >= 30:
                                break
                if total_invalid > 30:
                    print(f"  ... 还有 {total_invalid - 30} 个视频")
        
        return original_data, invalid_videos
        
    except Exception as e:
        print(f"❌ 预览过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def remove_invalid_videos_confirmed(original_data, invalid_videos, file_path):
    """
    确认删除无效的视频记录
    """
    try:
        # 备份原始文件
        backup_path = backup_file(file_path)
        if not backup_path:
            print("❌ 无法备份文件，操作终止")
            return False
        
        # 过滤数据
        print("\n🗑️  正在删除无效记录...")
        filtered_data = {}
        
        # 创建要删除的视频名称集合
        videos_to_remove = set()
        for reason, videos in invalid_videos.items():
            videos_to_remove.update(videos)
        
        for video_name, video_info in original_data.items():
            if video_name not in videos_to_remove:
                filtered_data[video_name] = video_info
        
        # 保存过滤后的数据
        print(f"💾 正在保存过滤后的数据到: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
        
        print("✅ 删除操作完成！")
        return True
        
    except Exception as e:
        print(f"❌ 删除过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数 - 交互式版本
    """
    start_time = time.time()
    
    # 预览无效记录
    original_data, invalid_videos = preview_invalid_videos()
    
    if original_data is None or invalid_videos is None:
        print("❌ 预览失败！")
        return
    
    total_invalid = sum(len(videos) for videos in invalid_videos.values())
    
    if total_invalid == 0:
        print("\n🎉 没有找到无效的视频记录，无需删除！")
        return
    
    # 用户确认
    print("\n" + "=" * 60)
    print("⚠️  确认删除操作")
    print("=" * 60)
    print(f"即将删除 {total_invalid} 个无效视频记录")
    print("原始文件将被备份")
    
    print("\n删除类型详情:")
    for reason, videos in invalid_videos.items():
        if videos:
            print(f"  - {reason}: {len(videos)} 个")
    
    while True:
        choice = input("\n是否继续删除？(y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            break
        elif choice in ['n', 'no', '否']:
            print("❌ 用户取消操作")
            return
        else:
            print("❓ 请输入 y(是) 或 n(否)")
    
    # 执行删除
    success = remove_invalid_videos_confirmed(
        original_data, 
        invalid_videos, 
        'video_process/video_description.json'
    )
    
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