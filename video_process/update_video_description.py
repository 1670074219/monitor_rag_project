#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path

def scan_and_update_video_description():
    """
    扫描saved_video目录下的视频文件，将6月23号和6月24号的视频
    按照指定格式添加到video_description.json文件中
    """
    
    # 定义路径
    video_dir = Path("./saved_video")
    json_file = Path("./video_description.json")
    
    # 创建绝对路径前缀
    base_path = "/root/data1/monitor_rag_project/video_process/saved_video"
    
    # 读取现有的JSON数据
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
    else:
        video_data = {}
    
    # 定义要处理的日期（6月23号和24号）
    target_dates = ['20250623', '20250624']
    
    # 扫描视频文件
    if video_dir.exists():
        for video_file in video_dir.glob("*.mp4"):
            filename = video_file.name
            
            # 解析文件名 (格式: camera{id}_{date}_{time}.mp4)
            pattern = r'camera(\d+)_(\d{8})_(\d{6})\.mp4'
            match = re.match(pattern, filename)
            
            if match:
                camera_id, date, time = match.groups()
                
                # 只处理6月23号和24号的视频
                if date in target_dates:
                    # 生成key (去掉.mp4后缀)
                    key = filename[:-4]  # 去掉.mp4
                    
                    # 如果这个视频还没有在JSON中，则添加
                    if key not in video_data:
                        video_data[key] = {
                            "video_path": f"{base_path}/{filename}",
                            "analyse_result": None,
                            "is_embedding": False,
                            "idx": None
                        }
                        print(f"添加新视频: {key}")
                    else:
                        print(f"视频已存在: {key}")
    
    # 按key排序（保持JSON文件的整洁）
    sorted_video_data = dict(sorted(video_data.items()))
    
    # 保存更新后的JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_video_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n视频描述文件已更新: {json_file}")
    print(f"总共处理了 {len(sorted_video_data)} 个视频")
    
    # 统计各日期的视频数量
    date_counts = {}
    for key in sorted_video_data.keys():
        match = re.search(r'(\d{8})', key)
        if match:
            date = match.group(1)
            if date in target_dates:
                date_counts[date] = date_counts.get(date, 0) + 1
    
    print("\n各日期视频统计:")
    for date, count in date_counts.items():
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        print(f"  {formatted_date}: {count} 个视频")

def main():
    """主函数"""
    print("开始扫描和更新视频描述文件...")
    try:
        scan_and_update_video_description()
        print("✅ 处理完成！")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")

if __name__ == "__main__":
    main() 