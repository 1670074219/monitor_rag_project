#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查和统计长视频的工具脚本 (JSON版本)
"""

import json
import logging
import cv2
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1
        
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps > 0:
            return frame_count / fps
        return -1
    except:
        return -1

def scan_long_videos_json(json_path: str, min_duration_minutes: float = 10.0) -> List[Tuple[str, str, float]]:
    """扫描JSON文件中的长视频"""
    long_videos = []
    min_duration_seconds = min_duration_minutes * 60
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        
        # 获取所有未分析的视频
        unanalyzed_videos = []
        for v_k, v_info in video_data.items():
            analyse_result = v_info.get('analyse_result')
            if analyse_result is None or analyse_result == '':
                unanalyzed_videos.append((v_k, v_info.get('video_path')))
        
        logger.info(f"正在检查 {len(unanalyzed_videos)} 个视频的时长...")
        
        for i, (v_name, video_path) in enumerate(unanalyzed_videos, 1):
            if i % 50 == 0:
                logger.info(f"已检查 {i}/{len(unanalyzed_videos)} 个视频...")
                
            if not video_path or not os.path.exists(video_path):
                continue
                
            duration = get_video_duration(video_path)
            if duration > min_duration_seconds:
                long_videos.append((v_name, video_path, duration))
        
    except Exception as e:
        logger.error(f"扫描长视频失败: {e}")
    
    return long_videos

def get_json_summary(json_path: str):
    """获取JSON文件摘要信息"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        
        total_count = len(video_data)
        analyzed_count = 0
        skipped_count = 0
        unanalyzed_count = 0
        
        for v_k, v_info in video_data.items():
            analyse_result = v_info.get('analyse_result')
            
            if analyse_result is None or analyse_result == '':
                unanalyzed_count += 1
            elif analyse_result.startswith('跳过分析:'):
                skipped_count += 1
            else:
                analyzed_count += 1
        
        return total_count, analyzed_count, skipped_count, unanalyzed_count
        
    except Exception as e:
        logger.error(f"获取JSON摘要失败: {e}")
        return 0, 0, 0, 0

def mark_long_videos_as_skipped_json(json_path: str, long_videos: List[Tuple[str, str, float]]):
    """将长视频标记为跳过"""
    if not long_videos:
        logger.info("没有找到长视频需要标记")
        return
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)
        
        for v_name, video_path, duration in long_videos:
            if v_name in video_data:
                skip_reason = f"跳过分析: 视频时长超过10分钟 (实际时长: {duration/60:.2f}分钟)"
                video_data[v_name]['analyse_result'] = skip_reason
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(video_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✅ 成功标记 {len(long_videos)} 个长视频为跳过")
        
    except Exception as e:
        logger.error(f"标记长视频失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🎬 长视频检查工具 (JSON版本)")
    print("=" * 60)
    
    json_path = os.path.join(base_dir, "video_description.json")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON文件不存在: {json_path}")
        return
    
    # 显示JSON文件摘要
    total, analyzed, skipped, unanalyzed = get_json_summary(json_path)
    print(f"\n📊 当前JSON文件状态:")
    print(f"  总记录数: {total}")
    print(f"  ✅ 已分析: {analyzed}")
    print(f"  ⏸️ 已跳过: {skipped}")
    print(f"  ⏳ 待分析: {unanalyzed}")
    
    if total > 0:
        progress = (analyzed / total) * 100
        print(f"  📈 分析进度: {progress:.2f}%")
    
    # 扫描长视频
    print(f"\n🔍 开始扫描超过10分钟的视频...")
    long_videos = scan_long_videos_json(json_path, 10.0)
    
    if long_videos:
        print(f"\n⚠️ 找到 {len(long_videos)} 个超过10分钟的视频:")
        for v_name, video_path, duration in long_videos:
            print(f"  📹 {v_name}: {duration/60:.2f}分钟")
        
        # 询问是否标记为跳过
        confirm = input(f"\n是否将这些长视频标记为跳过? (y/N): ").strip().lower()
        if confirm == 'y':
            mark_long_videos_as_skipped_json(json_path, long_videos)
        else:
            print("操作已取消")
    else:
        print("✅ 没有找到超过10分钟的未分析视频")

if __name__ == "__main__":
    main() 