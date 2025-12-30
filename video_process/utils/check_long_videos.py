#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查和统计长视频的工具脚本
"""

import mysql.connector
from mysql.connector import Error
import logging
import cv2
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.151',
    'port': 3306,
    'database': 'monitor_rag',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4'
}

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

def scan_long_videos(min_duration_minutes: float = 10.0) -> List[Tuple[str, str, float]]:
    """扫描数据库中的长视频"""
    long_videos = []
    min_duration_seconds = min_duration_minutes * 60
    
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # 获取所有未分析的视频
        cursor.execute("""
            SELECT v_name, video_path 
            FROM v_dsp 
            WHERE analyse_result IS NULL OR analyse_result = ''
        """)
        
        videos = cursor.fetchall()
        logger.info(f"正在检查 {len(videos)} 个视频的时长...")
        
        for i, (v_name, video_path) in enumerate(videos, 1):
            if i % 50 == 0:
                logger.info(f"已检查 {i}/{len(videos)} 个视频...")
                
            if not os.path.exists(video_path):
                continue
                
            duration = get_video_duration(video_path)
            if duration > min_duration_seconds:
                long_videos.append((v_name, video_path, duration))
        
        cursor.close()
        connection.close()
        
    except Error as e:
        logger.error(f"数据库操作失败: {e}")
    
    return long_videos

def get_database_summary():
    """获取数据库摘要信息"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        # 总记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp")
        total_count = cursor.fetchone()[0]
        
        # 已分析的记录数（不包括跳过的）
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NOT NULL AND analyse_result != '' AND analyse_result NOT LIKE '跳过分析:%'")
        analyzed_count = cursor.fetchone()[0]
        
        # 跳过的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result LIKE '跳过分析:%'")
        skipped_count = cursor.fetchone()[0]
        
        # 未分析的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NULL OR analyse_result = ''")
        unanalyzed_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return total_count, analyzed_count, skipped_count, unanalyzed_count
        
    except Error as e:
        logger.error(f"数据库操作失败: {e}")
        return 0, 0, 0, 0

def main():
    """主函数"""
    print("=" * 60)
    print("🎬 长视频检查工具")
    print("=" * 60)
    
    # 显示数据库摘要
    total, analyzed, skipped, unanalyzed = get_database_summary()
    print(f"\n📊 当前数据库状态:")
    print(f"  总记录数: {total}")
    print(f"  ✅ 已分析: {analyzed}")
    print(f"  ⏸️ 已跳过: {skipped}")
    print(f"  ⏳ 待分析: {unanalyzed}")
    
    if total > 0:
        progress = (analyzed / total) * 100
        print(f"  📈 分析进度: {progress:.2f}%")
    
    # 扫描长视频
    print(f"\n🔍 开始扫描超过10分钟的视频...")
    long_videos = scan_long_videos(10.0)
    
    if long_videos:
        print(f"\n⚠️ 找到 {len(long_videos)} 个超过10分钟的视频:")
        for v_name, video_path, duration in long_videos:
            print(f"  📹 {v_name}: {duration/60:.2f}分钟")
        
        # 询问是否标记为跳过
        confirm = input(f"\n是否将这些长视频标记为跳过? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                connection = mysql.connector.connect(**DB_CONFIG)
                cursor = connection.cursor()
                
                for v_name, video_path, duration in long_videos:
                    skip_reason = f"跳过分析: 视频时长超过10分钟 (实际时长: {duration/60:.2f}分钟)"
                    cursor.execute("""
                        UPDATE v_dsp 
                        SET analyse_result = %s 
                        WHERE v_name = %s
                    """, (skip_reason, v_name))
                
                connection.commit()
                cursor.close()
                connection.close()
                
                print(f"✅ 成功标记 {len(long_videos)} 个长视频为跳过")
                
            except Error as e:
                logger.error(f"标记长视频失败: {e}")
        else:
            print("操作已取消")
    else:
        print("✅ 没有找到超过10分钟的未分析视频")

if __name__ == "__main__":
    main() 