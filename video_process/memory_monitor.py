#!/usr/bin/env python3
import psutil
import json
import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_json_file_size(json_path: str):
    """分析JSON文件大小和内容"""
    if not os.path.exists(json_path):
        logger.error(f"JSON文件不存在: {json_path}")
        return
    
    file_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
    logger.info(f"JSON文件大小: {file_size:.2f} MB")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_videos = len(data)
        processed_videos = sum(1 for v in data.values() if v.get('analyse_result') is not None)
        unprocessed_videos = total_videos - processed_videos
        
        logger.info(f"总视频数: {total_videos}")
        logger.info(f"已处理: {processed_videos}")
        logger.info(f"未处理: {unprocessed_videos}")
        logger.info(f"预估内存占用 (JSON in memory): {file_size:.2f} MB")
        logger.info(f"如果每次读取都加载: {file_size * 3:.2f} MB (analyze + save + main)")
        
    except Exception as e:
        logger.error(f"分析JSON文件失败: {e}")

def monitor_memory_usage(duration_seconds: int = 60):
    """监控内存使用情况"""
    logger.info(f"开始监控内存使用，持续 {duration_seconds} 秒...")
    
    start_time = time.time()
    max_memory = 0
    
    while time.time() - start_time < duration_seconds:
        # 获取当前进程内存使用
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        if memory_mb > max_memory:
            max_memory = memory_mb
        
        logger.info(f"当前内存使用: {memory_mb:.2f} MB")
        time.sleep(5)
    
    logger.info(f"监控完成，最大内存使用: {max_memory:.2f} MB")

def suggest_optimizations(json_path: str):
    """根据JSON文件大小建议优化方案"""
    if not os.path.exists(json_path):
        return
    
    file_size_mb = os.path.getsize(json_path) / (1024 * 1024)
    
    print("\n=== 内存优化建议 ===")
    
    if file_size_mb < 10:
        print("✅ JSON文件较小 (<10MB)，当前方案应该可以正常工作")
    elif file_size_mb < 50:
        print("⚠️  JSON文件中等大小 (10-50MB)，建议优化:")
        print("   1. 减少每次处理的视频帧数")
        print("   2. 降低图片质量和分辨率")
        print("   3. 考虑分批处理")
    else:
        print("🚨 JSON文件很大 (>50MB)，强烈建议:")
        print("   1. 使用数据库替代JSON文件")
        print("   2. 实现流式JSON读取 (ijson)")
        print("   3. 分片存储已处理和未处理的视频")
        print("   4. 定期清理已处理的记录")
    
    print(f"\n当前优化方案可以减少约 {file_size_mb * 2:.1f}MB 内存使用")

if __name__ == "__main__":
    json_path = "video_description.json"
    
    print("=== 视频分析服务内存使用分析 ===\n")
    
    # 分析JSON文件
    analyze_json_file_size(json_path)
    
    # 给出优化建议
    suggest_optimizations(json_path)
    
    # 可选：监控内存使用
    # monitor_memory_usage(30) 