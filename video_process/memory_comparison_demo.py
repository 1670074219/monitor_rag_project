#!/usr/bin/env python3
import json
import tracemalloc
import gc
import os

def demo_original_approach(json_path):
    """演示原始方法的内存使用"""
    print("=== 原始方法 (重复加载JSON) ===")
    
    tracemalloc.start()
    
    # 模拟原始代码中的操作
    for i in range(3):  # 模拟处理3个视频
        # 1. get_video_info() - 加载整个JSON
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)  # 🚨 加载全部21632个记录！
        
        # 只取一个视频信息
        video_keys = list(video_data.keys())
        if video_keys:
            v_k = video_keys[i % len(video_keys)]
            video_info = video_data.get(v_k)
            print(f"获取视频 {v_k} 信息")
        
        # 2. save_to_json() - 再次加载整个JSON
        with open(json_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)  # 🚨 又加载全部21632个记录！
        
        print(f"第{i+1}次操作完成")
        
        # 检查内存使用
        current, peak = tracemalloc.get_traced_memory()
        print(f"当前内存: {current / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    print(f"原始方法总内存峰值: {peak / 1024 / 1024:.2f} MB\n")
    return peak

def demo_optimized_approach(json_path):
    """演示优化方法的内存使用"""
    print("=== 优化方法 (只加载需要的数据) ===")
    
    tracemalloc.start()
    
    # 1. 只加载一次，获取需要的视频列表
    with open(json_path, "r", encoding="utf-8") as f:
        all_video_data = json.load(f)
    
    # 只保存需要处理的视频信息
    needed_videos = {}
    count = 0
    for v_k, v_info in all_video_data.items():
        if count < 3:  # 只取3个
            needed_videos[v_k] = v_info.copy()
            count += 1
        else:
            break
    
    # 立即清理大JSON
    del all_video_data
    gc.collect()
    
    print(f"只保存了 {len(needed_videos)} 个视频信息到内存")
    
    # 2. 处理视频（从内存中获取，不再读取JSON）
    for i, (v_k, video_info) in enumerate(needed_videos.items()):
        print(f"处理视频 {v_k}")
        
        # 模拟保存结果（优化版本中的高效保存）
        # 这里仍需要读写JSON，但次数大大减少
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"第{i+1}次操作内存: {current / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    print(f"优化方法总内存峰值: {peak / 1024 / 1024:.2f} MB\n")
    return peak

def analyze_json_loading():
    """分析JSON加载机制"""
    print("=== JSON加载机制详解 ===")
    print("🔍 关键问题：json.load() 的工作原理")
    print()
    print("原始代码中的问题：")
    print("```python")
    print("# 每次都这样做：")
    print("with open('video_description.json', 'r') as f:")
    print("    video_data = json.load(f)  # ⚠️ 加载全部21632个记录到内存")
    print("    needed_info = video_data[video_key]  # 只用其中1个！")
    print("```")
    print()
    print("💥 结果：")
    print("- 每个视频处理需要加载JSON 2次")
    print("- 每次加载 = 7.73MB × 21632个记录")
    print("- 5个视频 = 2次×5个×7.73MB = 77.3MB+")
    print("- 加上视频处理内存 = 内存爆炸！")
    print()
    print("✅ 优化方案：")
    print("```python") 
    print("# 1. 启动时只加载需要的信息")
    print("needed_videos = extract_needed_videos(json_path)")
    print("del all_video_data  # 立即清理大JSON")
    print()
    print("# 2. 处理时从内存获取")
    print("video_info = self.processing_videos[video_key]  # 不读取文件！")
    print("```")

if __name__ == "__main__":
    json_path = "video_description.json"
    
    analyze_json_loading()
    
    print("=== 内存使用对比测试 ===")
    if not os.path.exists(json_path):
        print(f"❌ JSON文件不存在: {json_path}")
        print("无法进行实际测试，但原理已经解释清楚")
    else:
        print("✅ 找到JSON文件，开始对比测试...")
        
        # 测试原始方法
        original_peak = demo_original_approach(json_path)
        
        # 清理内存
        gc.collect()
        
        # 测试优化方法  
        optimized_peak = demo_optimized_approach(json_path)
        
        # 对比结果
        improvement = (original_peak - optimized_peak) / original_peak * 100
        print(f"🎯 内存使用改善: {improvement:.1f}%")
        print(f"📉 内存减少: {(original_peak - optimized_peak) / 1024 / 1024:.2f} MB") 