#!/usr/bin/env python3
"""
从视频文件中提取单帧图像的工具
用于为行人跟踪系统准备测试图像
"""

import cv2
import os
import sys

def extract_frame_from_video(video_path, output_path=None, frame_number=None, timestamp_seconds=None):
    """
    从视频中提取单帧图像
    
    :param video_path: 视频文件路径
    :param output_path: 输出图像路径（可选，默认为frame.jpg）
    :param frame_number: 指定提取第几帧（从0开始）
    :param timestamp_seconds: 指定提取第几秒的帧
    :return: 是否成功提取
    """
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return False
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 视频信息:")
    print(f"   文件: {video_path}")
    print(f"   分辨率: {width}x{height}")
    print(f"   总帧数: {total_frames}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   时长: {duration:.2f} 秒")
    
    # 确定要提取的帧
    if timestamp_seconds is not None:
        # 按时间戳提取
        target_frame = int(timestamp_seconds * fps)
        print(f"\n🎯 按时间提取: {timestamp_seconds}秒 (第{target_frame}帧)")
    elif frame_number is not None:
        # 按帧号提取
        target_frame = frame_number
        target_time = frame_number / fps if fps > 0 else 0
        print(f"\n🎯 按帧号提取: 第{frame_number}帧 ({target_time:.2f}秒)")
    else:
        # 默认提取中间帧
        target_frame = total_frames // 2
        target_time = target_frame / fps if fps > 0 else 0
        print(f"\n🎯 提取中间帧: 第{target_frame}帧 ({target_time:.2f}秒)")
    
    # 检查帧号是否有效
    if target_frame >= total_frames:
        print(f"❌ 帧号超出范围: {target_frame} >= {total_frames}")
        cap.release()
        return False
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # 读取帧
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ 无法读取第{target_frame}帧")
        cap.release()
        return False
    
    # 确定输出路径
    if output_path is None:
        # 从视频文件名生成图像文件名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{video_name}_frame_{target_frame}.jpg"
    
    # 保存图像
    success = cv2.imwrite(output_path, frame)
    
    cap.release()
    
    if success:
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"✅ 帧提取成功!")
        print(f"   输出文件: {output_path}")
        print(f"   文件大小: {file_size:.1f} KB")
        print(f"   图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
        return True
    else:
        print(f"❌ 保存图像失败: {output_path}")
        return False

def extract_multiple_frames(video_path, output_dir="extracted_frames", num_frames=5):
    """
    从视频中提取多个帧（均匀分布）
    
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param num_frames: 提取的帧数
    """
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 从视频中提取 {num_frames} 个帧")
    print(f"   总帧数: {total_frames}")
    
    # 计算要提取的帧号（均匀分布）
    if num_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_count = 0
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            timestamp = frame_idx / fps if fps > 0 else 0
            output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:06d}_{timestamp:.2f}s.jpg")
            
            if cv2.imwrite(output_path, frame):
                print(f"   ✅ 第{i+1}帧: {output_path}")
                extracted_count += 1
            else:
                print(f"   ❌ 第{i+1}帧保存失败")
        else:
            print(f"   ❌ 第{i+1}帧读取失败")
    
    cap.release()
    print(f"\n🎉 完成! 成功提取 {extracted_count}/{num_frames} 帧到 {output_dir}/")
    return extracted_count > 0

def interactive_frame_extraction(video_path):
    """
    交互式帧提取 - 让用户选择提取方式
    """
    
    print("🎬 交互式帧提取工具")
    print("=" * 50)
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 获取视频基本信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"📹 视频: {os.path.basename(video_path)}")
    print(f"   总帧数: {total_frames}, 时长: {duration:.2f}秒, 帧率: {fps:.2f}")
    
    print("\n请选择提取方式:")
    print("1. 提取中间帧 (推荐)")
    print("2. 按时间提取 (输入秒数)")
    print("3. 按帧号提取")
    print("4. 提取多个帧 (均匀分布)")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            # 中间帧
            extract_frame_from_video(video_path)
            
        elif choice == "2":
            # 按时间
            time_input = input(f"请输入时间 (0-{duration:.1f}秒): ").strip()
            try:
                timestamp = float(time_input)
                if 0 <= timestamp <= duration:
                    extract_frame_from_video(video_path, timestamp_seconds=timestamp)
                else:
                    print(f"❌ 时间超出范围: {timestamp}")
            except ValueError:
                print("❌ 时间格式错误")
                
        elif choice == "3":
            # 按帧号
            frame_input = input(f"请输入帧号 (0-{total_frames-1}): ").strip()
            try:
                frame_num = int(frame_input)
                if 0 <= frame_num < total_frames:
                    extract_frame_from_video(video_path, frame_number=frame_num)
                else:
                    print(f"❌ 帧号超出范围: {frame_num}")
            except ValueError:
                print("❌ 帧号格式错误")
                
        elif choice == "4":
            # 多个帧
            num_input = input("请输入要提取的帧数 (默认5): ").strip()
            try:
                num_frames = int(num_input) if num_input else 5
                extract_multiple_frames(video_path, num_frames=num_frames)
            except ValueError:
                print("❌ 帧数格式错误")
                
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")

def main():
    """主函数"""
    print("🎬 视频帧提取工具")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        # 支持一些快速选项
        if len(sys.argv) > 2:
            option = sys.argv[2]
            if option == "--middle":
                extract_frame_from_video(video_path)
            elif option.startswith("--time="):
                try:
                    timestamp = float(option.split("=")[1])
                    extract_frame_from_video(video_path, timestamp_seconds=timestamp)
                except:
                    print("❌ 时间参数格式错误")
            elif option.startswith("--frame="):
                try:
                    frame_num = int(option.split("=")[1])
                    extract_frame_from_video(video_path, frame_number=frame_num)
                except:
                    print("❌ 帧号参数格式错误")
            elif option.startswith("--multi="):
                try:
                    num_frames = int(option.split("=")[1])
                    extract_multiple_frames(video_path, num_frames=num_frames)
                except:
                    print("❌ 帧数参数格式错误")
            else:
                print(f"❌ 未知选项: {option}")
        else:
            # 交互式模式
            interactive_frame_extraction(video_path)
    else:
        # 没有参数，查找视频文件
        video_files = []
        
        # 在当前目录和saved_video目录查找视频文件
        search_dirs = [".", "saved_video"]
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(search_dir, file))
        
        if not video_files:
            print("❌ 未找到视频文件")
            print("\n使用方法:")
            print("  python extract_frame.py <视频文件>")
            print("  python extract_frame.py <视频文件> --middle")
            print("  python extract_frame.py <视频文件> --time=30.5")
            print("  python extract_frame.py <视频文件> --frame=100")
            print("  python extract_frame.py <视频文件> --multi=5")
            return
        
        print("📁 找到以下视频文件:")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {video_file}")
        
        if len(video_files) == 1:
            # 只有一个视频文件，直接使用
            interactive_frame_extraction(video_files[0])
        else:
            # 多个视频文件，让用户选择
            try:
                choice = input(f"\n请选择视频文件 (1-{len(video_files)}): ").strip()
                file_index = int(choice) - 1
                
                if 0 <= file_index < len(video_files):
                    interactive_frame_extraction(video_files[file_index])
                else:
                    print("❌ 选择超出范围")
                    
            except (ValueError, KeyboardInterrupt):
                print("\n👋 操作取消")

if __name__ == "__main__":
    main() 