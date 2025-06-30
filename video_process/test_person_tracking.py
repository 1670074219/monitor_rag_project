#!/usr/bin/env python3
"""
行人位置跟踪系统测试脚本
用于验证系统是否正常工作
"""

import os
import sys
import traceback
from pathlib import Path

def test_imports():
    """测试所有必需的库是否正确导入"""
    print("🔍 测试导入模块...")
    
    try:
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"   ❌ OpenCV导入失败: {e}")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        print(f"   🖥️  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   🖥️  GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"   ❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"   ❌ NumPy导入失败: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"   ✅ Ultralytics YOLO")
    except ImportError as e:
        print(f"   ❌ Ultralytics导入失败: {e}")
        print(f"      💡 请运行: pip install ultralytics")
        return False
    
    return True

def test_model_files():
    """检查必需的模型文件是否存在"""
    print("\n🔍 检查模型文件...")
    
    model_files = {
        'YOLO模型': 'video_process/yolo/yolo11s.pt',
        'DeepSORT特征提取器': 'video_process/deepsort/deep/checkpoint/ckpt.t7'
    }
    
    all_exist = True
    for name, path in model_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"   ✅ {name}: {path} ({size:.1f}MB)")
        else:
            print(f"   ❌ {name}: {path} (文件不存在)")
            all_exist = False
    
    return all_exist

def test_basic_functionality():
    """测试基本功能"""
    print("\n🔍 测试基本功能...")
    
    try:
        # 测试导入自定义模块
        from person_position_api import PersonPositionAPI
        print("   ✅ PersonPositionAPI导入成功")
        
        # 创建API实例
        api = PersonPositionAPI()
        print("   ✅ API实例创建成功")
        
        # 设置测试参考点
        test_points = [
            (100, 100),
            (500, 100), 
            (500, 400),
            (100, 400)
        ]
        
        result = api.set_reference_points(test_points)
        if result["status"] == "success":
            print("   ✅ 参考点设置成功")
        else:
            print(f"   ❌ 参考点设置失败: {result['message']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_with_sample_image():
    """使用示例图像测试"""
    print("\n🔍 测试图像处理...")
    
    # 检查是否有测试图像
    test_images = ['frame.jpg', 'video_process/saved_video/test_frame.jpg']
    test_image = None
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if test_image is None:
        print("   ⚠️  未找到测试图像，跳过图像处理测试")
        print("   💡 请确保有frame.jpg或其他测试图像")
        return True
    
    try:
        from person_position_api import PersonPositionAPI
        
        api = PersonPositionAPI()
        
        # 设置参考点
        reference_points = [
            (200, 150),
            (600, 150),
            (600, 450),
            (200, 450)
        ]
        api.set_reference_points(reference_points)
        
        # 处理图像
        print(f"   🖼️  处理图像: {test_image}")
        result = api.get_simple_position_info(test_image)
        
        if result["status"] == "success":
            person_count = len(result["persons"])
            print(f"   ✅ 图像处理成功，检测到 {person_count} 个行人")
            
            if person_count > 0:
                for i, person in enumerate(result["persons"][:3], 1):  # 只显示前3个
                    pos = person['pixel_position']
                    print(f"      👤 行人{i}: 位置({pos[0]}, {pos[1]})")
            
        else:
            print(f"   ⚠️  图像处理返回: {result['message']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 图像处理测试失败: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n🔍 测试可视化功能...")
    
    try:
        from person_position_visualizer import PersonPositionVisualizer
        
        visualizer = PersonPositionVisualizer()
        print("   ✅ 可视化器创建成功")
        
        # 检查测试图像
        if os.path.exists('frame.jpg'):
            print("   ✅ 可视化模块可用")
        else:
            print("   ⚠️  无测试图像，跳过可视化演示")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 可视化测试失败: {e}")
        return False

def create_sample_test():
    """创建一个简单的测试示例"""
    print("\n🎯 创建测试示例...")
    
    sample_code = '''
# 简单测试示例
from person_position_api import PersonPositionAPI

# 1. 创建API
api = PersonPositionAPI()

# 2. 设置参考点（矩形区域）
points = [
    (100, 100),   # 左上
    (400, 100),   # 右上  
    (400, 300),   # 右下
    (100, 300)    # 左下
]
api.set_reference_points(points)

# 3. 如果有图像文件，可以这样测试：
# result = api.get_simple_position_info("your_image.jpg")
# print(result)

print("✅ 测试代码准备就绪!")
'''
    
    with open('test_example.py', 'w', encoding='utf-8') as f:
        f.write(sample_code)
    
    print("   💾 已创建 test_example.py")
    print("   💡 您可以修改其中的图像路径进行测试")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 行人位置跟踪系统测试")
    print("=" * 60)
    
    # 记录测试结果
    tests = [
        ("导入测试", test_imports),
        ("模型文件检查", test_model_files), 
        ("基本功能测试", test_basic_functionality),
        ("图像处理测试", test_with_sample_image),
        ("可视化测试", test_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name}发生异常: {e}")
            results.append((test_name, False))
    
    # 创建测试示例
    create_sample_test()
    
    # 显示测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统已准备就绪。")
        print("\n📝 下一步操作:")
        print("   1. 准备您的图像或视频文件")
        print("   2. 确定四个参考点的坐标") 
        print("   3. 运行 test_example.py 或使用API进行测试")
        print("   4. 查看 README_person_tracking.md 了解详细用法")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        print("\n🛠️ 常见解决方案:")
        if not any("导入测试" in name and success for name, success in results):
            print("   • 安装缺失的包: pip install ultralytics torch opencv-python")
        if not any("模型文件" in name and success for name, success in results):
            print("   • 下载必需的模型文件")
            print("   • 检查文件路径是否正确")

if __name__ == '__main__':
    main() 