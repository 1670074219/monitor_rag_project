# 批量视频轨迹处理系统

## 概述

这个系统能够批量处理多个摄像头的视频文件，自动识别摄像头ID，应用对应的坐标转换参数，并生成包含人员轨迹数据的结果文件。

## 主要功能

- 🎥 **多摄像头支持**: 支持处理来自不同摄像头的视频
- 🔄 **自动坐标转换**: 根据摄像头配置自动应用像素到现实世界的坐标转换
- 📊 **批量处理**: 一次性处理多个视频文件
- 🎯 **人员追踪**: 使用YOLO+DeepSORT进行准确的人员检测和追踪
- 💾 **结果保存**: 将轨迹数据保存为结构化的JSON文件

## 文件结构

```
video_process/
├── person_trajectory_optimized.py     # 优化后的主处理脚本
├── test_trajectory_batch.py           # 测试脚本
├── camera_config.json                 # 摄像头配置文件
├── video_description.json             # 输入视频描述文件
├── video_description_with_trajectory.json  # 输出轨迹数据文件
├── yolo/
│   └── yolo11s.pt                     # YOLO模型文件
└── deepsort/
    └── deep/checkpoint/ckpt.t7        # DeepSORT模型文件
```

## 配置文件格式

### camera_config.json
包含每个摄像头的坐标转换参数：

```json
{
  "camera_config": [
    {
      "camera_id": "camera1",
      "camera_url": "rtsp://...",
      "pixel_coordinate": [
        [517, 95], [667, 95], [1182, 720], [276, 720]
      ],
      "real_coordinate": [
        [2731, 1028], [2582, 1028], [2582, 341], [2731, 341]
      ]
    }
  ]
}
```

### video_description.json
包含需要处理的视频信息：

```json
{
  "camera1_20250623_095558": {
    "video_path": "/path/to/video.mp4",
    "analyse_result": "视频分析结果...",
    "is_embedding": false,
    "idx": null
  }
}
```

## 输出格式

### video_description_with_trajectory.json
包含原始视频信息和轨迹数据：

```json
{
  "camera1_20250623_095558": {
    "video_path": "/path/to/video.mp4",
    "analyse_result": "视频分析结果...",
    "is_embedding": false,
    "idx": null,
    "trajectory_data": {
      "person_count": 1,
      "trajectories": [
        {
          "track_id": 2,
          "trajectory_length": 51,
          "coordinates": [
            [460.79, 987.86],
            [461.48, 986.48]
          ]
        }
      ],
      "coordinate_system": "real_world",
      "unit": "centimeters"
    }
  }
}
```

## 使用方法

### 1. 环境准备

确保已安装必要的依赖：

```bash
pip install opencv-python
pip install ultralytics
pip install torch torchvision
pip install numpy
```

### 2. 文件准备

确保以下文件存在：
- `camera_config.json` - 摄像头配置
- `video_description.json` - 输入视频描述
- `yolo/yolo11s.pt` - YOLO模型文件
- `deepsort/deep/checkpoint/ckpt.t7` - DeepSORT模型文件

### 3. 运行处理

```bash
# 方法1: 直接运行主脚本
cd video_process
python person_trajectory_optimized.py

# 方法2: 运行测试脚本（推荐）
python test_trajectory_batch.py
```

### 4. 查看结果

处理完成后，查看生成的 `video_description_with_trajectory.json` 文件。

## 核心功能说明

### 摄像头识别
系统通过视频文件名自动识别摄像头ID：
- `camera1_20250623_095558` → `camera1`
- `camera2_20250623_094828` → `camera2`

### 坐标转换
每个摄像头使用独立的单应性矩阵进行坐标转换：
1. 从 `camera_config.json` 读取像素坐标和现实世界坐标
2. 计算单应性矩阵
3. 将人员脚部中心点从像素坐标转换为现实世界坐标

### 人员追踪
1. **检测**: 使用YOLOv11检测视频中的人员
2. **追踪**: 使用DeepSORT为每个人分配唯一ID并追踪轨迹
3. **坐标转换**: 将像素坐标转换为现实世界坐标
4. **轨迹记录**: 记录每个人的完整移动轨迹

## 性能优化

- **帧跳跃**: 每5帧处理一次，平衡精度和性能
- **GPU支持**: 自动检测并使用GPU加速
- **内存管理**: 逐个处理视频文件，避免内存溢出

## 错误处理

系统包含完善的错误处理机制：
- 文件不存在检查
- 模型加载失败处理
- 视频读取异常处理
- 坐标转换异常处理

## 扩展功能

### 添加新摄像头
1. 在 `camera_config.json` 中添加新的摄像头配置
2. 确保视频文件名包含正确的摄像头ID

### 调整追踪参数
修改 `process_single_video` 函数中的DeepSORT参数：

```python
deepsort = DeepSort(
    max_dist=0.2,           # 最大距离阈值
    min_confidence=0.7,     # 最小置信度
    nms_max_overlap=0.5,    # NMS最大重叠
    max_iou_distance=0.7,   # 最大IOU距离
    max_age=70,             # 最大年龄
    n_init=3,               # 初始化帧数
    nn_budget=100           # 特征预算
)
```

## 故障排除

### 常见问题

1. **模型文件缺失**
   - 确保 `yolo11s.pt` 和 `ckpt.t7` 文件存在
   - 检查文件路径是否正确

2. **视频文件无法打开**
   - 检查视频文件路径
   - 确认视频文件格式支持

3. **坐标转换异常**
   - 检查摄像头配置中的坐标点数量（必须是4个）
   - 确认坐标值格式正确

4. **内存不足**
   - 增加帧跳跃间隔（修改 `frame_count % 5`）
   - 使用更小的模型

### 日志分析

系统会输出详细的处理日志：
- 模型加载状态
- 视频处理进度
- 轨迹检测结果
- 错误信息

## 更新日志

### v1.0.0
- 初始版本发布
- 支持多摄像头批量处理
- 实现自动坐标转换
- 添加完整的错误处理机制 