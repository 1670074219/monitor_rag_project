# 基于四个坐标点的行人像素坐标计算系统

## 📋 功能概述

这个系统可以根据您提供的四个参考坐标点，实时计算视频中行人相对于这些坐标点的像素坐标位置。

### 🎯 主要功能
- ✅ **四点参考系统**: 基于您指定的4个像素坐标建立参考坐标系
- ✅ **实时行人检测**: 使用YOLO进行高精度行人检测
- ✅ **目标跟踪**: 使用DeepSORT保持行人ID的连续性
- ✅ **位置计算**: 计算行人相对于参考点的详细位置信息
- ✅ **可视化显示**: 在图像/视频上直观显示所有信息
- ✅ **数据导出**: 支持JSON格式的结果导出

## 🚀 快速开始

### 1️⃣ 基础使用 (推荐)

```python
from person_position_api import PersonPositionAPI

# 1. 创建API实例
api = PersonPositionAPI()

# 2. 设置四个参考点 (像素坐标)
reference_points = [
    (200, 100),   # 点1: 左上角
    (800, 100),   # 点2: 右上角  
    (800, 600),   # 点3: 右下角
    (200, 600)    # 点4: 左下角
]
api.set_reference_points(reference_points)

# 3. 分析图像
result = api.get_simple_position_info("your_image.jpg")

# 4. 查看结果
if result["status"] == "success":
    for person in result["persons"]:
        print(f"行人ID: {person['person_id']}")
        print(f"像素位置: {person['pixel_position']}")
        print(f"最近参考点: P{person['closest_reference_point']['point_index']}")
        print(f"是否在参考区域内: {person['is_inside_area']}")
```

### 2️⃣ 可视化使用

```python
from person_position_visualizer import PersonPositionVisualizer

# 创建可视化器
visualizer = PersonPositionVisualizer()

# 可视化单帧图像
result = visualizer.visualize_frame(
    image_path="your_image.jpg",
    reference_points=reference_points,
    output_path="result_visualization.jpg",
    show_image=True
)
```

### 3️⃣ 视频处理

```python
from person_relative_position import PersonRelativePositionTracker

# 创建跟踪器
tracker = PersonRelativePositionTracker(reference_points)

# 处理整个视频
results = tracker.process_video(
    video_path="your_video.mp4",
    output_file="tracking_results.json",
    skip_frames=5,      # 每5帧处理一次
    max_frames=1000     # 最多处理1000帧
)
```

## 📊 输出数据格式

### 简化版本 (推荐日常使用)
```json
{
  "status": "success",
  "message": "检测到2个行人",
  "persons": [
    {
      "person_id": 1,
      "pixel_position": [450, 300],
      "closest_reference_point": {
        "point_index": 1,
        "distance": 125.5,
        "angle": 45.2
      },
      "relative_to_center": {
        "distance": 89.3,
        "angle": -12.7
      },
      "position_percentage": {
        "x_percent": 35.2,
        "y_percent": 67.8
      },
      "is_inside_area": true
    }
  ]
}
```

### 详细版本 (完整分析数据)
```json
{
  "person_pixel_position": [450, 300],
  "relative_to_each_point": [
    {
      "point_index": 1,
      "reference_point": [200, 100],
      "distance": 125.5,
      "angle_degrees": 45.2,
      "offset_x": 250.0,
      "offset_y": 200.0
    },
    // ... 其他3个参考点的信息
  ],
  "relative_to_center": {
    "center_position": [500, 350],
    "distance": 89.3,
    "angle_degrees": -12.7,
    "offset_x": -50.0,
    "offset_y": -50.0
  },
  "relative_position_percentage": {
    "x_percent": 35.2,
    "y_percent": 67.8
  },
  "is_inside_reference_area": true,
  "reference_bounds": {
    "min_x": 200,
    "max_x": 800,
    "min_y": 100,
    "max_y": 600,
    "center_x": 500,
    "center_y": 350
  }
}
```

## 🎨 可视化效果

系统会在图像上绘制：

- 🟡 **参考点**: 黄色圆圈标记，标注P1-P4
- 🟡 **参考区域**: 黄色矩形边界框
- 🟡 **中心点**: 参考区域的几何中心
- 🟢 **行人边界框**: 绿色矩形
- 🔵 **行人中心点**: 蓝色圆圈
- 🔴 **行人脚点**: 红色圆圈
- ⚪ **连接线**: 行人到最近参考点的灰色连线
- 📝 **信息标签**: 显示ID、位置、距离等信息

## 🔧 配置参数

### 参考点设置
```python
# 参考点顺序不限，但建议按顺时针或逆时针排列
reference_points = [
    (x1, y1),  # 点1
    (x2, y2),  # 点2  
    (x3, y3),  # 点3
    (x4, y4)   # 点4
]
```

### 跟踪器参数
```python
tracker = PersonRelativePositionTracker(
    reference_points=points,
    yolo_model_path='video_process/yolo/yolo11s.pt',      # YOLO模型路径
    deepsort_model_path='video_process/deepsort/deep/checkpoint/ckpt.t7'  # DeepSORT模型
)
```

### 视频处理参数
```python
results = tracker.process_video(
    video_path="input.mp4",
    output_file="results.json",
    skip_frames=5,      # 跳帧数：1=每帧处理，5=每5帧处理一次
    max_frames=None     # 最大帧数：None=处理整个视频
)
```

## 📈 性能优化建议

### 🚀 提高处理速度
- **跳帧处理**: 设置 `skip_frames=5` 或更高
- **限制帧数**: 设置 `max_frames` 参数
- **GPU加速**: 确保CUDA可用，系统自动使用GPU

### 🎯 提高检测精度
- **合适的参考点**: 选择清晰可见的区域角点
- **适当的参考区域**: 不要太大也不要太小
- **高质量视频**: 使用清晰度高的输入视频

### 💾 内存管理
- **批量处理**: 分批处理长视频
- **结果清理**: 及时清理不需要的跟踪结果

## 🛠️ 故障排除

### 常见问题

**Q: 为什么没有检测到行人？**
A: 
- 检查图像质量和清晰度
- 确认YOLO模型文件存在
- 尝试调整DeepSORT的 `min_confidence` 参数

**Q: 跟踪ID经常变化怎么办？**
A: 
- 检查DeepSORT特征提取器模型
- 调整 `max_age` 和 `n_init` 参数
- 确保视频帧率稳定

**Q: 位置计算不准确？**
A: 
- 重新选择更精确的参考点
- 确保参考点形成的区域覆盖主要活动区域
- 检查摄像头是否发生移动

**Q: 处理速度太慢？**
A: 
- 增加 `skip_frames` 参数
- 使用GPU加速
- 降低输入视频分辨率

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查组件状态
print("CUDA可用:", torch.cuda.is_available())
print("设备:", tracker.device)
```

## 📚 使用场景示例

### 🏢 监控场景
```python
# 监控走廊的四个角落作为参考点
corridor_points = [
    (100, 50),    # 走廊左上角
    (1180, 50),   # 走廊右上角
    (1180, 670),  # 走廊右下角
    (100, 670)    # 走廊左下角
]
```

### 🚪 门禁场景
```python
# 以门框作为参考区域
door_points = [
    (350, 200),   # 门框左上
    (550, 200),   # 门框右上
    (550, 600),   # 门框右下
    (350, 600)    # 门框左下
]
```

### 🚗 车辆通道
```python
# 车道标线作为参考
lane_points = [
    (200, 300),   # 车道左前
    (800, 280),   # 车道右前
    (900, 500),   # 车道右后
    (100, 520)    # 车道左后
]
```

## 📞 技术支持

如果遇到问题，请检查：
1. ✅ 所有依赖包是否安装完整
2. ✅ 模型文件是否存在并完整
3. ✅ 输入文件路径是否正确
4. ✅ 参考点坐标是否在图像范围内

---

🎉 **祝您使用愉快！** 如有问题，欢迎查看代码注释或进行调试。 