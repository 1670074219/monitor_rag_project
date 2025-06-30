# 行人轨迹分析系统使用说明

## 系统概述

这是一个完整的行人轨迹分析系统，能够：
1. 从视频中提取行人轨迹
2. 将像素坐标转换为真实世界坐标
3. 按照JSON描述的人数选择最长轨迹
4. 为前端3D显示提供API接口

## 文件结构

```
video_process/
├── add_trajectory_to_json.py         # 主要轨迹分析脚本
├── test_single_video_trajectory.py   # 单视频测试脚本
├── trajectory_api.py                 # 轨迹数据API服务器
├── video_description.json            # 原始视频描述数据
├── video_description_with_trajectory.json  # 包含轨迹的视频数据
└── deepsort/                         # DeepSORT跟踪算法
```

## 使用流程

### 1. 环境准备

确保已安装所需依赖：
```bash
pip install ultralytics torch torchvision opencv-python flask flask-cors numpy
```

### 2. 单视频测试

首先测试单个视频的轨迹提取：

```bash
cd /root/data1/monitor_rag_project
python video_process/test_single_video_trajectory.py
```

**输出示例：**
```
🚀 单视频轨迹分析测试
==================================================
🎬 测试视频: camera1_20250623_094900.mp4
   目标人数: 2
   使用设备: cuda
   总帧数: 900
   帧率: 30.0 FPS
   进度: 16.7% - 时间: 5.0s - 已检测轨迹: 2

📊 轨迹分析结果:
   检测到轨迹数: 2
   轨迹ID 1: 85 个点
      起点: [425.67, 890.23]
      终点: [410.45, 380.12]
   轨迹ID 2: 62 个点
      起点: [480.12, 350.45]
      终点: [520.34, 950.67]

🎯 选择轨迹:
   ID 1: 85 个坐标点
   ID 2: 62 个坐标点

✅ 测试成功!
   测试结果已保存: video_process/test_trajectory_result.json
```

### 3. 批量轨迹分析

对所有视频进行轨迹分析：

```bash
python video_process/add_trajectory_to_json.py
```

**处理过程：**
1. 读取 `video_description.json`
2. 提取每个视频描述中的人数
3. 运行YOLO + DeepSORT分析
4. 选择轨迹最长的N个ID（N为描述中的人数）
5. 保存到 `video_description_with_trajectory.json`

**输出示例：**
```
🚀 行人轨迹分析脚本
============================================================
📁 读取JSON文件: video_process/video_description.json
📊 找到 3 个视频条目

🎬 处理视频: camera1_20250623_094842
🎬 分析视频: camera1_20250623_094842.mp4
   目标人数: 2
   进度: 33.3% (300/900)
   ✅ 轨迹ID 1: 45 个坐标点
   ✅ 轨迹ID 2: 38 个坐标点
   🎯 选择了 2 条轨迹
   ✅ 轨迹数据已添加

🎉 处理完成!
   处理视频数: 3
   输出文件: video_process/video_description_with_trajectory.json
```

### 4. 启动API服务

为前端提供轨迹数据API：

```bash
python video_process/trajectory_api.py
```

**API端点：**
- `GET /api/trajectory/videos` - 获取视频列表
- `GET /api/trajectory/video/<video_id>` - 获取原始轨迹数据
- `GET /api/trajectory/video/<video_id>/frontend` - 获取前端3D格式轨迹
- `POST /api/trajectory/reload` - 重新加载数据
- `GET /api/trajectory/status` - 获取数据状态

## 数据格式

### JSON文件格式

处理后的JSON文件包含轨迹数据：

```json
{
  "camera1_20250623_094900": {
    "video_path": "/path/to/video.mp4",
    "analyse_result": "人数：2\n...",
    "is_embedding": false,
    "idx": null,
    "trajectory_data": {
      "person_count": 2,
      "coordinate_system": "real_world",
      "unit": "centimeters",
      "trajectories": [
        {
          "track_id": 1,
          "trajectory_length": 85,
          "coordinates": [
            [425.67, 890.23],
            [426.12, 888.45],
            ...
          ]
        },
        {
          "track_id": 2,
          "trajectory_length": 62,
          "coordinates": [
            [480.12, 350.45],
            [479.67, 352.34],
            ...
          ]
        }
      ]
    }
  }
}
```

### 前端3D格式

API返回的前端格式：

```json
{
  "success": true,
  "data": {
    "video_id": "camera1_20250623_094900",
    "person_count": 2,
    "trajectories": [
      {
        "track_id": 1,
        "trajectory_length": 85,
        "color": "#ff4444",
        "coordinates": [
          {"x": -0.375, "y": 0, "z": 2.047},
          {"x": -0.371, "y": 0, "z": 2.029},
          ...
        ]
      }
    ]
  }
}
```

## 坐标系统

### 像素坐标 → 真实世界坐标

使用四个标定点计算单应性矩阵：

```python
# 像素标定点
pixel_points = [
    (517, 95),    # 左上
    (667, 95),    # 右上
    (1182, 720),  # 右下
    (276, 720)    # 左下
]

# 真实世界坐标 (厘米)
real_world_points = [
    (400, 351),   # 左上
    (525, 351),   # 右上
    (525, 1020),  # 右下
    (400, 1020)   # 左下
]
```

### 真实世界坐标 → 3D场景坐标

```python
scene_x = (real_x - 462.5) / 100  # 偏移并缩放
scene_z = (real_y - 685.5) / 100  # Y轴映射到Z轴
scene_y = 0                       # 地面高度
```

## 前端集成

### 获取轨迹数据

```javascript
// 获取视频列表
const response = await fetch('http://localhost:5002/api/trajectory/videos');
const videoList = await response.json();

// 获取特定视频的3D轨迹数据
const trajectoryResponse = await fetch(
  `http://localhost:5002/api/trajectory/video/${videoId}/frontend`
);
const trajectoryData = await trajectoryResponse.json();
```

### 在3D场景中显示轨迹

```javascript
// 在ThreeDMap组件中添加轨迹显示方法
showTrajectories(trajectoryData) {
  const trajectories = trajectoryData.data.trajectories;
  
  trajectories.forEach(traj => {
    const points = traj.coordinates.map(coord => 
      new THREE.Vector3(coord.x, coord.y, coord.z)
    );
    
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ 
      color: traj.color, 
      linewidth: 3 
    });
    
    const line = new THREE.Line(geometry, material);
    this.scene.add(line);
  });
}
```

## 配置参数

### 轨迹分析参数

```python
# DeepSORT配置
deepsort = DeepSort(
    max_dist=0.2,          # 最大匹配距离
    min_confidence=0.3,    # 最小置信度
    nms_max_overlap=0.5,   # NMS重叠阈值
    max_iou_distance=0.7,  # 最大IOU距离
    max_age=70,            # 轨迹最大保持帧数
    n_init=3,              # 轨迹确认需要的帧数
    nn_budget=100          # 特征预算
)

# YOLO置信度阈值
confidence_threshold = 0.4

# 帧采样间隔
frame_interval = 3  # 每3帧处理一次
```

### 坐标转换参数

根据您的实际摄像头标定结果调整：

```python
# 标定点（需要根据实际情况修改）
pixel_points_calib = [(517, 95), (667, 95), (1182, 720), (276, 720)]
real_world_points_calib = [(400, 351), (525, 351), (525, 1020), (400, 1020)]

# 3D场景转换参数
scene_x_offset = 462.5
scene_z_offset = 685.5
scale_factor = 100
```

## 故障排除

### 常见问题

1. **轨迹检测失败**
   - 检查YOLO模型文件是否存在
   - 检查DeepSORT模型文件是否存在
   - 验证视频文件路径

2. **坐标转换错误**
   - 验证标定点是否正确
   - 检查单应性矩阵计算

3. **API连接失败**
   - 确认API服务器运行在正确端口
   - 检查CORS设置

### 调试工具

```bash
# 检查模型文件
ls -la video_process/yolo/yolo11s.pt
ls -la video_process/deepsort/deep/checkpoint/ckpt.t7

# 测试API状态
curl http://localhost:5002/api/trajectory/status

# 查看日志
python video_process/trajectory_api.py 2>&1 | tee trajectory_api.log
```

## 性能优化

### 处理速度优化

1. **帧采样间隔**：增大frame_interval减少处理帧数
2. **GPU加速**：确保CUDA可用
3. **批处理**：一次处理多个视频

### 存储优化

1. **坐标精度**：调整小数位数
2. **轨迹过滤**：移除过短的轨迹
3. **数据压缩**：使用更紧凑的数据格式

## 扩展功能

### 轨迹分析

1. **速度计算**：基于相邻点计算移动速度
2. **方向变化**：检测行走方向改变
3. **停留检测**：识别停留位置和时间
4. **交互分析**：检测人员之间的交互

### 可视化增强

1. **轨迹动画**：时间序列播放
2. **热力图**：显示活动密集区域
3. **统计图表**：轨迹长度、速度分布
4. **3D路径**：立体轨迹显示

---

**注意：** 首次运行前请确保所有模型文件已下载，并根据实际摄像头配置调整标定参数。 