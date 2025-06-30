# 3D轨迹显示功能使用说明

## 功能概述

此功能允许用户在3D平面图上显示选中事件的行人运行轨迹，支持多人轨迹的不同颜色区分。

## 主要特性

### ✨ 核心功能
- **事件轨迹显示**：点击事件后可显示该事件的所有行人轨迹
- **多人区分**：不同track_id使用不同颜色的轨迹线条
- **轨迹动画**：轨迹线条带有起点、终点和路径点标记
- **一键切换**：支持显示/隐藏轨迹的快速切换

### 🎨 视觉效果
- **轨迹线条**：半透明彩色线条，清晰显示移动路径
- **起点标记**：绿色球体标记轨迹起始位置
- **终点标记**：红色球体标记轨迹结束位置
- **路径点**：小球体标记轨迹关键点
- **颜色方案**：10种不同颜色自动分配给不同track_id

## 使用步骤

### 1. 启动服务
```bash
# 启动后端API服务器
python api_server.py

# 启动前端（另一个终端）
cd frontend
npm run dev
```

### 2. 查看事件
1. 打开前端界面：`http://localhost:5173`
2. 在3D地图上可以看到各种事件标记点
3. 点击任意事件标记查看事件详情

### 3. 显示轨迹
1. 在左侧事件详情面板中，如果事件有轨迹数据，会显示**"🛤️ 显示轨迹"**按钮
2. 点击按钮即可在3D地图上显示该事件的所有行人轨迹
3. 再次点击按钮可隐藏轨迹
4. 切换到其他有轨迹的事件会自动清除当前轨迹并显示新轨迹

### 4. 轨迹解读
- **不同颜色线条**：代表不同的行人（track_id）
- **绿色起点**：行人开始被检测到的位置
- **红色终点**：行人最后被检测到的位置
- **线条路径**：行人的完整移动轨迹

## API接口

### 后端接口

1. **获取事件列表（包含轨迹标志）**
   ```
   GET /api/events_3d
   响应: 事件列表，每个事件包含 has_trajectory 字段
   ```

2. **获取原始轨迹数据**
   ```
   GET /api/trajectory/<event_id>
   响应: 真实世界坐标的轨迹数据
   ```

3. **获取3D场景坐标轨迹数据**
   ```
   GET /api/trajectory/<event_id>/scene_coords
   响应: 转换为3D场景坐标的轨迹数据，包含颜色信息
   ```

### 前端组件方法

1. **显示轨迹**
   ```javascript
   await threeDMap.showEventTrajectory(eventId)
   ```

2. **清除轨迹**
   ```javascript
   threeDMap.clearTrajectories()
   ```

3. **切换轨迹显示**
   ```javascript
   await threeDMap.toggleTrajectory(eventId)
   ```

## 数据格式

### 轨迹数据结构
```json
{
  "event_id": "camera1_20250623_094900",
  "person_count": 2,
  "trajectories": [
    {
      "track_id": 1,
      "trajectory_length": 46,
      "color": "#ff4444",
      "coordinates": [
        {"x": -0.375, "y": 0, "z": 2.047},
        {"x": -0.371, "y": 0, "z": 2.029},
        ...
      ]
    }
  ]
}
```

### 颜色分配方案
```javascript
const colors = [
  '#ff4444',  // 红色 - Track ID 0, 10, 20...
  '#44ff44',  // 绿色 - Track ID 1, 11, 21...
  '#4444ff',  // 蓝色 - Track ID 2, 12, 22...
  '#ffff44',  // 黄色 - Track ID 3, 13, 23...
  '#ff44ff',  // 紫色 - Track ID 4, 14, 24...
  '#44ffff',  // 青色 - Track ID 5, 15, 25...
  '#ff8844',  // 橙色 - Track ID 6, 16, 26...
  '#8844ff',  // 紫罗兰 - Track ID 7, 17, 27...
  '#44ff88',  // 青绿 - Track ID 8, 18, 28...
  '#ff4488'   // 洋红 - Track ID 9, 19, 29...
]
```

## 测试验证

### 运行测试脚本
```bash
# 测试API接口
python video_process/test_trajectory_api.py
```

### 测试内容
- ✅ 获取事件列表（验证has_trajectory标志）
- ✅ 获取原始轨迹数据
- ✅ 获取3D场景坐标轨迹数据
- ✅ 验证坐标转换准确性
- ✅ 验证颜色分配机制

## 故障排除

### 常见问题

1. **事件没有轨迹按钮**
   - 检查事件是否有轨迹数据：该事件在`video_description_with_trajectory.json`中应包含`trajectory_data`字段
   - 确认后端API正确返回`has_trajectory: true`

2. **点击轨迹按钮无反应**
   - 检查浏览器控制台错误信息
   - 确认API服务器运行正常
   - 验证事件ID是否正确

3. **轨迹显示位置不准确**
   - 检查坐标转换参数是否与3D场景设置匹配
   - 验证真实世界坐标到场景坐标的转换公式

4. **轨迹颜色异常**
   - 检查track_id是否为有效数字
   - 验证颜色分配算法

### 调试步骤

1. **检查后端API**
   ```bash
   curl http://localhost:5000/api/events_3d | jq '.[] | select(.has_trajectory == true)'
   ```

2. **检查轨迹数据**
   ```bash
   curl http://localhost:5000/api/trajectory/camera1_20250623_094900/scene_coords
   ```

3. **查看前端控制台**
   - 打开浏览器开发者工具
   - 查看Console标签页的错误信息
   - 查看Network标签页的API请求状态

## 技术实现

### 坐标转换流程
1. **真实世界坐标** (厘米) → **3D场景坐标**
2. 转换公式：
   ```javascript
   scene_x = (real_x - 462.5) / 100
   scene_z = (real_y - 685.5) / 100
   scene_y = 0  // 地面高度
   ```

### 3D渲染技术
- **Three.js LineBasicMaterial**：轨迹线条渲染
- **Three.js SphereGeometry**：起点终点标记
- **BufferGeometry**：高效的轨迹点存储
- **透明度和动画**：增强视觉效果

### 性能优化
- **按需加载**：只在用户点击时加载轨迹数据
- **内存管理**：切换轨迹时自动清理前一个轨迹的几何体和材质
- **缓存机制**：3D组件缓存轨迹状态，避免重复请求

---

**提示**：首次使用前请确保已运行轨迹分析脚本生成轨迹数据，详见[轨迹分析系统文档](README_trajectory_system.md)。 