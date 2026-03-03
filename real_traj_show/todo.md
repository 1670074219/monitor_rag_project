🚀 智能监控与轨迹追踪系统 (Project To-Do List)
📦 模块一：基础设施与环境准备 (Infrastructure)
本模块主要负责数据库、搜索引擎及深度学习模型的初始化。
• [ ] 1.1 MySQL 数据库配置
• [ ] 创建 videos 表（存储视频元数据与路径）。
• [ ] 创建 video_vectors 表（存储视频ID、行人ID、轨迹JSON、图片相对路径 person_image_path）。
• [ ] 创建纯净版 video_logs_index 表（清理历史遗留的 faiss_idx 字段，存储文本与向量Blob）。
• [ ] 1.2 Elasticsearch 8.x 配置
• [ ] 建立 video_reid_index 索引及 Mapping（512维 dense_vector，用于行人图片特征）。
• [ ] 建立 video_logs_index 索引及 Mapping（768/1024维，文本与向量混合检索）。
• [ ] 1.3 算法模型权重就位
• [ ] 下载并配置 YOLOv11x 权重文件（yolo11x.pt）。
• [ ] 配置 DeepSORT Checkpoint。
• [ ] 配置 Torchreid (OSNet) 权重文件。
• [ ] 配置 BGE 中文文本向量化模型（bge_zh_v1.5）。
• [ ] 1.4 本地存储目录初始化
• [ ] 建立视频上传/读取目录。
• [ ] 建立并挂载行人截图目录（./person_crops），确保后端可将其作为静态资源提供给前端。
🧠 模块二：核心视觉算法流水线 (Vision Pipeline)
本模块负责从视频流中提取出干净、准确的行人特征和坐标。
• [ ] 2.1 目标检测与过滤 (YOLO)
• [ ] 设定高置信度阈值（conf > 0.65）。
• [ ] 加入严格的形态学过滤（剔除高度 < 80 像素、长宽比异常的假目标，如机器人）。
• [ ] 2.2 目标追踪 (DeepSORT)
• [ ] 将检测框送入 DeepSORT 维持时序 ID。
• [ ] 获取目标脚底坐标 (foot_x, foot_y) 用于轨迹记录。
• [ ] 2.3 视角坐标转换 (Homography)
• [ ] 编写标定脚本，获取摄像头与平面图的 4 对参考坐标。
• [ ] 维护 camera_config.json。
• [ ] 在代码中实时将视频像素坐标转化为楼层平面图坐标 (X, Y)。
• [ ] 2.4 高质量特征提取 (ReID)
• [ ] 废弃复杂的清晰度评分，改为**“缓存轨迹全生命周期截图”**。
• [ ] 轨迹结束后，精准截取 1/3 处的中间帧作为最佳特征图。
• [ ] 将最佳特征图送入 Torchreid 提取 512 维特征向量。
💾 模块三：数据清洗与持久化 (Data Processing)
本模块负责过滤脏数据，并将高价值数据落盘。
• [ ] 3.1 轨迹与背景去重过滤
• [ ] 过滤过短的瞬时轨迹（点数 < 6）。
• [ ] 计算“最大活动半径（max_dist）”，低于 5.0 像素的判定为墙壁反光/死背景，坚决丢弃。
• [ ] 3.2 图片与数据双写落盘
• [ ] 将提取的 best_crop 保存至本地磁盘（如 video_{id}/person_{idx}.jpg）。
• [ ] 组装轨迹 JSON，连同图片相对路径写入 MySQL。
• [ ] 将 512 维特征向量异步写入 Elasticsearch。
• [ ] 确保处理完一个视频后，及时清空图片缓存列表（data['all_crops'] = []）并调用 gc.collect() 防止内存泄漏。
🔌 模块四：后端服务与 API (Backend Services)
本模块负责为前端提供数据接口和实时通讯通道。
• [x] 4.1 基础查询 REST API
• [x] GET /api/videos：获取当前系统内的视频列表。
• [x] GET /api/videos/{video_id}/persons：根据视频 ID，返回该视频内所有行人的 ID 与图片路径（供前端展示“候选人列表”）。
• [x] 4.2 向量检索 API (Search Target)
• [x] POST /api/search/person：接收目标 video_id 和 person_index，查询 ES 提取向量。
• [x] 结合 camera_topology.json（摄像头拓扑图），按空间逻辑扩展搜索邻近摄像头。
• [x] 返回匹配成功的轨迹数据及目标对应的图片路径。
• [x] 4.3 实时数据流 (WebSocket)
• [x] 创建 ws://your_domain/ws/tracking 路由。
• [ ] 改造 analyze_video 循环：每处理一帧，立即算出目标的平面图 (X, Y) 坐标。
• [x] 将 {"track_id": x, "x": X, "y": Y} 序列化后实时 Push 给前端。
🖥️ 模块五：前端可视化交互 (Frontend UI)
本模块负责将冷冰冰的数据转化为直观的操作界面。
• [x] 5.1 候选人展示面板
• [x] 请求 persons 接口，用卡片或横向滚动列表展示视频中的行人截图。
• [x] 增加点击交互，点击某张人脸/身体截图，即触发跨镜头搜索。
• [x] 5.2 历史轨迹渲染 (Search Result View)
• [x] 接收搜索 API 返回的轨迹 JSON。
• [x] 在前端使用 Canvas 或 SVG，在 floorplan.jpg 静态底图上绘制出该目标的历史行进路线。
• [x] 5.3 实时追踪沙盘 (Real-time View)
• [x] 建立 WebSocket 连接监听后端坐标推送。
• [x] 使用 HTML5 Canvas，在楼层图上实时绘制移动的彩色小圆点。
• [x] 维护每个 ID 的历史坐标数组，用 lineTo 画出跟随圆点移动的“贪吃蛇”尾巴。
• [x] 设置超时清理机制（如超过 3 秒未收到某 ID 坐标，使其在图上淡出消失）。
🛠️ 模块六：联调、测试与优化 (Testing & Optimization)
• [ ] 6.1 极端场景测试
• [ ] 测试目标“折返跑”和“原地停留”场景，确保不会被静止过滤机制误杀。
• [ ] 测试极其拥挤的画面，观察 DeepSORT ID 跳变的频率。
• [ ] 6.2 性能压力测试
• [ ] 监控长视频处理时的内存占用，排查 OOM（Out of Memory）风险。
• [ ] 验证 ES 在插入上万条向量后的检索延迟（目标控制在 100ms 以内）。
• [ ] 6.3 部署封装
• [ ] 将依赖写入 requirements.txt。
• [ ] （可选）编写 Dockerfile 和 docker-compose.yml，将 MySQL, ES, Backend 一键拉起。
