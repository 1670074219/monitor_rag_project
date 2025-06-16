# 监控日志反查系统

这是一个基于RAG（检索增强生成）的智能监控日志反查系统，通过自然语言查询来搜索和分析监控视频记录。

## 系统架构

- **后端**: Python + Flask + FAISS + BGE嵌入模型
- **前端**: Vue.js 3 + Vite + ECharts
- **AI模型**: 使用BGE中文嵌入模型进行语义搜索，结合BM25算法

## 功能特性

1. 🔍 **智能搜索**: 支持自然语言查询监控记录
2. 🎥 **视频播放**: 直接播放相关的监控视频
3. 📊 **混合检索**: 结合向量检索和BM25算法，提高搜索准确性
4. 💬 **对话界面**: 友好的聊天式查询界面
5. 📈 **可视化展示**: 丰富的数据可视化界面

## 快速开始

### 1. 环境要求
- Python 3.8+
- Node.js 16+
- CUDA支持的GPU（用于模型推理）

### 2. 启动系统

使用提供的启动脚本：

```bash
./start_services.sh
```

或手动启动：

```bash
# 安装Python依赖
pip install -r requirements.txt

# 启动API服务器
python api_server.py

# 在另一个终端启动前端
cd frontend
npm install
npm run dev
```

### 3. 访问系统

- 前端界面: http://localhost:3001
- API接口: http://localhost:5000
- 健康检查: http://localhost:5000/api/health

## 使用方法

1. 打开前端界面
2. 点击右上角的"对话"按钮
3. 在聊天框中输入查询，例如：
   - "穿蓝色上衣的人"
   - "有人在操作设备"
   - "晚上的监控记录"
4. 系统会返回相关的日志记录，点击日志可以播放对应视频

## API接口

### 查询接口
```
POST /api/query
Content-Type: application/json

{
  "query": "查询内容",
  "k": 5,        // 可选，返回结果数量
  "alpha": 0.6   // 可选，FAISS权重
}
```

### 视频接口
```
GET /api/video/<video_name>
```

### 健康检查
```
GET /api/health
```

## 项目结构

```
├── video_process/          # 后端处理模块
│   ├── faiss_server.py    # FAISS检索服务
│   ├── video_analyse_server.py  # 视频分析服务
│   ├── video_capture_server.py  # 视频捕获服务
│   └── saved_video/       # 视频文件存储
├── frontend/              # 前端代码
│   ├── src/
│   │   ├── App.vue       # 主应用组件
│   │   └── components/   # 组件目录
│   └── package.json
├── api_server.py          # Flask API服务器
├── main.py               # 后端主程序入口
├── requirements.txt      # Python依赖
└── start_services.sh     # 启动脚本
```

## 注意事项

1. 确保BGE模型路径正确：`/root/data1/bge_zh_v1.5/`
2. 确保FAISS索引文件存在：`video_process/faiss_ifl2.index`
3. 确保视频描述文件存在：`video_process/video_description.json`
4. 如果出现模型加载错误，请检查CUDA环境和模型路径

## 故障排除

### API服务器启动失败
- 检查Python依赖是否正确安装
- 检查模型文件路径是否正确
- 查看控制台错误信息

### 前端无法访问API
- 确认API服务器在5000端口运行
- 检查防火墙设置
- 查看浏览器网络请求错误

### 视频无法播放
- 确认视频文件存在于`saved_video`目录
- 检查视频文件权限
- 确认浏览器支持MP4格式

## 开发说明

本系统设计为最小化对原有后端代码的修改，通过API服务器作为中间层，连接前端和现有的FAISS搜索功能。主要特点：

- 保持原有后端架构不变
- 通过Flask提供RESTful API
- 支持跨域请求
- 提供健康检查和统计接口 