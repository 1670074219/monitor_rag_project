import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import fs from 'fs' // 引入 Node.js 文件系统模块
import { fileURLToPath } from 'url'; // 用于处理 __dirname
import { searchForWorkspaceRoot } from 'vite'

// 获取当前文件的目录，因为 __dirname 在 ES modules 中不可用
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const API_TARGET = process.env.VITE_API_TARGET || 'http://127.0.0.1:5000'

// Plugin to force Content-Type for specific extensions
const forceContentTypePlugin = () => ({
  name: 'force-content-type',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      // Check if the request URL ends with .mp4 (case-insensitive)
      if (req.url && req.url.toLowerCase().endsWith('.mp4')) {
        // Force the Content-Type header
        res.setHeader('Content-Type', 'video/mp4');
        console.log(`Forcing Content-Type: video/mp4 for ${req.url}`); // Add log
      }
      // Pass the request to the next middleware
      next();
    });
  }
});

// 新的中间件插件，用于服务视频文件
const serveVideoPlugin = () => ({
  name: 'serve-video',
  configureServer(server) {
    const videoDir = '/root/data1/monitor_rag_project/video_process/saved_video'; // 当前项目的视频目录
    console.log(`Video serving configured for directory: ${videoDir}`);

    server.middlewares.use((req, res, next) => {
      if (req.url && req.url.startsWith('/save_video/')) {
        const videoName = req.url.substring('/save_video/'.length);
        // 对视频文件名进行解码，以防包含特殊字符 (如空格 '%20')
        const decodedVideoName = decodeURIComponent(videoName);
        const filePath = path.join(videoDir, decodedVideoName);

        console.log(`Attempting to serve video: ${filePath}`);

        // 检查文件是否存在且可读
        fs.access(filePath, fs.constants.R_OK, (err) => {
          if (err) {
            console.error(`Video file not found or not readable: ${filePath}`, err);
            res.statusCode = 404; // 设置 404 状态码
            res.end('Video Not Found'); // 发送错误信息
            // next(); // 或者调用 next() 让 Vite 处理 404，但自己处理更明确
          } else {
            // 文件存在且可读，提供服务
            console.log(`Serving video: ${filePath}`);
            fs.stat(filePath, (statErr, stats) => {
                 if (statErr) {
                      console.error(`Error getting stats for video file: ${filePath}`, statErr);
                      res.statusCode = 500;
                      res.end('Internal Server Error');
                      return;
                 }
                 // 设置正确的 Content-Type 和 Content-Length
                 res.writeHead(200, {
                   'Content-Type': 'video/mp4',
                   'Content-Length': stats.size,
                   'Accept-Ranges': 'bytes' // 支持范围请求，对视频播放很重要
                 });
                 // 创建可读流并管道到响应
                 const stream = fs.createReadStream(filePath);
                 stream.pipe(res);
                 stream.on('error', (streamErr) => {
                      console.error(`Error streaming video file: ${filePath}`, streamErr);
                      // 尝试结束响应，如果还能写的话
                      if (!res.writableEnded) {
                           res.statusCode = 500;
                           res.end('Error streaming video');
                      }
                 });
            });
          }
        });
      } else {
        // 如果不是 /save_video/ 请求，交给下一个中间件
        next();
      }
    });
  }
});

export default defineConfig({
  plugins: [
    vue(),
    // forceContentTypePlugin() // 移除旧插件
    serveVideoPlugin() // 添加新的视频服务插件
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  server: {
    host:'0.0.0.0',
    port: 3001,
    proxy: { // Added proxy configuration
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        // secure: false, // Uncomment if your backend is http and vite is https (not common in dev)
        // rewrite: (path) => path.replace(/^\/api/, '') // Uncomment if backend API paths don't start with /api
      },
      '/monitor': {
        target: API_TARGET,
        changeOrigin: true
      },
      '/video_feed': {
        target: API_TARGET,
        changeOrigin: true
      }
    },
    // Add fs.allow configuration
    fs: {
      // Allow serving files from the current project
      allow: [
        '/root/data1/monitor_rag_project/video_process/saved_video',
        '/root/data1/monitor_rag_project/frontend',
        '/root/data1/monitor_rag_project',
        searchForWorkspaceRoot(process.cwd()), // 启用工作区根目录访问
      ]
    },
    watch: {
      // 保持忽略 node_modules
      ignored: [
         // 忽略 node_modules（无论是否在 fs.allow 中）
        '**/node_modules/**',
        // 忽略视频目录（即使允许访问）
        '/root/data1/monitor_rag_project/video_process/saved_video/**',
        // 忽略 public 目录
        '**/public/**',
        // 忽略 .git 目录
        '**/.git/**',
        // 忽略项目根目录下的 floorplan.png
        'floorplan.png'
      ]
    }
  },
}) 