import os
import json
from queue import Queue
import threading
import subprocess
import time
import logging

from video_process.multi_camera_capture import VideoCaptureManager
from video_process.video_analyse_server import VideoAnalyServer
from video_process.faiss_server import FaissServer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_queue(video_queue: Queue, video_dsp_queue: Queue, video_description_path: str):
    json_lock = threading.Lock()
    with json_lock:
        with open(video_description_path, "r", encoding="utf-8") as f:
            video_data = json.load(f)

    for v_k, v_info in video_data.items():
        if v_info['video_path'] is not None and v_info['analyse_result'] is None:
            video_queue.put(v_k)
        elif v_info['analyse_result'] is not None and v_info['is_embedding'] is False:
            video_dsp_queue.put(v_k)
    
    logging.info(f"Init v_q put {video_queue.qsize()}, v_d_q put {video_dsp_queue.qsize()}")
        


def start_vllm_server():
    vllm_cmd = [
        "vllm", "serve", "/root/data1/Qwen2.5-VL-7B-Instruct-AWQ/",
        "--dtype", "auto",
        "--api-key", "token-abc123",
        "--served-model-name", "qwen2.5",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.8"
    ]

    logging.info("🚀 正在启动 vLLM 服务器...")
    vllm_process = subprocess.Popen(
        vllm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 给 vLLM 一点时间启动（你也可以更智能地检测端口是否 ready）
    time.sleep(10)  # 暂时简单粗暴等待 10 秒
    logging.info("✅ vLLM 服务器启动完成！")

    return vllm_process

if __name__ == "__main__":
    # 启动vllm大模型
    # vllm_process = start_vllm_server()

    # 配置路径
    base_dir = os.path.join(os.path.dirname(__file__), "video_process")
    logging.info(f"dir: {base_dir}")

    # 加载摄像头配置
    with open(os.path.join(base_dir, "camera_config.json"), "r") as f:
        camera_config = json.load(f)["camera_config"]

    # 队列
    video_queue = Queue()
    video_dsp_queue = Queue()

    init_queue(video_queue, video_dsp_queue, os.path.join(base_dir, "video_description.json"))
    logger.info(f"video_queue: {list(video_queue.queue)}, video_dsp_queue: {list(video_dsp_queue.queue)}")

    # 启动 VideoCaptureManager
    video_capture_server = VideoCaptureManager(
        camera_config=camera_config,
        yolo_path=os.path.join(base_dir, "yolo/yolo11s.pt"),
        saved_video_path=os.path.join(base_dir, "saved_video"),
        video_queue=video_queue,
        video_description_path=os.path.join(base_dir, "video_description.json")
    )

    # 启动 VideoAnalyServer
    video_analyse_server = VideoAnalyServer(
        video_queue=video_queue,
        model_name="qwen2.5",
        api_url="http://localhost:8000/v1",
        api_key="token-abc123",
        video_description_path=os.path.join(base_dir, "video_description.json"),
        video_dsp_queue=video_dsp_queue
    )
    
    # 启动 FaissServer
    faiss_server = FaissServer(
        emd_model_path="/root/data1/bge_zh_v1.5/",
        index_path="/root/data1/monitor_rag_project/video_process/faiss_cache/faiss_ifl2.index",
        video_dsp_queue=video_dsp_queue,
        video_description_path=os.path.join(base_dir, "video_description.json")
    )

    # 启动线程
    # video_capture_server 本身是普通类，run_all_cameras 会内部多线程，直接用线程包装
    capture_thread = threading.Thread(target=video_capture_server.run_all_cameras, daemon=True)
    analyse_thread = threading.Thread(target=video_analyse_server.run, daemon=True)
    faiss_thread = threading.Thread(target=faiss_server.run, daemon=True)

    # 启动所有线程
    capture_thread.start()
    analyse_thread.start()
    faiss_thread.start()

    logger.info("🎬 Video Capture Server, 🧠 Video Analyse Server, 🔍 Faiss Server 已启动！")

    # 设置信号处理器（只能在主线程中设置）
    import signal
    def signal_handler(sig, frame):
        logger.info("⛔️ 收到退出信号，正在清理资源...")
        # 清理视频录制进程
        video_capture_server.cleanup()
        # vllm_process.terminate()
        # vllm_process.wait()
        logger.info("✅ 清理完成，退出程序。")
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 主线程等待子线程
    try:
        capture_thread.join()
        analyse_thread.join()
        faiss_thread.join()
    except KeyboardInterrupt:
        logger.info("⛔️ 收到键盘中断，正在退出...")
        signal_handler(signal.SIGINT, None)

