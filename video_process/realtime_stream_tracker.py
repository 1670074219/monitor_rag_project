import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from urllib import error, request

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor


def load_camera_configs(config_path: Path) -> list:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("camera_config", [])


def get_camera_config(camera_configs: list, camera_id: str) -> dict:
    for item in camera_configs:
        if item.get("camera_id") == camera_id:
            return item
    raise ValueError(f"未找到 camera_id={camera_id} 的配置")


def build_homography(camera_cfg: dict) -> np.ndarray:
    src = np.array(camera_cfg.get("pixel_coordinate", []), dtype=np.float32)
    dst = np.array(camera_cfg.get("real_coordinate", []), dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("camera_config 中 pixel_coordinate / real_coordinate 必须是 4x2")

    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if matrix is None:
        raise ValueError("单应性矩阵计算失败，请检查 camera_config 点位")
    return matrix


def pixel_to_world(x: float, y: float, h_matrix: np.ndarray) -> tuple[float, float] | None:
    vec = np.array([[x], [y], [1.0]], dtype=np.float32)
    mapped = h_matrix @ vec
    if mapped[2, 0] == 0:
        return None
    return float(mapped[0, 0] / mapped[2, 0]), float(mapped[1, 0] / mapped[2, 0])


def parse_camera_numeric_suffix(camera_id: str) -> int:
    match = re.search(r"(\d+)$", camera_id)
    if match:
        return int(match.group(1))
    return abs(hash(camera_id)) % 1000


class PushWorker:
    def __init__(self, api_url: str, timeout_sec: float = 0.5, max_queue_size: int = 2000):
        self.api_url = api_url
        self.timeout_sec = timeout_sec
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def push(self, payload: dict):
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(payload)
            except queue.Empty:
                pass

    def _run(self):
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            req = request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=self.timeout_sec):
                    pass
            except (error.URLError, TimeoutError, ConnectionError):
                pass
            except Exception:
                pass


def xyxy_to_xywh(dets_xyxy_conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xywh = np.zeros((len(dets_xyxy_conf), 4), dtype=np.float32)
    xywh[:, 0] = (dets_xyxy_conf[:, 0] + dets_xyxy_conf[:, 2]) / 2.0
    xywh[:, 1] = (dets_xyxy_conf[:, 1] + dets_xyxy_conf[:, 3]) / 2.0
    xywh[:, 2] = dets_xyxy_conf[:, 2] - dets_xyxy_conf[:, 0]
    xywh[:, 3] = dets_xyxy_conf[:, 3] - dets_xyxy_conf[:, 1]
    confs = dets_xyxy_conf[:, 4]
    return xywh, confs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时摄像头轨迹推送服务（YOLO + DeepSORT）")
    parser.add_argument("--camera", required=True, help="camera_config 中的 camera_id，如 camera1")
    parser.add_argument("--camera-config", default=str(CURRENT_DIR / "camera_config.json"), help="camera_config.json 路径")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/tracking/push", help="轨迹推送接口")
    parser.add_argument("--yolo-model", default=str(CURRENT_DIR / "models" / "yolo" / "yolo11s.pt"), help="YOLO 权重路径")
    parser.add_argument(
        "--deepsort-ckpt",
        default=str(CURRENT_DIR / "deepsort" / "deep" / "checkpoint" / "ckpt.t7"),
        help="DeepSORT ReID 权重路径",
    )
    parser.add_argument("--conf-thres", type=float, default=0.5, help="行人检测置信度阈值")
    parser.add_argument("--frame-stride", type=int, default=2, help="每 N 帧处理一次")
    parser.add_argument("--offset-step", type=int, default=10000, help="跨摄像头 track_id 前缀步长")
    parser.add_argument("--preview", action="store_true", help="本地预览检测与跟踪画面")
    return parser


def can_open_preview_window() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    return True


def main():
    args = build_parser().parse_args()
    enable_preview = bool(args.preview)

    if enable_preview and not can_open_preview_window():
        print("[WARN] 当前环境无图形显示（DISPLAY/WAYLAND_DISPLAY 未设置），已自动禁用 --preview")
        enable_preview = False

    camera_config_path = Path(args.camera_config).resolve()
    yolo_path = Path(args.yolo_model).resolve()
    deepsort_ckpt = Path(args.deepsort_ckpt).resolve()

    if not camera_config_path.exists():
        raise FileNotFoundError(f"camera_config 不存在: {camera_config_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO 权重不存在: {yolo_path}")
    if not deepsort_ckpt.exists():
        raise FileNotFoundError(f"DeepSORT 权重不存在: {deepsort_ckpt}")

    camera_configs = load_camera_configs(camera_config_path)
    camera_cfg = get_camera_config(camera_configs, args.camera)
    camera_url = camera_cfg["camera_url"]
    h_matrix = build_homography(camera_cfg)

    cam_num = parse_camera_numeric_suffix(args.camera)
    print(f"[INFO] 启动摄像头: {args.camera} ({camera_url})")
    print(f"[INFO] track_id 全局前缀基数: {cam_num * args.offset_step}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(yolo_path)).to(device)
    deepsort = DeepSort(
        max_dist=0.2,
        min_confidence=0.3,
        nms_max_overlap=0.5,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
    )
    extractor = Extractor(str(deepsort_ckpt), use_cuda=torch.cuda.is_available())

    push_worker = PushWorker(api_url=args.api_url)
    push_worker.start()

    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_index = 0
    reconnect_wait_sec = 2.0

    try:
        while True:
            if not cap.isOpened():
                print("[WARN] 视频流未打开，尝试重连...")
                cap.release()
                time.sleep(reconnect_wait_sec)
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] 读取视频帧失败，重连中...")
                cap.release()
                time.sleep(reconnect_wait_sec)
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_index += 1
            if args.frame_stride > 1 and frame_index % args.frame_stride != 0:
                continue

            result = yolo(frame, verbose=False, classes=[0])
            detections = []

            if result and result[0].boxes is not None and len(result[0].boxes) > 0:
                boxes = result[0].boxes
                xyxy = boxes.xyxy.detach().cpu().numpy()
                conf = boxes.conf.detach().cpu().numpy()

                for idx in range(len(xyxy)):
                    score = float(conf[idx])
                    if score < args.conf_thres:
                        continue
                    x1, y1, x2, y2 = xyxy[idx]
                    detections.append([x1, y1, x2, y2, score])

            if detections:
                dets_arr = np.array(detections, dtype=np.float32)
                xywh, confs = xyxy_to_xywh(dets_arr)
                deepsort.update(xywh, confs, frame, extractor)
            else:
                deepsort.update([], [], frame, extractor)

            for track in deepsort.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                x1, y1, x2, y2 = track.to_tlbr()
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                mapped = pixel_to_world(foot_x, foot_y, h_matrix)
                if mapped is None:
                    continue

                world_x, world_y = mapped
                local_track_id = int(track.track_id)
                global_track_id = cam_num * args.offset_step + local_track_id

                payload = {
                    "track_id": global_track_id,
                    "x": round(world_x, 2),
                    "y": round(world_y, 2),
                }
                push_worker.push(payload)

                if enable_preview:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"gid={global_track_id}",
                        (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            if enable_preview:
                cv2.imshow(f"{args.camera} realtime tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，准备退出")
    finally:
        push_worker.stop()
        cap.release()
        if enable_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
