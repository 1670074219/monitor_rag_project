import os
import cv2
import json
import argparse
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PROCESS_ROOT = os.path.dirname(CURRENT_DIR)

import sys
if VIDEO_PROCESS_ROOT not in sys.path:
    sys.path.insert(0, VIDEO_PROCESS_ROOT)

from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor


def _abs_from_root(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(VIDEO_PROCESS_ROOT, path_value))


class TrajectoryFrameExtractor:
    def __init__(
        self,
        yolo_model_path: str,
        deepsort_checkpoint: str,
        conf_thres: float = 0.60,
        frame_stride: int = 5,
        min_height: int = 50,
        max_wh_ratio: float = 0.8,
    ):
        self.conf_thres = conf_thres
        self.frame_stride = frame_stride
        self.min_height = min_height
        self.max_wh_ratio = max_wh_ratio

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 使用设备: {self.device}")

        yolo_model_path = _abs_from_root(yolo_model_path)
        deepsort_checkpoint = _abs_from_root(deepsort_checkpoint)

        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"YOLO 模型不存在: {yolo_model_path}")
        if not os.path.exists(deepsort_checkpoint):
            raise FileNotFoundError(f"DeepSORT checkpoint 不存在: {deepsort_checkpoint}")

        print(f"📦 加载 YOLO: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path).to(self.device)

        print(f"📦 加载 DeepSORT ReID 特征提取器: {deepsort_checkpoint}")
        self.ds_extractor = Extractor(deepsort_checkpoint, use_cuda=torch.cuda.is_available())

        self.deepsort = DeepSort(
            max_dist=0.2,
            min_confidence=0.3,
            nms_max_overlap=0.5,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
        )

    def _detect_persons(self, frame_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.yolo(img_rgb, verbose=False)

        detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                if int(box.cls) != 0 or float(box.conf) <= self.conf_thres:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w_box = x2 - x1
                h_box = y2 - y1

                if h_box < self.min_height or w_box > h_box * self.max_wh_ratio:
                    continue

                detections.append([x1, y1, x2, y2, float(box.conf.item())])

        return detections

    def extract(self, video_path: str, output_dir: str):
        video_path = os.path.normpath(video_path)
        output_dir = os.path.normpath(output_dir)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        frame_idx = 0
        saved_count = 0
        track_save_counts = {}
        metadata = []

        print(f"▶️ 开始提取轨迹人像: {video_path}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.frame_stride != 0:
                continue

            detections = self._detect_persons(frame)

            if detections:
                dets = np.array(detections)
                xywh = np.zeros((len(dets), 4))
                xywh[:, 0] = (dets[:, 0] + dets[:, 2]) / 2
                xywh[:, 1] = (dets[:, 1] + dets[:, 3]) / 2
                xywh[:, 2] = dets[:, 2] - dets[:, 0]
                xywh[:, 3] = dets[:, 3] - dets[:, 1]
                confs = dets[:, 4]
                self.deepsort.update(xywh, confs, frame, self.ds_extractor)
            else:
                self.deepsort.update([], [], frame, self.ds_extractor)

            h, w = frame.shape[:2]
            for track in self.deepsort.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = int(track.track_id)
                bbox = track.to_tlbr()
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                track_dir = os.path.join(output_dir, f"track_{track_id}")
                os.makedirs(track_dir, exist_ok=True)

                track_save_counts[track_id] = track_save_counts.get(track_id, 0) + 1
                img_name = f"f{frame_idx:06d}_track{track_id}_n{track_save_counts[track_id]:04d}.jpg"
                img_path = os.path.join(track_dir, img_name)

                if cv2.imwrite(img_path, crop):
                    saved_count += 1
                    foot_x, foot_y = int((x1 + x2) / 2), y2
                    metadata.append(
                        {
                            "track_id": track_id,
                            "frame_index": frame_idx,
                            "image_path": os.path.relpath(img_path, output_dir).replace("\\", "/"),
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "foot_point": [foot_x, foot_y],
                        }
                    )

        cap.release()

        summary = {
            "video_path": video_path,
            "frame_stride": self.frame_stride,
            "confidence_threshold": self.conf_thres,
            "saved_images": saved_count,
            "track_count": len(track_save_counts),
            "images_per_track": track_save_counts,
            "items": metadata,
        }

        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"✅ 提取完成: 共保存 {saved_count} 张人像，轨迹ID数 {len(track_save_counts)}")
        print(f"📄 元数据已保存: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="提取用于轨迹分析的行人裁剪图")
    parser.add_argument("video_path", type=str, help="输入视频路径")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trajectory_person_frames",
        help="输出目录（会自动创建）",
    )
    parser.add_argument("--frame_stride", type=int, default=5, help="抽帧步长")
    parser.add_argument("--conf", type=float, default=0.60, help="YOLO 人体检测置信度阈值")
    parser.add_argument("--min_height", type=int, default=50, help="最小框高")
    parser.add_argument("--max_wh_ratio", type=float, default=0.8, help="宽高比上限（w <= h * ratio）")
    parser.add_argument("--yolo_model", type=str, default="yolo/yolo11x.pt", help="YOLO 模型路径（相对 video_process 根目录或绝对路径）")
    parser.add_argument(
        "--deepsort_ckpt",
        type=str,
        default="deepsort/deep/checkpoint/ckpt.t7",
        help="DeepSORT ReID checkpoint 路径（相对 video_process 根目录或绝对路径）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extractor = TrajectoryFrameExtractor(
        yolo_model_path=args.yolo_model,
        deepsort_checkpoint=args.deepsort_ckpt,
        conf_thres=args.conf,
        frame_stride=args.frame_stride,
        min_height=args.min_height,
        max_wh_ratio=args.max_wh_ratio,
    )
    extractor.extract(args.video_path, args.output_dir)
