import numpy as np
import torch
from .detection import Detection
from .tracker import Tracker
from . import nn_matching


class DeepSort(object):
    def __init__(self, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        # 根据字符串 'cosine' 创建距离度量对象
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_dist, nn_budget)
        
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img, feature_extractor):
        try:
            self.height, self.width = ori_img.shape[:2]
            
            # 检查输入是否有效
            if len(bbox_xywh) == 0:
                return np.array([])
                
            # generate detections
            features = self._get_features(bbox_xywh, ori_img, feature_extractor)
            
            # 确保 features 是正确的格式
            if len(features) == 0:
                return np.array([])
                
            bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
            detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
                confidences) if conf > self.min_confidence]

            if len(detections) == 0:
                return np.array([])

            # update tracker
            self.tracker.predict()
            self.tracker.update(detections)

            # output bbox identities
            outputs = []
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=int))
                
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
            else:
                outputs = np.array([])
            return outputs
            
        except Exception as e:
            print(f"DeepSORT update 错误: {e}")
            return np.array([])

    """
    TODO:
        Convert bbox from xc_yc_w_h to top_left_w_h
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from top_left_w_h to xc_yc_w_h
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img, feature_extractor):
        try:
            im_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = self._xywh_to_xyxy(box)
                # 确保坐标在有效范围内
                if x2 > x1 and y2 > y1:
                    im = ori_img[y1:y2, x1:x2]
                    if im.size > 0:  # 确保图像不为空
                        im_crops.append(im)
                        
            if im_crops:
                features = feature_extractor(im_crops)
                # 确保返回的特征是 numpy 数组
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                return features
            else:
                return np.array([])
        except Exception as e:
            print(f"特征提取错误: {e}")
            return np.array([]) 