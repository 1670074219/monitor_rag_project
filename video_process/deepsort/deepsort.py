import numpy as np
import torch
from .detection import Detection
from .tracker import Tracker
from . import nn_matching


class DeepSort(object):
    def __init__(self, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.feature_dim = None

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
            
            # 确保特征数量与边界框数量匹配
            if len(features) != len(bbox_tlwh):
                min_len = min(len(features), len(bbox_tlwh), len(confidences))
                features = features[:min_len]
                bbox_tlwh = bbox_tlwh[:min_len]
                confidences = confidences[:min_len]
            
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
            valid_indices = []
            
            for i, box in enumerate(bbox_xywh):
                x1, y1, x2, y2 = self._xywh_to_xyxy(box)
                # 确保坐标在有效范围内
                if x2 > x1 and y2 > y1:
                    im = ori_img[y1:y2, x1:x2]
                    if im.size > 0:  # 确保图像不为空
                        im_crops.append(im)
                        valid_indices.append(i)
                        
            if im_crops:
                features = feature_extractor(im_crops)
                features = np.asarray(features, dtype=np.float32)
                if features.ndim == 1 and features.size > 0:
                    features = features.reshape(1, -1)

                if features.ndim != 2 or features.shape[0] == 0:
                    dim = self.feature_dim if self.feature_dim is not None else 0
                    return np.zeros((len(bbox_xywh), dim), dtype=np.float32)

                self.feature_dim = features.shape[1]
                    
                # 创建完整的特征数组，无效位置填充零
                full_features = np.zeros((len(bbox_xywh), self.feature_dim), dtype=np.float32)
                for i, valid_idx in enumerate(valid_indices):
                    full_features[valid_idx] = features[i]
                    
                return full_features
            else:
                # 返回与输入数量匹配的零特征
                dim = self.feature_dim if self.feature_dim is not None else 0
                return np.zeros((len(bbox_xywh), dim), dtype=np.float32)
        except Exception as e:
            # 返回与输入数量匹配的零特征
            dim = self.feature_dim if self.feature_dim is not None else 0
            return np.zeros((len(bbox_xywh), dim), dtype=np.float32)