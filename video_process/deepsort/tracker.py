# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories.
    tracks : list[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track states forward in time."""
        for t in self.tracks:
            t.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : list[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        try:
            # Run matching cascade.
            matches, unmatched_tracks, unmatched_detections = \
                self._match(detections)

            # Update track set.
            for track_idx, detection_idx in matches:
                self.tracks[track_idx].update(
                    self.kf, detections[detection_idx])
            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mark_missed()
            for detection_idx in unmatched_detections:
                self._initiate_track(detections[detection_idx])
            self.tracks = [t for t in self.tracks if not t.is_deleted()]

            # Update distance metric.
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
            features, targets = [], []
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                features += track.features
                targets += [track.track_id for _ in track.features]
                track.features = []
            
            if len(features) > 0:
                self.metric.partial_fit(
                    np.asarray(features), np.asarray(targets), active_targets)
            
        except Exception as e:
            print(f"  Tracker.update错误: {e}")
            # 重置tracker状态，避免持续错误
            self.tracks = []
            return []

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            try:
                features = np.array([dets[i].feature for i in detection_indices])
                targets = np.array([tracks[i].track_id for i in track_indices])
                
                # 注意：distance返回的是(features数量, targets数量)，需要转置为(tracks数量, detections数量)
                distance_matrix = self.metric.distance(features, targets)
                
                # 转置以匹配期望的维度：(track_indices数量, detection_indices数量)
                cost_matrix = distance_matrix.T
                
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices)

                return cost_matrix
            except Exception as e:
                print(f"    gated_metric错误: {e}")
                # 返回一个高成本矩阵作为备选
                return np.full((len(track_indices), len(detection_indices)), 999999.0)

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        try:
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(
                    gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks)
        except Exception as e:
            print(f"    _match: matching_cascade错误: {e}")
            matches_a, unmatched_tracks_a, unmatched_detections = [], confirmed_tracks, list(range(len(detections)))

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        try:
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)
        except Exception as e:
            print(f"    _match: IOU匹配错误: {e}")
            matches_b, unmatched_tracks_b = [], iou_track_candidates

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, self.nn_budget))
        self._next_id += 1 