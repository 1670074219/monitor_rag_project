# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        An NxL matrix where the entry (i, j) is the squared distance between
        the i-th sample in `a` and the j-th sample in `b`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in `a` and `b` are unit length vectors.
        Otherwise, a normalization step is performed before computing the
        cosine distance.

    Returns
    -------
    ndarray
        An NxL matrix where the entry (i, j) is the cosine distance between
        the i-th sample in `a` and the j-th sample in `b`.

    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    
    if not data_is_normalized:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a = a / (a_norm + 1e-12)  # 添加小值避免除零
        b = b / (b_norm + 1e-12)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each test sample, searches
    through the gallery for the closest sample.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold : float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fixes the gallery size at this value.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the gallery with new features.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of N targets associated with the features.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            # 确保 feature 是 numpy 数组
            feature = np.asarray(feature, dtype=np.float32)
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the features against.

        Returns
        -------
        ndarray
            Cost matrix of shape len(features), len(targets), where entry (i, j)
            contains the closest squared Euclidean distance between `features[i]`
            and `targets[j]`.

        """
        cost_matrix = np.zeros((len(features), len(targets)))
        for i, feat in enumerate(features):
            for j, target in enumerate(targets):
                if target in self.samples and len(self.samples[target]) > 0:
                    gallery_features = np.array(self.samples[target])
                    query_feature = np.array([feat])
                    distances = self._metric(gallery_features, query_feature)
                    cost_matrix[i, j] = np.min(distances)
                else:
                    cost_matrix[i, j] = float('inf')
        return cost_matrix

    def _metric(self, gallery, query):
        """
        Helper function to compute the distance for a single query.
        This function is redefined in the constructor based on the metric.
        """
        # This will be replaced by _nn_euclidean_distance or _nn_cosine_distance
        pass 