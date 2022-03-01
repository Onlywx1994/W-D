import numpy as np
from collections import defaultdict


class FtrlEstimator:
    def __init__(self, alpha, beta, L1, L2):

        self._alpha = alpha
        self._beta = beta
        self._L1 = L1
        self._L2 = L2

        self._n = defaultdict(float)
        self._z = defaultdict(float)

        self._w = {}

        self._current_feat_ids = None
        self._current_feat_vals = None

    def predict_logit(self, feature_ids, feature_values):

        self._current_feat_ids = feature_ids
        self._current_feat_vals = feature_values

        logit = 0
        self._w.clear()

        for feat_id, feat_val in zip(feature_ids, feature_values):
            z = self._z[feat_id]
            sign_z = -1. if z < 0 else 1

            if abs(z) > self._L1:
                w = (sign_z * self._L1 - z) / ((self._beta + np.sqrt(self._n[feat_id])) / self._alpha + self._L2)
                self._w[feat_id] = w
                logit += w * feat_val
        return logit

    def update(self, pred_prob, label):
        grad2logit = pred_prob - label
        for feat_id, feat_val in zip(self._current_feat_ids, self._current_feat_vals):
            g = grad2logit * feat_val
            g2 = g * g
            n = self._n[feat_id]
            self._z[feat_id] += g
            if feat_id in self._w:
                sigma = (np.sqrt(n + g2) - np.sqrt(n)) // self._alpha
                self._z[feat_id] -= sigma * self._w[feat_id]

            self._n[feat_id] = n + g2
