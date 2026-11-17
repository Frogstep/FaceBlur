"""Compact ByteTrack (Zhang et al., 2022): Kalman motion model plus two-stage
IoU association, where low-confidence detections are used only to continue
existing tracks. Gives stable IDs to faces across frames and bridges short
detection gaps.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Kalman:
    """Constant-velocity Kalman filter on (cx, cy, aspect, h)."""

    _std_pos = 1.0 / 20
    _std_vel = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = (np.eye(8) - K @ self.H) @ cov
        return mean, cov


def _xyxy_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def _z_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


class _Track:
    def __init__(self, tid, mean, cov, frame):
        self.id = tid
        self.mean, self.cov = mean, cov
        self.last_frame = frame
        self.hits = 1

    @property
    def box(self):
        return _z_to_xyxy(self.mean)


class ByteTrack:
    def __init__(self, fps: float, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.55, match_iou: float = 0.2,
                 lost_seconds: float = 1.0):
        self.kf = _Kalman()
        self.high, self.low, self.new = high_thresh, low_thresh, new_track_thresh
        self.match_iou = match_iou
        self.max_lost = max(1, int(round(fps * lost_seconds)))
        self.tracks: list[_Track] = []
        self._next_id = 1
        self.frame = -1

    def _associate(self, tracks, boxes, idxs, thr):
        if not tracks or not len(idxs):
            return [], list(range(len(tracks))), list(idxs)
        tb = np.array([t.box for t in tracks])
        iou = iou_matrix(tb, boxes[idxs])
        r, c = linear_sum_assignment(-iou)
        matches, mt, md = [], set(), set()
        for i, j in zip(r, c):
            if iou[i, j] >= thr:
                matches.append((i, idxs[j]))
                mt.add(i)
                md.add(j)
        un_t = [i for i in range(len(tracks)) if i not in mt]
        un_d = [idxs[j] for j in range(len(idxs)) if j not in md]
        return matches, un_t, un_d

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Returns a track id per detection (0 = not tracked)."""
        self.frame += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        ids = np.zeros(len(boxes), np.int64)

        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        hi = [i for i in range(len(boxes)) if scores[i] >= self.high]
        lo = [i for i in range(len(boxes)) if self.low <= scores[i] < self.high]

        m1, un_t, un_hi = self._associate(self.tracks, boxes, hi, self.match_iou)
        rest = [self.tracks[i] for i in un_t]
        m2, _, _ = self._associate(rest, boxes, lo, self.match_iou + 0.2)

        for tracks, matches in ((self.tracks, m1), (rest, m2)):
            for ti, di in matches:
                t = tracks[ti]
                t.mean, t.cov = self.kf.update(t.mean, t.cov, _xyxy_to_z(boxes[di]))
                t.last_frame = self.frame
                t.hits += 1
                ids[di] = t.id

        for di in un_hi:
            if scores[di] >= self.new:
                mean, cov = self.kf.initiate(_xyxy_to_z(boxes[di]))
                t = _Track(self._next_id, mean, cov, self.frame)
                self._next_id += 1
                self.tracks.append(t)
                ids[di] = t.id

        self.tracks = [t for t in self.tracks if self.frame - t.last_frame <= self.max_lost]
        return ids
