"""Builds the final per-frame mask (SAM 2 mask plus padded ellipses from the
target boxes) and applies an irreversible blur / pixelate / fill."""

from __future__ import annotations

import cv2
import numpy as np


def ellipse_mask(shape, boxes, pad: float = 0.3) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - bh * 0.05  # shift up a bit for forehead/hair
        ax, ay = bw * (1 + pad) / 2, bh * (1 + pad) / 2 * 1.12
        cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 255, -1)
    return m > 0


def combine(shape, sam_mask: np.ndarray | None, boxes, pad: float = 0.3) -> np.ndarray:
    m = ellipse_mask(shape, boxes, pad)
    if sam_mask is not None and sam_mask.any():
        ys, xs = np.nonzero(sam_mask)
        size = max(ys.max() - ys.min(), xs.max() - xs.min(), 8)
        k = max(3, int(size * 0.08)) | 1  # grow SAM's edge slightly so no skin peeks out
        grown = cv2.dilate(sam_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m |= grown > 0
    return m


def apply(frame: np.ndarray, mask: np.ndarray, style: str = "pixelate", strength: float = 1.0) -> np.ndarray:
    if not mask.any():
        return frame
    out = frame.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        size = max(bw, bh)
        feather = max(2, int(size * 0.06))
        x1, y1 = max(0, x - 2 * feather), max(0, y - 2 * feather)
        x2, y2 = min(w, x + bw + 2 * feather), min(h, y + bh + 2 * feather)
        roi = frame[y1:y2, x1:x2]
        comp = (labels[y1:y2, x1:x2] == i).astype(np.uint8)

        if style == "fill":
            eff = np.zeros_like(roi)
        else:
            # ~8 blocks across the face: far below what identity recognition needs
            block = max(4, int(size / (8 / strength)))
            sw, sh = max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
            eff = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            if style == "blur":
                sigma = max(2.0, block * 0.8)
                eff = cv2.GaussianBlur(eff, (0, 0), sigma)

        # feather outward only: grow first, then soften, so the core stays fully covered
        grown = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1,) * 2))
        alpha = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), feather / 2)
        alpha = np.maximum(alpha, comp.astype(np.float32))[..., None]
        out[y1:y2, x1:x2] = (eff * alpha + out[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)
    return out


def draw_debug(frame: np.ndarray, sam_mask, plan_boxes, others, status: str, source: str) -> np.ndarray:
    img = frame.copy()
    if sam_mask is not None and sam_mask.any():
        cnts, _ = cv2.findContours(sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    for b in plan_boxes:
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 255), 2)
    for b in others:
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 1)
    cv2.putText(img, f"sam:{status} box:{source or '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img
