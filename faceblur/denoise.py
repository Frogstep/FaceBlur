"""Non-local means denoising for very dark frames (detection copy only).

Filtered at the face detector's input resolution: the detector shrinks every
frame to DETECT_SIZE anyway, so filtering more pixels than that is wasted
work (about 2x less for 1080p, 9x less for 4K). Measured on 300 frames of a
very dark video against full-resolution filtering: 99% of the confident face
matches are kept, and 13% more are found.

A PyTorch GPU version of the same filter was tried: it found fewer faces and
competed with face detection for the GPU, so the OpenCV CPU filter is used on
every machine (it runs in parallel worker threads, see FaceAnalyzer.analyze_stream).
"""

from __future__ import annotations

import cv2
import numpy as np

H = 5  # filter strength for luminance and colour (OpenCV h / hColor)
TEMPLATE = 5  # patch size
SEARCH = 15  # search window size
DETECT_SIZE = 1280  # the face detector's input size (long side)


def denoise(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = DETECT_SIZE / max(h, w)
    if scale >= 0.95:
        return cv2.fastNlMeansDenoisingColored(bgr, None, H, H, TEMPLATE, SEARCH)
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.fastNlMeansDenoisingColored(small, None, H, H, TEMPLATE, SEARCH)
    # back to frame size: face embeddings are cut from this image at full resolution
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
