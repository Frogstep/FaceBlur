"""JPEG output for the API: thumbnails and video frames with face-card boxes."""

from __future__ import annotations

import cv2
import numpy as np

# BGR; the same colours as the web app's cards
BLUR = (60, 60, 235)  # red: this face is blurred
KEEP = (90, 200, 60)  # green: this face stays visible


def jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def resize_to_width(img: np.ndarray, width: int | None) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if not width or width >= w:
        return img, 1.0
    s = width / w
    return cv2.resize(img, (width, max(1, round(h * s))), interpolation=cv2.INTER_AREA), s


def draw_cards(frame: np.ndarray, analysis, blurred: list[bool], f: int, width: int | None) -> np.ndarray:
    """The frame scaled to `width`, with every detection of a face card boxed and
    numbered (#card): red when the card is blurred, green when it stays visible."""
    img, s = resize_to_width(frame, width)
    img = img.copy()
    thick = max(1, round(img.shape[1] / 500))
    for i in analysis.detections_in(f):
        k = int(analysis.card_of_det[i])
        if k < 0:
            continue
        x1, y1, x2, y2 = (analysis.dets.box[i] * s).astype(int)
        color = BLUR if blurred[k] else KEEP
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick * (2 if blurred[k] else 1))
        text = f"#{analysis.cards[k].number}"
        scale = max(0.5, img.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 4, ty + 3), color, -1)
        cv2.putText(img, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    return img
