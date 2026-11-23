"""Detection at any angle: faces found in a turned copy of the frame are put
back where they are in the frame.

  .venv/Scripts/python -m pytest tests/test_faces.py -q
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from faceblur.faces import ANY_ANGLE, SCRFD, _ROTATE, _unrotate
from faceblur.project import Edit, Project

W, H = 64, 36


def _spot(img: np.ndarray) -> tuple[float, float]:
    """Center of the lit pixel, in continuous coordinates."""
    y, x = np.argwhere(img > 0)[0]
    return x + 0.5, y + 0.5


@pytest.mark.parametrize("angle", ANY_ANGLE[1:])
def test_points_found_in_a_turned_frame_map_back(angle):
    img = np.zeros((H, W), np.uint8)
    img[7, 50] = 255
    turned = cv2.rotate(img, _ROTATE[angle])
    x, y = _spot(turned)
    boxes = np.array([[x - 3, y - 1, x + 3, y + 1]], np.float32)
    kps = np.array([[[x, y]] * 5], np.float32)
    b, k = _unrotate(boxes, kps, angle, W, H)
    assert np.allclose(k[0, 0], (50.5, 7.5))
    assert np.allclose((b[0, :2] + b[0, 2:]) / 2, (50.5, 7.5))
    assert b[0, 0] < b[0, 2] and b[0, 1] < b[0, 3]
    w, h = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
    assert (w, h) == ((2, 6) if angle in (90, 270) else (6, 2))


class _FakeSCRFD(SCRFD):
    """Finds one "face": the lit pixel, but only when the frame is not turned
    (like the real model, which finds upright faces only)."""

    def __init__(self, upright: tuple[int, int]):
        self.upright = upright

    def detect_raw(self, bgr):
        if bgr.shape[:2] != self.upright:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        x, y = _spot(bgr[..., 0])
        return (np.array([[x - 8, y - 8, x + 8, y + 8]], np.float32), np.array([0.9], np.float32),
                np.array([[[x, y]] * 5], np.float32))


def test_a_face_only_found_turned_is_found_with_any_angle():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[700, 640] = 255  # bottom center, like a face of someone lying down
    det = _FakeSCRFD(upright=(1280, 720))  # only the 90°-turned frame has this shape
    assert det.detect([frame]) == []
    faces = det.detect([frame], rotations=ANY_ANGLE)
    assert len(faces) == 1
    assert np.allclose((faces[0].box[:2] + faces[0].box[2:]) / 2, (640.5, 700.5))


def test_project_keeps_any_angle(tmp_path):
    p = Project.new(tmp_path / "v.mp4")
    p.edit = Edit(any_angle=True)
    assert not p.edit.active  # same frames and pixels as no edit
    p.save()
    assert Project.load(p.path).edit.any_angle
