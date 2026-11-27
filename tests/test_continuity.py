"""Weak unlabeled detections next to a blurred face: adopted only near the
person's own detections and only when they are about the face's size."""

from __future__ import annotations

import numpy as np

from faceblur.identity import Detections, plan_labeled
from faceblur.review import UNLABELED


def _dets(n: int, items: list[tuple[int, list[float], float]]) -> Detections:
    k = len(items)
    return Detections(num_frames=n,
                      frame=np.array([f for f, _, _ in items], np.int64),
                      box=np.array([b for _, b, _ in items], np.float32).reshape(k, 4),
                      score=np.array([s for _, _, s in items], np.float32),
                      kps=np.zeros((k, 5, 2), np.float32),
                      emb=np.zeros((k, 512), np.float32),
                      track=np.zeros(k, np.int64))


def _plan(n, items, labels):
    dets = _dets(n, items)
    return plan_labeled(dets, np.array(labels, np.int64), 0, 30.0, max_gap_s=1.0, hold_s=0.0)


FACE = [100, 100, 140, 150]


def test_weak_face_sized_box_is_adopted():
    plan = _plan(100, [(10, FACE, 0.8), (20, [102, 101, 142, 151], 0.3)], [0, UNLABELED])
    assert plan.source[20] == "near"


def test_much_larger_weak_box_is_not_adopted():
    # overlaps the face well (IoU > 0.3) but has 2.5x its area: a body, not this face
    big = [90, 90, 155, 168]
    plan = _plan(100, [(10, FACE, 0.8), (20, big, 0.3)], [0, UNLABELED])
    assert plan.source[20] != "near"
    assert not plan.boxes[20] or plan.source[20] == "interp"


def test_adopted_boxes_do_not_extend_the_chain():
    # weak boxes every 25 frames walking away from the only real detection
    # at frame 0: only the first is within a second of it
    items = [(0, FACE, 0.8)] + [(25 * k, FACE, 0.3) for k in range(1, 5)]
    plan = _plan(200, items, [0] + [UNLABELED] * 4)
    assert plan.source[25] == "near"
    assert all(not plan.boxes[f] for f in (50, 75, 100))
