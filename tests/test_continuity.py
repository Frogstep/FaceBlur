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
    return plan_labeled(dets, np.array(labels, np.int64), 0, 30.0, max_gap_s=1.0, hold_s=0.0,
                        frame_size=(400, 300))


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


def test_body_sized_box_in_the_faces_own_track_is_dropped():
    # the face for 20 frames, then the tracker ties a box 6x its area to it
    items = [(f, FACE, 0.5) for f in range(20)] + [(25, [60, 60, 180, 210], 0.4)]
    plan = _plan(100, items, [0] * 21)
    assert not plan.boxes[25]
    assert all(plan.source[f] != "interp" for f in range(20, 25))


def test_long_run_of_body_sized_boxes_is_dropped():
    # face for 3 s, then the track jumps to a body-sized box for 1.5 s
    items = [(f, FACE, 0.5) for f in range(90)] + [(f, [60, 60, 180, 210], 0.4) for f in range(90, 135)]
    plan = _plan(200, items, [0] * 135)
    assert all(not plan.boxes[f] or plan.source[f] == "hold" for f in range(90, 135))


def test_growing_face_is_kept():
    # a face that grows 5x in area over two seconds stays blurred all the way
    items = []
    for f in range(61):
        s = 40 * (1 + 1.25 * f / 60)
        items.append((f, [100, 100, 100 + s, 100 + 1.2 * s], 0.6))
    plan = _plan(90, items, [0] * 61)
    assert all(plan.source[f] == "det" for f in range(61))


def test_hold_copies_the_nearest_box():
    a, b = [0, 0, 40, 50], [200, 0, 240, 50]
    dets = _dets(40, [(10, a, 0.8), (12, b, 0.8)])
    plan = plan_labeled(dets, np.array([0, 0], np.int64), 0, 30.0, max_gap_s=0.0, hold_s=0.25)
    assert list(plan.boxes[13][0]) == b
    assert list(plan.boxes[9][0]) == a


def test_loose_box_around_a_small_face_is_kept():
    # 3.2x the face's area but only 1.3% of the frame: kept (over-blurring is safe)
    items = [(f, FACE, 0.5) for f in range(20)] + [(25, [80, 80, 160, 160], 0.4)]
    dets = _dets(100, items)
    plan = plan_labeled(dets, np.zeros(21, np.int64), 0, 30.0, max_gap_s=1.0, hold_s=0.0,
                        frame_size=(800, 600))
    assert plan.source[25] == "det"


def test_adopted_boxes_do_not_extend_the_chain():
    # weak boxes every 25 frames walking away from the only real detection
    # at frame 0: only the first is within a second of it
    items = [(0, FACE, 0.8)] + [(25 * k, FACE, 0.3) for k in range(1, 5)]
    plan = _plan(200, items, [0] + [UNLABELED] * 4)
    assert plan.source[25] == "near"
    assert all(not plan.boxes[f] for f in (50, 75, 100))
