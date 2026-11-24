"""Drawn boxes: forward-only duration, older ±span boxes, "until the next
blurred part"."""

from __future__ import annotations

import json

import numpy as np

from faceblur.identity import Detections, ManualBox, next_blurred_frame, plan_labeled
from faceblur.project import Manual, Project


def test_new_box_reaches_forward_only():
    m = Manual(100, [0, 0, 10, 10], length=25)
    assert (m.first, m.last) == (100, 124)
    assert Manual(100, [0, 0, 10, 10], length=1).last == 100


def test_old_box_keeps_its_span():
    m = Manual(100, [0, 0, 10, 10], span=15)
    assert (m.first, m.last) == (85, 115)


def test_project_round_trip_and_old_files(tmp_path):
    video = tmp_path / "v.mp4"
    p = Project.new(video)
    p.manual = [Manual(10, [1, 2, 3, 4], length=40)]
    path = p.save()
    back = Project.load(path).manual[0]
    assert (back.first, back.last, back.span) == (10, 49, None)

    # a version 4 file: boxes reached span frames both ways (default 15)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["version"] = 4
    d["manual"] = [{"frame": 50, "box": [1, 2, 3, 4], "person": 0, "span": 7},
                   {"frame": 80, "box": [1, 2, 3, 4], "person": 0}]
    path.write_text(json.dumps(d), encoding="utf-8")
    old = Project.load(path).manual
    assert [(m.first, m.last) for m in old] == [(43, 57), (65, 95)]


def test_next_blurred_frame():
    lv = np.array([2, 2, 0, 0, 0, 1, 2, 0, 0], np.int8)
    assert next_blurred_frame(lv, 2) == 5  # in a gap: where it starts again
    assert next_blurred_frame(lv, 4) == 5
    assert next_blurred_frame(lv, 0) == 5  # on a blurred frame: after the gap that follows
    assert next_blurred_frame(lv, 7) is None  # nothing later
    assert next_blurred_frame(np.zeros(4), 0) is None


def test_plan_masks_only_the_box_frames():
    n = 30
    dets = Detections(num_frames=n, frame=np.zeros(0, np.int64), box=np.zeros((0, 4), np.float32),
                      score=np.zeros(0, np.float32), kps=np.zeros((0, 5, 2), np.float32),
                      emb=np.zeros((0, 512), np.float32), track=np.zeros(0, np.int64))
    box = ManualBox(10, np.array([0, 0, 20, 20], np.float32), 10, 14)
    plan = plan_labeled(dets, np.zeros(0, np.int64), 0, 30.0, manual=[box])
    covered = [f for f in range(n) if plan.boxes[f]]
    assert covered == [10, 11, 12, 13, 14]
    assert plan.strong[10] is not None
