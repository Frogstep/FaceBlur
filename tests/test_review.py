"""Blur choice: blurred cards are linked into masking tracks by position and
time only, and older projects (people, "someone else", "not a face") load as
blur / not blur.

  .venv/Scripts/python -m pytest tests/test_review.py -q
"""

from __future__ import annotations

import numpy as np

from faceblur.identity import Detections
from faceblur.review import BLUR, NOT_A_FACE, SOMEONE_ELSE, UNLABELED, build_cards, chain_cards, is_blur

FPS = 25.0


def dets_from_tracks(tracks: dict[int, tuple[range, list[float]]], n: int = 200) -> Detections:
    """tracks: tracker id -> (frames, box)."""
    rng = np.random.default_rng(1)
    fr, bx, tr = [], [], []
    for t, (frames, box) in tracks.items():
        for f in frames:
            fr.append(f)
            bx.append(box)
            tr.append(t)
    m = len(fr)
    emb = rng.normal(size=(m, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return Detections(np.array(fr), np.array(bx, np.float32), np.full(m, 0.9, np.float32),
                      np.zeros((m, 5, 2), np.float32), emb, np.array(tr), n)


def card_of_track(dets, cards, t: int) -> int:
    return next(k for k, c in enumerate(cards) if c.track == t)


def test_broken_track_of_one_face_is_linked():
    # the tracker lost the face for 5 frames, same place: one masking track
    dets = dets_from_tracks({1: (range(0, 40), [100, 100, 160, 160]), 2: (range(45, 90), [104, 102, 164, 162])})
    cards = build_cards(dets)
    a, b = card_of_track(dets, cards, 1), card_of_track(dets, cards, 2)
    assert chain_cards(dets, cards, [a, b], FPS) == [[a, b]]


def test_faces_on_screen_together_are_never_linked():
    dets = dets_from_tracks({1: (range(0, 60), [100, 100, 160, 160]), 2: (range(10, 70), [400, 100, 460, 160])})
    cards = build_cards(dets)
    a, b = card_of_track(dets, cards, 1), card_of_track(dets, cards, 2)
    assert sorted(map(sorted, chain_cards(dets, cards, [a, b], FPS))) == [[a], [b]]


def test_far_away_or_much_later_is_another_track():
    dets = dets_from_tracks({
        1: (range(0, 40), [100, 100, 160, 160]),
        2: (range(45, 80), [500, 300, 560, 360]),  # soon, but elsewhere
        3: (range(140, 180), [100, 100, 160, 160]),  # same place, 2.4 s later
    })
    cards = build_cards(dets)
    ks = [card_of_track(dets, cards, t) for t in (1, 2, 3)]
    assert len(chain_cards(dets, cards, ks, FPS)) == 3


def test_older_project_labels_mean_blur_or_not():
    assert is_blur(BLUR) and is_blur(2)  # person 3 of an older project
    assert not any(is_blur(x) for x in (SOMEONE_ELSE, NOT_A_FACE, UNLABELED))


def test_plans_one_per_track_and_drawn_box(tmp_path):
    from fractions import Fraction

    from faceblur.pipeline import Analysis
    from faceblur.project import Manual, Project
    from faceblur.video_io import VideoInfo

    dets = dets_from_tracks({
        1: (range(0, 40), [100, 100, 160, 160]),
        2: (range(45, 90), [104, 102, 164, 162]),  # continues track 1
        3: (range(0, 90), [400, 100, 460, 160]),  # another face, not blurred
    })
    info = VideoInfo(tmp_path / "v.mp4", 640, 480, Fraction(25), 200, 8.0, False, "bt601", "tv", False)
    a = Analysis(tmp_path / "v.mp4", info, dets, wd=tmp_path / "work")
    project = Project(video=str(tmp_path / "v.mp4"))
    assert a.blurred(project) == [False, False, False]
    assert a.plans(project) == ([], [])
    a.set_blur(project, [card_of_track(dets, a.cards, 1), card_of_track(dets, a.cards, 2)], True)
    project.manual.append(Manual(150, [10, 10, 50, 50]))
    inputs, plans = a.plans(project)
    assert len(plans) == 2  # the linked face + the drawn box
    face = plans[0]
    assert all(face.boxes[f] for f in range(0, 90))  # the 5-frame gap is bridged
    assert not any(face.boxes[f] for f in range(96, 200))  # after a 0.25 s hold
    # the visible face is never in the blurred face's boxes
    assert all(not any(b[0] > 300 for b in face.boxes[f]) for f in range(90))
    assert plans[1].source[150] == "manual" and not plans[1].boxes[0]
    # switching a card back
    a.set_blur(project, [card_of_track(dets, a.cards, 2)], False)
    assert a.blurred(project).count(True) == 1
