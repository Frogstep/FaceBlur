"""SAM 2 tracking ranges, with a fake predictor (no model, no GPU): every
blurred face shares one state per chunk, is tracked from its first prompt
(forwards and backwards) and stops once SAM 2 has lost it beyond its boxes.

  .venv/Scripts/python -m pytest tests/test_segment.py -q

Needs PyTorch (segment.py imports it), so it is skipped in CI, which runs without it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from faceblur import segment  # noqa: E402
from faceblur.identity import TargetPlan  # noqa: E402
from faceblur.segment import Segmenter  # noqa: E402

H = W = 32
N = 60


class FakeSam:
    """Stands in for SAM2VideoPredictor: object `obj_id` is "seen" on the
    frames in visible[obj_id]; records every tracking step."""

    image_size = 16

    def __init__(self, visible: dict[int, range], chunk: int = N):
        self.visible = visible  # absolute frame numbers
        self.chunk = chunk
        self.start = -chunk  # first frame of the current chunk
        self.steps: list[tuple[int, int, bool]] = []  # (obj_id, frame, reverse)
        self.device = torch.device("cpu")

    def init_state(self) -> dict:
        self.start += self.chunk
        return {"output_dict_per_obj": {}, "frames_tracked_per_obj": {}, "ids": [], "device": "cpu"}

    def _obj_id_to_idx(self, state, obj_id):
        if obj_id not in state["ids"]:
            state["ids"].append(obj_id)
            k = len(state["ids"]) - 1
            state["output_dict_per_obj"][k] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}, "id": obj_id}
            state["frames_tracked_per_obj"][k] = {}
        return state["ids"].index(obj_id)

    def _mask(self, obj_id, f):
        m = torch.full((1, 1, H, W), -1.0)
        if self.start + f in self.visible[obj_id]:
            m[..., 4:12, 4:12] = 1.0
        return m

    def _cond(self, state, frame_idx, obj_id):
        d = state["output_dict_per_obj"][self._obj_id_to_idx(state, obj_id)]
        d["cond_frame_outputs"][frame_idx] = {"pred_masks": self._mask(obj_id, frame_idx)}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self._cond(state, frame_idx, obj_id)

    def add_new_points_or_box(self, state, frame_idx, obj_id, box):
        self._cond(state, frame_idx, obj_id)

    def propagate_in_video_preflight(self, state):
        pass

    def _run_single_frame_inference(self, *, inference_state, output_dict, frame_idx, reverse, **_):
        self.steps.append((output_dict["id"], self.start + frame_idx, reverse))
        return {}, self._mask(output_dict["id"], frame_idx)

    def _get_orig_video_res_output(self, state, pred):
        return None, pred


def plan(box_frames: range, n: int = N) -> TargetPlan:
    box = np.array([4, 4, 12, 12], np.float32)
    boxes = [[box] if f in box_frames else [] for f in range(n)]
    strong = [box if f in box_frames else None for f in range(n)]
    return TargetPlan(boxes, strong, ["det" if b else "" for b in boxes], [[] for _ in range(n)],
                      np.zeros(0, np.float32))


@pytest.fixture(autouse=True)
def fake_state(monkeypatch):
    monkeypatch.setattr(segment, "_init_state", lambda p, images, h, w: p.init_state())
    monkeypatch.setattr(segment, "_prep", lambda frames, size: None)


def frames(n: int = N):
    return (np.zeros((H, W, 3), np.uint8) for _ in range(n))


def test_each_face_is_tracked_only_over_its_own_frames():
    # A has boxes on 20..29 and SAM 2 sees it on 15..34; B is on screen throughout
    sam = FakeSam({1: range(15, 35), 2: range(0, N)})
    out = list(Segmenter(sam, chunk=N, lost=5).run(frames(), [plan(range(20, 30)), plan(range(0, N))]))
    a_fwd = sorted(f for o, f, r in sam.steps if o == 1 and not r)
    a_back = sorted(f for o, f, r in sam.steps if o == 1 and r)
    # forwards: from the first prompt until lost for 5 frames after the face left (35..39)
    assert a_fwd and a_fwd[0] > 20 and a_fwd[-1] == 39
    # backwards: from the first prompt down to the start of the chunk (lost only on 14..10 -> stops at 10)
    assert a_back == list(range(10, 20))
    # B starts on frame 0: forwards only, every frame
    assert not any(r for o, f, r in sam.steps if o == 2)
    # the union of the masks covers A where SAM 2 sees it, B everywhere
    assert all(m.any() for _, m, _ in out)
    assert len(out) == N


def test_chunk_without_prompts_or_carry_is_skipped():
    sam = FakeSam({1: range(0, 0)})
    out = list(Segmenter(sam, chunk=30, lost=5).run(frames(), [plan(range(0, 0))]))
    assert len(out) == N and not sam.steps and not any(m.any() for _, m, _ in out)


def test_mask_is_carried_into_the_next_chunk():
    # boxes only in the first chunk; SAM 2 keeps seeing the face 5 frames into the next
    sam = FakeSam({1: range(0, 35)}, chunk=30)
    out = list(Segmenter(sam, chunk=30, lost=3).run(frames(), [plan(range(0, 30))]))
    steps = [f for _, f, _ in sam.steps]
    # chunk 1: every frame but the prompts (0, then every 10 frames)
    assert [f for f in steps if f < 30] == [f for f in range(30) if f % 10]
    # chunk 2 starts from the carried mask on 30 and stops once lost on 35..37
    assert [f for f in steps if f >= 30] == list(range(31, 38))
    assert out[34][1].any() and not out[35][1].any()
