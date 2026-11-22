"""SAM 2 video segmentation of the blurred faces, in chunks so long videos fit
in memory. The faces' confident boxes are used as prompts (re-acquiring a face
after it leaves and returns); the last mask of each chunk seeds the next one.
All faces share one SAM 2 state per chunk, so each frame is encoded once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import torch

from .identity import TargetPlan

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# (config, checkpoint) per model size, fastest first
MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
CHOICES = ("auto", *MODELS)  # Settings.sam_model


def model_for(device: str, choice: str = "auto") -> str:
    """The model size to use. "auto": large on the GPU; on the CPU the large
    model takes seconds per frame, so the tiny one when it has been downloaded."""
    if choice in MODELS:
        return choice
    if device == "cpu" and is_downloaded("tiny"):
        return "tiny"
    return "large"


def is_downloaded(size: str) -> bool:
    return (MODELS_DIR / MODELS[size][1]).exists()


def load_predictor(device: str = "cuda", size: str | None = None):
    import warnings

    import sam2.sam2_video_predictor as svp
    from sam2.build_sam import build_sam2_video_predictor

    svp.tqdm = lambda x, **_: x  # silence SAM 2's per-chunk progress bars
    # the optional compiled hole-filling extension is not built; results are unaffected
    warnings.filterwarnings("ignore", message="(?s).*post-processing step.*")
    cfg, ckpt = MODELS[size or model_for(device)]
    if device == "cpu":
        import os

        torch.set_num_threads(os.cpu_count() or 4)
    return build_sam2_video_predictor(cfg, str(MODELS_DIR / ckpt), device=device)


def _autocast(predictor):
    """bfloat16 on the GPU (faster, as SAM 2 recommends); full precision on the CPU."""
    import contextlib

    if predictor.device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _prep(frames: list[np.ndarray], size: int) -> torch.Tensor:
    out = torch.empty(len(frames), 3, size, size, dtype=torch.float32)
    for i, f in enumerate(frames):
        rgb = cv2.cvtColor(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        out[i] = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    out -= _MEAN
    out /= _STD
    return out


def _init_state(predictor, images: torch.Tensor, h: int, w: int):
    import sam2.sam2_video_predictor as svp

    orig = svp.load_video_frames
    svp.load_video_frames = lambda **_: (images, h, w)
    try:
        return predictor.init_state(video_path="<memory>", offload_video_to_cpu=True)
    finally:
        svp.load_video_frames = orig


def _pad_box(b: np.ndarray, pad: float, w: int, h: int) -> np.ndarray:
    bw, bh = b[2] - b[0], b[3] - b[1]
    out = np.array([b[0] - bw * pad, b[1] - bh * pad * 1.3, b[2] + bw * pad, b[3] + bh * pad])
    return np.clip(out, 0, [w - 1, h - 1, w - 1, h - 1])


def _mask_bbox(m: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def validate(mask: np.ndarray, plan: TargetPlan, f: int) -> tuple[np.ndarray, str]:
    """Keeps SAM 2 from spreading beyond the face or jumping to another person.
    Returns the cleaned mask and a status string for the debug report."""
    h, w = mask.shape
    if not mask.any():
        return mask, "empty"
    boxes = plan.boxes[f]
    if boxes:
        # clip to a generous region around the known target boxes
        region = np.zeros_like(mask)
        for b in boxes:
            x1, y1, x2, y2 = _pad_box(b, 0.6, w, h).astype(int)
            region[y1:y2, x1:x2] = True
        clipped = mask & region
        return clipped, "clipped" if clipped.sum() < mask.sum() * 0.98 else "ok"
    # no box from detection or bridging: trust SAM 2, but not if it is huge
    # or sitting on a face that is confidently someone else
    if mask.mean() > 0.10:
        return np.zeros_like(mask), "rejected-large"
    mb = _mask_bbox(mask)
    for ob in plan.others[f]:
        if _iou(mb, ob) > 0.3:
            return np.zeros_like(mask), "rejected-other"
    return mask, "sam-only"


class Segmenter:
    """Tracks every person of the plans with SAM 2 in one inference state per
    chunk, so each frame goes through the image encoder once however many
    people are blurred. Each person is tracked only over their own frames:
    from their first prompt (or the mask carried over from the previous chunk)
    until SAM 2 has lost the face for `lost` frames after their last box, and
    backwards from their first prompt the same way."""

    def __init__(self, predictor, chunk: int = 120, reprompt_every: int = 10, lost: int = 10, cancel=None):
        self.p = predictor
        self.chunk = chunk
        self.reprompt_every = reprompt_every
        self.lost = lost
        self.cancel = cancel or (lambda: False)
        self.frames_encoded = 0  # statistics: frames that went through SAM 2
        self.steps = 0  # person-frames tracked

    def _prompt_frames(self, plan: TargetPlan, start: int, end: int) -> list[int]:
        """Frames inside [start, end) to prompt with a confident box: the first
        confident frame of each run of confident frames, then every N frames."""
        out, last = [], -10**9
        prev = False
        for f in range(start, end):
            ok = plan.strong[f] is not None
            if ok and (not prev or f - last >= self.reprompt_every):
                out.append(f)
                last = f
            prev = ok
        return out

    def _step(self, state, obj_idx: int, fi: int, reverse: bool) -> np.ndarray:
        """One SAM 2 step of one object on one frame: the body of SAM 2's
        propagate_in_video for a single object (sam2 is pinned to one commit
        in the installers). Image features are cached for the latest frame,
        so the other objects on the same frame reuse them."""
        d = state["output_dict_per_obj"][obj_idx]
        if fi in d["cond_frame_outputs"]:
            pred = d["cond_frame_outputs"][fi]["pred_masks"].to(state["device"], non_blocking=True)
        else:
            out, pred = self.p._run_single_frame_inference(
                inference_state=state, output_dict=d, frame_idx=fi, batch_size=1, is_init_cond_frame=False,
                point_inputs=None, mask_inputs=None, reverse=reverse, run_mem_encoder=True)
            d["non_cond_frame_outputs"][fi] = out
        state["frames_tracked_per_obj"][obj_idx][fi] = {"reverse": reverse}
        _, logits = self.p._get_orig_video_res_output(state, pred)
        self.steps += 1
        return (logits[0, 0] > 0).cpu().numpy()

    def _track_chunk(self, state, plans: list[TargetPlan], start: int, end: int,
                     carry: list[np.ndarray | None], masks: list[list[np.ndarray]]) -> None:
        """Runs SAM 2 for everyone over the chunk already loaded in `state`;
        fills masks[person][frame in the chunk]."""
        h, w = masks[0][0].shape
        n = end - start
        n_plan = min(end, len(plans[0].boxes))
        # per tracked person: (person, SAM 2 object index, first tracked frame,
        # first and last frame with a box); frames relative to the chunk
        objs: list[tuple[int, int, int, int, int]] = []
        for pi, plan in enumerate(plans):
            prompts = self._prompt_frames(plan, start, n_plan)
            if not prompts and carry[pi] is None:
                continue
            obj_idx = self.p._obj_id_to_idx(state, pi + 1)
            use_carry = carry[pi] is not None and start not in prompts
            if use_carry:
                self.p.add_new_mask(state, frame_idx=0, obj_id=pi + 1, mask=torch.from_numpy(carry[pi]))
            for f in prompts:
                box = _pad_box(plan.strong[f], 0.1, w, h)
                self.p.add_new_points_or_box(state, frame_idx=f - start, obj_id=pi + 1, box=box)
            first = 0 if use_carry else prompts[0] - start
            covered = [f - start for f in range(start, n_plan) if plan.boxes[f]]
            objs.append((pi, obj_idx, first, covered[0] if covered else first, covered[-1] if covered else first))
        if not objs:
            return
        self.p.propagate_in_video_preflight(state)
        encoded: set[int] = set()

        def track(order, reverse: bool, group) -> None:
            """Tracks `group` frame by frame in `order`. A person stops for good
            once SAM 2 has lost them for `lost` frames beyond their boxes."""
            empty = {o[0]: 0 for o in group}
            stopped: set[int] = set()
            for fi in order:
                if len(stopped) == len(group):
                    return
                todo = [o for o in group if o[0] not in stopped and (fi < o[2] if reverse else fi >= o[2])]
                if not todo:
                    continue
                if self.cancel():
                    from .identity import Cancelled

                    raise Cancelled()
                encoded.add(fi)
                for pi, obj_idx, _first, cov0, cov1 in todo:
                    m = self._step(state, obj_idx, fi, reverse)
                    masks[pi][fi] = m
                    beyond = fi < cov0 if reverse else fi > cov1
                    empty[pi] = empty[pi] + 1 if beyond and not m.any() else 0
                    if empty[pi] >= self.lost:
                        stopped.add(pi)

        # forward from each person's first prompt (or carried mask) ...
        track(range(min(o[2] for o in objs), n), False, objs)
        # ... and backwards from it, for people who first appear inside this chunk
        back = [o for o in objs if o[2] > 0]
        if back:
            track(range(max(o[2] for o in back) - 1, -1, -1), True, back)
        self.frames_encoded += len(encoded)

    @torch.inference_mode()
    def run(self, frames: Iterable[np.ndarray], plans: list[TargetPlan]) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Tracks every person in `plans` and yields (frame, union of cleaned
        masks, status) for every frame in order."""
        it = iter(frames)
        start = 0
        carry: list[np.ndarray | None] = [None] * len(plans)  # last valid mask per person
        n_total = len(plans[0].boxes)
        while True:
            buf = []
            for fr in it:
                buf.append(fr)
                if len(buf) == self.chunk:
                    break
            if not buf:
                return
            end = start + len(buf)
            h, w = buf[0].shape[:2]
            per_person = [[np.zeros((h, w), bool) for _ in buf] for _ in plans]
            needed = any(bool(self._prompt_frames(p, start, min(end, n_total))) or c is not None
                         for p, c in zip(plans, carry))
            if needed:
                with _autocast(self.p):
                    state = _init_state(self.p, _prep(buf, self.p.image_size), h, w)
                    self._track_chunk(state, plans, start, end, carry, per_person)
                    del state
                torch.cuda.empty_cache()
            for k, fr in enumerate(buf):
                f = start + k
                union = np.zeros((h, w), bool)
                statuses = []
                for pi, plan in enumerate(plans):
                    m = per_person[pi][k]
                    cm, status = validate(m, plan, f) if f < n_total else (m, "ok")
                    union |= cm
                    statuses.append(status)
                    if k == len(buf) - 1:
                        carry[pi] = cm if cm.any() else None
                yield fr, union, "+".join(statuses)
            start = end
