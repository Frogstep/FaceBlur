"""SAM 2 video segmentation of the target face, in chunks so long videos fit in
memory. ArcFace-confirmed boxes from the identity stage are used as prompts
(re-acquiring the face after it leaves and returns); the last mask of each
chunk seeds the next one.
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


# (config, checkpoint) per model size
MODELS = {
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
}


def model_for(device: str) -> str:
    """Large on the GPU. On the CPU the large model takes seconds per frame, so
    the tiny one is used when it has been downloaded."""
    if device == "cpu" and (MODELS_DIR / MODELS["tiny"][1]).exists():
        return "tiny"
    return "large"


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
    def __init__(self, predictor, chunk: int = 120, reprompt_every: int = 10, cancel=None):
        self.p = predictor
        self.chunk = chunk
        self.reprompt_every = reprompt_every
        self.cancel = cancel or (lambda: False)

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

    def _track_person(self, state, plan: TargetPlan, start: int, end: int,
                      carry: np.ndarray | None, masks: list[np.ndarray]) -> None:
        """Runs SAM 2 for one person over the chunk already loaded in `state`."""
        h, w = masks[0].shape
        prompts = self._prompt_frames(plan, start, min(end, len(plan.boxes)))
        if not prompts and carry is None:
            return
        self.p.reset_state(state)
        use_carry = carry is not None and start not in prompts
        if use_carry:
            self.p.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(carry))
        for f in prompts:
            box = _pad_box(plan.strong[f], 0.1, w, h)
            self.p.add_new_points_or_box(state, frame_idx=f - start, obj_id=1, box=box)
        first = 0 if use_carry else prompts[0] - start
        for rev in (False, True):
            if rev and first == 0:
                continue
            for fi, _, logits in self.p.propagate_in_video(state, start_frame_idx=first, reverse=rev):
                if self.cancel():
                    from .identity import Cancelled

                    raise Cancelled()
                masks[fi] = (logits[0, 0] > 0).cpu().numpy()

    @torch.inference_mode()
    def run(self, frames: Iterable[np.ndarray], plans: list[TargetPlan]) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        """Tracks every person in `plans` separately and yields
        (frame, union of cleaned masks, status) for every frame in order."""
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
            needed = [bool(self._prompt_frames(p, start, min(end, n_total))) or c is not None
                      for p, c in zip(plans, carry)]
            if any(needed):
                with _autocast(self.p):
                    state = _init_state(self.p, _prep(buf, self.p.image_size), h, w)
                    for pi, plan in enumerate(plans):
                        if needed[pi]:
                            self._track_person(state, plan, start, end, carry[pi], per_person[pi])
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
