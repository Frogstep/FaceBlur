"""Detection scan (cached, with face tracks) and per-frame target selection
from the user's labels, with continuity, gap bridging and drawn boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .bytetrack import ByteTrack
from .faces import FaceAnalyzer
from .video_io import FrameReader, VideoInfo


@dataclass
class Detections:
    frame: np.ndarray  # (N,) int
    box: np.ndarray  # (N,4) float32
    score: np.ndarray  # (N,)
    kps: np.ndarray  # (N,5,2)
    emb: np.ndarray  # (N,512)
    track: np.ndarray  # (N,) int, 0 = untracked
    num_frames: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, frame=self.frame, box=self.box, score=self.score, kps=self.kps,
                            emb=self.emb, track=self.track, num_frames=self.num_frames)

    @classmethod
    def load(cls, path: Path) -> "Detections":
        z = np.load(path)
        return cls(z["frame"], z["box"], z["score"], z["kps"], z["emb"], z["track"], int(z["num_frames"]))

    def quality(self) -> np.ndarray:
        """Detection quality used to weight embeddings: confident and large faces count more."""
        h = self.box[:, 3] - self.box[:, 1]
        return self.score * np.clip(h / 80.0, 0.1, 1.0)

    def by_frame(self) -> list[np.ndarray]:
        order = np.argsort(self.frame, kind="stable")
        bounds = np.searchsorted(self.frame[order], np.arange(self.num_frames + 1))
        return [order[bounds[i]:bounds[i + 1]] for i in range(self.num_frames)]


class Cancelled(Exception):
    pass


def scan(info: VideoInfo, analyzer: FaceAnalyzer, cache: Path | None = None,
         desc: str = "detecting faces", progress=None, cancel=None, any_angle: bool = False) -> Detections:
    """progress(done, total) and cancel() -> bool are optional hooks for the GUI;
    without progress a console progress bar is shown."""
    if cache is not None and cache.exists():
        return Detections.load(cache)
    tracker = ByteTrack(info.fps_float)
    fr, bx, sc, kp, em, tr = [], [], [], [], [], []
    n = 0
    stream = analyzer.analyze_stream(FrameReader(info), any_angle=any_angle)
    results = stream if progress is not None else tqdm(stream, total=info.num_frames, desc=desc, unit="f")
    for i, (_frame, faces) in enumerate(results):
        if cancel and cancel():
            stream.close()
            raise Cancelled()
        if progress and i % 5 == 0:
            progress(i, info.num_frames)
        ids = tracker.update(np.array([f.box for f in faces]).reshape(-1, 4),
                             np.array([f.score for f in faces]))
        for f, tid in zip(faces, ids):
            fr.append(i)
            bx.append(f.box)
            sc.append(f.score)
            kp.append(f.kps)
            em.append(f.emb)
            tr.append(tid)
        n = i + 1
    dets = Detections(
        np.array(fr, np.int64), np.array(bx, np.float32).reshape(-1, 4), np.array(sc, np.float32),
        np.array(kp, np.float32).reshape(-1, 5, 2), np.array(em, np.float32).reshape(-1, 512),
        np.array(tr, np.int64), n,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        dets.save(cache)
    return dets


# ---------------------------------------------------------------- identities

def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def select_refs(dets: Detections, idx: np.ndarray, k: int = 8) -> np.ndarray:
    """Weighted mean embedding plus up to k diverse high-quality samples
    (different angles and lighting)."""
    q = dets.quality()[idx]
    good = idx[q >= np.quantile(q, 0.5)] if len(idx) > 4 else idx
    mean = _norm((dets.emb[idx] * dets.quality()[idx, None]).sum(0))
    chosen = [int(good[np.argmax(dets.quality()[good])])]
    while len(chosen) < min(k, len(good)):
        sims = dets.emb[good] @ dets.emb[chosen].T
        nxt = int(good[np.argmin(sims.max(1))])
        if nxt in chosen:
            break
        chosen.append(nxt)
    return np.vstack([mean[None], dets.emb[chosen]]).astype(np.float32)


def similarity(embs: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Similarity to the identity: best of mean-reference and individual references
    (individual matches are slightly discounted)."""
    if not len(embs):
        return np.zeros(0, np.float32)
    s = embs @ refs.T
    return np.maximum(s[:, 0], 0.95 * s[:, 1:].max(1) if refs.shape[0] > 1 else s[:, 0])


# ------------------------------------------------------------ target selection

@dataclass
class TargetPlan:
    """Per-frame target face boxes and how they were obtained."""
    boxes: list[list[np.ndarray]]  # per frame: list of xyxy boxes to mask
    strong: list[np.ndarray | None]  # per frame: the best confident box (SAM 2 prompt), or None
    source: list[str]  # per frame: "det", "interp", "hold", "" (none)
    others: list[list[np.ndarray]]  # per frame: confidently non-target faces
    sim: np.ndarray  # per detection similarity to the target
    target_dets: list[list[int]] = None  # per frame: detection indices taken as target


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


@dataclass
class ManualBox:
    """User-drawn "mask here too" box: masked on frames first..last (inclusive)
    and used as a SAM 2 prompt on its own frame."""
    frame: int
    box: np.ndarray
    first: int
    last: int


def next_blurred_frame(levels: np.ndarray, frame: int) -> int | None:
    """Where blurring starts again after `frame`: the first blurred frame
    (levels > 0) after the unblurred run that `frame` is in, or that follows
    it when `frame` itself is blurred. None if nothing is blurred later."""
    lv = np.asarray(levels)
    g = frame
    while g < len(lv) and lv[g] > 0:  # skip the blurred run we are in
        g += 1
    later = np.nonzero(lv[g:] > 0)[0]
    return int(g + later[0]) if len(later) else None


def plan_labeled(dets: Detections, det_label: np.ndarray, person: int, fps: float = 30.0,
                 max_gap_s: float = 1.0, hold_s: float = 0.25, excluded: set[int] | None = None,
                 manual: list[ManualBox] | None = None) -> TargetPlan:
    """Decides, frame by frame, where to mask `person`, using only the user's labels.

    det_label: label per detection (review.py: person index, SOMEONE_ELSE,
    NOT_A_FACE, or UNLABELED for weak detections outside any card). Detections
    labeled `person` are masked; other people's and "someone else" faces are
    never masked for this person; weak unlabeled detections are masked only
    where this person's face was within the last/next second (continuity).
    excluded: detection indices the user marked "not this person" on a frame.
    manual: user-drawn boxes."""
    from .review import NOT_A_FACE, UNLABELED

    n = dets.num_frames
    excluded = excluded or set()
    boxes: list[list[np.ndarray]] = [[] for _ in range(n)]
    strong: list[np.ndarray | None] = [None] * n
    source = [""] * n
    others: list[list[np.ndarray]] = [[] for _ in range(n)]
    target_dets: list[list[int]] = [[] for _ in range(n)]
    candidates: list[list[int]] = [[] for _ in range(n)]
    for f, idx in enumerate(dets.by_frame()):
        best = -1.0
        for i in idx:
            lab = int(det_label[i])
            if i in excluded:
                others[f].append(dets.box[i])
            elif lab == person:
                boxes[f].append(dets.box[i])
                target_dets[f].append(int(i))
                source[f] = "det"
                if dets.score[i] >= 0.5 and dets.score[i] > best:  # SAM 2 prompt
                    best = float(dets.score[i])
                    strong[f] = dets.box[i]
            elif lab == UNLABELED:
                candidates[f].append(int(i))
            elif lab != NOT_A_FACE:  # another person, or someone else
                others[f].append(dets.box[i])

    max_gap = int(round(max_gap_s * fps))
    hold = int(round(hold_s * fps))

    def main_box(f):
        return strong[f] if strong[f] is not None else boxes[f][0]

    # spatial continuity: a weak unlabeled detection (dark, profile, blur)
    # where this person's face was within the last/next second is this person.
    for order in (range(n), range(n - 1, -1, -1)):
        last_f, last_box = None, None
        for f in order:
            if boxes[f]:
                last_f, last_box = f, main_box(f)
                continue
            if last_f is None or abs(f - last_f) > max_gap:
                continue
            for i in candidates[f]:
                if _box_iou(dets.box[i], last_box) > 0.3:
                    boxes[f].append(dets.box[i])
                    target_dets[f].append(i)
                    source[f] = "near"
            if boxes[f]:
                last_f, last_box = f, main_box(f)

    # gap bridging: interpolate across short gaps, hold briefly at the ends
    have = [f for f in range(n) if boxes[f]]

    # boxes the user excluded: gap bridging must never re-mask them
    excl_boxes: dict[int, list[np.ndarray]] = {}
    for i in excluded:
        excl_boxes.setdefault(int(dets.frame[i]), []).append(dets.box[i])

    def blocked(f, b) -> bool:
        return any(_box_iou(b, e) > 0.3 for e in excl_boxes.get(f, ()))

    for a, b in zip(have, have[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            ba, bb = main_box(a), main_box(b)
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                box = (1 - t) * ba + t * bb
                if not blocked(f, box):
                    boxes[f] = [box]
                    source[f] = "interp"
    have_set = set(f for f in range(n) if boxes[f])
    for f in sorted(have_set):
        for d in range(1, hold + 1):
            for g in (f - d, f + d):
                if 0 <= g < n and not boxes[g] and not blocked(g, main_box(f)):
                    boxes[g] = [main_box(f)]
                    source[g] = "hold"

    # manual "mask here too" boxes: masked over their span, SAM 2 prompt on their frame
    for mb in manual or []:
        b = np.asarray(mb.box, np.float32)
        for g in range(max(0, mb.first), min(n, mb.last + 1)):
            boxes[g].append(b)
            if not source[g] or source[g] == "hold":
                source[g] = "manual"
        if 0 <= mb.frame < n:
            strong[mb.frame] = b
    return TargetPlan(boxes, strong, source, others, np.zeros(0, np.float32), target_dets)
