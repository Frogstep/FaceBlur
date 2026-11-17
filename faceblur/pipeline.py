"""Shared pipeline used by both the CLI and the GUI: face scan -> face cards
the user labels -> per-person plans (with manual corrections) -> render ->
safety pass.

Long operations take progress(stage, done, total) and cancel() -> bool hooks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .identity import Cancelled, Detections, ManualBox, TargetPlan, plan_labeled, scan, similarity
from .project import Label, Project
from .review import UNLABELED, Card, build_cards, card_of_detection, detection_labels, person_refs
from .video_io import FrameReader, FrameWriter, VideoInfo, probe

ROOT = Path(__file__).resolve().parent.parent
Progress = Callable[[str, int, int], None]
Cancel = Callable[[], bool]

__all__ = ["Analysis", "Cancelled", "RenderResult", "render", "safety_check", "get_analyzer", "workdir"]


def workdir(video: str | Path) -> Path:
    d = ROOT / "work" / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


_ANALYZER: dict = {}  # loaded once per device (they take a few seconds)


def get_analyzer():
    from . import device
    from .faces import FaceAnalyzer

    device.check_forced_gpu()
    key = device.faces_on_gpu()
    if key not in _ANALYZER:
        _ANALYZER.clear()
        _ANALYZER[key] = FaceAnalyzer()
    return _ANALYZER[key]


_PREDICTOR: dict = {}  # SAM 2 is loaded once per device (up to 1 GB of weights)


def get_predictor():
    from . import device
    from .segment import load_predictor

    device.check_forced_gpu()
    key = device.sam_device()
    if key not in _PREDICTOR:
        _PREDICTOR.clear()
        _PREDICTOR[key] = load_predictor(key)
    return _PREDICTOR[key]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class PersonInput:
    index: int  # position in project.people
    name: str
    refs: np.ndarray | None  # reference embeddings (safety check only)
    det_idx: np.ndarray  # detections the user labeled as this person
    manual: list[ManualBox] = field(default_factory=list)


class Analysis:
    """Everything known about a video after the face scan: detections and the
    face cards the user labels."""

    def __init__(self, video: Path, info: VideoInfo, dets: Detections):
        self.video = video
        self.info = info
        self.dets = dets
        self.wd = workdir(video)
        self.cards: list[Card] = build_cards(dets)
        self.card_of_det = card_of_detection(dets, self.cards)
        self._frame_index = dets.by_frame()

    @classmethod
    def open(cls, video: str | Path, progress: Progress | None = None, cancel: Cancel | None = None) -> "Analysis":
        video = Path(video).resolve()
        info = probe(video)
        cache = workdir(video) / "detections.npz"
        if cache.exists():
            dets = Detections.load(cache)
        else:
            if progress:
                progress("Loading face models", 0, 0)
            analyzer = get_analyzer()
            cb = (lambda d, t: progress("Detecting faces", d, t)) if progress else None
            dets = scan(info, analyzer, cache, progress=cb, cancel=cancel)
        return cls(video, info, dets)

    def detections_in(self, frame: int) -> np.ndarray:
        if 0 <= frame < len(self._frame_index):
            return self._frame_index[frame]
        return np.zeros(0, np.int64)

    def find_detection(self, frame: int, box) -> int | None:
        best, best_iou = None, 0.5
        for i in self.detections_in(frame):
            v = _iou(self.dets.box[i], box)
            if v > best_iou:
                best, best_iou = int(i), v
        return best

    # ------------------------------------------------------------- labels

    def card_labels(self, project: Project) -> list[int]:
        """The user's label per card (UNLABELED where none)."""
        labels = [UNLABELED] * len(self.cards)
        for lab in project.labels:
            i = self.find_detection(lab.frame, lab.box)
            if i is not None and self.card_of_det[i] >= 0:
                labels[int(self.card_of_det[i])] = lab.label
        return labels

    def set_label(self, project: Project, card: int, label: int) -> None:
        """Label a card by its position in self.cards (UNLABELED clears it)."""
        c = self.cards[card]
        keep = []
        for lab in project.labels:
            i = self.find_detection(lab.frame, lab.box)
            if i is None or self.card_of_det[i] != card:
                keep.append(lab)
        if label != UNLABELED:
            a = c.anchor
            keep.append(Label(int(self.dets.frame[a]), [round(float(v), 1) for v in self.dets.box[a]], int(label)))
        project.labels = keep

    def unlabeled(self, project: Project) -> list[int]:
        """Card numbers still waiting for a label."""
        return [c.number for c, lab in zip(self.cards, self.card_labels(project)) if lab == UNLABELED]

    # ------------------------------------------------------------- plans

    def excluded(self, project: Project) -> set[int]:
        out: set[int] = set()
        for ex in project.exclusions:
            i = self.find_detection(ex.frame, ex.box)
            if i is None:
                continue
            t = int(self.dets.track[i])
            if ex.scope == "track" and t != 0:
                out.update(np.where(self.dets.track == t)[0].tolist())
            else:
                out.add(i)
        return out

    def person_inputs(self, project: Project) -> list[PersonInput]:
        det_label = detection_labels(self.dets, self.cards, self.card_labels(project))
        people = []
        for pi, person in enumerate(project.people):
            idx = np.where(det_label == pi)[0]
            manual = [ManualBox(m.frame, np.array(m.box, np.float32), m.span)
                      for m in project.manual if m.person == pi]
            people.append(PersonInput(pi, person.name, person_refs(self.dets, idx), idx, manual))
        return people

    def plans(self, project: Project) -> tuple[list[PersonInput], list[TargetPlan]]:
        """One plan per person who has labeled faces or drawn boxes."""
        det_label = detection_labels(self.dets, self.cards, self.card_labels(project))
        inputs = [p for p in self.person_inputs(project) if len(p.det_idx) or p.manual]
        excluded = self.excluded(project)
        plans = [plan_labeled(self.dets, det_label, p.index, self.info.fps_float,
                              excluded=excluded, manual=p.manual) for p in inputs]
        return inputs, plans


def face_crop(frame: np.ndarray, box, size: int = 160) -> np.ndarray:
    """Square crop around a face box, with some context, resized to size x size."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    s = max(x2 - x1, y2 - y1) * 0.8
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - s)), int(max(0, cy - s))
    c, d = int(min(w, cx + s)), int(min(h, cy + s))
    if c <= a or d <= b:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(frame[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)


def card_thumbnails(analysis: Analysis, size: int = 96, progress: Progress | None = None,
                    cancel: Cancel | None = None) -> dict[int, np.ndarray]:
    """Thumbnail per sample detection of every card, cached in the work folder
    (one frame seek per distinct frame)."""
    from .video_io import FrameGrabber

    cache = analysis.wd / f"thumbs_{size}.npz"
    want = sorted({i for c in analysis.cards for i in c.samples})
    if cache.exists():
        z = np.load(cache)
        if all(f"d{i}" in z.files for i in want):
            return {i: z[f"d{i}"] for i in want}
    dets = analysis.dets
    by_frame: dict[int, list[int]] = {}
    for i in want:
        by_frame.setdefault(int(dets.frame[i]), []).append(i)
    grab = FrameGrabber(analysis.info, cache_size=2)
    out: dict[int, np.ndarray] = {}
    for k, (f, idx) in enumerate(sorted(by_frame.items())):
        if cancel and cancel():
            raise Cancelled()
        if progress:
            progress("Preparing face cards", k, len(by_frame))
        img = grab.get(f)
        for i in idx:
            out[i] = face_crop(img, dets.box[i], size) if img is not None else np.zeros((size, size, 3), np.uint8)
    np.savez(cache, **{f"d{i}": v for i, v in out.items()})
    return out


# ------------------------------------------------------------------ render

@dataclass
class RenderResult:
    output: Path
    frames: int
    unmasked: int
    sam_status: dict
    fps: float
    log: Path


def render(project: Project, analysis: Analysis, progress: Progress | None = None,
           cancel: Cancel | None = None, debug: bool = False, log: Callable[[str], None] = print) -> RenderResult:
    from .mask import apply, combine, draw_debug

    missing = analysis.unlabeled(project)
    if missing:
        raise ValueError(f"{len(missing)} face card(s) are not labeled yet: "
                         f"{', '.join('#' + str(k) for k in missing[:20])}{' ...' if len(missing) > 20 else ''}. "
                         f"Label every card (a person, someone else, or not a face) before rendering.")
    inputs, plans = analysis.plans(project)
    if not plans:
        raise ValueError("Nobody is marked to be masked: label at least one face card as a person "
                         "(or draw a 'mask here too' box).")
    s = project.settings
    info, dets, wd = analysis.info, analysis.dets, analysis.wd
    n = info.num_frames
    from . import device

    device.check_forced_gpu()
    log(device.summary())
    if s.use_sam and device.sam_device() == "cpu":
        from .segment import model_for

        log(f"SAM 2 runs on the CPU ({model_for('cpu')} model): this is slow. "
            f"Turning SAM 2 off (ellipse masks) is much faster.")
    for p, plan in zip(inputs, plans):
        covered = sum(1 for b in plan.boxes if b)
        log(f"{p.name}: face box on {covered}/{dets.num_frames} frames "
            f"(det {plan.source.count('det')}, interp {plan.source.count('interp')}, "
            f"hold {plan.source.count('hold')}, manual {plan.source.count('manual')})")

    out = Path(project.output or analysis.video.with_name(analysis.video.stem + "_masked.mp4"))
    frames = FrameReader(info)
    if s.use_sam:
        from .segment import Segmenter

        if progress:
            progress("Loading SAM 2", 0, 0)
        stream = Segmenter(get_predictor(), cancel=cancel).run(frames, plans)
    else:
        stream = ((fr, None, "off") for fr in frames)

    np.savez(wd / "target_refs.npz", *[p.refs for p in inputs if p.refs is not None])
    stats: dict[str, int] = {}
    per_frame = []
    t0 = time.time()
    debug_w = FrameWriter(wd / "debug.mp4", info, cq=28) if debug else None
    tmp = out.with_name(out.stem + ".partial" + out.suffix)
    try:
        with FrameWriter(tmp, info, cq=s.cq) as writer:
            for f, (frame, sam_mask, status) in enumerate(stream):
                if cancel and cancel():
                    raise Cancelled()
                inside = f < dets.num_frames
                boxes = [b for p in plans for b in p.boxes[f]] if inside else []
                source = "+".join(p.source[f] or "-" for p in plans) if inside else ""
                mask = combine(frame.shape, sam_mask, boxes, pad=s.pad)
                writer.write(apply(frame, mask, s.style, s.strength))
                stats[status] = stats.get(status, 0) + 1
                per_frame.append({"frame": f, "sam": status, "box": source, "masked": bool(mask.any())})
                if debug_w:
                    others = [b for p in plans for b in p.others[f]] if inside else []
                    debug_w.write(draw_debug(frame, sam_mask, boxes, others, status, source))
                if progress and f % 2 == 0:
                    progress("Rendering", f, n)
    except BaseException:
        if debug_w:
            debug_w.__exit__(Cancelled, None, None)
        tmp.unlink(missing_ok=True)
        raise
    if debug_w:
        debug_w.close()
    out.unlink(missing_ok=True)
    tmp.rename(out)
    dt = max(time.time() - t0, 1e-6)
    log_path = wd / "render_log.json"
    log_path.write_text(json.dumps({"output": str(out), "sam": stats, "frames": per_frame}, indent=1), encoding="utf-8")
    unmasked = sum(1 for p in per_frame if not p["masked"])
    return RenderResult(out, len(per_frame), unmasked, stats, len(per_frame) / dt, log_path)


# ------------------------------------------------------------------ safety

def safety_check(masked: str | Path, refs_list: list[np.ndarray], wd: Path,
                 progress: Progress | None = None, cancel: Cancel | None = None, thr: float = 0.28) -> dict:
    """Re-detects faces in the rendered video and flags frames where any of the
    masked people is still recognizable. Writes flagged/*.jpg and safety_report.json."""
    import cv2

    analyzer = get_analyzer()
    info = probe(masked)
    flag_dir = wd / "flagged"
    flag_dir.mkdir(exist_ok=True)
    for p in flag_dir.glob("*.jpg"):
        p.unlink()
    flagged = []
    if refs_list:
        stream = analyzer.analyze_stream(FrameReader(info))
        for f, (frame, faces) in enumerate(stream):
            if cancel and cancel():
                stream.close()
                raise Cancelled()
            if progress and f % 5 == 0:
                progress("Safety check", f, info.num_frames)
            if not faces:
                continue
            embs = np.array([x.emb for x in faces])
            sims = np.max([similarity(embs, r) for r in refs_list], axis=0)
            # a weak match only counts on a clearly detected face; texture that the
            # (deliberately over-sensitive) detector mistakes for a face is ignored
            hits = [(fc, float(s)) for fc, s in zip(faces, sims)
                    if s >= 0.40 or (s >= thr and fc.score >= 0.5)]
            if hits:
                flagged.append({"frame": f, "time": round(f / info.fps_float, 3),
                                "sim": [round(s, 3) for _, s in hits],
                                "box": [[round(float(v), 1) for v in fc.box] for fc, _ in hits]})
                img = frame.copy()
                for fc, s in hits:
                    cv2.rectangle(img, tuple(map(int, fc.box[:2])), tuple(map(int, fc.box[2:])), (0, 0, 255), 2)
                    cv2.putText(img, f"{s:.2f}", (int(fc.box[0]), int(fc.box[1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(str(flag_dir / f"{f:06d}.jpg"), img)
    report = {"video": str(masked), "threshold": thr, "frames": info.num_frames, "flagged": flagged}
    (wd / "safety_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_target_refs(wd: Path) -> list[np.ndarray]:
    p = wd / "target_refs.npz"
    if not p.exists():
        return []
    z = np.load(p)
    return [z[k] for k in z.files]
