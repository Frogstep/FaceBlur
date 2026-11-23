"""Shared pipeline used by the CLI, the GUI and the web server: face scan ->
face cards the user marks to blur (every card starts as not blurred) -> one
masking plan per track of blurred cards and per drawn box (with manual
corrections) -> render -> safety pass.

Long operations take progress(stage, done, total) and cancel() -> bool hooks.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .identity import Cancelled, Detections, ManualBox, TargetPlan, plan_labeled, scan, similarity
from .project import Edit, Label, Project
from .review import (BLUR, SOMEONE_ELSE, UNLABELED, Card, build_cards, card_of_detection, chain_cards, is_blur,
                     person_refs)
from .video_io import FrameReader, FrameWriter, VideoInfo, probe

ROOT = Path(__file__).resolve().parent.parent
Progress = Callable[[str, int, int], None]
Cancel = Callable[[], bool]

__all__ = ["Analysis", "Cancelled", "RenderResult", "render", "safety_check", "get_analyzer", "workdir"]


TEMP_KEEP_DAYS = 2  # leftovers older than this (e.g. from a crash) are deleted


def temp_dir() -> Path:
    """The app's folder in the system temp directory (%TEMP%\\FaceBlur on Windows),
    for videos while they are being written. Removes stale leftovers."""
    d = Path(tempfile.gettempdir()) / "FaceBlur"
    d.mkdir(parents=True, exist_ok=True)
    old = time.time() - TEMP_KEEP_DAYS * 86400
    for p in d.glob("*.partial.*"):
        try:
            if p.stat().st_mtime < old:
                p.unlink()
        except OSError:  # in use by another render, or already gone
            pass
    return d


def scan_key(info: VideoInfo, any_angle: bool = False) -> str:
    """Names the scan results of this cut & crop and detection mode. "cfr":
    frames counted on a constant frame-rate clock (scans from before that
    counted the frames a video holds, which differs when it has gaps)."""
    return "_".join(p for p in (info.edit_key(), "any" if any_angle else "", "cfr") if p)


def detections_cache(wd: Path, info: VideoInfo, any_angle: bool = False) -> Path:
    """Cached detections for this cut & crop (a changed edit is scanned again)."""
    return wd / f"detections_{scan_key(info, any_angle)}.npz"


def clear_scan_cache(wd: Path) -> None:
    """Forget the detections and card thumbnails of every edit."""
    for p in [*wd.glob("detections*.npz"), *wd.glob("thumbs_*.npz")]:
        p.unlink(missing_ok=True)


def open_video(video: str | Path, edit: Edit | None = None) -> VideoInfo:
    """probe() with the project's cut & crop applied."""
    info = probe(video)
    if edit is not None and edit.active:
        info = info.turned(edit.turn).with_edit(edit.keep, edit.crop)
    return info


def workdir(video: str | Path, root: str | Path | None = None) -> Path:
    """Cache folder for a video: work/<video name>/ by default, or `root` itself
    when given (the web server keeps one folder per uploaded video)."""
    d = Path(root) if root is not None else ROOT / "work" / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


_ANALYZER: dict = {}  # loaded once per device (they take a few seconds)
_MODEL_LOCK = threading.Lock()  # the web server may ask from several threads


def get_analyzer():
    from . import device
    from .faces import FaceAnalyzer

    device.check_forced_gpu()
    key = device.faces_on_gpu()
    with _MODEL_LOCK:
        if key not in _ANALYZER:
            _ANALYZER.clear()
            _ANALYZER[key] = FaceAnalyzer()
        return _ANALYZER[key]


_PREDICTOR: dict = {}  # one SAM 2 model kept loaded (up to 1 GB of weights)


def sam_model(choice: str = "auto") -> str:
    """The SAM 2.1 size a render uses for the setting `choice` on this PC."""
    from . import device
    from .segment import model_for

    return model_for(device.sam_device(), choice)


def get_predictor(choice: str = "auto", progress: Progress | None = None):
    """The SAM 2 predictor for Settings.sam_model `choice`; a size that is not
    downloaded yet is downloaded first (progress shows it)."""
    from . import device
    from .segment import is_downloaded, load_predictor

    device.check_forced_gpu()
    dev = device.sam_device()
    size = sam_model(choice)
    key = (dev, size)
    with _MODEL_LOCK:
        if key not in _PREDICTOR:
            if not is_downloaded(size):
                from .setup_models import fetch_sam

                if progress:
                    progress("Downloading the SAM 2 model", 0, 0)
                fetch_sam(size, (lambda d, t: progress("Downloading the SAM 2 model", d >> 20, t >> 20))
                          if progress else None)
            if progress:
                progress("Loading SAM 2", 0, 0)
            _PREDICTOR.clear()
            if dev == "cuda":
                import torch

                torch.cuda.empty_cache()  # free the previous model before loading the next
            _PREDICTOR[key] = load_predictor(dev, size)
        return _PREDICTOR[key]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class PersonInput:
    """One masking track: linked blurred cards, or one drawn box."""
    index: int  # track number (0, 1, ...)
    name: str  # for logs: the cards in it ("#3+#7") or "drawn box"
    refs: np.ndarray | None  # reference embeddings (safety check only)
    det_idx: np.ndarray  # detections of the track's cards
    manual: list[ManualBox] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)  # card positions


class Analysis:
    """Everything known about a video after the face scan: detections and the
    face cards the user labels."""

    def __init__(self, video: Path, info: VideoInfo, dets: Detections, wd: Path | None = None,
                 any_angle: bool = False):
        self.video = video
        self.info = info
        self.any_angle = any_angle  # faces were looked for at every angle (Edit.any_angle)
        self.dets = dets
        self.wd = workdir(video, wd)
        self.cards: list[Card] = build_cards(dets)
        self.card_of_det = card_of_detection(dets, self.cards)
        self._frame_index = dets.by_frame()
        self._plan_cache: dict[tuple, TargetPlan] = {}

    @classmethod
    def open(cls, video: str | Path, progress: Progress | None = None, cancel: Cancel | None = None,
             wd: str | Path | None = None, edit: Edit | None = None) -> "Analysis":
        """wd: cache folder (default work/<video name>/, see workdir); edit: the
        project's cut & crop (faces are only looked for in the edited video)."""
        video = Path(video).resolve()
        info = open_video(video, edit)
        wd = workdir(video, wd)
        any_angle = bool(edit and edit.any_angle)
        cache = detections_cache(wd, info, any_angle)
        if cache.exists():
            dets = Detections.load(cache)
        else:
            if progress:
                progress("Loading face models", 0, 0)
            analyzer = get_analyzer()
            cb = (lambda d, t: progress("Detecting faces", d, t)) if progress else None
            dets = scan(info, analyzer, cache, progress=cb, cancel=cancel, any_angle=any_angle)
        return cls(video, info, dets, wd, any_angle)

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

    def blurred(self, project: Project) -> list[bool]:
        """Per card: True when the user marked it to blur (cards start unblurred)."""
        return [is_blur(lab) for lab in self.card_labels(project)]

    def set_blur(self, project: Project, cards, blur: bool) -> None:
        """Mark cards (positions in self.cards) to blur, or back to not blurred."""
        for k in cards:
            self.set_label(project, int(k), BLUR if blur else UNLABELED)

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

    def tracks(self, project: Project) -> tuple[list[PersonInput], np.ndarray]:
        """The masking tracks, and the label per detection that plan_labeled
        reads: the track number for blurred cards, SOMEONE_ELSE for every other
        card (never masked), UNLABELED for weak detections outside any card
        (masked only next to a blurred face)."""
        blurred = self.blurred(project)
        det_label = np.full(len(self.dets.frame), UNLABELED, np.int64)
        for c in self.cards:
            det_label[c.det_idx] = SOMEONE_ELSE
        chains = chain_cards(self.dets, self.cards, [k for k, b in enumerate(blurred) if b], self.info.fps_float)
        out = []
        for t, chain in enumerate(chains):
            idx = np.sort(np.concatenate([self.cards[k].det_idx for k in chain]))
            det_label[idx] = t
            name = "+".join(f"#{self.cards[k].number}" for k in chain)
            out.append(PersonInput(t, name, person_refs(self.dets, idx), idx, [], list(chain)))
        # each drawn box is its own track: it is a SAM 2 prompt of its own
        for m in project.manual:
            box = ManualBox(m.frame, np.array(m.box, np.float32), m.span)
            out.append(PersonInput(len(out), f"drawn box (frame {m.frame})", None, np.zeros(0, np.int64), [box]))
        return out, det_label

    def plans(self, project: Project) -> tuple[list[PersonInput], list[TargetPlan]]:
        """One masking plan per track of blurred cards and per drawn box."""
        inputs, det_label = self.tracks(project)
        excluded = self.excluded(project)
        # A track's plan depends only on its own cards (or drawn box) and the
        # exclusions: every other card's detections count as "someone else"
        # whether blurred or not. So switching one card reuses the other plans
        # instead of re-planning every track (seconds with hundreds of cards).
        ex_key = frozenset(excluded)
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = self._plan_cache = {}
        plans = []
        for p in inputs:
            key = (tuple(p.cards), tuple((m.frame, tuple(m.box.tolist()), m.span) for m in p.manual), ex_key)
            plan = cache.get(key)
            if plan is None:
                plan = plan_labeled(self.dets, det_label, p.index, self.info.fps_float,
                                    excluded=excluded, manual=p.manual)
                if len(cache) > 4000:
                    cache.clear()
                cache[key] = plan
            plans.append(plan)
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

    key = scan_key(analysis.info, analysis.any_angle)
    cache = analysis.wd / f"thumbs_{size}_{key}.npz"
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

    inputs, plans = analysis.plans(project)
    if not plans:
        raise ValueError("No face is marked to blur: mark at least one face card to blur "
                         "(or draw a box to blur an area).")
    s = project.settings
    info, dets, wd = analysis.info, analysis.dets, analysis.wd
    n = info.num_frames
    from . import device

    device.check_forced_gpu()
    log(device.summary())
    if s.use_sam:
        size = sam_model(s.sam_model)
        log(f"SAM 2.1 model: {size}" + (" (auto)" if s.sam_model == "auto" else ""))
        if device.sam_device() == "cpu":
            log(f"SAM 2 runs on the CPU ({size} model): this is slow. "
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

        stream = Segmenter(get_predictor(s.sam_model, progress), cancel=cancel).run(frames, plans)
    else:
        stream = ((fr, None, "off") for fr in frames)

    np.savez(wd / "target_refs.npz", *[p.refs for p in inputs if p.refs is not None])
    _save_face_positions(wd, analysis, project, plans)
    stats: dict[str, int] = {}
    per_frame = []
    t0 = time.time()
    debug_w = FrameWriter(wd / "debug.mp4", info, cq=28) if debug else None
    # written in the temp folder and moved into place when complete, so a
    # cancelled or failed render never leaves half a video next to the output
    tmp = temp_dir() / f"{out.stem}.{uuid.uuid4().hex[:8]}.partial{out.suffix}"
    try:
        with FrameWriter(tmp, info, cq=s.cq, fmt=s.codec, quality=s.quality) as writer:
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
    if progress:  # a copy when the temp folder is on another drive (seconds for a large video)
        progress("Saving the video", 0, 0)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        shutil.move(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)  # left over only when the move failed
    dt = max(time.time() - t0, 1e-6)
    log_path = wd / "render_log.json"
    log_path.write_text(json.dumps({"output": str(out), "sam": stats, "frames": per_frame}, indent=1), encoding="utf-8")
    unmasked = sum(1 for p in per_frame if not p["masked"])
    return RenderResult(out, len(per_frame), unmasked, stats, len(per_frame) / dt, log_path)


# ------------------------------------------------------------------ safety

def _save_face_positions(wd: Path, analysis: Analysis, project: Project, plans: list[TargetPlan]) -> None:
    """For the safety check: where faces were left visible on purpose (cards not
    blurred, and "don't blur here"), and where faces were blurred, per frame."""
    dets = analysis.dets
    target = [set(i for p in plans for i in p.target_dets[f]) for f in range(dets.num_frames)]
    blurred = analysis.blurred(project)
    excluded = analysis.excluded(project)
    vis_f, vis_b = [], []
    for k, c in enumerate(analysis.cards):
        for i in c.det_idx:
            f = int(dets.frame[i])
            if (not blurred[k] or int(i) in excluded) and int(i) not in target[f]:
                vis_f.append(f)
                vis_b.append(dets.box[i])
    blur_f, blur_b = [], []
    for p in plans:
        for f, boxes in enumerate(p.boxes):
            for b in boxes:
                blur_f.append(f)
                blur_b.append(np.asarray(b, np.float32))
    np.savez(wd / "face_positions.npz",
             visible_frame=np.array(vis_f, np.int64), visible_box=np.array(vis_b, np.float32).reshape(-1, 4),
             blur_frame=np.array(blur_f, np.int64), blur_box=np.array(blur_b, np.float32).reshape(-1, 4))


def _load_face_positions(wd: Path) -> tuple[dict, dict]:
    """(visible boxes per frame, blurred boxes per frame) saved by the last render."""
    p = wd / "face_positions.npz"
    vis: dict[int, list] = {}
    blur: dict[int, list] = {}
    if p.exists():
        z = np.load(p)
        for f, b in zip(z["visible_frame"], z["visible_box"]):
            vis.setdefault(int(f), []).append(b)
        for f, b in zip(z["blur_frame"], z["blur_box"]):
            blur.setdefault(int(f), []).append(b)
    return vis, blur


def safety_check(masked: str | Path, refs_list: list[np.ndarray], wd: Path,
                 progress: Progress | None = None, cancel: Cancel | None = None, thr: float = 0.28,
                 any_angle: bool = False) -> dict:
    """Re-detects faces in the rendered video and flags frames where any of the
    blurred faces is still recognizable. Writes flagged/*.jpg and safety_report.json.

    A face found where the user left a face visible on purpose is not flagged
    (family members often look alike), unless a blurred face was in the same
    place on that frame."""
    import cv2

    analyzer = get_analyzer()
    info = probe(masked)
    visible, blurred_at = _load_face_positions(wd)
    ignored = 0
    flag_dir = wd / "flagged"
    flag_dir.mkdir(exist_ok=True)
    for p in flag_dir.glob("*.jpg"):
        p.unlink()
    flagged = []
    if refs_list:
        stream = analyzer.analyze_stream(FrameReader(info), any_angle=any_angle)
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
            kept = [(fc, s) for fc, s in hits
                    if not (any(_iou(fc.box, b) > 0.3 for b in visible.get(f, ()))
                            and not any(_iou(fc.box, b) > 0.1 for b in blurred_at.get(f, ())))]
            ignored += len(hits) - len(kept)
            hits = kept
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
    report = {"video": str(masked), "threshold": thr, "frames": info.num_frames, "flagged": flagged,
              "ignored_visible": ignored}  # look-alike matches on faces left visible on purpose
    (wd / "safety_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_target_refs(wd: Path) -> list[np.ndarray]:
    p = wd / "target_refs.npz"
    if not p.exists():
        return []
    z = np.load(p)
    return [z[k] for k in z.files]
