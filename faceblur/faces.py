"""Face detection (SCRFD) and identity embedding (ArcFace) on onnxruntime-gpu,
plus the low-light pre-processing used on the detection copy of each frame.

The ONNX models are InsightFace's buffalo_l pack (det_10g.onnx, w600k_r50.onnx),
loaded directly so the insightface pip package (which needs a C++ compiler on
Windows) is not required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def preload_cuda() -> None:
    """Load CUDA/cuDNN for onnxruntime from PyTorch.

    PyTorch and onnxruntime must use ONE copy of cuDNN: two different versions
    with the same DLL names in one process make Windows mix their parts
    ("Entry Point Not Found" in cudnn_*64_9.dll). Both are built for CUDA 13,
    so importing torch first and calling preload_dlls() makes onnxruntime use
    the DLLs torch already loaded."""
    import torch  # noqa: F401  (must come first)
    import onnxruntime as ort

    try:
        ort.preload_dlls()
    except Exception:  # older onnxruntime without preload_dlls, or CPU-only PC
        pass


def _session(path: Path):
    import onnxruntime as ort

    preload_cuda()
    from .device import faces_on_gpu

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if faces_on_gpu() else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


# ---------------------------------------------------------------- enhancement

def enhance_for_detection(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on luminance plus adaptive gamma for dark frames.

    Applied only to the copy used for detection and embedding, never to the
    output frame.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean = float(l.mean()) / 255.0
    if mean < 0.40:
        # brighten so the mean lands near 0.45
        gamma = np.clip(np.log(0.45) / np.log(max(mean, 0.02)), 1.0, 3.0)
        lut = (np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255.0).clip(0, 255).astype(np.uint8)
        l = cv2.LUT(l, lut)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if mean < 0.15:
        # very dark footage is usually noisy too; light denoise helps the detector
        from .denoise import denoise

        out = denoise(out)
    return out


# ------------------------------------------------------------------ detection

@dataclass
class Face:
    box: np.ndarray  # x1, y1, x2, y2 in frame pixels
    score: float
    kps: np.ndarray  # 5x2 landmarks
    emb: np.ndarray | None = field(default=None)  # 512-d normalized
    sim: float = 0.0  # similarity to the reference identity


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


ANY_ANGLE = (0, 90, 180, 270)  # SCRFD.detect rotations for faces at any angle
_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _unrotate_points(p: np.ndarray, angle: int, w: int, h: int) -> np.ndarray:
    """Points found in the image turned `angle` degrees clockwise, back in the
    w x h original."""
    x, y = p[..., 0], p[..., 1]
    if angle == 90:
        x, y = y, h - x
    elif angle == 180:
        x, y = w - x, h - y
    elif angle == 270:
        x, y = w - y, x
    return np.stack([x, y], axis=-1)


def _unrotate(boxes: np.ndarray, kps: np.ndarray, angle: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    a = _unrotate_points(boxes[:, :2], angle, w, h)
    b = _unrotate_points(boxes[:, 2:], angle, w, h)
    boxes = np.concatenate([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.float32)
    # landmarks keep their meaning (left eye stays left eye of the face);
    # ArcFace alignment turns the face upright from them
    return boxes, _unrotate_points(kps, angle, w, h).astype(np.float32)


class SCRFD:
    def __init__(self, path: Path = MODELS_DIR / "det_10g.onnx", input_size: int = 1280,
                 conf: float = 0.3):
        self.sess = _session(path)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.conf = conf
        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self._centers: dict = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return self._centers[key]

    def detect_raw(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns boxes (N,4), scores (N,), kps (N,5,2) before NMS."""
        size = self.input_size
        h, w = bgr.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR if scale < 1 else cv2.INTER_CUBIC)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})
        n = len(self.strides)
        all_b, all_s, all_k = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i][:, 0]
            bbox = outs[i + n] * stride
            kps = outs[i + 2 * n] * stride
            fh, fw = size // stride, size // stride
            centers = self._anchor_centers(fh, fw, stride)
            idx = np.where(scores >= self.conf)[0]
            if not idx.size:
                continue
            c = centers[idx]
            b = np.concatenate([c - bbox[idx, :2], c + bbox[idx, 2:]], axis=1)
            k = kps[idx].reshape(-1, 5, 2) + c[:, None, :]
            all_b.append(b)
            all_s.append(scores[idx])
            all_k.append(k)
        if not all_b:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)
        return (np.concatenate(all_b) / scale, np.concatenate(all_s), np.concatenate(all_k) / scale)

    def detect(self, images: list[np.ndarray], upscale_small: bool = True,
               rotations: tuple[int, ...] = (0,)) -> list[Face]:
        """Detect on several versions of the same frame (e.g. original and
        enhanced) and merge with NMS. Over-detects on purpose; identity
        matching decides what is the target.

        rotations: also look at each version turned by these angles (degrees
        clockwise, multiples of 90). The detector only finds roughly upright
        faces, so a camera on its side or a person lying down needs ANY_ANGLE."""
        bs, ss, ks = [], [], []
        for img in images:
            h, w = img.shape[:2]
            factor = 1.0
            if upscale_small and min(h, w) < 720:
                factor = 2.0
                img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            for angle in rotations:
                b, s, k = self.detect_raw(img if angle == 0 else cv2.rotate(img, _ROTATE[angle]))
                if angle:
                    b, k = _unrotate(b, k, angle, img.shape[1], img.shape[0])
                bs.append(b / factor)
                ss.append(s)
                ks.append(k / factor)
        boxes, scores, kps = np.concatenate(bs), np.concatenate(ss), np.concatenate(ks)
        if not len(scores):
            return []
        keep = _nms(boxes, scores, 0.4)
        faces = []
        for i in keep:
            b = boxes[i]
            if (b[2] - b[0]) < 6 or (b[3] - b[1]) < 6:
                continue
            faces.append(Face(box=b.astype(np.float32), score=float(scores[i]), kps=kps[i].astype(np.float32)))
        return faces


# ------------------------------------------------------------------ embedding

_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_face(bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_DST * (size / 112.0), method=cv2.LMEDS)
    if m is None:
        m = cv2.getAffineTransform(kps[:3], _ARCFACE_DST[:3] * (size / 112.0))
    return cv2.warpAffine(bgr, m, (size, size), borderValue=0)


class ArcFace:
    def __init__(self, path: Path = MODELS_DIR / "w600k_r50.onnx"):
        self.sess = _session(path)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, bgr: np.ndarray, faces: list[Face]) -> None:
        if not faces:
            return
        crops = [align_face(bgr, f.kps) for f in faces]
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        embs = []
        for i in range(0, len(blob), 32):
            embs.append(self.sess.run(None, {self.input_name: blob[i:i + 32]})[0])
        embs = np.concatenate(embs)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        for f, e in zip(faces, embs):
            f.emb = e.astype(np.float32)


class FaceAnalyzer:
    """Detect + embed on the original and the enhanced copy of a frame."""

    def __init__(self, det_size: int = 1280, conf: float = 0.3):
        self.det = SCRFD(input_size=det_size, conf=conf)
        self.rec = ArcFace()

    def providers(self) -> list[str]:
        return self.det.sess.get_providers()

    def analyze(self, bgr: np.ndarray, enh: np.ndarray | None = None, any_angle: bool = False) -> list[Face]:
        """enh: the frame already passed through enhance_for_detection (optional);
        any_angle: also find sideways and upside-down faces (about 4x slower)."""
        if enh is None:
            enh = enhance_for_detection(bgr)
        faces = self.det.detect([bgr, enh], rotations=ANY_ANGLE if any_angle else (0,))
        # embed from the enhanced copy so dark frames give usable embeddings;
        # the reference is built the same way so the two stay comparable
        self.rec.embed(enh, faces)
        return faces

    def analyze_stream(self, frames, workers: int | None = None, any_angle: bool = False):
        """Yields (frame, faces) for each frame, in order.

        The CPU-heavy enhancement (CLAHE, gamma, denoise of dark frames) runs
        ahead in a thread pool (OpenCV releases the GIL), while detection and
        embedding run here in order. Results are identical to analyze() frame
        by frame; it is only faster."""
        import os
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        # measured on 16 cores: 8 workers is fastest; more only compete with OpenCV's own threads
        workers = workers or max(1, min(8, (os.cpu_count() or 2) // 2))
        ahead = 2 * workers
        pending: deque = deque()
        it = iter(frames)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="enhance") as pool:
            def fill():
                while len(pending) < ahead:
                    try:
                        fr = next(it)
                    except StopIteration:
                        return
                    pending.append((fr, pool.submit(enhance_for_detection, fr)))
            try:
                fill()
                while pending:
                    fr, fut = pending.popleft()
                    fill()
                    yield fr, self.analyze(fr, fut.result(), any_angle)
            finally:
                for _, fut in pending:
                    fut.cancel()
                if hasattr(it, "close"):  # stop the decoder if we were cancelled early
                    it.close()
