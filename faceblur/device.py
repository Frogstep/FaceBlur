"""Where processing runs: "auto" (NVIDIA GPU when it works, else CPU), "gpu" or "cpu".

Every GPU step has a CPU fallback, so the app also runs on a PC without an
NVIDIA card (much slower, same algorithm):

  decode     FFmpeg NVDEC          -> FFmpeg software decode
  faces      onnxruntime CUDA      -> onnxruntime CPU
  SAM 2      PyTorch CUDA (large)  -> PyTorch CPU (tiny model if downloaded)
  encode     hevc_nvenc            -> libx265

Set with set_device(), the --device CLI option, the app's settings, or the
FACEBLUR_DEVICE environment variable.
"""

from __future__ import annotations

import functools
import os
import subprocess

_PREF = os.environ.get("FACEBLUR_DEVICE", "auto").lower()
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}


def set_device(pref: str) -> None:
    global _PREF
    pref = (pref or "auto").lower()
    if pref not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be auto, gpu or cpu, not {pref!r}")
    _PREF = pref


def preference() -> str:
    return _PREF


@functools.lru_cache(maxsize=1)
def cuda_torch() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def cuda_onnx() -> bool:
    """True if onnxruntime can really create a CUDA session (DLLs present, GPU usable)."""
    try:
        from .faces import MODELS_DIR, preload_cuda

        preload_cuda()
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False

        so = ort.SessionOptions()
        so.log_severity_level = 4
        s = ort.InferenceSession(str(MODELS_DIR / "w600k_r50.onnx"), so,
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        return s.get_providers()[0] == "CUDAExecutionProvider"
    except Exception:
        return False


def _ffmpeg_works(args: list[str]) -> bool:
    from .video_io import _ffbin

    try:
        exe = _ffbin("ffmpeg")
        r = subprocess.run([exe, "-v", "error", "-nostdin", *args], capture_output=True, timeout=30, **_NOWIN)
        return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def nvenc_works() -> bool:
    """Encode a few frames with NVENC; listing the encoder is not enough (no GPU, old driver)."""
    return _ffmpeg_works(["-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "hevc_nvenc",
                          "-f", "null", "-"])


@functools.lru_cache(maxsize=1)
def nvdec_works() -> bool:
    return _ffmpeg_works(["-hwaccel", "cuda", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                          "-f", "null", "-"]) and cuda_torch()


def faces_on_gpu() -> bool:
    return _PREF != "cpu" and cuda_onnx()


def sam_device() -> str:
    return "cuda" if _PREF != "cpu" and cuda_torch() else "cpu"


def decode_on_gpu() -> bool:
    return _PREF != "cpu" and nvdec_works()


def encoder() -> str:
    return "hevc_nvenc" if _PREF != "cpu" and nvenc_works() else "libx265"


def check_forced_gpu() -> None:
    """With device 'gpu', fail clearly instead of silently running on the CPU."""
    if _PREF == "gpu" and not (cuda_onnx() and cuda_torch()):
        raise RuntimeError("Device is set to GPU, but no usable NVIDIA GPU was found "
                           "(onnxruntime CUDA: %s, PyTorch CUDA: %s). Use 'auto' or 'cpu'."
                           % (cuda_onnx(), cuda_torch()))


def summary() -> str:
    """One line for logs and the status bar."""
    parts = [f"faces: {'GPU' if faces_on_gpu() else 'CPU'}", f"SAM 2: {sam_device().upper().replace('CUDA', 'GPU')}",
             f"decode: {'GPU' if decode_on_gpu() else 'CPU'}",
             f"encode: {'GPU (NVENC)' if encoder() == 'hevc_nvenc' else 'CPU (x265)'}"]
    return f"device {_PREF}: " + ", ".join(parts)
