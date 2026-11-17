"""FFmpeg-based decode and encode via subprocess pipes.

Decode produces BGR uint8 frames in display orientation. Encode writes HEVC
(NVENC) and copies the audio from the source file unchanged.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


# no console window flashing when called from the GUI on Windows
_NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}


_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg"


def _ffbin(name: str) -> str:
    """The FFmpeg downloaded by the installer (tools/ffmpeg), else the one on PATH."""
    local = _TOOLS / (f"{name}.exe" if sys.platform == "win32" else name)
    if local.exists():
        return str(local)
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"{name} not found (run the installer, or put FFmpeg on PATH)")
    return exe


@dataclass
class VideoInfo:
    path: Path
    width: int  # display width (after rotation)
    height: int
    fps: Fraction
    num_frames: int
    duration: float
    has_audio: bool
    color_matrix: str  # swscale matrix name: bt709 / bt601 / bt2020
    color_range: str  # tv / pc
    vfr: bool

    @property
    def fps_float(self) -> float:
        return float(self.fps)


_MATRIX_MAP = {
    "bt709": "bt709",
    "smpte170m": "bt601",
    "bt470bg": "bt601",
    "bt601": "bt601",
    "bt2020nc": "bt2020",
    "bt2020c": "bt2020",
}


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = subprocess.run(
        [
            _ffbin("ffprobe"), "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        capture_output=True, text=True, check=True, **_NOWIN,
    ).stdout
    info = json.loads(out)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])

    w, h = int(vs["width"]), int(vs["height"])
    rotation = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if "rotate" in vs.get("tags", {}):
        rotation = int(vs["tags"]["rotate"])
    if abs(rotation) % 180 == 90:
        w, h = h, w

    def frac(s: str) -> Fraction:
        try:
            f = Fraction(s)
            return f if f > 0 else Fraction(0)
        except (ValueError, ZeroDivisionError):
            return Fraction(0)

    r_fps, a_fps = frac(vs.get("r_frame_rate", "0/1")), frac(vs.get("avg_frame_rate", "0/1"))
    fps = a_fps or r_fps or Fraction(30)
    vfr = bool(r_fps and a_fps and abs(float(r_fps) - float(a_fps)) > 0.01)

    duration = float(vs.get("duration") or info["format"].get("duration") or 0.0)
    nb = int(vs.get("nb_frames") or 0) or int(round(duration * float(fps)))

    cs = vs.get("color_space", "")
    matrix = _MATRIX_MAP.get(cs) or ("bt709" if h >= 720 else "bt601")
    rng = "pc" if vs.get("color_range") == "pc" else "tv"

    return VideoInfo(path, w, h, fps, nb, duration, has_audio, matrix, rng, vfr)


class FrameReader:
    """Iterates BGR frames from a video using FFmpeg (NVDEC decode when the device allows)."""

    def __init__(self, info: VideoInfo, hwaccel: bool | None = None, start_frame: int = 0):
        self.info = info
        if hwaccel is None:
            from .device import decode_on_gpu

            hwaccel = decode_on_gpu()
        cmd = [_ffbin("ffmpeg"), "-v", "error", "-nostdin"]
        if hwaccel:
            cmd += ["-hwaccel", "cuda"]
        if start_frame > 0:  # accurate seek, same convention as FrameGrabber
            cmd += ["-ss", f"{max(0.0, (start_frame - 0.25) / info.fps_float):.6f}"]
        cmd += [
            "-i", str(info.path),
            "-map", "0:v:0",
            "-vf", f"scale=in_color_matrix={info.color_matrix}:in_range={info.color_range}",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._cmd = cmd
        self._frame_bytes = info.width * info.height * 3

    def __iter__(self) -> Iterator[np.ndarray]:
        proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=self._frame_bytes * 2, **_NOWIN)
        finished = False
        try:
            while True:
                buf = proc.stdout.read(self._frame_bytes)
                if len(buf) < self._frame_bytes:
                    finished = True
                    break
                yield np.frombuffer(buf, np.uint8).reshape(self.info.height, self.info.width, 3)
        finally:
            if not finished:  # consumer stopped early: just stop ffmpeg
                proc.kill()
            proc.stdout.close()
            err = proc.stderr.read().decode(errors="replace")
            proc.wait()
            if finished and proc.returncode != 0:
                raise RuntimeError(f"ffmpeg decode failed:\n{err}")


class FrameGrabber:
    """Random access to single frames (for scrubbing in the GUI). Uses FFmpeg's
    accurate input seeking, so frame N here is the same as frame N from FrameReader."""

    def __init__(self, info: VideoInfo, cache_size: int = 48):
        self.info = info
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []
        self._size = cache_size

    def get(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(idx, self.info.num_frames - 1))
        if idx in self._cache:
            return self._cache[idx]
        t = max(0.0, (idx - 0.25) / self.info.fps_float)
        cmd = [
            _ffbin("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{t:.6f}", "-i", str(self.info.path),
            "-map", "0:v:0", "-frames:v", "1",
            "-vf", f"scale=in_color_matrix={self.info.color_matrix}:in_range={self.info.color_range}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        out = subprocess.run(cmd, capture_output=True, **_NOWIN).stdout
        n = self.info.width * self.info.height * 3
        if len(out) < n:
            return None
        frame = np.frombuffer(out[:n], np.uint8).reshape(self.info.height, self.info.width, 3)
        self._cache[idx] = frame
        self._order.append(idx)
        if len(self._order) > self._size:
            self._cache.pop(self._order.pop(0), None)
        return frame


class FrameWriter:
    """Encodes BGR frames to HEVC (NVENC on the GPU, x265 on the CPU) and copies
    audio from the source."""

    def __init__(self, out_path: str | Path, info: VideoInfo, cq: int = 19,
                 codec: str | None = None):
        if codec is None:
            from .device import encoder

            codec = encoder()
        self.codec = codec
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _ffbin("ffmpeg"), "-v", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{info.width}x{info.height}",
            "-framerate", str(info.fps),
            "-i", "-",
            "-i", str(info.path),
            "-map", "0:v:0",
        ]
        if info.has_audio:
            cmd += ["-map", "1:a?", "-c:a", "copy"]
        cmd += [
            "-map_metadata", "1",
            "-vf", f"scale=out_color_matrix={info.color_matrix}:out_range={info.color_range},format=yuv420p",
            "-c:v", codec,
        ]
        if codec.endswith("_nvenc"):
            cmd += ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
        elif codec == "libx265":  # similar quality to NVENC at the same number
            cmd += ["-preset", "medium", "-crf", str(cq + 1), "-x265-params", "log-level=error"]
        else:
            cmd += ["-crf", str(cq)]
        cs = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt2020nc"}[info.color_matrix]
        cmd += ["-colorspace", cs, "-color_range", info.color_range]
        if codec in ("hevc_nvenc", "libx265"):
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", str(self.out_path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_NOWIN)

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed:\n{err}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.kill()
