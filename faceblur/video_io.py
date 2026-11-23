"""FFmpeg-based decode and encode via subprocess pipes.

Decode produces BGR uint8 frames in display orientation. Encode writes H.265 or
H.264 (NVENC on the GPU, x265/x264 on the CPU) at a chosen quality level and
copies the audio from the source file unchanged.

Cut & crop (VideoInfo.with_edit) are applied while decoding: the readers seek to
each kept range and crop each frame, so cut parts are never decoded and the
source file is never changed. The writer cuts the audio to match.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
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
    has_audio: bool  # at least one audio track that can go into the output
    color_matrix: str  # swscale matrix name: bt709 / bt601 / bt2020
    color_range: str  # tv / pc
    vfr: bool
    audio: tuple[tuple[int, str], ...] = ()  # usable audio tracks: (stream index, codec name)
    skipped_audio: int = 0  # audio tracks FFmpeg cannot read (left out of the output)
    # turn, cut & crop (see turned, with_edit); width, height, num_frames and
    # duration above are then those of the edited video
    keep: tuple[tuple[int, int], ...] = ()  # kept [start, end) source frames; () = all of them
    crop: tuple[int, int, int, int] | None = None  # x, y, w, h in display pixels (after turning)
    turn: int = 0  # degrees clockwise (0/90/180/270), applied before the crop
    source_frames: int = 0  # frames in the source when cut (0: same as num_frames)

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def edited(self) -> bool:
        return bool(self.keep or self.crop or self.turn)

    @property
    def segments(self) -> tuple[tuple[int, int], ...]:
        """The source frame ranges that make up the video, in order."""
        return self.keep or ((0, self.source_frames or self.num_frames),)

    def to_source(self, frame: int) -> int:
        """Source frame number of a frame of the (edited) video."""
        for a, b in self.segments:
            if frame < b - a:
                return a + frame
            frame -= b - a
        return self.segments[-1][1] + frame

    def from_source(self, src: int, after: bool = False) -> int | None:
        """Frame of the (edited) video showing source frame `src`; None when it
        was cut, or with after=True the first kept frame after it (None at the end)."""
        k = 0
        for a, b in self.segments:
            if src < a:
                return k if after else None
            if src < b:
                return k + src - a
            k += b - a
        return None

    def edit_key(self) -> str:
        """Short id of the cut & crop ("" when unedited), for cache file names."""
        if not self.edited:
            return ""
        spec = [list(map(list, self.keep)), list(self.crop) if self.crop else None]
        if self.turn:  # (unturned edits keep the keys they had before turning existed)
            spec.append(self.turn)
        spec = json.dumps(spec)
        return hashlib.sha1(spec.encode()).hexdigest()[:10]

    def turned(self, turn: int) -> "VideoInfo":
        """This (unedited) video turned `turn` degrees clockwise (a multiple of
        90), e.g. for a camera that filmed on its side. Width and height swap
        for 90 and 270. A crop given to with_edit afterwards is in the turned
        picture."""
        if self.edited:
            raise ValueError("this video is already edited")
        turn = int(turn) % 360
        if turn % 90:
            raise ValueError(f"the video can only be turned by 90, 180 or 270 degrees (got {turn})")
        if not turn:
            return self
        w, h = (self.height, self.width) if turn % 180 else (self.width, self.height)
        return replace(self, width=w, height=h, turn=turn)

    def with_edit(self, keep=None, crop=None) -> "VideoInfo":
        """This (unedited) video with only the `keep` ranges ([start, end) source
        frames; empty = all) and cropped to `crop` ([x, y, w, h] display pixels;
        None = full frame). Ranges are clipped, sorted and merged; the crop is
        clipped to the frame and rounded to even numbers (4:2:0 video needs
        them). Returns self when the edit changes nothing. May follow turned()."""
        if self.keep or self.crop:
            raise ValueError("this video is already edited")
        n = self.num_frames
        merged: list[list[int]] = []
        for a, b in sorted((max(0, int(a)), min(n, int(b))) for a, b in keep or ()):
            if b <= a:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if keep and not merged:
            raise ValueError("the cut leaves no frames")
        if merged == [[0, n]]:
            merged = []
        box = fit_crop(crop, self.width, self.height)
        if not merged and box is None:
            return self
        keep_t = tuple((a, b) for a, b in merged)
        frames = sum(b - a for a, b in keep_t) if keep_t else n
        return replace(self, width=box[2] if box else self.width, height=box[3] if box else self.height,
                       num_frames=frames, duration=frames / self.fps_float if keep_t else self.duration,
                       keep=keep_t, crop=box, source_frames=n)


MIN_CROP = 16


def fit_crop(crop, width: int, height: int) -> tuple[int, int, int, int] | None:
    """[x, y, w, h] clipped to a width x height frame and rounded down to even
    numbers (4:2:0 video needs them); None for no crop or the full frame."""
    if not crop:
        return None
    x, y, w, h = (int(round(float(v))) for v in crop)
    x, y = min(max(0, x), width), min(max(0, y), height)
    w, h = min(w, width - x), min(h, height - y)
    x, y, w, h = x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2
    if w < MIN_CROP or h < MIN_CROP:
        raise ValueError(f"the crop is too small ({w}x{h}; at least {MIN_CROP}x{MIN_CROP})")
    if x == 0 and y == 0 and w >= width - 1 and h >= height - 1:
        return None
    return x, y, w, h


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
        # ffprobe writes UTF-8; without an explicit encoding Windows decodes with
        # the locale code page, fails on non-ASCII paths, and .stdout comes back None
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True, **_NOWIN,
    ).stdout
    info = json.loads(out)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    # some phone apps write an empty placeholder audio track (no codec, 0 Hz, 0
    # channels); copying it makes the MP4 muxer refuse the whole output
    audio, skipped = [], 0
    for s in info["streams"]:
        if s["codec_type"] != "audio":
            continue
        if s.get("codec_name") not in (None, "", "unknown", "none") \
                and int(s.get("sample_rate") or 0) > 0 and int(s.get("channels") or 0) > 0:
            audio.append((int(s["index"]), s["codec_name"]))
        else:
            skipped += 1

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

    return VideoInfo(path, w, h, fps, nb, duration, bool(audio), matrix, rng, vfr, tuple(audio), skipped)


def _cfr(info: VideoInfo) -> list[str]:
    """Frames on a constant frame-rate clock: frame N is what the video shows at
    N / fps. Some cameras leave frame slots empty (an AVI that says 30 fps but
    holds 25 real frames a second) and phones film at a variable rate; read
    back to back, such frames drift away from the sound and from frames found
    by seeking. A missing slot is filled with a neighbouring real frame."""
    return ["-fps_mode", "cfr", "-r", str(info.fps)]


_TURN_FILTER = {90: "transpose=clock", 180: "hflip,vflip", 270: "transpose=cclock"}


def _decode_filter(info: VideoInfo) -> str:
    vf = f"scale=in_color_matrix={info.color_matrix}:in_range={info.color_range}"
    if info.crop:
        x, y, w, h = info.crop
        vf = f"crop={w}:{h}:{x}:{y}," + vf
    if info.turn:
        vf = f"{_TURN_FILTER[info.turn]}," + vf
    return vf


def turn_crop(crop, turn: int, width: int, height: int):
    """[x, y, w, h] in a width x height picture, moved to where it is after
    turning the picture `turn` degrees clockwise (None stays None)."""
    if not crop:
        return crop
    x, y, w, h = crop
    turn %= 360
    if turn == 90:
        return [height - (y + h), x, h, w]
    if turn == 180:
        return [width - (x + w), height - (y + h), w, h]
    if turn == 270:
        return [y, width - (x + w), h, w]
    return list(crop)


class FrameReader:
    """Iterates BGR frames from a video using FFmpeg (NVDEC decode when the device
    allows). An edited video is read one kept range at a time (one FFmpeg run
    each, starting with a seek), so the parts cut out are never decoded."""

    def __init__(self, info: VideoInfo, hwaccel: bool | None = None, start_frame: int = 0):
        self.info = info
        if hwaccel is None:
            from .device import decode_on_gpu

            hwaccel = decode_on_gpu()
        self._hwaccel = hwaccel
        # (first source frame, number of frames or None = to the end) per FFmpeg run;
        # start_frame is a frame of the edited video
        self._parts: list[tuple[int, int | None]] = []
        skip, segs = max(0, start_frame), info.segments
        src_end = info.source_frames or info.num_frames
        for k, (a, b) in enumerate(segs):
            if skip >= b - a:
                skip -= b - a
                continue
            to_end = k == len(segs) - 1 and b >= src_end
            self._parts.append((a + skip, None if to_end else b - a - skip))
            skip = 0
        self._frame_bytes = info.width * info.height * 3

    def _cmd(self, start: int, count: int | None) -> list[str]:
        cmd = [_ffbin("ffmpeg"), "-v", "error", "-nostdin"]
        if self._hwaccel:
            cmd += ["-hwaccel", "cuda"]
        if start > 0:  # accurate seek, same convention as FrameGrabber
            cmd += ["-ss", f"{max(0.0, (start - 0.25) / self.info.fps_float):.6f}"]
        cmd += ["-i", str(self.info.path), "-map", "0:v:0", "-vf", _decode_filter(self.info)]
        if count is not None:
            cmd += ["-frames:v", str(count)]
        cmd += [*_cfr(self.info), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return cmd

    def __iter__(self) -> Iterator[np.ndarray]:
        for start, count in self._parts:
            yield from self._run(self._cmd(start, count))

    def _run(self, cmd: list[str]) -> Iterator[np.ndarray]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    accurate input seeking, so frame N here is the same as frame N from FrameReader
    (also for an edited video: N counts edited frames)."""

    def __init__(self, info: VideoInfo, cache_size: int = 48):
        self.info = info
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []
        self._size = cache_size

    def get(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(idx, self.info.num_frames - 1))
        if idx in self._cache:
            return self._cache[idx]
        t = max(0.0, (self.info.to_source(idx) - 0.25) / self.info.fps_float)
        cmd = [
            _ffbin("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{t:.6f}", "-i", str(self.info.path),
            "-map", "0:v:0", "-frames:v", "1",
            "-vf", _decode_filter(self.info), *_cfr(self.info),
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


# output formats: (GPU encoder, CPU encoder)
ENCODERS = {"h265": ("hevc_nvenc", "libx265"), "h264": ("h264_nvenc", "libx264")}
# smallest frame each NVENC encoder takes (rounded up to even); smaller ones (e.g. a
# tight crop) go to the CPU encoder, which is fast at that size anyway
NVENC_MIN = {"hevc_nvenc": (130, 34), "h264_nvenc": (146, 50)}
# quality levels -> NVENC CQ / x264 CRF (lower = better, bigger file). H.264 needs
# lower numbers than H.265 for the same picture; "high" for H.265 is the old fixed 19.
QUALITY_LEVELS = ("best", "high", "medium", "small")
QUALITY = {"h265": {"best": 16, "high": 19, "medium": 23, "small": 28},
           "h264": {"best": 15, "high": 18, "medium": 22, "small": 26}}

# audio codecs the MP4 container can carry as they are
_MP4_AUDIO = {"aac", "mp3", "alac", "ac3", "eac3", "opus", "flac"}


def format_of(codec: str) -> str:
    """'h264' or 'h265' for an FFmpeg encoder name."""
    return "h264" if "264" in codec else "h265"


def quality_value(fmt: str, quality: str = "high", cq: int | None = None) -> int:
    """The CQ/CRF number for a format and quality level; cq, when given, wins."""
    if cq is not None:
        return int(cq)
    if quality not in QUALITY_LEVELS:
        raise ValueError(f"quality must be one of {', '.join(QUALITY_LEVELS)}, not {quality!r}")
    return QUALITY[fmt][quality]


def encoder_args(codec: str, value: int) -> list[str]:
    """FFmpeg options for one encoder at CQ/CRF value."""
    if codec.endswith("_nvenc"):
        return ["-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", str(value), "-b:v", "0"]
    if codec == "libx265":  # similar quality to NVENC at the same number
        return ["-preset", "medium", "-crf", str(value + 1), "-x265-params", "log-level=error"]
    if codec == "libx264":
        return ["-preset", "medium", "-crf", str(value), "-profile:v", "high"]
    return ["-crf", str(value)]


class FrameWriter:
    """Encodes BGR frames to H.265 or H.264 (NVENC on the GPU, x265/x264 on the CPU)
    and copies audio from the source.

    fmt ('h265'/'h264') and quality (QUALITY_LEVELS) pick the encoder and its
    setting; cq overrides the level's number; codec forces an FFmpeg encoder."""

    def __init__(self, out_path: str | Path, info: VideoInfo, cq: int | None = None,
                 codec: str | None = None, fmt: str = "h265", quality: str = "high"):
        if codec is None:
            from .device import encoder

            if fmt not in ENCODERS:
                raise ValueError(f"format must be one of {', '.join(ENCODERS)}, not {fmt!r}")
            codec = encoder(fmt)
        fmt = format_of(codec)
        small = NVENC_MIN.get(codec)
        if small and (info.width < small[0] or info.height < small[1]):
            codec = ENCODERS[fmt][1]
        self.codec = codec
        self.quality = quality_value(fmt, quality, cq)
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
        # only the audio tracks FFmpeg can read (see probe); copied unchanged when
        # MP4 can hold them, otherwise converted to AAC
        if info.keep and info.audio:
            # cut: the kept ranges of each track joined up, so it can only be AAC
            fps, src_end, graph = info.fps_float, info.source_frames, []
            for k, (index, _) in enumerate(info.audio):
                parts = []
                for i, (a, b) in enumerate(info.keep):
                    end = "" if b >= src_end else f":end={b / fps:.6f}"
                    graph.append(f"[1:{index}]atrim=start={a / fps:.6f}{end},asetpts=PTS-STARTPTS[a{k}p{i}]")
                    parts.append(f"[a{k}p{i}]")
                graph.append("".join(parts) + f"concat=n={len(parts)}:v=0:a=1[a{k}]")
            cmd += ["-filter_complex", ";".join(graph)]
            for k in range(len(info.audio)):
                cmd += ["-map", f"[a{k}]", f"-c:a:{k}", "aac", f"-b:a:{k}", "192k"]
        else:
            for k, (index, acodec) in enumerate(info.audio):
                cmd += ["-map", f"1:{index}"]
                if acodec in _MP4_AUDIO:
                    cmd += [f"-c:a:{k}", "copy"]
                else:
                    cmd += [f"-c:a:{k}", "aac", f"-b:a:{k}", "192k"]
        if info.keep:
            cmd += ["-map_chapters", "-1"]  # their times would be wrong after a cut
        cmd += [
            "-map_metadata", "1",
            "-vf", f"scale=out_color_matrix={info.color_matrix}:out_range={info.color_range},format=yuv420p",
            "-c:v", codec,
        ]
        cmd += encoder_args(codec, self.quality)
        cs = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt2020nc"}[info.color_matrix]
        cmd += ["-colorspace", cs, "-color_range", info.color_range]
        if fmt == "h265":
            cmd += ["-tag:v", "hvc1"]  # so Apple players open it
        cmd += ["-movflags", "+faststart", str(self.out_path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **_NOWIN)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError:
            # FFmpeg has exited (on Windows this shows as EINVAL, elsewhere as a
            # broken pipe); its own message says why
            self._proc.kill()
            err = self._proc.stderr.read().decode(errors="replace").strip()
            self._proc.wait()
            raise RuntimeError(f"FFmpeg stopped while encoding {self.out_path.name}:\n{err or '(no message)'}") from None

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
