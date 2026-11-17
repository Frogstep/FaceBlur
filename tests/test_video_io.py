"""FFmpeg I/O: audio tracks the output can take, and FFmpeg's own message when
encoding fails.

  .venv/Scripts/python -m pytest tests/test_video_io.py -q
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from faceblur import video_io
from faceblur.video_io import FrameWriter, _ffbin, probe


def _fake_ffprobe(monkeypatch, streams):
    out = json.dumps({"streams": streams, "format": {"duration": "2.0"}})
    real = subprocess.run

    def run(cmd, *a, **kw):
        if "ffprobe" in Path(cmd[0]).name.lower():
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        return real(cmd, *a, **kw)

    monkeypatch.setattr(video_io.subprocess, "run", run)


VIDEO = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1080, "height": 2340,
         "r_frame_rate": "60/1", "avg_frame_rate": "60/1", "nb_frames": "120"}


def test_empty_phone_audio_track_is_skipped(monkeypatch):
    # what some phone camera apps write: an audio track with no codec, 0 Hz, 0 channels
    _fake_ffprobe(monkeypatch, [VIDEO, {"index": 1, "codec_type": "audio", "codec_name": "unknown",
                                        "sample_rate": "0", "channels": 0}])
    info = probe("phone.mp4")
    assert not info.has_audio and info.audio == () and info.skipped_audio == 1


def test_usable_audio_tracks_are_kept(monkeypatch):
    _fake_ffprobe(monkeypatch, [
        VIDEO,
        {"index": 1, "codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2},
        {"index": 2, "codec_type": "audio", "codec_name": "unknown", "sample_rate": "0", "channels": 0},
        {"index": 3, "codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "48000", "channels": 1},
    ])
    info = probe("x.mov")
    assert info.has_audio and info.audio == ((1, "aac"), (3, "pcm_s16le")) and info.skipped_audio == 1


@pytest.fixture(scope="module")
def small_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v") / "small.mp4"
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=48000",
                    "-ac", "1", "-c:a", "pcm_s16le", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest",
                    str(out.with_suffix(".mov"))], check=True)
    return out.with_suffix(".mov")


def _encode(info, out: Path, codec="libx264") -> None:
    frame = np.zeros((info.height, info.width, 3), np.uint8)
    with FrameWriter(out, info, codec=codec) as w:
        for _ in range(info.num_frames):
            w.write(frame)


def test_audio_mp4_cannot_hold_is_converted(small_video, tmp_path):
    info = probe(small_video)
    assert info.audio == ((1, "pcm_s16le"),)
    out = tmp_path / "out.mp4"
    _encode(info, out)
    streams = json.loads(subprocess.run([_ffbin("ffprobe"), "-v", "error", "-print_format", "json",
                                         "-show_streams", str(out)], capture_output=True, text=True).stdout)["streams"]
    assert [(s["codec_type"], s["codec_name"]) for s in streams] == [("video", "h264"), ("audio", "aac")]


def test_ffmpeg_error_reaches_the_caller(small_video, tmp_path):
    info = replace(probe(small_video), audio=((7, "aac"),))  # a track that does not exist
    with pytest.raises(RuntimeError) as e:
        _encode(info, tmp_path / "out.mp4")
    msg = str(e.value)
    assert "FFmpeg stopped while encoding out.mp4" in msg and "1:7" in msg
    assert "Errno 22" not in msg
