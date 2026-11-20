"""Output format (H.265 / H.264) and quality level: encoder choice, its settings,
real encodes, and projects saved before the choice existed.

  .venv/Scripts/python -m pytest tests/test_encoding.py -q
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from faceblur import device
from faceblur.project import Project, Settings
from faceblur.video_io import (FrameWriter, _ffbin, encoder_args, probe, quality_value)


def test_quality_levels_follow_the_format():
    assert quality_value("h265", "high") == 19  # the old fixed setting
    assert quality_value("h264", "high") < quality_value("h265", "high")  # H.264 needs a lower number
    for fmt in ("h265", "h264"):
        values = [quality_value(fmt, q) for q in ("best", "high", "medium", "small")]
        assert values == sorted(values) and len(set(values)) == 4
    assert quality_value("h264", "small", cq=30) == 30  # an exact number wins
    with pytest.raises(ValueError):
        quality_value("h265", "ultra")


def test_encoder_settings():
    assert encoder_args("hevc_nvenc", 19)[encoder_args("hevc_nvenc", 19).index("-cq") + 1] == "19"
    assert "-cq" in encoder_args("h264_nvenc", 18)
    x265 = encoder_args("libx265", 19)
    assert x265[x265.index("-crf") + 1] == "20"  # x265 matches NVENC one step higher
    x264 = encoder_args("libx264", 18)
    assert x264[x264.index("-crf") + 1] == "18" and "high" in x264


def test_cpu_device_picks_the_cpu_encoders(monkeypatch):
    monkeypatch.setattr(device, "_PREF", "cpu")
    assert device.encoder("h265") == "libx265" and device.encoder("h264") == "libx264"


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("enc") / "clip.mp4"
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _frames(info, n=10):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (info.height, info.width, 3), dtype=np.uint8)
    return [np.roll(base, 3 * i, axis=1) for i in range(n)]


def _write(info, out: Path, **kw) -> FrameWriter:
    with FrameWriter(out, info, **kw) as w:
        for f in _frames(info):
            w.write(f)
    return w


def _video_stream(path: Path) -> dict:
    out = subprocess.run([_ffbin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-print_format", "json",
                          "-show_streams", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


@pytest.mark.parametrize("fmt, codec_name, tag", [("h265", "hevc", "hvc1"), ("h264", "h264", "avc1")])
def test_cpu_encode_writes_the_chosen_format(clip, tmp_path, monkeypatch, fmt, codec_name, tag):
    monkeypatch.setattr(device, "_PREF", "cpu")
    out = tmp_path / f"{fmt}.mp4"
    w = _write(probe(clip), out, fmt=fmt, quality="medium")
    s = _video_stream(out)
    assert w.codec == ("libx265" if fmt == "h265" else "libx264")
    assert (s["codec_name"], s["codec_tag_string"]) == (codec_name, tag)


def test_quality_changes_the_file_size(clip, tmp_path):
    info = probe(clip)
    best = _write(info, tmp_path / "best.mp4", codec="libx264", quality="best")
    small = _write(info, tmp_path / "small.mp4", codec="libx264", quality="small")
    assert (best.quality, small.quality) == (15, 26)
    assert (tmp_path / "best.mp4").stat().st_size > 1.5 * (tmp_path / "small.mp4").stat().st_size


@pytest.mark.parametrize("gpu, cpu", [("hevc_nvenc", "libx265"), ("h264_nvenc", "libx264")])
def test_frames_too_small_for_nvenc_go_to_the_cpu(clip, tmp_path, gpu, cpu):
    from dataclasses import replace

    info = replace(probe(clip), width=96, height=32)
    w = _write(info, tmp_path / "tiny.mp4", codec=gpu)
    assert w.codec == cpu


@pytest.mark.parametrize("fmt", ["h265", "h264"])
def test_gpu_encode(clip, tmp_path, fmt):
    from faceblur.video_io import ENCODERS

    if not device.nvenc_works(ENCODERS[fmt][0]):
        pytest.skip("no working NVENC on this PC")
    out = tmp_path / f"gpu_{fmt}.mp4"
    w = _write(probe(clip), out, codec=ENCODERS[fmt][0])
    assert w.codec.endswith("_nvenc")
    assert _video_stream(out)["codec_name"] == ("hevc" if fmt == "h265" else "h264")


def _project_file(tmp_path: Path, version: int, settings: dict) -> Path:
    p = tmp_path / "v.faceblur.json"
    p.write_text(json.dumps({"video": "v.mp4", "output": "v_masked.mp4", "version": version,
                             "settings": settings}), encoding="utf-8")
    return p


def test_old_projects_render_as_before(tmp_path):
    s = Project.load(_project_file(tmp_path, 3, {"style": "blur", "cq": 19})).settings
    assert (s.codec, s.quality, s.cq) == ("h265", "high", None)  # same output as before: CQ 19
    s = Project.load(_project_file(tmp_path, 3, {"cq": 24})).settings
    assert s.cq == 24  # a number set on purpose (API) is kept


def test_format_and_quality_are_saved(tmp_path):
    p = Project.load(_project_file(tmp_path, 4, {}))
    assert p.settings == Settings()
    p.settings.codec, p.settings.quality = "h264", "small"
    p.save()
    s = Project.load(p.path).settings
    assert (s.codec, s.quality, s.cq) == ("h264", "small", None)
