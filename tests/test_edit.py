"""Cut & crop: the edited video the readers produce, the audio the writer cuts to
match, the project file and the CLI's --keep / --crop.

  .venv/Scripts/python -m pytest tests/test_edit.py -q
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from faceblur.cli import parse_crop, parse_keep, parse_time
from faceblur.project import Edit, Project
from faceblur.video_io import FrameGrabber, FrameReader, FrameWriter, VideoInfo, _ffbin, probe

FPS = 10
SECONDS = 3


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """3 s, 10 fps, 160x120, with a mono AAC track."""
    out = tmp_path_factory.mktemp("edit") / "clip.mp4"
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", "-f", "lavfi",
                    "-i", f"testsrc=size=160x120:rate={FPS}:duration={SECONDS}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={SECONDS}:sample_rate=48000",
                    "-ac", "1", "-c:a", "aac", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "15",
                    "-shortest", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def source_frames(clip) -> list[np.ndarray]:
    return list(FrameReader(probe(clip), hwaccel=False))


KEEP = [[3, 8], [15, 22], [27, 30]]  # the last part runs to the end of the video
CROP = [20, 10, 100, 80]


def _fake_info(w=1081, h=1920, n=300) -> VideoInfo:
    return VideoInfo(Path("x.mp4"), w, h, Fraction(30), n, n / 30, False, "bt709", "tv", False)


# ------------------------------------------------------------------ the edit

def test_edit_is_clipped_sorted_merged_and_even():
    e = _fake_info().with_edit([[250, 400], [0, 10], [5, 20]], [1, 3, 600.4, 700])
    assert e.keep == ((0, 20), (250, 300))
    assert e.crop == (0, 2, 600, 700)
    assert (e.width, e.height, e.num_frames, e.source_frames) == (600, 700, 70, 300)
    assert e.duration == pytest.approx(70 / 30)
    assert [e.to_source(f) for f in (0, 19, 20, 69)] == [0, 19, 250, 299]
    assert e.edit_key() and e.edit_key() != _fake_info().with_edit([[0, 20]]).edit_key()


def test_edit_that_changes_nothing_is_no_edit():
    info = _fake_info()
    assert info.with_edit([[0, 300]], [0, 0, 1081, 1920]) is info
    assert info.with_edit() is info and info.edit_key() == ""


def test_crop_only_keeps_every_frame():
    e = _fake_info().with_edit(None, [100, 100, 500, 500])
    assert e.keep == () and e.num_frames == 300 and e.segments == ((0, 300),)
    assert e.to_source(123) == 123


def test_edit_that_leaves_nothing_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        _fake_info().with_edit([[400, 500]])
    with pytest.raises(ValueError, match="too small"):
        _fake_info().with_edit(None, [1070, 0, 50, 50])


# ------------------------------------------------------------------ reading

def test_reader_reads_only_kept_parts_cropped(clip, source_frames):
    info = probe(clip).with_edit(KEEP, CROP)
    frames = list(FrameReader(info, hwaccel=False))
    assert len(frames) == info.num_frames == 5 + 7 + 3
    assert frames[0].shape == (80, 100, 3)
    x, y, w, h = CROP
    src = [f[y:y + h, x:x + w].astype(np.int16) for f in source_frames]
    for k, fr in enumerate(frames):
        diffs = [np.abs(fr.astype(np.int16) - s).mean() for s in src]
        assert int(np.argmin(diffs)) == info.to_source(k), f"edited frame {k}"
        assert min(diffs) < 3


def test_reader_starts_inside_the_edit(clip):
    info = probe(clip).with_edit(KEEP, CROP)
    every = list(FrameReader(info, hwaccel=False))
    for start in (0, 4, 5, 6, 12, 14):
        part = list(FrameReader(info, hwaccel=False, start_frame=start))
        assert len(part) == len(every) - start
        assert all(np.array_equal(a, b) for a, b in zip(part, every[start:]))
    assert list(FrameReader(info, hwaccel=False, start_frame=99)) == []


def test_grabber_matches_reader_across_cuts(clip):
    info = probe(clip).with_edit(KEEP, CROP)
    every = list(FrameReader(info, hwaccel=False))
    grab = FrameGrabber(info)
    for k in (0, 4, 5, 11, 12, 14):
        assert np.array_equal(grab.get(k), every[k]), f"edited frame {k}"


# ------------------------------------------------------------------ writing

def _streams(path: Path) -> list[dict]:
    out = subprocess.run([_ffbin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams",
                          "-count_frames", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


@pytest.mark.parametrize("keep", [KEEP, [[0, 12]], [[5, 30]]])
def test_cut_audio_matches_the_video(clip, tmp_path, keep):
    info = probe(clip).with_edit(keep, CROP)
    out = tmp_path / "cut.mp4"
    with FrameWriter(out, info, codec="libx264") as w:
        for fr in FrameReader(info, hwaccel=False):
            w.write(fr)
    v, a = _streams(out)
    assert (v["codec_type"], a["codec_type"], a["codec_name"]) == ("video", "audio", "aac")
    assert (int(v["width"]), int(v["height"])) == (100, 80)
    assert int(v["nb_read_frames"]) == info.num_frames
    assert float(a["duration"]) == pytest.approx(info.num_frames / FPS, abs=0.06)


def test_crop_only_still_copies_audio(clip, tmp_path):
    info = probe(clip).with_edit(None, CROP)
    out = tmp_path / "crop.mp4"
    with FrameWriter(out, info, codec="libx264") as w:
        for fr in FrameReader(info, hwaccel=False):
            w.write(fr)
    v, a = _streams(out)
    assert int(v["nb_read_frames"]) == SECONDS * FPS
    assert float(a["duration"]) == pytest.approx(SECONDS, abs=0.06)


# ------------------------------------------------------------------ project & CLI

def test_project_keeps_the_edit(tmp_path):
    video = tmp_path / "v.mp4"
    p = Project.new(video)
    p.edit = Edit(keep=[[3, 8], [15, 22]], crop=[20, 10, 100, 80])
    p.save()
    q = Project.load(p.path)
    assert q.edit == p.edit and q.edit.active
    data = json.loads(Path(p.path).read_text(encoding="utf-8"))
    del data["edit"]  # a project from before cut & crop
    Path(p.path).write_text(json.dumps(data), encoding="utf-8")
    assert not Project.load(p.path).edit.active


def test_cli_times_and_ranges():
    assert parse_time("90") == 90 and parse_time("1:30.5") == 90.5 and parse_time("1:02:03") == 3723
    assert parse_keep("0:00.3-0:00.8, 2-", 10, 30) == [[3, 8], [20, 30]]
    assert parse_keep("-1", 10, 30) == [[0, 10]]
    assert parse_crop("0, 420, 1080,1080") == [0, 420, 1080, 1080]
    for bad in ("5-2", "1:x-2", "3"):
        with pytest.raises(ValueError):
            parse_keep(bad, 10, 30)
    with pytest.raises(ValueError):
        parse_crop("1,2,3")


def test_each_edit_has_its_own_scan_cache(tmp_path):
    from faceblur.pipeline import clear_scan_cache, detections_cache

    info = _fake_info()
    paths = {detections_cache(tmp_path, i) for i in
             (info, info.with_edit([[0, 20]]), info.with_edit([[0, 21]]), info.with_edit(None, CROP))}
    assert len(paths) == 4 and detections_cache(tmp_path, info).name == "detections.npz"
    for p in [*paths, tmp_path / "thumbs_96.npz", tmp_path / "thumbs_96_abc.npz", tmp_path / "project.json"]:
        p.write_bytes(b"x")
    clear_scan_cache(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["project.json"]


# ------------------------------------------------------------------ temp folder

def _empty_detections(n: int):
    from faceblur.identity import Detections

    return Detections(np.zeros(0, np.int64), np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                      np.zeros((0, 5, 2), np.float32), np.zeros((0, 512), np.float32), np.zeros(0, np.int64), n)


def test_render_writes_in_the_temp_folder(clip, tmp_path, monkeypatch):
    import tempfile

    from faceblur.pipeline import Analysis, render
    from faceblur.project import Manual

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    temp = tmp_path / "tmp" / "FaceBlur"
    temp.mkdir(parents=True)
    stale = temp / "old.abc.partial.mp4"
    stale.write_bytes(b"x")
    import os
    os.utime(stale, (0, 0))  # from a crash long ago

    info = probe(clip).with_edit(KEEP, CROP)
    analysis = Analysis(clip, info, _empty_detections(info.num_frames), tmp_path / "work")
    project = Project.new(clip)
    project.output = str(tmp_path / "out" / "masked.mp4")
    project.manual = [Manual(frame=2, box=[10, 10, 50, 50], span=30)]
    seen = []

    def progress(stage, done, total):
        seen.append((stage, sorted(p.name for p in temp.iterdir())))

    result = render(project, analysis, progress=progress, log=lambda *_: None)
    during = [names for stage, names in seen if stage == "Rendering" and names]  # FFmpeg creates it a bit later
    assert during and all(len(n) == 1 and n[0].startswith("masked.") and n[0].endswith(".partial.mp4")
                          for n in during)
    assert "Saving the video" in [s for s, _ in seen]
    assert result.output == Path(project.output) and result.output.stat().st_size > 0
    assert list(temp.iterdir()) == []  # moved into place; the stale leftover is gone too
    assert not list(result.output.parent.glob("*.partial*"))
