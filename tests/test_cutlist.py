"""The cut model (split / remove / in / out) and the desktop app's edit mode.

  .venv/Scripts/python -m pytest tests/test_cutlist.py -q
"""

from __future__ import annotations

import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from faceblur.edit import CutList
from faceblur.video_io import VideoInfo, _ffbin, fit_crop


def test_split_and_remove():
    c = CutList(100)
    assert c.keep() == [] and c.kept_frames() == 100
    assert c.split(30) and c.split(60) and not c.split(30) and not c.split(0) and not c.split(100)
    assert c.split_points() == [30, 60]
    assert c.toggle_at(45)  # the middle part
    assert c.keep() == [[0, 30], [60, 100]] and c.removed_at(30) and not c.removed_at(29)
    assert c.toggle_at(45) and c.keep() == []  # back again; the split points stay
    assert c.split_points() == [30, 60]


def test_last_part_cannot_be_removed():
    c = CutList(100)
    assert not c.toggle_at(10)
    c.split(50)
    assert c.toggle_at(10) and not c.toggle_at(70)
    assert c.keep() == [[50, 100]]


def test_in_and_out_points():
    c = CutList(100)
    assert c.set_in(20) and c.keep() == [[20, 100]]
    assert c.set_out(79) and c.keep() == [[20, 80]]  # the out frame itself stays
    assert c.kept_frames() == 60
    assert c.set_in(10) and c.keep() == [[10, 80]]  # moving the in point earlier brings frames back
    assert not c.set_in(10)  # nothing changes


def test_in_point_on_a_removed_part_keeps_from_there():
    c = CutList(100)
    c.split(50)
    c.toggle_at(60)  # 50-100 removed
    assert c.set_in(70) and c.keep() == [[70, 100]]


def test_from_keep_and_undo_state():
    c = CutList(100, [[60, 80], [10, 30]])
    assert c.keep() == [[10, 30], [60, 80]]
    assert [p[2] for p in c.parts] == [True, False, True, False, True]
    st = c.state()
    c.toggle_at(70)
    c.restore(st)
    assert c.keep() == [[10, 30], [60, 80]]
    assert CutList(100, [[0, 100]]).keep() == []


def test_from_source_and_fit_crop():
    info = VideoInfo(Path("x.mp4"), 200, 100, Fraction(10), 100, 10.0, False, "bt709", "tv", False)
    e = info.with_edit([[10, 30], [60, 80]])
    assert [e.from_source(f) for f in (10, 29, 60, 79)] == [0, 19, 20, 39]
    assert e.from_source(5) is None and e.from_source(5, after=True) == 0
    assert e.from_source(40, after=True) == 20 and e.from_source(90, after=True) is None
    assert fit_crop((3.4, 5.6, 101, 51), 200, 100) == (2, 6, 100, 50)
    assert fit_crop((0, 0, 200, 100), 200, 100) is None and fit_crop(None, 200, 100) is None
    assert fit_crop((150, 0, 400, 100), 200, 100) == (150, 0, 50, 100)  # clipped to the frame


# ------------------------------------------------------------------ the desktop app

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setOrganizationName("FaceBlurTest")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from faceblur.gui import app as gui
    from faceblur import pipeline

    video = tmp_path / "clip.mp4"
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)  # work/ goes to the test folder
    w = gui.MainWindow()
    started = []
    monkeypatch.setattr(w, "_run_task", lambda fn, done, label: started.append(label))
    w.open_path(str(video))
    yield w, started
    w.dirty = False
    w.close()


def test_new_video_opens_for_editing_without_scanning(window):
    w, started = window
    assert w.editing and started == []
    assert w.timeline.parts is not None and w.view.crop_mode
    assert not w.tabs.isTabEnabled(1) and not w.render_btn.isEnabled()


def test_edit_then_detect_faces(window):
    w, started = window
    w.seek(10)
    w._cut_in()
    w.seek(20)
    w._cut_split()
    w.seek(25)
    w._cut_toggle()  # 20-30 removed
    w._on_crop_changed((10.3, 20, 161, 121))
    assert w.project.edit.keep == [[10, 20]] and w.project.edit.crop == [10, 20, 160, 120]
    w._edit_undo()
    assert w.project.edit.crop is None
    w._edit_redo()
    assert w.project.edit.crop == [10, 20, 160, 120]
    w._finish_edit()
    assert not w.editing and started == ["Detecting faces"]
    assert (w.info.width, w.info.height, w.info.num_frames) == (160, 120, 10)
    assert w.timeline.parts is None and w.timeline.n == 10


def test_changing_the_edit_asks_before_clearing_choices(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from faceblur.project import Label

    w, _ = window
    w.project.labels.append(Label(0, [1, 2, 3, 4], 0))
    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: answers.pop(0))
    w.seek(5)
    w._cut_in()  # No: nothing changes, the choice stays
    assert w.project.edit.keep == [] and w.project.labels
    w._cut_in()  # Yes: cut, and the choice is cleared
    assert w.project.edit.keep == [[5, 30]] and not w.project.labels
    w.seek(8)
    w._cut_in()  # asked once per edit session
    assert w.project.edit.keep == [[8, 30]] and answers == []


def test_reset_project_starts_over(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from faceblur.pipeline import workdir
    from faceblur.project import Label

    w, started = window
    video = Path(w.project.video)
    w.seek(5)
    w._cut_in()
    w._on_crop_changed((10, 20, 160, 120))
    w._finish_edit()
    started.clear()  # (the scan it starts is not run here)
    w.seek(3)
    w._add_manual([10, 10, 40, 40])  # a drawn "blur this area"
    w._recompute()  # (normally on a short timer)
    assert w.corr_list.count() == 1
    w.project.labels.append(Label(0, [1, 2, 3, 4], 0))
    w.project.settings.style = "fill"
    w.save_project()
    wd = workdir(video)
    (wd / "detections.npz").write_bytes(b"x")
    (wd / "flagged").mkdir()
    (wd / "flagged" / "000001.jpg").write_bytes(b"x")
    output = video.with_name("clip_masked.mp4")
    output.write_bytes(b"rendered")
    project_file = Path(w.project.path)
    assert project_file.exists()

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    w.reset_project()  # cancelled: nothing is touched
    assert project_file.exists() and (wd / "detections.npz").exists() and w.project.edit.keep

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.reset_project()
    assert not project_file.exists() and not (wd / "detections.npz").exists() and not (wd / "flagged").exists()
    assert video.exists() and output.read_bytes() == b"rendered"  # the user's files stay
    p = w.project
    assert not p.edit.active and not p.labels and not p.manual and p.settings.style == "pixelate"
    assert w.corr_list.count() == 0  # the drawn box is gone from the list too
    assert w.editing and w.info.num_frames == 30 and w.cuts.keep() == [] and started == []


def test_cancelled_scan_goes_back_to_cut_and_crop(window):
    """After cancelling the face scan, the cut & crop and "Faces at any angle"
    can be changed before detecting again."""
    w, started = window
    w._finish_edit()  # Detect faces
    assert not w.editing and started == ["Detecting faces"] and w.analysis is None
    w._on_cancelled("Detecting faces")  # what the task reports when stopped
    w._on_task_finished()
    assert w.editing and w.any_angle_cb.isEnabled()
    w.any_angle_cb.setChecked(True)
    assert w.project.edit.any_angle and "Detect faces" in w.edit_btn.text()
    w._finish_edit()
    assert started == ["Detecting faces", "Detecting faces"]


def test_failed_scan_goes_back_to_cut_and_crop(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    w, started = window
    w._finish_edit()
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    w._on_failed("decoder error\n\ndetails")
    w._on_task_finished()
    assert w.editing and w.any_angle_cb.isEnabled()
