"""FaceBlur main window.

Workflow: open a video -> faces are detected (cached) and shown as face cards,
all not blurred -> the user clicks the faces to blur in the Review tab -> fix
individual frames (click a box: "don't blur here"; drag a box: "blur this
area") -> render -> review the safety check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

from ..identity import Cancelled  # noqa: F401  (re-exported for workers)
from ..pipeline import Analysis, card_thumbnails, load_target_refs, render, safety_check
from ..project import Exclusion, Manual, Project
from ..video_io import FrameGrabber, probe
from . import i18n
from .i18n import HELP_EN, LANGUAGES, tr, trn
from .review_tab import ReviewPanel
from .widgets import BLUR_COLOR, EXCLUDED_COLOR, KEEP_COLOR, OTHER_COLOR, Overlay, Timeline, VideoView
from .workers import Player, Task

VIDEO_EXTS = "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.m2ts *.wmv"
_WINDOWS: list = []  # keeps the current main window alive across language switches


def video_filter() -> str:
    return f"{tr('Videos')} ({VIDEO_EXTS});;{tr('All files')} (*)"


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceBlur")
        self._fit_to_screen()
        self.setAcceptDrops(True)

        self.project: Project | None = None
        self.analysis: Analysis | None = None
        self.grabber: FrameGrabber | None = None
        self.out_grabber: FrameGrabber | None = None
        self.info = None
        self.frame = 0
        self.plans = []
        self.excluded: set[int] = set()
        self.blurred: list[bool] = []  # per face card: marked to blur
        self.flagged: list[dict] = []
        self.dirty = False
        self.task: Task | None = None
        self.player: Player | None = None

        self._build_ui()
        self._build_menu()
        self._recompute_timer = QTimer(self, singleShot=True, interval=120)
        self._recompute_timer.timeout.connect(self._recompute)
        # every change is saved to <video>.faceblur.json shortly after it is made
        self._autosave_timer = QTimer(self, singleShot=True, interval=1000)
        self._autosave_timer.timeout.connect(lambda: self.dirty and self.save_project())
        self._set_enabled()

    # ================================================================ UI
    def _build_ui(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)

        # ---- left: video, timeline, transport
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        self.view = VideoView()
        self.view.itemClicked.connect(self._on_item_clicked)
        self.view.boxDrawn.connect(self._on_box_drawn)
        lv.addWidget(self.view, 1)

        # job strip: visible while detecting / rendering / checking
        self.job_strip = QWidget()
        js = QHBoxLayout(self.job_strip)
        js.setContentsMargins(0, 2, 0, 2)
        self.job_label = QLabel()
        self.job_label.setStyleSheet("font-weight: bold;")
        self.job_bar = QProgressBar()
        self.job_bar.setTextVisible(True)
        self.job_bar.setMinimumWidth(260)
        job_cancel = QPushButton(tr("Cancel"))
        job_cancel.clicked.connect(self.cancel_task)
        js.addWidget(self.job_label)
        js.addWidget(self.job_bar, 1)
        js.addWidget(job_cancel)
        self.job_strip.setVisible(False)
        lv.addWidget(self.job_strip)
        self._stage = None  # (name, start time, frames done at start) for the time estimate

        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        lv.addWidget(self.timeline)

        bar = QHBoxLayout()
        def tbtn(text, tip, fn):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tr(tip))
            b.setMinimumSize(44, 36)
            b.setStyleSheet("QToolButton { font-size: 18px; padding: 2px 8px; }")
            b.clicked.connect(fn)
            bar.addWidget(b)
            return b
        tbtn("⏮", "Previous second (Shift+Left)", lambda: self.step(-self._fps_int()))
        tbtn("◀", "Previous frame (Left)", lambda: self.step(-1))
        self.play_btn = tbtn("▶", "Play / pause (Space)", self.toggle_play)
        tbtn("▶|", "Next frame (Right)", lambda: self.step(1))
        tbtn("⏭", "Next second (Shift+Right)", lambda: self.step(self._fps_int()))
        self.pos_label = QLabel("–")
        self.pos_label.setMinimumWidth(190)
        bar.addWidget(self.pos_label)
        bar.addStretch(1)
        self.view_orig = QRadioButton(tr("Original"))
        self.view_out = QRadioButton(tr("Rendered output"))
        self.view_orig.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.view_orig)
        grp.addButton(self.view_out)
        self.view_orig.toggled.connect(lambda _: self.show_frame())
        bar.addWidget(self.view_orig)
        bar.addWidget(self.view_out)
        self.preview_cb = QCheckBox(tr("Preview mask"))
        self.preview_cb.setToolTip(tr("Show the mask on the original (box-based; the render also uses SAM 2)"))
        self.preview_cb.toggled.connect(lambda _: self.show_frame())
        self.boxes_cb = QCheckBox(tr("Show face boxes"))
        self.boxes_cb.setChecked(True)
        self.boxes_cb.toggled.connect(lambda _: self.show_frame())
        bar.addWidget(self.preview_cb)
        bar.addWidget(self.boxes_cb)
        lv.addLayout(bar)
        hint = QLabel(tr("Click a face box to blur it or leave it visible. "
                         "Drag a rectangle to blur an area the detector missed."))
        hint.setStyleSheet("color: #999;")
        hint.setWordWrap(True)
        lv.addWidget(hint)
        self.tabs = QTabWidget()
        self.tabs.addTab(left, tr("Video"))
        self.review = ReviewPanel()
        self.review.blurCards.connect(self.set_blur)
        self.review.jump.connect(self._jump_from_review)
        self.tabs.addTab(self.review, tr("Review faces"))
        split.addWidget(self.tabs)

        # ---- right: side panel
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(6, 6, 6, 6)

        # face review summary (the cards themselves are in the Review faces tab)
        g = QGroupBox(tr("Face review"))
        fv = QVBoxLayout(g)
        self.faces_note = QLabel(tr("Open a video to detect faces."))
        self.faces_note.setWordWrap(True)
        fv.addWidget(self.faces_note)
        self.review_btn = QPushButton(tr("Open the face review"))
        self.review_btn.clicked.connect(lambda: self.tabs.setCurrentWidget(self.review))
        fv.addWidget(self.review_btn)
        sv.addWidget(g)

        # mask settings
        g = QGroupBox(tr("Mask"))
        form = QFormLayout(g)
        self.style_combo = QComboBox()
        self.style_combo.addItem(tr("Pixelate"), "pixelate")
        self.style_combo.addItem(tr("Blur (pixelated first)"), "blur")
        self.style_combo.addItem(tr("Solid black"), "fill")
        self.strength_spin = QDoubleSpinBox(minimum=0.5, maximum=4.0, singleStep=0.25, value=1.0)
        self.strength_spin.setToolTip(tr("Higher = coarser blocks"))
        self.pad_spin = QDoubleSpinBox(minimum=0.0, maximum=1.5, singleStep=0.05, value=0.3)
        self.pad_spin.setToolTip(tr("Extra margin around each face box"))
        self.sam_cb = QCheckBox(tr("Precise outline with SAM 2 (slower)"))
        self.sam_cb.setChecked(False)
        form.addRow(tr("Style"), self.style_combo)
        form.addRow(tr("Strength"), self.strength_spin)
        form.addRow(tr("Margin"), self.pad_spin)
        form.addRow(self.sam_cb)
        for w, sig in ((self.style_combo, "currentIndexChanged"), (self.strength_spin, "valueChanged"),
                       (self.pad_spin, "valueChanged"), (self.sam_cb, "toggled")):
            getattr(w, sig).connect(self._settings_changed)
        sv.addWidget(g)

        # corrections
        g = QGroupBox(tr("Corrections"))
        cv = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Drawn box covers ±")))
        self.span_spin = QSpinBox(minimum=0, maximum=600, value=15)
        self.span_spin.setSuffix(tr(" frames"))
        row.addWidget(self.span_spin)
        row.addStretch(1)
        cv.addLayout(row)
        self.corr_list = QListWidget()
        self.corr_list.setMaximumHeight(90)
        self.corr_list.setMinimumHeight(48)
        self.corr_list.itemDoubleClicked.connect(lambda it: self.seek(it.data(Qt.ItemDataRole.UserRole + 1)))
        cv.addWidget(self.corr_list)
        rm = QPushButton(tr("Remove selected correction"))
        rm.clicked.connect(self._remove_correction)
        cv.addWidget(rm)
        sv.addWidget(g)

        # processing device (scrolls with the panel)
        g = QGroupBox(tr("Processing"))
        rv = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Device")))
        self.device_combo = QComboBox()
        self.device_combo.addItem(tr("Auto (GPU if available)"), "auto")
        self.device_combo.addItem("GPU (NVIDIA)", "gpu")
        self.device_combo.addItem(tr("CPU only (slow)"), "cpu")
        self.device_combo.setToolTip(tr("Where face detection, SAM 2 and encoding run. "
                                        "CPU works on any PC but is much slower."))
        row.addWidget(self.device_combo, 1)
        rv.addLayout(row)
        self.device_label = QLabel("")
        self.device_label.setWordWrap(True)
        self.device_label.setStyleSheet("color: #999;")
        rv.addWidget(self.device_label)
        from .. import device as _dev

        saved = str(QSettings().value("device", _dev.preference()))
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(saved)))
        _dev.set_device(self.device_combo.currentData())
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        QTimer.singleShot(0, self._show_device)
        sv.addWidget(g)

        # render footer: pinned under the scrolling panel, always visible
        footer = QGroupBox(tr("Render"))
        rv = QVBoxLayout(footer)
        rv.setSpacing(4)
        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText(tr("Output file"))
        self.out_edit.editingFinished.connect(self._output_changed)
        browse = QToolButton(text="…")
        browse.setToolTip(tr("Choose the output file"))
        browse.clicked.connect(self._browse_output)
        row.addWidget(self.out_edit)
        row.addWidget(browse)
        rv.addLayout(row)
        self.safety_cb = QCheckBox(tr("Run safety check after rendering"))
        self.safety_cb.setToolTip(tr("Re-detect faces in the output and flag frames where a masked face "
                                     "may still be recognizable (slower)"))
        self.safety_cb.setChecked(False)
        self.safety_cb.toggled.connect(self._on_safety_toggled)
        rv.addWidget(self.safety_cb)
        row = QHBoxLayout()
        self.render_btn = QPushButton(tr("Render  (Ctrl+R)"))
        # blue (same as the progress bars' highlight) only while it can be clicked
        self.render_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
            "QPushButton:enabled { background: #3d78dc; color: white; border: 1px solid #2f63bd; border-radius: 3px; }"
            "QPushButton:enabled:hover { background: #4a86ea; }"
            "QPushButton:enabled:pressed { background: #2f63bd; }")
        self.render_btn.clicked.connect(self.start_render)
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.cancel_task)
        row.addWidget(self.render_btn, 2)
        row.addWidget(self.cancel_btn, 1)
        rv.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMaximumHeight(18)
        rv.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setMaximumHeight(60)
        rv.addWidget(self.status)
        sc = QShortcut(QKeySequence("Ctrl+R"), self)
        sc.activated.connect(self.start_render)

        # safety
        g = QGroupBox(tr("Safety check"))
        scv = QVBoxLayout(g)
        self.safety_label = QLabel(self._safety_idle_text())
        self.safety_label.setWordWrap(True)
        scv.addWidget(self.safety_label)
        self.safety_list = QListWidget()
        self.safety_list.setMaximumHeight(120)
        self.safety_list.itemClicked.connect(self._on_flag_clicked)
        scv.addWidget(self.safety_list)
        sv.addWidget(g)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        side.setMinimumWidth(360)
        scroll.setMinimumWidth(380)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        rl.addWidget(scroll, 1)
        rl.addWidget(footer)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setSizes([1080, 420])

        # keyboard transport works while the video side has focus (so it never
        # steals arrow keys from text fields in the side panel)
        self.view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.timeline.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for key, fn in (("Left", lambda: self.step(-1)), ("Right", lambda: self.step(1)),
                        ("Shift+Left", lambda: self.step(-self._fps_int())),
                        ("Shift+Right", lambda: self.step(self._fps_int())),
                        ("Home", lambda: self.seek(0)), ("End", lambda: self.seek(10**9)),
                        ("Space", self.toggle_play)):
            sc = QShortcut(QKeySequence(key), left)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu(tr("&File"))
        for text, key, fn in (("&Open video or project…", QKeySequence.StandardKey.Open, self.open_dialog),
                              ("&Save project", QKeySequence.StandardKey.Save, self.save_project),
                              ("Save project &as…", QKeySequence.StandardKey.SaveAs, self.save_project_as)):
            a = QAction(tr(text), self)
            a.setShortcut(key)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        self.rescan_action = QAction(tr("Detect faces again (discard cache)"), self)
        self.rescan_action.triggered.connect(self._rescan)
        m.addAction(self.rescan_action)
        m.addSeparator()
        a = QAction(tr("E&xit"), self)
        a.triggered.connect(self.close)
        m.addAction(a)
        lang = self.menuBar().addMenu(tr("&Language"))
        self.lang_actions = {}
        group = QActionGroup(self)
        for code, name in LANGUAGES.items():
            a = QAction(name, self, checkable=True, checked=code == i18n.language())
            a.triggered.connect(lambda _=False, c=code: self._set_language(c))
            group.addAction(a)
            lang.addAction(a)
            self.lang_actions[code] = a
        h = self.menuBar().addMenu(tr("&Help"))
        a = QAction(tr("How to use"), self)
        a.triggered.connect(self._help)
        h.addAction(a)

    def _set_enabled(self) -> None:
        busy = self.task is not None
        ready = self.analysis is not None
        self.render_btn.setEnabled(ready and not busy)
        self.render_btn.setText(tr("Render  (Ctrl+R)"))
        self.device_combo.setEnabled(not busy)  # never switch device in the middle of a job
        self.cancel_btn.setEnabled(busy)
        self.view.drawing_enabled = ready and not busy
        self.rescan_action.setEnabled(self.project is not None and not busy)
        out_ok = self.out_grabber is not None
        self.view_out.setEnabled(out_ok)
        if not out_ok and self.view_out.isChecked():
            self.view_orig.setChecked(True)

    def _fit_to_screen(self) -> None:
        """Restore the last window geometry, or fit the window to the screen's
        available area (without the taskbar) on first start."""
        saved = QSettings().value("geometry")
        if saved is not None and self.restoreGeometry(saved):
            return
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 850)
            return
        area = screen.availableGeometry()
        w, h = min(1600, int(area.width() * 0.94)), min(1000, int(area.height() * 0.92))
        self.resize(w, h)
        self.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)

    def _fps_int(self) -> int:
        return int(round(self.info.fps_float)) if self.info else 30

    # ============================================================ open/save
    def open_dialog(self) -> None:
        start = QSettings().value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(self, tr("Open video or project"), start,
                                              f"{tr('Video or project')} (*.faceblur.json {VIDEO_EXTS});;"
                                              f"{video_filter()}")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        if not self._confirm_discard():
            return
        p = Path(path)
        QSettings().setValue("last_dir", str(p.parent))
        try:
            if p.name.endswith(".faceblur.json"):
                project = Project.load(p)
            elif Project.default_path(p).exists():
                project = Project.load(Project.default_path(p))
            else:
                project = Project.new(p)
            info = probe(project.video)
        except Exception as e:
            QMessageBox.critical(self, tr("Cannot open"), f"{path}\n\n{e}")
            return
        self._stop_player()
        if self.task is not None:  # e.g. still scanning the previous video
            self.task.cancel()
            self.task.wait(10000)
            self.task = None
        self.project, self.info, self.analysis = project, info, None
        self.grabber = FrameGrabber(info)
        self.out_grabber = None
        self.frame, self.plans, self.flagged, self.blurred = 0, [], [], []
        self.dirty = False
        self.setWindowTitle(f"FaceBlur — {Path(project.video).name}")
        self.timeline.set_length(info.num_frames, info.fps_float)
        self.timeline.set_tracks([])
        self.timeline.set_flags([])
        self._load_settings_to_ui()
        self._clear_faces(tr("Detecting faces…"))
        self.safety_list.clear()
        self.safety_label.setText(self._safety_idle_text())
        self.show_frame()
        self._load_existing_output()
        if info.vfr:
            self.status.setText(tr("Note: variable frame rate video; the output will use a constant frame rate."))
        self._run_task(self._analysis_job(project.video), self._on_analysis_done, tr("Detecting faces"))

    def _analysis_job(self, video: str):
        def job(progress, cancel):
            analysis = Analysis.open(video, progress=progress, cancel=cancel)
            thumbs = card_thumbnails(analysis, 84, progress=progress, cancel=cancel)
            return analysis, thumbs
        return job

    def _on_analysis_done(self, result) -> None:
        analysis, thumbs = result
        if self.project is None or analysis.video != Path(self.project.video).resolve():
            return  # result for a video that is no longer open
        self.analysis = analysis
        a = self.analysis
        self.review.set_cards(a.cards, a.dets, a.info.fps_float, thumbs)
        n = len(a.cards)
        self.status.setText(tr("Found {cards} in {frames} frames.", n=n, cards=trn(n, "{n} face card", "{n} face cards"),
                               frames=a.info.num_frames))
        self._recompute()
        if a.cards and not any(self.blurred) and not self.project.manual:
            self.tabs.setCurrentWidget(self.review)  # nothing chosen yet: start with the faces
            self.review.setFocus()
        else:
            self.view.setFocus()

    def _rescan(self) -> None:
        if self.project is None:
            return
        if QMessageBox.question(self, tr("Detect faces again"),
                                tr("Discard cached detections and scan the whole video again?\n"
                                   "Face cards will change, so all face labels and corrections are cleared.")) \
                != QMessageBox.StandardButton.Yes:
            return
        from ..pipeline import workdir

        (workdir(self.project.video) / "detections.npz").unlink(missing_ok=True)
        self.project.labels.clear()
        self.project.exclusions.clear()
        self.analysis = None
        self._mark_dirty()
        self._clear_faces(tr("Detecting faces…"))
        self._run_task(self._analysis_job(self.project.video), self._on_analysis_done, tr("Detecting faces"))

    def save_project(self) -> bool:
        if self.project is None:
            return False
        try:
            path = self.project.save()
        except OSError as e:
            QMessageBox.critical(self, tr("Save failed"), str(e))
            return False
        self.dirty = False
        self.setWindowTitle(f"FaceBlur — {Path(self.project.video).name}")
        self.statusBar().showMessage(tr("Saved {path}", path=path), 4000)
        return True

    def save_project_as(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save project"), self.project.path,
                                              f"{tr('FaceBlur project')} (*.faceblur.json)")
        if path:
            if not path.endswith(".faceblur.json"):
                path = str(Path(path).with_suffix("")) + ".faceblur.json"
            self.project.path = path
            self.save_project()

    def _confirm_discard(self) -> bool:
        if not self.dirty or self.project is None:
            return True
        r = QMessageBox.question(self, tr("Unsaved changes"), tr("Save changes to the project first?"),
                                 QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return r == QMessageBox.StandardButton.Discard

    def _mark_dirty(self) -> None:
        self.dirty = True
        if self.project:
            self.setWindowTitle(f"FaceBlur — {Path(self.project.video).name} *")
            self._autosave_timer.start()

    def closeEvent(self, e) -> None:
        if self.task is not None and QMessageBox.question(
                self, tr("Busy"), tr("A job is running. Cancel it and quit?")) != QMessageBox.StandardButton.Yes:
            e.ignore()
            return
        if not self._confirm_discard():
            e.ignore()
            return
        self._stop_player()
        if self.task is not None:
            self.task.cancel()
            self.task.wait(5000)
        QSettings().setValue("geometry", self.saveGeometry())
        e.accept()

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        urls = e.mimeData().urls()
        if urls:
            self.open_path(urls[0].toLocalFile())

    # ============================================================ settings
    def _load_settings_to_ui(self) -> None:
        s = self.project.settings
        widgets = (self.style_combo, self.strength_spin, self.pad_spin, self.sam_cb)
        for w in widgets:
            w.blockSignals(True)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(s.style)))
        self.strength_spin.setValue(s.strength)
        self.pad_spin.setValue(s.pad)
        self.sam_cb.setChecked(s.use_sam)
        for w in widgets:
            w.blockSignals(False)
        self.out_edit.setText(self.project.output)

    def _settings_changed(self, *_):
        if self.project is None:
            return
        s = self.project.settings
        s.style = self.style_combo.currentData()
        s.strength = self.strength_spin.value()
        s.pad = self.pad_spin.value()
        s.use_sam = self.sam_cb.isChecked()
        self._mark_dirty()
        self.show_frame()

    def _device_changed(self, *_):
        from .. import device

        device.set_device(self.device_combo.currentData())
        QSettings().setValue("device", device.preference())
        self._show_device()

    def _show_device(self) -> None:
        from .. import device

        if device.preference() != "cpu" and not (device.cuda_onnx() and device.cuda_torch()):
            note = tr("No usable NVIDIA GPU found; everything runs on the CPU (slow).")
            if device.preference() == "gpu":
                note = tr("No usable NVIDIA GPU found. Choose Auto or CPU.")
        else:  # "faces: GPU, SAM 2: GPU, decode: …" with the labels translated
            parts = device.summary().split(": ", 1)[1].split(", ")
            note = ", ".join(tr(k) + ":" + v for k, _, v in (x.partition(":") for x in parts))
        self.device_label.setText(note)

    def _output_changed(self) -> None:
        if self.project and self.out_edit.text().strip() != self.project.output:
            self.project.output = self.out_edit.text().strip()
            self._mark_dirty()

    def _browse_output(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Output video"), self.project.output, f"{tr('MP4 video')} (*.mp4)")
        if path:
            self.out_edit.setText(path)
            self._output_changed()

    # ============================================================ blur choice
    def set_blur(self, cards: list[int], blur: bool) -> None:
        """Mark face cards (positions in analysis.cards) to blur, or not."""
        if self.analysis is None:
            return
        self.analysis.set_blur(self.project, cards, blur)
        self._blur_changed()

    def _jump_from_review(self, frame: int) -> None:
        self.tabs.setCurrentIndex(0)
        self.seek(frame)
        self.view.setFocus()

    def _blur_changed(self) -> None:
        self._mark_dirty()
        if self.analysis is not None:
            self.blurred = self.analysis.blurred(self.project)
        self._refresh_review()
        self._schedule_recompute()

    # ============================================================ face review
    def _clear_faces(self, note: str) -> None:
        self.faces_note.setText(note)
        self.blurred = []
        self.review.set_cards([], None, 30.0, {})
        self.review.set_blurred([])

    def _refresh_review(self) -> None:
        if self.project is None or self.analysis is None:
            return
        self.review.set_blurred(self.blurred)
        n, total = sum(self.blurred), len(self.blurred)
        n_manual = len(self.project.manual)
        if not total:
            self.faces_note.setText(tr("No faces found. You can still drag boxes onto the video to blur areas."))
        else:
            text = trn(n, "<b>{n}</b> of {total} faces will be blurred.", "<b>{n}</b> of {total} faces will be blurred.",
                       total=total)
            if n_manual:
                text += " " + trn(n_manual, "Plus {n} drawn area.", "Plus {n} drawn areas.")
            if not n and not n_manual:
                text += "<br><span style='color:#ffb020'>" + tr("Click the faces to blur in the Review faces tab.") \
                        + "</span>"
            self.faces_note.setText(text)
        self.tabs.setTabText(1, tr("Review faces ({n} blurred)", n=n))
        self._set_enabled()

    # ============================================================ plans
    def _schedule_recompute(self) -> None:
        self._recompute_timer.start()

    def _recompute(self) -> None:
        self._refresh_corrections()
        if self.analysis is None or self.project is None:
            self.show_frame()
            return
        self.blurred = self.analysis.blurred(self.project)
        self._refresh_review()
        _, self.plans = self.analysis.plans(self.project)
        self.excluded = self.analysis.excluded(self.project)
        # one band for everything blurred: bright = face detected, dim = bridged or drawn
        n = self.analysis.dets.num_frames
        levels = np.zeros(n, np.int8)
        for plan in self.plans:
            for f, s in enumerate(plan.source):
                levels[f] = max(levels[f], 2 if s == "det" else (1 if s else 0))
        self.timeline.set_tracks([(BLUR_COLOR, levels)] if self.plans else [])
        self.show_frame()

    def _refresh_corrections(self) -> None:
        self.corr_list.clear()
        if self.project is None:
            return
        fps = self.info.fps_float if self.info else 30.0
        for k, ex in enumerate(self.project.exclusions):
            what = tr("whole track") if ex.scope == "track" else tr("this frame")
            it = QListWidgetItem(tr("✕  {time} (frame {frame}): not masked, {what}",
                                    time=fmt_time(ex.frame / fps), frame=ex.frame, what=what))
            it.setData(Qt.ItemDataRole.UserRole, ("ex", k))
            it.setData(Qt.ItemDataRole.UserRole + 1, ex.frame)
            it.setForeground(KEEP_COLOR)
            self.corr_list.addItem(it)
        for k, m in enumerate(self.project.manual):
            it = QListWidgetItem(tr("▢  {time} (frame {frame}): blurred area, ±{span} frames",
                                    time=fmt_time(m.frame / fps), frame=m.frame, span=m.span))
            it.setData(Qt.ItemDataRole.UserRole, ("manual", k))
            it.setData(Qt.ItemDataRole.UserRole + 1, m.frame)
            it.setForeground(BLUR_COLOR)
            self.corr_list.addItem(it)

    def _remove_correction(self) -> None:
        it = self.corr_list.currentItem()
        if it is None:
            return
        kind, k = it.data(Qt.ItemDataRole.UserRole)
        if kind == "ex":
            del self.project.exclusions[k]
        else:
            del self.project.manual[k]
        self._mark_dirty()
        self._refresh_review()
        self._schedule_recompute()

    # ============================================================ frame display
    def seek(self, f: int) -> None:
        self._stop_player()
        if self.info is None:
            return
        self.frame = int(min(max(f, 0), self.info.num_frames - 1))
        self.show_frame()

    def step(self, d: int) -> None:
        self.seek(self.frame + d)

    def show_frame(self, img: np.ndarray | None = None) -> None:
        if self.info is None:
            return
        f = self.frame
        output = self.view_out.isChecked() and self.out_grabber is not None
        if img is None:
            img = (self.out_grabber if output else self.grabber).get(f)
        if img is None:
            return
        if self.preview_cb.isChecked() and not output and self.plans:
            from ..mask import apply, ellipse_mask

            boxes = [b for p in self.plans for b in p.boxes[f]] if f < len(self.plans[0].boxes) else []
            if boxes:
                s = self.project.settings
                img = apply(img, ellipse_mask(img.shape, boxes, s.pad), s.style, s.strength)
        self.view.set_image(img)
        self.view.set_overlays(self._overlays(f, output) if self.boxes_cb.isChecked() else [])
        self.timeline.set_position(f)
        n = self.info.num_frames
        self.pos_label.setText(f"{fmt_time(f / self.info.fps_float)} / {fmt_time(n / self.info.fps_float)}"
                               + tr("   frame {f} / {last}", f=f, last=n - 1))

    def _overlays(self, f: int, output: bool) -> list[Overlay]:
        out: list[Overlay] = []
        if output:
            for item in self.flagged:
                if item["frame"] == f:
                    for b, s in zip(item["box"], item["sim"]):
                        out.append(Overlay(tuple(b), EXCLUDED_COLOR, label=tr("still recognizable {s}", s=f"{s:.2f}"),
                                           kind="flag"))
            return out
        if self.analysis is None or f >= self.analysis.dets.num_frames:
            return out
        dets = self.analysis.dets
        target: set[int] = set()  # detections that will be blurred on this frame
        for plan in self.plans:
            target.update(plan.target_dets[f])
        for i in self.analysis.detections_in(f):
            i = int(i)
            k = int(self.analysis.card_of_det[i])
            tag = f"#{self.analysis.cards[k].number}" if k >= 0 else ""
            box = tuple(float(v) for v in dets.box[i])
            if i in self.excluded:
                out.append(Overlay(box, KEEP_COLOR, Qt.PenStyle.DashLine, 2, tr("{tag} not blurred here", tag=tag),
                                   "det", i))
            elif i in target:
                text = tr("{tag} blur", tag=tag) if k >= 0 else tr("blur (nearby)")
                out.append(Overlay(box, BLUR_COLOR, Qt.PenStyle.SolidLine, 3, text, "det", i))
            elif k < 0:
                out.append(Overlay(box, OTHER_COLOR, Qt.PenStyle.DotLine, 1, "", "det", i))  # weak, in no card
            else:
                out.append(Overlay(box, KEEP_COLOR, Qt.PenStyle.SolidLine, 1.5, tag, "det", i))
        for plan in self.plans:
            if plan.source[f] in ("interp", "hold"):
                for b in plan.boxes[f]:
                    out.append(Overlay(tuple(float(v) for v in b), BLUR_COLOR, Qt.PenStyle.DashLine, 1.5,
                                       tr("blur (bridged)"), "bridge"))
        for k, m in enumerate(self.project.manual):
            if abs(f - m.frame) <= m.span:
                exact = f == m.frame
                out.append(Overlay(tuple(m.box), BLUR_COLOR, Qt.PenStyle.DotLine, 2.5 if exact else 1.5,
                                   tr("blur (drawn)"), "manual", k))
        return out

    # ============================================================ playback
    def toggle_play(self) -> None:
        if self.player is not None:
            self._stop_player()
            return
        if self.info is None:
            return
        if self.frame >= self.info.num_frames - 1:
            self.frame = 0
        info = self.info
        if self.view_out.isChecked() and self.out_grabber is not None:
            info = self.out_grabber.info
        self.player = Player(info, self.frame, self)
        self.player.tick.connect(self._on_play_tick)
        self.player.finished.connect(self._on_play_finished)
        self.player.start()
        self.play_btn.setText("⏸")

    def _on_play_tick(self) -> None:
        if self.player is None:
            return
        item = self.player.latest()
        if item is not None:
            self.frame, img = item
            self.show_frame(img)

    def _on_play_finished(self) -> None:
        if self.player is not None and self.sender() is self.player:
            self.player = None
            self.play_btn.setText("▶")

    def _stop_player(self) -> None:
        if self.player is not None:
            p, self.player = self.player, None
            p.stop()
            p.wait(2000)
            self.play_btn.setText("▶")

    # ============================================================ corrections
    def _on_item_clicked(self, kind: str, ref: int, gpos) -> None:
        if self.project is None or self.task is not None:
            return
        self._stop_player()
        menu = QMenu(self)
        if kind == "manual":
            m = self.project.manual[ref]
            menu.addAction(tr("Drawn area, frame {frame} ±{span}", frame=m.frame, span=m.span)).setEnabled(False)
            a = menu.addAction(tr("Remove this drawn area"))
            a.triggered.connect(lambda: self._remove_manual(ref))
            menu.exec(gpos)
            return
        i = ref
        dets = self.analysis.dets
        k = int(self.analysis.card_of_det[i])
        box = [float(v) for v in dets.box[i]]
        if i in self.excluded:
            a = menu.addAction(tr("Blur this face here again"))
            a.triggered.connect(lambda: self._unexclude(i))
            menu.exec(gpos)
            return
        if k >= 0:
            num = self.analysis.cards[k].number
            if self.blurred[k]:
                a = menu.addAction(tr("Don't blur face #{num}", num=num))
                a.triggered.connect(lambda: self.set_blur([k], False))
            else:
                a = menu.addAction(tr("Blur face #{num}", num=num))
                a.triggered.connect(lambda: self.set_blur([k], True))
            menu.addAction(tr("Show face #{num} in the list", num=num)).triggered.connect(
                lambda: (self.tabs.setCurrentWidget(self.review), self.review.select(k)))
        is_target = any(i in plan.target_dets[self.frame] for plan in self.plans)
        menu.addSeparator()
        if is_target:
            a = menu.addAction(tr("Don't blur here — this frame only"))
            a.triggered.connect(lambda: self._exclude(self.frame, box, "frame"))
            if int(dets.track[i]) != 0:
                a = menu.addAction(tr("Don't blur here — while followed continuously"))
                a.setToolTip(tr("Every frame where this face is followed continuously"))
                a.triggered.connect(lambda: self._exclude(self.frame, box, "track"))
        elif k < 0 or self.blurred[k]:
            a = menu.addAction(tr("Blur this face here too (±{n} frames)", n=self.span_spin.value()))
            a.triggered.connect(lambda: self._add_manual(box))
        menu.exec(gpos)

    def _on_box_drawn(self, box: list, gpos) -> None:
        if self.project is None or self.task is not None:
            return
        self._stop_player()
        menu = QMenu(self)
        a = menu.addAction(tr("Blur this area (±{n} frames)", n=self.span_spin.value()))
        a.triggered.connect(lambda: self._add_manual(box))
        menu.exec(gpos)

    def _exclude(self, frame: int, box, scope: str) -> None:
        self.project.exclusions.append(Exclusion(frame, [round(v, 1) for v in box], scope))
        self._mark_dirty()
        self._schedule_recompute()

    def _unexclude(self, det: int) -> None:
        a = self.analysis
        keep = []
        for ex in self.project.exclusions:
            i = a.find_detection(ex.frame, ex.box)
            covers = i == det or (ex.scope == "track" and i is not None and a.dets.track[i] != 0
                                  and a.dets.track[i] == a.dets.track[det])
            if not covers:
                keep.append(ex)
        self.project.exclusions = keep
        self._mark_dirty()
        self._schedule_recompute()

    def _add_manual(self, box) -> None:
        self.project.manual.append(Manual(self.frame, [round(float(v), 1) for v in box], 0, self.span_spin.value()))
        self._mark_dirty()
        self._refresh_review()
        self._schedule_recompute()

    def _remove_manual(self, k: int) -> None:
        del self.project.manual[k]
        self._mark_dirty()
        self._refresh_review()
        self._schedule_recompute()

    # ============================================================ render
    def _run_task(self, fn, on_done, label: str) -> None:
        self.task = Task(fn, self)
        self.task.progress.connect(self._on_progress)
        self.task.done.connect(on_done)
        self.task.failed.connect(self._on_failed)
        self.task.cancelled.connect(lambda: self._on_cancelled(label))
        self.task.finished.connect(self._on_task_finished)
        self.progress.setRange(0, 0)
        self.progress.setFormat(label)
        self.status.setText(f"{label}…")
        self._stage = None
        self.job_label.setText(label)
        self.job_bar.setRange(0, 0)
        self.job_bar.setFormat(tr("starting…"))
        self.job_strip.setVisible(True)
        self.task.start()
        self._set_enabled()

    def _on_progress(self, stage: str, done: int, total: int) -> None:
        import time

        now = time.monotonic()
        if self._stage is None or self._stage[0] != stage:
            self._stage = (stage, now, done)
        _, t0, d0 = self._stage
        raw, stage = stage, tr(stage)
        self.job_label.setText(stage)
        if total <= 0:
            for bar in (self.progress, self.job_bar):
                bar.setRange(0, 0)
                bar.setFormat(stage)
            return
        elapsed = now - t0
        rate = (done - d0) / elapsed if elapsed > 0 else 0.0
        detail = tr("%p%  ·  frame {done} / {total}", done=f"{done:,}", total=f"{total:,}")
        if rate > 0 and elapsed >= 3:
            left = (total - done) / rate
            detail += tr("  ·  {rate} fps  ·  about {left} left", rate=f"{rate:.1f}", left=self._fmt_left(left))
        for bar in (self.progress, self.job_bar):
            bar.setRange(0, total)
            bar.setValue(done)
        self.job_bar.setFormat(detail)
        self.progress.setFormat(f"{stage}  %p%")
        if raw == "Detecting faces":
            self.faces_note.setText(tr("Detecting faces: {pct}% ({done} of {total} frames)",
                                       pct=100 * done // total, done=f"{done:,}", total=f"{total:,}")
                                    + (tr(", about {left} left.", left=self._fmt_left((total - done) / rate))
                                       if rate > 0 and elapsed >= 3 else ".")
                                    + tr("\nYou can scrub the video meanwhile; results are cached for next time."))

    @staticmethod
    def _fmt_left(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return tr("{s} s", s=max(seconds, 1))
        if seconds < 3600:
            return tr("{m} min {s} s", m=seconds // 60, s=f"{seconds % 60:02d}")
        return tr("{h} h {m} min", h=seconds // 3600, m=f"{seconds % 3600 // 60:02d}")

    def _on_cancelled(self, label: str) -> None:
        self.status.setText(tr("{label} cancelled.", label=label))
        if self.analysis is None:
            self.faces_note.setText(tr("Face detection was cancelled. Use File › Detect faces again to restart it."))

    def _on_failed(self, msg: str) -> None:
        self.status.setText(tr("Failed: ") + msg.splitlines()[0])
        box = QMessageBox(QMessageBox.Icon.Critical, tr("Error"), msg.split("\n\n")[0], parent=self)
        box.setDetailedText(msg)
        box.exec()

    def _on_task_finished(self) -> None:
        self.task = None
        self.job_strip.setVisible(False)
        self._stage = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self._set_enabled()

    def cancel_task(self) -> None:
        if self.task is not None:
            self.task.cancel()
            self.status.setText(tr("Cancelling…"))

    def start_render(self) -> None:
        if self.analysis is None or self.task is not None:
            return
        self._output_changed()
        _, plans = self.analysis.plans(self.project)
        if not plans:
            QMessageBox.information(self, tr("Nothing to blur"),
                                    tr("No face is marked to blur yet. Click the faces to blur in the "
                                       "Review faces tab, or drag a box onto the video."))
            self.tabs.setCurrentWidget(self.review)
            return
        out = Path(self.project.output or "")
        if not self.project.output:
            QMessageBox.information(self, tr("Output"), tr("Choose an output file first."))
            return
        if out.resolve() == Path(self.project.video).resolve():
            QMessageBox.warning(self, tr("Output"), tr("The output must be a different file from the original video."))
            return
        if out.exists() and QMessageBox.question(self, tr("Overwrite"), tr("{name} exists. Overwrite it?", name=out.name)) \
                != QMessageBox.StandardButton.Yes:
            return
        self.save_project()
        self._stop_player()
        self.out_grabber = None
        self.flagged = []
        self.safety_list.clear()
        check = self.safety_cb.isChecked()
        self.safety_label.setText(tr("Waiting for render…") if check else self._safety_idle_text())
        self.timeline.set_flags([])
        self._set_enabled()
        project, analysis = self.project, self.analysis
        self._render_log = []

        def job(progress, cancel):
            result = render(project, analysis, progress=progress, cancel=cancel, log=self._render_log.append)
            if not check:
                # drop an older report so it isn't shown later as if it belonged to this output
                (analysis.wd / "safety_report.json").unlink(missing_ok=True)
                return result, None
            report = safety_check(result.output, load_target_refs(analysis.wd), analysis.wd,
                                  progress=progress, cancel=cancel)
            return result, report

        self._run_task(job, self._on_render_done, tr("Rendering"))

    def _on_render_done(self, payload) -> None:
        result, report = payload
        self.status.setText(tr("Wrote {name}: {frames} frames at {fps} fps. Frames with no mask at all: {unmasked}.",
                               name=result.output.name, frames=result.frames, fps=f"{result.fps:.1f}",
                               unmasked=result.unmasked) + "\n" + "\n".join(self._render_log))
        self._show_report(result.output, report)
        if report is None:
            self.safety_label.setText(tr("Skipped (turned off for this render)."))
        self.view_out.setChecked(True)
        self.show_frame()

    def _safety_idle_text(self) -> str:
        return (tr("Runs automatically after rendering.") if self.safety_cb.isChecked()
                else tr("Off. Tick “Run safety check after rendering” to run it."))

    def _on_safety_toggled(self, _checked: bool) -> None:
        # only update the hint while no result is on display
        if self.task is None and self.out_grabber is None and not self.flagged:
            self.safety_label.setText(self._safety_idle_text())

    def _load_existing_output(self) -> None:
        """Reopen a project that was rendered before: show its output and report."""
        import json

        from ..pipeline import workdir

        out = Path(self.project.output) if self.project.output else None
        rep = workdir(self.project.video) / "safety_report.json"
        if out and out.exists():
            report = None
            if rep.exists():
                try:
                    data = json.loads(rep.read_text(encoding="utf-8"))
                    if Path(data.get("video", "")).resolve() == out.resolve():
                        report = data
                except (OSError, ValueError):
                    pass
            self._show_report(out, report)

    def _show_report(self, output: Path, report: dict | None) -> None:
        try:
            self.out_grabber = FrameGrabber(probe(output))
        except Exception:
            self.out_grabber = None
        self.flagged = report["flagged"] if report else []
        self.safety_list.clear()
        fps = self.info.fps_float
        if report is None:
            self.safety_label.setText(tr("Output found; no safety check on record for it."))
        elif not self.flagged:
            self.safety_label.setText(f"<span style='color:#3ddc84'><b>{tr('Passed.')}</b></span> "
                                      + tr("No masked face is recognizable on any output frame."))
        else:
            need = tr("{n} frame(s) need a look.", n=len(self.flagged))
            self.safety_label.setText(f"<span style='color:#ff4d4d'><b>{need}</b></span> "
                                      + tr("A masked face may still be recognizable. Click to view, "
                                           "fix with a drawn box, and render again."))
            for item in self.flagged:
                it = QListWidgetItem(tr("{time} (frame {frame}) — similarity {sim}",
                                        time=fmt_time(item["frame"] / fps), frame=item["frame"],
                                        sim=f"{max(item['sim']):.2f}"))
                it.setData(Qt.ItemDataRole.UserRole, item["frame"])
                self.safety_list.addItem(it)
        self.timeline.set_flags([x["frame"] for x in self.flagged])
        self._set_enabled()

    def _on_flag_clicked(self, it: QListWidgetItem) -> None:
        if self.out_grabber is not None:
            self.view_out.setChecked(True)
        self.seek(int(it.data(Qt.ItemDataRole.UserRole)))

    def _help(self) -> None:
        QMessageBox.information(self, tr("How to use"), tr(HELP_EN))

    def _set_language(self, code: str) -> None:
        """Switch the UI language by rebuilding the window. The open project is
        saved and reopened (face detections come from the cache)."""
        if code == i18n.language():
            return
        if self.task is not None:
            QMessageBox.information(self, tr("Language"),
                                    tr("Finish or cancel the current job before changing the language."))
            self.lang_actions[i18n.language()].setChecked(True)
            return
        if self.dirty and not self.save_project():
            self.lang_actions[i18n.language()].setChecked(True)
            return
        i18n.set_language(code)
        QSettings().setValue("language", code)
        path = self.project.path if self.project is not None else None
        tab = self.tabs.currentIndex()
        win = MainWindow()
        win.restoreGeometry(self.saveGeometry())
        _WINDOWS[:] = [win]
        win.show()
        self.close()
        self.deleteLater()
        if path:
            win.open_path(path)
            win.tabs.setCurrentIndex(tab)

def _dark(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    base, alt, text = QColor(37, 37, 40), QColor(45, 45, 50), QColor(225, 225, 225)
    for role, c in ((QPalette.ColorRole.Window, base), (QPalette.ColorRole.WindowText, text),
                    (QPalette.ColorRole.Base, QColor(28, 28, 30)), (QPalette.ColorRole.AlternateBase, alt),
                    (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, alt),
                    (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.ToolTipBase, alt),
                    (QPalette.ColorRole.ToolTipText, text), (QPalette.ColorRole.Highlight, QColor(61, 120, 220)),
                    (QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))):
        pal.setColor(role, c)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(110, 110, 110))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(110, 110, 110))
    app.setPalette(pal)


def run(path: str | None = None) -> int:
    import torch  # noqa: F401  (load PyTorch's CUDA DLLs before onnxruntime's)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("FaceBlur")
    app.setApplicationName("FaceBlur")
    _dark(app)
    i18n.set_language(str(QSettings().value("language", i18n.system_language())))
    win = MainWindow()
    _WINDOWS[:] = [win]
    win.show()
    if path:
        QTimer.singleShot(0, lambda: win.open_path(path))
    return app.exec()


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
