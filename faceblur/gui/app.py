"""FaceBlur main window.

Workflow: open a video -> cut & crop it if wanted (the view shows the whole
source; faces are only looked for in what is kept) -> Detect faces (cached) ->
face cards, all not blurred -> the user clicks the faces to blur in the Review tab -> fix
individual frames (click a box: "don't blur here"; drag a box: "blur this
area") -> render -> review the safety check.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import (QAction, QActionGroup, QColor, QIcon, QKeySequence, QPainter, QPalette, QPen, QPixmap,
                           QPolygonF, QShortcut)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSplitter, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

from .. import __version__
from ..edit import CutList
from ..identity import Cancelled  # noqa: F401  (re-exported for workers)
from ..pipeline import Analysis, card_thumbnails, detections_cache, load_target_refs, render, safety_check, workdir
from ..project import Edit, Exclusion, Manual, Project
from ..video_io import FrameGrabber, fit_crop, probe, turn_crop
from . import i18n
from .i18n import HELP_EN, LANGUAGES, tr, trn
from .review_tab import ReviewPanel
from .widgets import BLUR_COLOR, EXCLUDED_COLOR, KEEP_COLOR, OTHER_COLOR, Overlay, Timeline, VideoView
from .workers import Player, Task

VIDEO_EXTS = "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.m2ts *.wmv"
_WINDOWS: list = []  # keeps the current main window alive across language switches


# blue (same as the progress bars' highlight) only while it can be clicked
PRIMARY_STYLE = (
    "QPushButton { font-weight: bold; padding: 6px; }"
    "QPushButton:enabled { background: #3d78dc; color: white; border: 1px solid #2f63bd; border-radius: 3px; }"
    "QPushButton:enabled:hover { background: #4a86ea; }"
    "QPushButton:enabled:pressed { background: #2f63bd; }")
FACES_HINT = ("Click a face box to blur it or leave it visible. "
              "Drag a rectangle to blur an area the detector missed.")
EDIT_HINT = ("Crop: drag the corners or edges of the frame (drag outside it to draw a new crop). "
             "Cut: split at the playhead (S) and remove the parts you don't need (Delete), or cut the "
             "start and end (I / O). Faces are only looked for in what is left.")
ASPECTS = (("Free", 0.0), ("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0), ("4:3", 4 / 3), ("4:5", 4 / 5))


def video_filter() -> str:
    return f"{tr('Videos')} ({VIDEO_EXTS});;{tr('All files')} (*)"


def fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _transport_icon(kind: str, color: QColor) -> QIcon:
    """Player buttons drawn in one style: "play", "pause", "frame" (one frame
    on: triangle and bar), "second" (one second on: two triangles and bar);
    "back_..." is the mirror image."""
    px = QPixmap(44, 44)  # drawn at 2x, shown at 22x22
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    if kind.startswith("back_"):
        kind = kind[5:]
        p.translate(44, 0)
        p.scale(-1, 1)

    def tri(x0, x1):  # pointing right, from x0 to x1
        p.drawPolygon(QPolygonF([QPointF(x0, 9), QPointF(x1, 22), QPointF(x0, 35)]))

    if kind == "play":
        tri(13, 35)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(11, 9, 8, 26), 1.5, 1.5)
        p.drawRoundedRect(QRectF(25, 9, 8, 26), 1.5, 1.5)
    elif kind == "frame":
        tri(11, 29)
        p.drawRoundedRect(QRectF(29, 9, 5, 26), 1.5, 1.5)
    elif kind == "second":
        tri(5, 19)
        tri(18, 32)
        p.drawRoundedRect(QRectF(32, 9, 5, 26), 1.5, 1.5)
    p.end()
    return QIcon(px)


def _turn_icon(clockwise: bool, color: QColor) -> QIcon:
    """A three-quarter circle with an arrowhead: turn the picture left or right."""
    px = QPixmap(40, 40)  # drawn at 2x, shown at 20x20
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(color, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    if not clockwise:  # the mirror image
        p.translate(40, 0)
        p.scale(-1, 1)
    # arc from 3 o'clock going clockwise round to 12 o'clock (Qt angles run counterclockwise)
    p.drawArc(QRectF(7, 8, 26, 26), 0, -270 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    # arrowhead at 12 o'clock, pointing right = going clockwise
    p.drawPolygon(QPolygonF([QPointF(30, 8), QPointF(19, 1), QPointF(19, 15)]))
    p.end()
    return QIcon(px)


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
        # cut & crop
        self.src_info = None  # the source video, turned but uncut and uncropped (self.info is the edited one)
        self._raw_info = None  # the source video as it is in the file
        self.src_grabber: FrameGrabber | None = None
        self.editing = False  # edit mode: the view and timeline show the source
        self.cuts: CutList | None = None
        self._undo: list = []
        self._redo: list = []
        self._edit_confirmed = False  # the user agreed to clear face choices for this edit session
        self._scanned_edit: Edit | None = None  # the cut & crop the current face scan was made for
        self._back_to_edit = False  # a face scan stopped without result: show cut & crop when it ends
        self._play_info = None  # what the player plays in edit mode (the kept parts, uncropped)

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
        icon_color = self.palette().buttonText().color()
        self._play_icons = {k: _transport_icon(k, icon_color) for k in ("play", "pause")}

        def tbtn(kind, tip, fn):
            b = QToolButton()
            b.setIcon(self._play_icons.get(kind) or _transport_icon(kind, icon_color))
            b.setIconSize(QSize(22, 22))
            b.setToolTip(tr(tip))
            b.setMinimumSize(44, 36)
            b.clicked.connect(fn)
            bar.addWidget(b)
            return b
        tbtn("back_second", "Previous second (Shift+Left)", lambda: self.step(-self._fps_int()))
        tbtn("back_frame", "Previous frame (Left)", lambda: self.step(-1))
        self.play_btn = tbtn("play", "Play / pause (Space)", self.toggle_play)
        tbtn("frame", "Next frame (Right)", lambda: self.step(1))
        tbtn("second", "Next second (Shift+Right)", lambda: self.step(self._fps_int()))
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

        # cut & crop tools: visible in edit mode
        self.edit_bar = QWidget()
        eb = QHBoxLayout(self.edit_bar)
        eb.setContentsMargins(0, 0, 0, 0)

        def ebtn(text, tip, fn):
            b = QPushButton(tr(text))
            b.setToolTip(tr(tip))
            b.clicked.connect(fn)
            eb.addWidget(b)
            return b
        ebtn("⇤ Start here", "Cut everything before the playhead (I)", self._cut_in)
        ebtn("End here ⇥", "Cut everything after the playhead (O)", self._cut_out)
        ebtn("Split", "Split the part under the playhead (S)", self._cut_split)
        self.cut_toggle_btn = ebtn("Remove part", "Remove the part under the playhead, or bring it back (Delete)",
                                   self._cut_toggle)
        eb.addSpacing(18)
        for text, tip, by in (("Turn left", "Turn the video left (90° counterclockwise), e.g. when the camera "
                                            "filmed on its side", 270),
                              ("Turn right", "Turn the video right (90° clockwise), e.g. when the camera "
                                             "filmed on its side", 90)):
            b = ebtn(text, tip, lambda _=False, by=by: self._turn(by))
            b.setIcon(_turn_icon(clockwise=by == 90, color=b.palette().buttonText().color()))
            b.setIconSize(QSize(20, 20))
        eb.addSpacing(18)
        eb.addWidget(QLabel(tr("Crop")))
        self.aspect_combo = QComboBox()
        for label, ar in ASPECTS:
            self.aspect_combo.addItem(tr(label), ar)
        self.aspect_combo.setToolTip(tr("Shape of the crop"))
        self.aspect_combo.currentIndexChanged.connect(self._aspect_changed)
        eb.addWidget(self.aspect_combo)
        ebtn("Full frame", "Remove the crop", self._crop_reset)
        eb.addStretch(1)
        self.undo_btn = ebtn("Undo", "Undo (Ctrl+Z)", self._edit_undo)
        self.redo_btn = ebtn("Redo", "Redo (Ctrl+Y)", self._edit_redo)
        self.edit_bar.setVisible(False)
        lv.addWidget(self.edit_bar)

        self.hint = QLabel(tr(FACES_HINT))
        self.hint.setStyleSheet("color: #999;")
        self.hint.setWordWrap(True)
        lv.addWidget(self.hint)
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

        # open a video or project (same as File › Open)
        open_btn = QPushButton(tr("Open video or project…"))
        open_btn.setMinimumHeight(36)
        open_btn.clicked.connect(self.open_dialog)
        self.reset_btn = QPushButton(tr("Reset project…"))
        self.reset_btn.setMinimumHeight(36)
        self.reset_btn.setToolTip(tr("Start over with this video: deletes the detected faces, your choices, "
                                     "corrections, cut & crop and the work folder"))
        self.reset_btn.clicked.connect(self.reset_project)
        row = QHBoxLayout()
        row.addWidget(open_btn, 3)
        row.addWidget(self.reset_btn, 2)
        sv.addLayout(row)

        # cut & crop: before faces are detected
        g = QGroupBox(tr("Cut & crop").replace("&", "&&"))  # a single & would underline the next letter
        ev = QVBoxLayout(g)
        self.edit_summary = QLabel("–")
        self.edit_summary.setWordWrap(True)
        ev.addWidget(self.edit_summary)
        # how faces are looked for: set next to the Detect faces button (in cut & crop mode)
        self.any_angle_cb = QCheckBox(tr("Faces at any angle (slower)"))
        self.any_angle_cb.setToolTip(tr("Also find sideways and upside-down faces (camera on its side, "
                                        "people lying down). Face detection takes about 4 times longer."))
        self.any_angle_cb.toggled.connect(self._any_angle_toggled)
        ev.addWidget(self.any_angle_cb)
        self.edit_btn = QPushButton()
        self.edit_btn.setMinimumHeight(36)
        self.edit_btn.clicked.connect(self._edit_button)
        ev.addWidget(self.edit_btn)
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
        self.sam_model_combo = QComboBox()
        self._fill_sam_models()
        self.sam_model_combo.setToolTip(tr("Bigger models follow the face outline more exactly but render slower. "
                                           "A model that is not downloaded yet is downloaded on the first render."))
        self.sam_model_combo.setEnabled(False)
        self.sam_cb.toggled.connect(self.sam_model_combo.setEnabled)
        form.addRow(tr("Style"), self.style_combo)
        form.addRow(tr("Strength"), self.strength_spin)
        form.addRow(tr("Margin"), self.pad_spin)
        form.addRow(self.sam_cb)
        form.addRow(tr("SAM 2 model"), self.sam_model_combo)
        for w, sig in ((self.style_combo, "currentIndexChanged"), (self.strength_spin, "valueChanged"),
                       (self.pad_spin, "valueChanged"), (self.sam_cb, "toggled"),
                       (self.sam_model_combo, "currentIndexChanged")):
            getattr(w, sig).connect(self._settings_changed)
        sv.addWidget(g)

        # corrections
        g = QGroupBox(tr("Corrections"))
        cv = QVBoxLayout(g)
        self.corr_list = QListWidget()
        self.corr_list.setMaximumHeight(90)
        self.corr_list.setMinimumHeight(48)
        self.corr_list.setToolTip(tr("Double-click to go there. Right-click a drawn area to change its duration."))
        self.corr_list.itemDoubleClicked.connect(lambda it: self.seek(it.data(Qt.ItemDataRole.UserRole + 1)))
        self.corr_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.corr_list.customContextMenuRequested.connect(self._correction_menu)
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
        row = QHBoxLayout()
        self.codec_combo = QComboBox()
        self.codec_combo.addItem(tr("H.265 (smaller file)"), "h265")
        self.codec_combo.addItem(tr("H.264 (plays everywhere)"), "h264")
        self.codec_combo.setToolTip(tr("H.265 needs about half the size for the same picture; "
                                       "H.264 also plays in every browser and on old devices"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItem(tr("Best"), "best")
        self.quality_combo.addItem(tr("High"), "high")
        self.quality_combo.addItem(tr("Medium"), "medium")
        self.quality_combo.addItem(tr("Small file"), "small")
        self.quality_combo.setCurrentIndex(self.quality_combo.findData("high"))  # the default until a project opens
        self.quality_combo.setToolTip(tr("Higher quality makes a bigger file"))
        row.addWidget(QLabel(tr("Format")))
        row.addWidget(self.codec_combo, 3)
        row.addWidget(QLabel(tr("Quality")))
        row.addWidget(self.quality_combo, 2)
        rv.addLayout(row)
        for w in (self.codec_combo, self.quality_combo):
            w.currentIndexChanged.connect(self._settings_changed)
        self.safety_cb = QCheckBox(tr("Run safety check after rendering"))
        self.safety_cb.setToolTip(tr("Re-detect faces in the output and flag frames where a masked face "
                                     "may still be recognizable (slower)"))
        self.safety_cb.setChecked(False)
        self.safety_cb.toggled.connect(self._on_safety_toggled)
        rv.addWidget(self.safety_cb)
        row = QHBoxLayout()
        self.render_btn = QPushButton(tr("Render  (Ctrl+R)"))
        self.render_btn.setStyleSheet(PRIMARY_STYLE)
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
                        ("Space", self.toggle_play),
                        # cut & crop (edit mode only)
                        ("I", self._cut_in), ("O", self._cut_out), ("S", self._cut_split),
                        ("Delete", self._cut_toggle), ("Ctrl+Z", self._edit_undo), ("Ctrl+Y", self._edit_redo),
                        ("Ctrl+Shift+Z", self._edit_redo)):
            sc = QShortcut(QKeySequence(key), left)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self.view.cropChanged.connect(self._on_crop_changed)

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
        self.reset_action = QAction(tr("Reset project…"), self)
        self.reset_action.triggered.connect(self.reset_project)
        m.addAction(self.reset_action)
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
        h.addSeparator()
        a = QAction(f"FaceBlur {__version__}", self)  # update.bat installs newer versions
        a.setEnabled(False)
        h.addAction(a)

    def _set_enabled(self) -> None:
        busy = self.task is not None
        ready = self.analysis is not None
        editing = self.editing
        self.render_btn.setEnabled(ready and not busy and not editing)
        self.render_btn.setText(tr("Render  (Ctrl+R)"))
        self.device_combo.setEnabled(not busy)  # never switch device in the middle of a job
        self.cancel_btn.setEnabled(busy)
        self.view.drawing_enabled = ready and not busy and not editing
        self.rescan_action.setEnabled(self.project is not None and not busy and not editing)
        self.reset_btn.setEnabled(self.project is not None and not busy)
        self.reset_action.setEnabled(self.project is not None and not busy)
        out_ok = self.out_grabber is not None
        self.view_out.setEnabled(out_ok and not editing)
        if (not out_ok or editing) and self.view_out.isChecked():
            self.view_orig.setChecked(True)
        self.preview_cb.setEnabled(not editing)
        self.boxes_cb.setEnabled(not editing)
        self.tabs.setTabEnabled(1, not editing)
        self.edit_bar.setEnabled(not busy)
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(bool(self._redo))
        self.edit_btn.setEnabled(self.project is not None and not busy)
        self.any_angle_cb.setEnabled(editing and not busy)
        self.any_angle_cb.setToolTip(tr("Also find sideways and upside-down faces (camera on its side, "
                                        "people lying down). Face detection takes about 4 times longer.")
                                     if editing else
                                     tr("Click “Change cut & crop…” to change this; faces are then detected again."))
        if editing:
            unchanged = self.analysis is not None and self._scanned_edit == self.project.edit
            self.edit_btn.setText(tr("Back to the faces (nothing changed)") if unchanged
                                  else tr("Detect faces  ▶"))
            self.edit_btn.setStyleSheet(PRIMARY_STYLE)
        else:
            self.edit_btn.setText(tr("Change cut & crop…").replace("&", "&&"))
            self.edit_btn.setStyleSheet("")

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
            raw = probe(project.video)
            src = raw.turned(project.edit.turn)
            info = src.with_edit(project.edit.keep, project.edit.crop) if project.edit.active else src
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
        self._raw_info = raw
        self.src_info, self.src_grabber = src, (FrameGrabber(src) if src is not info else self.grabber)
        self.cuts = CutList(src.num_frames, project.edit.keep)
        self._undo.clear()
        self._redo.clear()
        self._scanned_edit = None
        self.editing = False
        self.view.set_crop_mode(False)
        self.edit_bar.setVisible(False)
        self.timeline.set_parts(None)
        self.hint.setText(tr(FACES_HINT))
        self.out_grabber = None
        self.frame, self.plans, self.flagged, self.blurred = 0, [], [], []
        self.dirty = False
        self.setWindowTitle(f"FaceBlur — {Path(project.video).name}")
        self.timeline.set_length(info.num_frames, info.fps_float)
        self.timeline.set_tracks([])
        self.timeline.set_flags([])
        self._load_settings_to_ui()
        self._clear_faces()
        self._refresh_corrections()  # the drawn boxes and exclusions of this project, not the last one's
        self.safety_list.clear()
        self.safety_label.setText(self._safety_idle_text())
        self.show_frame()
        self._load_existing_output()
        self._update_edit_summary()
        if info.vfr:
            self.status.setText(tr("Note: variable frame rate video; the output will use a constant frame rate."))
        if detections_cache(workdir(project.video), info, project.edit.any_angle).exists():  # scanned before
            self._run_task(self._analysis_job(project.video), self._on_analysis_done, tr("Detecting faces"))
        else:  # new: cut & crop first, so faces are only looked for where needed
            self._enter_edit()
            if not info.vfr:
                self.status.setText(tr("Cut and crop the video if you need to, then click Detect faces."))

    def _analysis_job(self, video: str):
        edit = copy.deepcopy(self.project.edit)

        def job(progress, cancel):
            analysis = Analysis.open(video, progress=progress, cancel=cancel, edit=edit)
            thumbs = card_thumbnails(analysis, 84, progress=progress, cancel=cancel)
            return analysis, thumbs, edit
        return job

    def _on_analysis_done(self, result) -> None:
        analysis, thumbs, edit = result
        if self.project is None or analysis.video != Path(self.project.video).resolve():
            return  # result for a video that is no longer open
        if self.editing or edit != self.project.edit:
            return  # made for a cut & crop that has changed since
        self._scanned_edit = edit
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
        from ..pipeline import clear_scan_cache

        clear_scan_cache(workdir(self.project.video))
        self.project.labels.clear()
        self.project.exclusions.clear()
        self.analysis = None
        self._mark_dirty()
        self._clear_faces()
        self._run_task(self._analysis_job(self.project.video), self._on_analysis_done, tr("Detecting faces"))

    def reset_project(self) -> None:
        """Start over with the open video: delete its work folder (detections,
        thumbnails, reports) and its project file, then open it as a new video.
        The video and rendered outputs are kept."""
        if self.project is None or self.task is not None:
            return
        import shutil

        video = Path(self.project.video)
        wd = workdir(video)
        files = {Path(self.project.path), Project.default_path(video)}
        msg = tr("Reset the project for {name}?\n\n"
                 "This deletes the detected faces, your choice of faces to blur, the corrections, "
                 "the cut & crop and the settings, and the work folder:\n{work}\n\n"
                 "The video itself and rendered videos are kept. This cannot be undone.",
                 name=video.name, work=wd)
        if QMessageBox.question(self, tr("Reset project"), msg,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
            return
        self._stop_player()
        self._autosave_timer.stop()
        self.analysis = None
        try:
            if wd.exists():
                shutil.rmtree(wd)
            for f in files:
                f.unlink(missing_ok=True)
        except OSError as e:
            QMessageBox.critical(self, tr("Reset project"), tr("Could not delete everything:") + f"\n{e}")
        self.dirty = False
        self.aspect_combo.blockSignals(True)
        self.aspect_combo.setCurrentIndex(0)
        self.aspect_combo.blockSignals(False)
        self.view.crop_aspect = None
        self.open_path(str(video))
        self.statusBar().showMessage(tr("Project reset: {name} is open as a new video.", name=video.name), 6000)

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
        widgets = (self.style_combo, self.strength_spin, self.pad_spin, self.sam_cb, self.sam_model_combo,
                   self.codec_combo, self.quality_combo)
        for w in widgets:
            w.blockSignals(True)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(s.style)))
        self.codec_combo.setCurrentIndex(max(0, self.codec_combo.findData(s.codec)))
        self.quality_combo.setCurrentIndex(max(0, self.quality_combo.findData(s.quality)))
        self.quality_combo.setToolTip(tr("Higher quality makes a bigger file") if s.cq is None else
                                      tr("This project uses the exact setting {n} (lower = better); "
                                         "choosing a level replaces it", n=s.cq))
        self.strength_spin.setValue(s.strength)
        self.pad_spin.setValue(s.pad)
        self.sam_cb.setChecked(s.use_sam)
        self.sam_model_combo.setEnabled(s.use_sam)
        self.sam_model_combo.setCurrentIndex(max(0, self.sam_model_combo.findData(s.sam_model)))
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
        s.sam_model = self.sam_model_combo.currentData() or "auto"
        s.codec = self.codec_combo.currentData()
        if self.quality_combo.currentData() != s.quality:
            s.quality = self.quality_combo.currentData()
            s.cq = None  # a chosen level replaces an exact number set elsewhere (CLI or API)
            self.quality_combo.setToolTip(tr("Higher quality makes a bigger file"))
        self._mark_dirty()
        self.show_frame()

    def _fill_sam_models(self, detect: bool = False) -> None:
        """The SAM 2 model choices; sizes that are not downloaded yet say so.
        detect: also name the size "auto" means on this PC (checks the GPU,
        which loads PyTorch: not while the window is being built)."""
        from .. import device
        from ..setup_models import SAM_MODELS

        names = {"tiny": tr("Tiny (fastest)"), "small": tr("Small"), "base_plus": tr("Base+"),
                 "large": tr("Large (most precise)")}
        on_disk = {size: (it.folder / it.name).exists() for size, it in SAM_MODELS.items()}
        keep = self.sam_model_combo.currentData()
        self.sam_model_combo.blockSignals(True)
        self.sam_model_combo.clear()
        if detect:  # the same rule as segment.model_for, without importing PyTorch here
            auto = "tiny" if device.sam_device() == "cpu" and on_disk["tiny"] else "large"
            self.sam_model_combo.addItem(tr("Auto ({model})", model={"tiny": "Tiny", "large": "Large"}[auto]), "auto")
        else:
            self.sam_model_combo.addItem(tr("Auto"), "auto")
        for size, name in names.items():
            self.sam_model_combo.addItem(name if on_disk[size] else tr("{model} — downloads on first use",
                                                                       model=name), size)
        self.sam_model_combo.setCurrentIndex(max(0, self.sam_model_combo.findData(keep)))
        self.sam_model_combo.blockSignals(False)

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
        self._fill_sam_models(detect=True)  # "auto" means another size on the CPU

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
    def _clear_faces(self) -> None:
        self.blurred = []
        self.review.set_cards([], None, 30.0, {})
        self.review.set_blurred([])

    def _refresh_review(self) -> None:
        if self.project is None or self.analysis is None:
            return
        self.review.set_blurred(self.blurred)
        n = sum(self.blurred)
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
        self.timeline.set_tracks([(BLUR_COLOR, self._blur_levels())] if self.plans else [])
        self.show_frame()

    def _blur_levels(self, without_manual: int | None = None) -> np.ndarray:
        """The blur band per frame (2 = face detected, 1 = bridged or drawn,
        0 = not blurred), optionally leaving out one drawn box."""
        n = self.analysis.dets.num_frames
        levels = np.zeros(n, np.int8)
        skip = None  # drawn boxes are the last plans, one each, in project order
        if without_manual is not None:
            skip = len(self.plans) - len(self.project.manual) + without_manual
        for k, plan in enumerate(self.plans):
            if k == skip:
                continue
            lv = getattr(plan, "_levels", None)  # plans are cached and reused, so is this
            if lv is None:
                src = np.asarray(plan.source)
                lv = plan._levels = np.where(src == "det", 2, (src != "").astype(np.int8)).astype(np.int8)
            np.maximum(levels, lv[:n], out=levels)
        return levels

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
            it = QListWidgetItem(tr("▢  {start} → {end} (frame {frame}): blurred area, {secs} s",
                                    start=fmt_time(m.first / fps), end=fmt_time((m.last + 1) / fps),
                                    frame=m.frame, secs=f"{(m.last - m.first + 1) / fps:.1f}"))
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
        n = (self.src_info if self.editing else self.info).num_frames
        self.frame = int(min(max(f, 0), n - 1))
        self.show_frame()

    def step(self, d: int) -> None:
        self.seek(self.frame + d)

    def show_frame(self, img: np.ndarray | None = None) -> None:
        if self.info is None:
            return
        if self.editing:
            self._show_source_frame(img)
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
            if m.first <= f <= m.last:
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
        self._play_info = None
        if self.editing:  # play the kept parts (uncropped, to see the crop), skipping what is cut
            keep = self.cuts.keep()
            info = self._play_info = self.src_info.with_edit(keep) if keep else self.src_info
            start = info.from_source(self.frame, after=True)
            if start is None or start >= info.num_frames - 1:
                start = 0
        else:
            if self.frame >= self.info.num_frames - 1:
                self.frame = 0
            info, start = self.info, self.frame
            if self.view_out.isChecked() and self.out_grabber is not None:
                info = self.out_grabber.info
        self.player = Player(info, start, self)
        self.player.tick.connect(self._on_play_tick)
        self.player.finished.connect(self._on_play_finished)
        self.player.start()
        self.play_btn.setIcon(self._play_icons["pause"])

    def _on_play_tick(self) -> None:
        if self.player is None:
            return
        item = self.player.latest()
        if item is not None:
            k, img = item
            self.frame = self._play_info.to_source(k) if self._play_info is not None else k
            self.show_frame(img)

    def _on_play_finished(self) -> None:
        if self.player is not None and self.sender() is self.player:
            self.player = None
            self.play_btn.setIcon(self._play_icons["play"])

    def _stop_player(self) -> None:
        if self.player is not None:
            p, self.player = self.player, None
            p.stop()
            p.wait(2000)
            self.play_btn.setIcon(self._play_icons["play"])

    # ============================================================ cut & crop
    def _edit_button(self) -> None:
        if self.editing:
            self._finish_edit()
        else:
            self._enter_edit()

    def _enter_edit(self) -> None:
        """Show the whole source with the cut parts and the crop, for editing."""
        if self.project is None or self.src_info is None or self.task is not None:
            return
        self._stop_player()
        src_frame = self.frame if self.editing else self.info.to_source(self.frame)
        self.editing = True
        self._edit_confirmed = False
        self._undo.clear()
        self._redo.clear()
        self.view_orig.setChecked(True)
        self.view.set_crop_mode(True)
        self.view.set_crop(self.project.edit.crop)
        self.edit_bar.setVisible(True)
        self.hint.setText(tr(EDIT_HINT))
        self.tabs.setCurrentIndex(0)
        self.timeline.set_length(self.src_info.num_frames, self.src_info.fps_float)
        self.timeline.set_parts(self.cuts.parts)
        self.frame = min(max(src_frame, 0), self.src_info.num_frames - 1)
        self._update_edit_summary()
        self._set_enabled()
        self.show_frame()
        self.view.setFocus()

    def _finish_edit(self) -> None:
        """Leave edit mode: detect faces in the edited video (or go back to the
        faces found before, when the cut & crop did not change)."""
        e = self.project.edit
        try:
            info = self.src_info.with_edit(e.keep, e.crop) if e.active else self.src_info
        except ValueError as err:
            QMessageBox.warning(self, tr("Cut & crop"), str(err))
            return
        self._stop_player()
        unchanged = self.analysis is not None and self._scanned_edit == e
        src_frame = self.frame
        self.editing = False
        self.view.set_crop_mode(False)
        self.edit_bar.setVisible(False)
        self.timeline.set_parts(None)
        self.hint.setText(tr(FACES_HINT))
        self._undo.clear()
        self._redo.clear()
        if not unchanged:
            self.info = info
            self.grabber = FrameGrabber(info) if info is not self.src_info else self.src_grabber
        self.timeline.set_length(self.info.num_frames, self.info.fps_float)
        k = self.info.from_source(src_frame, after=True)
        self.frame = k if k is not None else 0
        self._update_edit_summary()
        if unchanged:
            self._set_enabled()
            self.show_frame()
            return
        self.analysis, self.plans, self.excluded = None, [], set()
        self._scanned_edit = None
        self._clear_faces()
        self.timeline.set_tracks([])
        if self.dirty:
            self.save_project()
        self.show_frame()
        self._run_task(self._analysis_job(self.project.video), self._on_analysis_done, tr("Detecting faces"))

    def _edit_state(self) -> tuple:
        crop = self.project.edit.crop
        return self.cuts.state(), tuple(crop) if crop else None, self.project.edit.turn

    def _set_edit_state(self, state: tuple) -> None:
        parts, crop, turn = state
        self.cuts.restore(parts)
        self.project.edit.crop = list(crop) if crop else None
        self.project.edit.turn = turn

    def _edit_change(self, fn) -> None:
        """Run fn(), which changes the cut or the crop and returns True when it
        did, with undo. The first change of an edit session asks before clearing
        the face choices made for the old cut & crop."""
        if not self.editing or self.task is not None:
            return
        before = self._edit_state()
        if not fn():
            return
        if not self._confirm_edit_change():
            self._set_edit_state(before)
            self._edit_applied()
            return
        self._undo.append(before)
        del self._undo[:-200]
        self._redo.clear()
        self._edit_applied()

    def _confirm_edit_change(self) -> bool:
        p = self.project
        if self._edit_confirmed or not (p.labels or p.exclusions or p.manual):
            return True
        n = len(p.labels)
        fixes = len(p.exclusions) + len(p.manual)
        msg = tr("Changing the cut or crop means detecting the faces again. This clears your choices "
                 "({faces} chosen to blur, {fixes} corrections). Continue?", faces=n, fixes=fixes)
        if QMessageBox.question(self, tr("Cut & crop"), msg) != QMessageBox.StandardButton.Yes:
            return False
        p.labels.clear()
        p.exclusions.clear()
        p.manual.clear()
        self._edit_confirmed = True
        self._refresh_corrections()
        return True

    def _edit_applied(self) -> None:
        self.project.edit.keep = self.cuts.keep()
        if self.src_info.turn != self.project.edit.turn:  # show (and crop) the turned picture
            self.src_info = self._raw_info.turned(self.project.edit.turn)
            self.src_grabber = FrameGrabber(self.src_info)
            self._play_info = None
        self.view.set_crop(self.project.edit.crop)
        self.timeline.set_parts(self.cuts.parts)
        self._mark_dirty()
        self._update_edit_summary()
        self._set_enabled()
        self.show_frame()

    def _cut_in(self) -> None:
        self._edit_change(lambda: self.cuts.set_in(self.frame))

    def _cut_out(self) -> None:
        self._edit_change(lambda: self.cuts.set_out(self.frame))

    def _cut_split(self) -> None:
        self._edit_change(lambda: self.cuts.split(self.frame))

    def _cut_toggle(self) -> None:
        if not self.editing:
            return
        a, b, removed = self.cuts.parts[self.cuts.index_at(self.frame)]
        if not removed and self.cuts.kept_frames() <= b - a:
            self.statusBar().showMessage(tr("At least one part of the video must stay."), 4000)
            return
        self._edit_change(lambda: self.cuts.toggle_at(self.frame))

    def _edit_undo(self) -> None:
        if not self.editing or not self._undo or self.task is not None:
            return
        self._redo.append(self._edit_state())
        self._set_edit_state(self._undo.pop())
        self._edit_applied()

    def _edit_redo(self) -> None:
        if not self.editing or not self._redo or self.task is not None:
            return
        self._undo.append(self._edit_state())
        self._set_edit_state(self._redo.pop())
        self._edit_applied()

    def _turn(self, by: int) -> None:
        """Turn the video `by` degrees clockwise; the crop turns with it."""
        def apply():
            e = self.project.edit
            e.crop = turn_crop(e.crop, by, self.src_info.width, self.src_info.height)
            e.turn = (e.turn + by) % 360
            return True
        self._stop_player()
        self._edit_change(apply)

    def _on_crop_changed(self, rect) -> None:
        def apply():
            try:
                c = fit_crop(rect, self.src_info.width, self.src_info.height)
            except ValueError:  # too small
                return False
            new = list(c) if c else None
            if new == self.project.edit.crop:
                return False
            self.project.edit.crop = new
            return True
        self._edit_change(apply)
        self.view.set_crop(self.project.edit.crop)  # rounded, or back to before when nothing changed

    def _aspect_changed(self, *_) -> None:
        ar = self.aspect_combo.currentData() or None
        self.view.crop_aspect = ar
        if ar is None or not self.editing:
            self.view.update()
            return
        # the largest crop of this shape inside the current one, centred
        x, y, w, h = self.view.crop_rect()
        nw, nh = (h * ar, h) if w / h > ar else (w, w / ar)
        self._on_crop_changed((x + (w - nw) / 2, y + (h - nh) / 2, nw, nh))

    def _any_angle_toggled(self, on: bool) -> None:
        if self.project is not None and self.editing and self.project.edit.any_angle != on:
            self.project.edit.any_angle = on
            self._mark_dirty()
            self._set_enabled()  # the button reads "Detect faces" again when the mode changed

    def _crop_reset(self) -> None:
        if self.aspect_combo.currentIndex() != 0:
            self.aspect_combo.blockSignals(True)
            self.aspect_combo.setCurrentIndex(0)
            self.aspect_combo.blockSignals(False)
            self.view.crop_aspect = None
        self._on_crop_changed(None)

    def _update_edit_summary(self) -> None:
        if self.project is None or self.src_info is None:
            self.edit_summary.setText("–")
            return
        src, e = self.src_info, self.project.edit
        fps = src.fps_float
        total = fmt_time(src.num_frames / fps)
        if e.keep:
            kept = sum(b - a for a, b in e.keep)
            when = tr("Keeps {kept} of {total} ({parts})", kept=fmt_time(kept / fps), total=total,
                      parts=trn(len(e.keep), "{n} part", "{n} parts"))
        else:
            when = tr("The whole video ({total})", total=total)
        if e.crop:
            frame = tr("cropped to {w} × {h}", w=e.crop[2], h=e.crop[3])
        else:
            frame = tr("full frame {w} × {h}", w=src.width, h=src.height)
        turned = {90: tr(", turned right"), 180: tr(", upside down"), 270: tr(", turned left")}.get(e.turn, "")
        self.edit_summary.setText(f"{when}, {frame}{turned}.")
        self.any_angle_cb.blockSignals(True)
        self.any_angle_cb.setChecked(e.any_angle)
        self.any_angle_cb.blockSignals(False)

    def _show_source_frame(self, img: np.ndarray | None) -> None:
        f = self.frame
        if img is None:
            img = self.src_grabber.get(f)
        if img is None:
            return
        self.view.set_image(img)
        self.view.set_overlays([])
        self.timeline.set_position(f)
        n, fps = self.src_info.num_frames, self.src_info.fps_float
        cut = self.cuts.removed_at(f)
        self.pos_label.setText(f"{fmt_time(f / fps)} / {fmt_time(n / fps)}"
                               + tr("   frame {f} / {last}", f=f, last=n - 1) + (tr("  · cut") if cut else ""))
        self.cut_toggle_btn.setText(tr("Bring part back") if cut else tr("Remove part"))

    # ============================================================ corrections
    def _on_item_clicked(self, kind: str, ref: int, gpos) -> None:
        if self.project is None or self.task is not None:
            return
        self._stop_player()
        menu = QMenu(self)
        if kind == "manual":
            self._manual_menu(menu, ref)
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
            a = menu.addAction(tr("Blur this face from here…"))
            a.triggered.connect(lambda: self._add_manual(box))
        menu.exec(gpos)

    def _on_box_drawn(self, box: list, gpos) -> None:
        if self.project is None or self.task is not None:
            return
        self._stop_player()
        self._add_manual(box)

    def _manual_menu(self, menu: QMenu, k: int) -> None:
        m = self.project.manual[k]
        fps = self.info.fps_float
        menu.addAction(tr("Drawn area, {start} → {end}", start=fmt_time(m.first / fps),
                          end=fmt_time((m.last + 1) / fps))).setEnabled(False)
        menu.addAction(tr("Change duration…")).triggered.connect(lambda: self._edit_manual(k))
        menu.addAction(tr("Remove this drawn area")).triggered.connect(lambda: self._remove_manual(k))

    def _correction_menu(self, pos) -> None:
        it = self.corr_list.itemAt(pos)
        if it is None or self.project is None or self.task is not None:
            return
        kind, k = it.data(Qt.ItemDataRole.UserRole)
        if kind != "manual":
            return
        menu = QMenu(self)
        self._manual_menu(menu, k)
        menu.exec(self.corr_list.viewport().mapToGlobal(pos))

    def _ask_duration(self, frame: int, frames: int, without_manual: int | None, title: str,
                      box=None) -> int | None:
        """The duration dialog: frames a drawn box on `frame` covers, or None
        if cancelled. A new `box` stays visible on the video meanwhile."""
        from .manual_dialog import DurationDialog

        n = self.info.num_frames
        if self.analysis is not None and self.plans:
            levels = self._blur_levels(without_manual)
        else:
            levels = np.zeros(n, np.int8)
        if box is not None:
            shown = self._overlays(self.frame, False) if self.boxes_cb.isChecked() else []
            self.view.set_overlays(shown + [Overlay(tuple(float(v) for v in box), QColor("#ffd400"),
                                                    Qt.PenStyle.DashLine, 2.5, tr("new area"), "new")])
        dlg = DurationDialog(self, frame, n, self.info.fps_float, levels, frames, self.timeline, fmt_time, title)
        ok = bool(dlg.exec())
        if box is not None and not ok:
            self.show_frame()  # drop the pending box
        return dlg.frames if ok else None

    def _last_manual_seconds(self) -> float:
        try:
            return max(0.1, float(QSettings().value("manual_seconds", 1.0)))
        except (TypeError, ValueError):
            return 1.0

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
        fps = self.info.fps_float
        frames = self._ask_duration(self.frame, max(1, round(self._last_manual_seconds() * fps)), None,
                                    tr("Blur this area"), box)
        if frames is None:
            return
        QSettings().setValue("manual_seconds", frames / fps)
        self.project.manual.append(Manual(self.frame, [round(float(v), 1) for v in box], length=frames))
        self._mark_dirty()
        self._refresh_review()
        self._schedule_recompute()

    def _edit_manual(self, k: int) -> None:
        m = self.project.manual[k]
        # an older box (±span) becomes a forward-only one from its frame
        frames = self._ask_duration(m.frame, m.last - m.frame + 1, k, tr("Change duration"))
        if frames is None:
            return
        QSettings().setValue("manual_seconds", frames / self.info.fps_float)
        m.span, m.length = None, frames
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
        stage = tr(stage)
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
        self._back_to_edit = self.analysis is None

    def _on_failed(self, msg: str) -> None:
        self._back_to_edit = self.analysis is None
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
        back, self._back_to_edit = self._back_to_edit, False
        if back and self.project is not None and self.analysis is None and not self.editing:
            # the face scan was cancelled or failed: back to where it was started from, so
            # the cut & crop and "Faces at any angle" can be changed before trying again
            status = self.status.text()
            self._enter_edit()
            self.status.setText(status)

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
                                  progress=progress, cancel=cancel, any_angle=analysis.any_angle)
            return result, report

        self._run_task(job, self._on_render_done, tr("Rendering"))

    def _on_render_done(self, payload) -> None:
        result, report = payload
        self._fill_sam_models(detect=True)  # a model downloaded by this render loses its note
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
