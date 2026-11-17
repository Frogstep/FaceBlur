"""Custom widgets: the video view (frame + clickable face boxes + drag-to-draw),
and the coverage timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .i18n import tr

PERSON_COLORS = [QColor(c) for c in ("#3ddc84", "#4aa3ff", "#ffb020", "#e36cff", "#00d5d5", "#ff7a59")]
OTHER_COLOR = QColor(170, 170, 170)
EXCLUDED_COLOR = QColor("#ff4d4d")  # safety flags: still recognizable in the output
BLUR_COLOR = QColor("#ff4d4d")  # a face that will be blurred
KEEP_COLOR = QColor("#3ddc84")  # a face that stays visible


def person_color(i: int) -> QColor:
    return PERSON_COLORS[i % len(PERSON_COLORS)]


def bgr_to_qimage(img: np.ndarray) -> QImage:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QImage(img.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


@dataclass
class Overlay:
    box: tuple[float, float, float, float]  # video pixel coords
    color: QColor
    style: Qt.PenStyle = Qt.PenStyle.SolidLine
    width: float = 2.0
    label: str = ""
    kind: str = "det"  # det / manual / bridge
    ref: int = -1  # detection index or manual-box index


class VideoView(QWidget):
    """Shows one frame with overlays. Click a box -> itemClicked; drag on the
    frame -> boxDrawn (both carry the global mouse position for a menu)."""

    itemClicked = Signal(str, int, QPoint)
    boxDrawn = Signal(list, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._pix: QPixmap | None = None
        self._overlays: list[Overlay] = []
        self._drag_from: QPointF | None = None
        self._drag_to: QPointF | None = None
        self._message = tr("Open a video to start (File > Open, or drop a file here)")
        self.drawing_enabled = True

    # ----------------------------------------------------------- content
    def set_image(self, img: np.ndarray | None) -> None:
        self._pix = QPixmap.fromImage(bgr_to_qimage(img)) if img is not None else None
        self.update()

    def set_overlays(self, overlays: list[Overlay]) -> None:
        self._overlays = overlays
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self.update()

    # ----------------------------------------------------------- geometry
    def _frame_rect(self) -> QRectF:
        if self._pix is None:
            return QRectF()
        pw, ph = self._pix.width(), self._pix.height()
        s = min(self.width() / pw, self.height() / ph)
        w, h = pw * s, ph * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _scale(self) -> float:
        r = self._frame_rect()
        return r.width() / self._pix.width() if self._pix else 1.0

    def to_view(self, x: float, y: float) -> QPointF:
        r = self._frame_rect()
        s = self._scale()
        return QPointF(r.x() + x * s, r.y() + y * s)

    def to_video(self, p: QPointF) -> tuple[float, float]:
        r = self._frame_rect()
        s = self._scale()
        x = (p.x() - r.x()) / s
        y = (p.y() - r.y()) / s
        return (min(max(x, 0), self._pix.width()), min(max(y, 0), self._pix.height()))

    def _view_rect(self, box) -> QRectF:
        a = self.to_view(box[0], box[1])
        b = self.to_view(box[2], box[3])
        return QRectF(a, b)

    def _hit(self, pos: QPointF) -> Overlay | None:
        hits = [o for o in self._overlays if o.kind in ("det", "manual") and self._view_rect(o.box).contains(pos)]
        if not hits:
            return None
        return min(hits, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]))

    # ----------------------------------------------------------- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 20))
        if self._pix is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(self._frame_rect(), self._pix, QRectF(self._pix.rect()))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPointSizeF(9)
        font.setBold(True)
        p.setFont(font)
        for o in self._overlays:
            r = self._view_rect(o.box)
            p.setPen(QPen(o.color, o.width, o.style))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            if o.label:
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(o.label) + 8
                tag = QRectF(r.x(), r.y() - fm.height() - 2, tw, fm.height() + 2)
                if tag.y() < 0:
                    tag.moveTop(r.y())
                p.fillRect(tag, QColor(0, 0, 0, 170))
                p.setPen(o.color)
                p.drawText(tag, Qt.AlignmentFlag.AlignCenter, o.label)
        if self._drag_from is not None and self._drag_to is not None:
            p.setPen(QPen(QColor("#ff5cf0"), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self._drag_from, self._drag_to).normalized())

    # ----------------------------------------------------------- mouse
    def mousePressEvent(self, e):
        if self._pix is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if self._frame_rect().contains(e.position()):
            self._drag_from = e.position()
            self._drag_to = None

    def mouseMoveEvent(self, e):
        if self._drag_from is not None and self.drawing_enabled:
            if (e.position() - self._drag_from).manhattanLength() > 6:
                self._drag_to = e.position()
                self.update()
            return
        if self._pix is not None:
            hit = self._hit(e.position())
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, e):
        if self._drag_from is None:
            return
        start, end = self._drag_from, self._drag_to
        self._drag_from = self._drag_to = None
        self.update()
        gpos = e.globalPosition().toPoint()
        if end is not None:
            x1, y1 = self.to_video(start)
            x2, y2 = self.to_video(end)
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                self.boxDrawn.emit(box, gpos)
            return
        hit = self._hit(start)
        if hit is not None:
            self.itemClicked.emit(hit.kind, hit.ref, gpos)


class Timeline(QWidget):
    """One coverage band per person (bright = face detected, dim = bridged or
    manual), red ticks for frames the safety check flagged, and the playhead.
    Click or drag to seek."""

    seek = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.n = 0
        self.pos = 0
        self.fps = 30.0
        self.tracks: list[tuple[QColor, np.ndarray]] = []
        self.flags: list[int] = []
        self._cache_key = None
        self._cache: list[np.ndarray] = []

    def sizeHint(self) -> QSize:
        return QSize(600, 22 + 10 * max(1, len(self.tracks)))

    def set_length(self, n: int, fps: float) -> None:
        self.n, self.fps = n, fps
        self._cache_key = None
        self.update()

    def set_tracks(self, tracks) -> None:
        self.tracks = tracks
        self._cache_key = None
        self.setMinimumHeight(22 + 10 * max(1, len(tracks)))
        self.updateGeometry()
        self.update()

    def set_flags(self, frames: list[int]) -> None:
        self.flags = frames
        self.update()

    def set_position(self, f: int) -> None:
        self.pos = f
        self.update()

    def _columns(self, w: int) -> list[np.ndarray]:
        key = (w, self.n, id(self.tracks))
        if key != self._cache_key:
            self._cache = []
            for _, levels in self.tracks:
                # the container's frame count can be one more than the frames that
                # decode (and were scanned): pad the levels to the timeline length
                lv = np.zeros(self.n, np.int64)
                lv[: min(self.n, len(levels))] = levels[: self.n]
                if w >= self.n:  # more pixels than frames: each column shows its frame
                    out = lv[(np.arange(w) * self.n // w).clip(0, self.n - 1)]
                else:  # more frames than pixels: each column shows the strongest frame
                    out = np.zeros(w, np.int64)
                    np.maximum.at(out, (np.arange(self.n) * w // self.n), lv)
                self._cache.append(out)
            self._cache_key = key
        return self._cache

    def _x(self, f: int) -> float:
        return f * self.width() / max(self.n, 1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self.n <= 0:
            return
        w = self.width()
        # time ticks every 10 s (or 60 s for long videos)
        step = 10 if self.n / self.fps <= 600 else 60
        p.setPen(QColor(70, 70, 76))
        t = 0
        while t * self.fps < self.n:
            x = self._x(int(t * self.fps))
            p.drawLine(int(x), 0, int(x), 4)
            t += step
        top = 8
        for (color, _), cols in zip(self.tracks, self._columns(w)):
            dim = QColor(color)
            dim.setAlpha(90)
            for level, c in ((1, dim), (2, color)):
                xs = np.where(cols == level)[0]
                if not len(xs):
                    continue
                # draw runs of equal columns as rectangles
                breaks = np.where(np.diff(xs) > 1)[0]
                starts = np.r_[xs[0], xs[breaks + 1]]
                ends = np.r_[xs[breaks], xs[-1]]
                for a, b in zip(starts, ends):
                    p.fillRect(int(a), top, int(b - a + 1), 7, c)
            top += 10
        p.setPen(QPen(EXCLUDED_COLOR, 1))
        for f in self.flags:
            x = int(self._x(f))
            p.drawLine(x, 0, x, self.height())
        x = int(self._x(self.pos))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(x, 0, x, self.height())

    def _seek_to(self, x: float) -> None:
        if self.n:
            self.seek.emit(int(min(max(x / max(self.width(), 1) * self.n, 0), self.n - 1)))

    def mousePressEvent(self, e):
        self._seek_to(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(e.position().x())
