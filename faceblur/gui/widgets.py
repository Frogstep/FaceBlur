"""Custom widgets: the video view (frame + clickable face boxes + drag-to-draw),
and the coverage timeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
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


CROP_MIN = 32  # smallest crop the handles allow, in video pixels
_HANDLE = 7  # half size of a crop handle, in screen pixels
_CURSORS = {"nw": Qt.CursorShape.SizeFDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor, "sw": Qt.CursorShape.SizeBDiagCursor,
            "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
            "w": Qt.CursorShape.SizeHorCursor, "e": Qt.CursorShape.SizeHorCursor,
            "move": Qt.CursorShape.SizeAllCursor}


class VideoView(QWidget):
    """Shows one frame with overlays. Click a box -> itemClicked; drag on the
    frame -> boxDrawn (both carry the global mouse position for a menu).

    In crop mode it shows a crop rectangle instead: drag its corners or edges,
    drag inside to move it, drag outside to draw a new one -> cropChanged."""

    itemClicked = Signal(str, int, QPoint)
    boxDrawn = Signal(list, QPoint)
    cropChanged = Signal(object)  # (x, y, w, h) in video pixels

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
        # crop mode
        self.crop_mode = False
        self.crop_aspect: float | None = None  # width / height, or None = free
        self._crop: tuple[float, float, float, float] | None = None  # None = full frame
        self._crop_drag: tuple | None = None  # (handle, press point in video px, rect at press)

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

    def set_crop_mode(self, on: bool) -> None:
        self.crop_mode = on
        self._crop_drag = None
        self._drag_from = self._drag_to = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_crop(self, crop) -> None:
        self._crop = tuple(float(v) for v in crop) if crop else None
        self.update()

    def crop_rect(self) -> tuple[float, float, float, float]:
        """The crop in video pixels (the full frame when there is none)."""
        if self._crop is not None:
            return self._crop
        return (0.0, 0.0, float(self._pix.width()), float(self._pix.height())) if self._pix else (0, 0, 0, 0)

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
        if self.crop_mode:
            self._paint_crop(p)

    def _crop_view_rect(self) -> QRectF:
        x, y, w, h = self.crop_rect()
        return self._view_rect((x, y, x + w, y + h))

    def _handles(self) -> dict[str, QPointF]:
        r = self._crop_view_rect()
        pts = {"nw": r.topLeft(), "ne": r.topRight(), "sw": r.bottomLeft(), "se": r.bottomRight()}
        if self.crop_aspect is None:  # edges change one side, which would break a fixed aspect
            c = r.center()
            pts.update(n=QPointF(c.x(), r.top()), s=QPointF(c.x(), r.bottom()),
                       w=QPointF(r.left(), c.y()), e=QPointF(r.right(), c.y()))
        return pts

    def _paint_crop(self, p: QPainter) -> None:
        frame, r = self._frame_rect(), self._crop_view_rect()
        shade = QColor(0, 0, 0, 160)
        p.fillRect(QRectF(frame.left(), frame.top(), frame.width(), r.top() - frame.top()), shade)
        p.fillRect(QRectF(frame.left(), r.bottom(), frame.width(), frame.bottom() - r.bottom()), shade)
        p.fillRect(QRectF(frame.left(), r.top(), r.left() - frame.left(), r.height()), shade)
        p.fillRect(QRectF(r.right(), r.top(), frame.right() - r.right(), r.height()), shade)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))  # rule of thirds
        for k in (1, 2):
            x = r.left() + r.width() * k / 3
            y = r.top() + r.height() * k / 3
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawRect(r)
        p.setBrush(QColor(255, 255, 255))
        for pt in self._handles().values():
            p.drawRect(QRectF(pt.x() - _HANDLE + 2, pt.y() - _HANDLE + 2, 2 * _HANDLE - 4, 2 * _HANDLE - 4))
        _, _, w, h = self.crop_rect()
        label = f"{int(w) // 2 * 2} × {int(h) // 2 * 2}"
        fm = p.fontMetrics()
        tag = QRectF(r.x() + 6, r.y() + 6, fm.horizontalAdvance(label) + 10, fm.height() + 4)
        p.fillRect(tag, QColor(0, 0, 0, 170))
        p.setPen(QColor(255, 255, 255))
        p.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)

    def _crop_hit(self, pos: QPointF) -> str | None:
        for name, pt in self._handles().items():
            if abs(pos.x() - pt.x()) <= _HANDLE + 2 and abs(pos.y() - pt.y()) <= _HANDLE + 2:
                return name
        if self._crop_view_rect().contains(pos):
            return "move"
        return None

    def _crop_drag_to(self, pos: QPointF) -> None:
        handle, (px, py), (x, y, w, h) = self._crop_drag
        W, H = float(self._pix.width()), float(self._pix.height())
        vx, vy = self.to_video(pos)
        mn = min(CROP_MIN, W, H)
        if handle == "move":
            nx = min(max(x + vx - px, 0.0), W - w)
            ny = min(max(y + vy - py, 0.0), H - h)
            self._crop = (nx, ny, w, h)
            return
        if handle == "new" or len(handle) == 2:  # a corner: the opposite corner stays put
            if handle == "new":
                ax, ay = px, py
            else:
                ax = x + w if "w" in handle else x
                ay = y + h if "n" in handle else y
            right, down = vx >= ax, vy >= ay
            room_w = (W - ax) if right else ax
            room_h = (H - ay) if down else ay
            nw, nh = abs(vx - ax), abs(vy - ay)
            ar = self.crop_aspect
            if ar:
                nw = max(nw, nh * ar)
                nw = min(nw, room_w, room_h * ar)
                nw = max(nw, min(mn, room_w, room_h * ar))
                nh = nw / ar
            else:
                nw = min(max(nw, min(mn, room_w)), room_w)
                nh = min(max(nh, min(mn, room_h)), room_h)
            self._crop = (ax if right else ax - nw, ay if down else ay - nh, nw, nh)
            return
        # an edge (free aspect only)
        x1, y1, x2, y2 = x, y, x + w, y + h
        if handle == "w":
            x1 = min(max(vx, 0.0), x2 - mn)
        elif handle == "e":
            x2 = max(min(vx, W), x1 + mn)
        elif handle == "n":
            y1 = min(max(vy, 0.0), y2 - mn)
        elif handle == "s":
            y2 = max(min(vy, H), y1 + mn)
        self._crop = (x1, y1, x2 - x1, y2 - y1)

    # ----------------------------------------------------------- mouse
    def mousePressEvent(self, e):
        if self._pix is None or e.button() != Qt.MouseButton.LeftButton:
            return
        if self.crop_mode:
            hit = self._crop_hit(e.position())
            if hit is None and not self._frame_rect().contains(e.position()):
                return
            self._crop_drag = (hit or "new", self.to_video(e.position()), self.crop_rect())
            return
        if self._frame_rect().contains(e.position()):
            self._drag_from = e.position()
            self._drag_to = None

    def mouseMoveEvent(self, e):
        if self.crop_mode:
            if self._crop_drag is not None:
                if self._crop_drag[0] == "new" and (e.position() - self.to_view(*self._crop_drag[1])).manhattanLength() < 6:
                    return  # a click, not a drag
                self._crop_drag_to(e.position())
                self.update()
            elif self._pix is not None:
                hit = self._crop_hit(e.position())
                self.setCursor(_CURSORS.get(hit, Qt.CursorShape.CrossCursor))
            return
        if self._drag_from is not None and self.drawing_enabled:
            if (e.position() - self._drag_from).manhattanLength() > 6:
                self._drag_to = e.position()
                self.update()
            return
        if self._pix is not None:
            hit = self._hit(e.position())
            self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, e):
        if self.crop_mode:
            if self._crop_drag is not None:
                before = self._crop_drag[2]
                self._crop_drag = None
                if self.crop_rect() != before:
                    self.cropChanged.emit(self.crop_rect())
            return
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
        self.parts: list | None = None  # cut & crop: [start, end, removed] per part of the source
        self.preview: tuple[int, int] | None = None  # frames (first, last) a drawn box would cover
        self._cache_key = None
        self._cache: list[np.ndarray] = []

    def set_parts(self, parts) -> None:
        """Show the parts of a cut (None: back to face tracks)."""
        self.parts = [tuple(p) for p in parts] if parts is not None else None
        self.setMinimumHeight(40 if parts is not None else 22 + 10 * max(1, len(self.tracks)))
        self.updateGeometry()
        self.update()

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

    def set_preview(self, frames: tuple[int, int] | None) -> None:
        """Highlight the frames (first, last) a drawn box would cover (None: off)."""
        self.preview = frames
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
        if self.parts is not None:
            self._paint_parts(p)
            return
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
        if self.preview is not None:
            a, b = self.preview
            x1 = int(self._x(a))
            x2 = max(int(self._x(b + 1)), x1 + 2)
            r = QRectF(x1, 6, x2 - x1, max(top, 18) - 4)
            p.fillRect(r, QColor(255, 212, 0, 110))
            p.setPen(QPen(QColor("#ffd400"), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
        p.setPen(QPen(EXCLUDED_COLOR, 1))
        for f in self.flags:
            x = int(self._x(f))
            p.drawLine(x, 0, x, self.height())
        x = int(self._x(self.pos))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(x, 0, x, self.height())

    def _paint_parts(self, p: QPainter) -> None:
        top, h = 8, self.height() - 12
        current = None
        for a, b, removed in self.parts:
            x1, x2 = int(self._x(a)), max(int(self._x(b)), int(self._x(a)) + 1)
            r = QRectF(x1, top, x2 - x1, h)
            if removed:
                p.fillRect(r, QColor(55, 25, 28))
                p.fillRect(r, QBrush(QColor(150, 60, 66), Qt.BrushStyle.BDiagPattern))
            else:
                p.fillRect(r, QColor(61, 120, 220, 150))
            if a <= self.pos < b:
                current = r
        p.setPen(QPen(QColor("#ffd400"), 2))  # split points
        for a, _, _ in self.parts[1:]:
            x = int(self._x(a))
            p.drawLine(x, top - 4, x, top + h + 2)
        if current is not None:  # the part Delete / Restore act on
            p.setPen(QPen(QColor(255, 255, 255, 200), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(current.adjusted(1, 1, -1, -1))
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
