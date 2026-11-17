"""The 'Review faces' tab: every face card with thumbnails across it. Every
face starts as not blurred; clicking a card switches it between blurred and
not blurred."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from .i18n import tr, trn
from .widgets import BLUR_COLOR, KEEP_COLOR, bgr_to_qimage

THUMB = 84


class CardRow(QFrame):
    """One face card: number + time range, blur state, thumbnails. A click
    anywhere on it switches it."""

    clicked = Signal(int)  # card position
    jump = Signal(int)  # frame

    def __init__(self, pos: int, number: int, info: str, thumbs: list[tuple[int, np.ndarray]], parent=None):
        super().__init__(parent)
        self.pos = pos
        self.number = number
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("Click to blur this face, or to leave it visible"))
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 6, 8, 6)
        left = QVBoxLayout()
        self.num = QLabel(f"<b style='font-size:15pt'>#{number}</b>")
        self.state = QLabel()
        self.state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state.setMinimumWidth(130)
        self.info = QLabel(info)
        self.info.setStyleSheet("color: #aaa;")
        show = QPushButton(tr("Show in video"))
        show.setToolTip(tr("Show this face in the video"))
        show.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.first_frame = thumbs[0][0] if thumbs else 0
        show.clicked.connect(lambda: self.jump.emit(self.first_frame))
        left.addWidget(self.num)
        left.addWidget(self.state)
        left.addWidget(self.info)
        left.addWidget(show)
        left.addStretch(1)
        h.addLayout(left)
        for _frame, img in thumbs:
            t = QLabel()
            t.setFixedSize(THUMB, THUMB)
            t.setPixmap(QPixmap.fromImage(bgr_to_qimage(img)).scaled(
                THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            h.addWidget(t)
        h.addStretch(1)
        self.blur = False
        self.current = False

    def set_state(self, blur: bool, current: bool) -> None:
        self.blur, self.current = blur, current
        red, green = BLUR_COLOR.name(), KEEP_COLOR.name()
        if blur:
            frame = f"border: 4px solid {red}; background: rgba(255, 77, 77, 40);"
            self.state.setText(f"● {tr('BLURRED')}")
            self.state.setStyleSheet(f"background: {red}; color: white; font-weight: bold; "
                                     f"border-radius: 9px; padding: 3px 10px;")
        else:
            frame = "border: 2px solid #555; background: transparent;"
            self.state.setText(f"○ {tr('Not blurred')}")
            self.state.setStyleSheet(f"color: {green}; border: 2px solid {green}; font-weight: bold; "
                                     f"border-radius: 9px; padding: 2px 9px;")
        if current:  # the keyboard position
            frame += " border-style: double;" if blur else " border: 2px solid #8ab4ff;"
        self.setStyleSheet(f"CardRow {{ {frame} border-radius: 6px; }}")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.pos)


class ReviewPanel(QWidget):
    """Lists all cards; emits blurCards(card positions, blur) and jump(frame).

    Mouse: click a card to switch it. Keys: Up/Down move, Space switches the
    current card, Enter shows it in the video."""

    blurCards = Signal(list, bool)
    jump = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.header = QLabel()
        self.header.setWordWrap(True)
        v.addWidget(self.header)
        help_ = QLabel(tr("Every face stays visible until you choose to blur it. <b>Click a face to blur it</b>; "
                          "click it again to leave it visible. Keys: <b>↑/↓</b> move, <b>Space</b> switch, "
                          "<b>Enter</b> show in the video."))
        help_.setWordWrap(True)
        help_.setStyleSheet("color: #aaa;")
        v.addWidget(help_)

        bar = QHBoxLayout()
        for text, blur in ((tr("Blur all"), True), (tr("Blur none"), False)):
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _=False, blur=blur: self._set_all(blur))
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(QLabel(tr("Show")))
        self.filter = QComboBox()
        self.filter.addItem(tr("All faces"), "all")
        self.filter.addItem(tr("Blurred"), "blur")
        self.filter.addItem(tr("Not blurred"), "keep")
        self.filter.currentIndexChanged.connect(lambda _: self._apply_filter())
        bar.addWidget(self.filter)
        v.addLayout(bar)

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.list = QWidget()
        self.lay = QVBoxLayout(self.list)
        self.lay.setSpacing(4)
        self.lay.addStretch(1)
        self.area.setWidget(self.list)
        v.addWidget(self.area, 1)

        self.rows: list[CardRow] = []
        self.blurred: list[bool] = []
        self.current: int | None = None

    # ---------------------------------------------------------------- content
    def set_cards(self, cards, dets, fps: float, thumbs: dict[int, np.ndarray]) -> None:
        for r in self.rows:
            r.setParent(None)
            r.deleteLater()
        self.rows, self.current = [], None
        for k, c in enumerate(cards):
            fr = dets.frame[c.det_idx]
            m0, s0 = divmod(fr.min() / fps, 60)
            m1, s1 = divmod(fr.max() / fps, 60)
            info = f"{int(m0)}:{s0:04.1f} – {int(m1)}:{s1:04.1f}\n" + trn(len(c.det_idx), "{n} frame", "{n} frames")
            tl = [(int(dets.frame[i]), thumbs[i]) for i in c.samples if i in thumbs]
            row = CardRow(k, c.number, info, tl)
            row.clicked.connect(self._on_clicked)
            row.jump.connect(self.jump)
            self.lay.insertWidget(self.lay.count() - 1, row)
            self.rows.append(row)

    def set_blurred(self, blurred: list[bool]) -> None:
        self.blurred = blurred
        n, total = sum(blurred), len(blurred)
        if not total:
            self.header.setText(f"<b>{tr('No faces were found in this video.')}</b>")
        else:
            color = BLUR_COLOR.name() if n else "#aaa"
            text = trn(n, "{n} of {total} faces will be blurred", "{n} of {total} faces will be blurred", total=total)
            self.header.setText(f"<b style='font-size:13pt; color:{color}'>{text}</b>")
        self._refresh()
        self._apply_filter()

    def _refresh(self) -> None:
        for r in self.rows:
            if r.pos < len(self.blurred):
                r.set_state(self.blurred[r.pos], r.pos == self.current)

    def _apply_filter(self) -> None:
        mode = self.filter.currentData()
        for r in self.rows:
            blur = self.blurred[r.pos] if r.pos < len(self.blurred) else False
            r.setVisible(mode == "all" or (mode == "blur") == blur)

    # ---------------------------------------------------------------- switching
    def _on_clicked(self, pos: int) -> None:
        self.setFocus()
        self.current = pos
        self.toggle(pos)

    def toggle(self, pos: int) -> None:
        if 0 <= pos < len(self.blurred):
            self.blurCards.emit([pos], not self.blurred[pos])

    def _set_all(self, blur: bool) -> None:
        cards = [k for k, b in enumerate(self.blurred) if b != blur]
        if not cards:
            return
        if not blur and QMessageBox.question(
                self, tr("Blur none"), tr("Remove the blur from all {n} faces?", n=len(cards))) \
                != QMessageBox.StandardButton.Yes:
            return
        self.blurCards.emit(cards, blur)

    def select(self, pos: int) -> None:
        """Make a card the current one and scroll to it."""
        self.current = pos
        self._refresh()
        self.area.ensureWidgetVisible(self.rows[pos], 0, 40)

    def _visible_positions(self) -> list[int]:
        return [r.pos for r in self.rows if r.isVisible()]

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Space and self.current is not None:
            self.toggle(self.current)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.current is not None:
            self.jump.emit(self.rows[self.current].first_frame)
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.rows:
            vis = self._visible_positions()
            if vis:
                i = vis.index(self.current) if self.current in vis else -1
                i = min(len(vis) - 1, i + 1) if key == Qt.Key.Key_Down else max(0, i - 1)
                self.select(vis[i])
        else:
            super().keyPressEvent(e)
