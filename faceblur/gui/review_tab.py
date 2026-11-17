"""The 'Review faces' tab: every face card with thumbnails across it; the user
selects cards and labels them (a person, someone else, or not a face)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from ..review import NOT_A_FACE, SOMEONE_ELSE, UNLABELED
from .i18n import tr
from .widgets import bgr_to_qimage, person_color

THUMB = 84


def label_text(label: int, names: list[str]) -> str:
    if label == UNLABELED:
        return tr("Not labeled")
    if label == SOMEONE_ELSE:
        return tr("Someone else (not masked)")
    if label == NOT_A_FACE:
        return tr("Not a face")
    return tr("{name} (masked)", name=names[label]) if 0 <= label < len(names) else "?"


def label_color(label: int) -> str:
    if label == UNLABELED:
        return "#e0e0e0"
    if label == SOMEONE_ELSE:
        return "#8a8a8a"
    if label == NOT_A_FACE:
        return "#5a5a5a"
    return person_color(label).name()


class CardRow(QFrame):
    """One face card: number + time range, thumbnails, label badge."""

    pressed = Signal(int, object)  # card position, Qt.KeyboardModifiers
    jump = Signal(int)  # frame

    def __init__(self, pos: int, number: int, info: str, thumbs: list[tuple[int, np.ndarray]], parent=None):
        super().__init__(parent)
        self.pos = pos
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 4, 6, 4)
        left = QVBoxLayout()
        self.num = QLabel(f"<b style='font-size:15pt'>#{number}</b>")
        self.info = QLabel(info)
        self.info.setStyleSheet("color: #aaa;")
        self.badge = QLabel()
        self.badge.setMinimumWidth(170)
        left.addWidget(self.num)
        left.addWidget(self.info)
        left.addWidget(self.badge)
        left.addStretch(1)
        h.addLayout(left)
        for frame, img in thumbs:
            t = QLabel()
            t.setFixedSize(THUMB, THUMB)
            t.setPixmap(QPixmap.fromImage(bgr_to_qimage(img)).scaled(
                THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            t.setToolTip(tr("Double-click to show this frame in the video"))
            t.mouseDoubleClickEvent = lambda _e, f=frame: self.jump.emit(f)
            h.addWidget(t)
        h.addStretch(1)
        self.selected = False
        self.label = UNLABELED

    def set_state(self, label: int, names: list[str], selected: bool) -> None:
        self.label, self.selected = label, selected
        color = label_color(label)
        bg = "#2b3b55" if selected else "transparent"
        width = 3 if label >= 0 or selected else 2
        style = "dashed" if label == UNLABELED else "solid"
        self.setStyleSheet(f"CardRow {{ border: {width}px {style} {color}; border-radius: 5px; background: {bg}; }}")
        weight = "bold" if label != UNLABELED else "normal"
        self.badge.setText(f"<span style='color:{color}; font-weight:{weight}'>{label_text(label, names)}</span>")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.pressed.emit(self.pos, e.modifiers())


class ReviewPanel(QWidget):
    """Lists all cards; emits labelCards(card positions, label) and jump(frame).

    Keys: 1-9 = person 1-9, O = someone else, N = not a face, Backspace/Delete =
    clear, Ctrl+A = select all shown, Space = next unlabeled card."""

    labelCards = Signal(list, int)
    newPersonFor = Signal(list)  # label the cards as a new person
    jump = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.header = QLabel()
        self.header.setWordWrap(True)
        v.addWidget(self.header)
        help_ = QLabel(tr("Click a card to select it (Ctrl+click: several, Shift+click: a range), then label it "
                          "with a button below or a key: <b>1–9</b> person, <b>O</b> someone else, "
                          "<b>N</b> not a face, <b>Backspace</b> clear. <b>Space</b> jumps to the next unlabeled "
                          "card. Double-click a thumbnail to see it in the video. Similar faces are listed next to "
                          "each other; nothing is joined automatically."))
        help_.setWordWrap(True)
        help_.setStyleSheet("color: #aaa;")
        v.addWidget(help_)

        bar = QHBoxLayout()
        self.buttons = QHBoxLayout()
        bar.addLayout(self.buttons)
        bar.addStretch(1)
        bar.addWidget(QLabel(tr("Show")))
        self.filter = QComboBox()
        self.filter.addItem(tr("All cards"), "all")
        self.filter.addItem(tr("Not labeled"), "unlabeled")
        self.filter.addItem(tr("Masked people"), "people")
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
        self.labels: list[int] = []
        self.names: list[str] = []
        self.selection: set[int] = set()
        self._anchor = None

    # ---------------------------------------------------------------- content
    def set_cards(self, cards, dets, fps: float, thumbs: dict[int, np.ndarray]) -> None:
        for r in self.rows:
            r.setParent(None)
            r.deleteLater()
        self.rows, self.selection, self._anchor = [], set(), None
        for k, c in enumerate(cards):
            fr = dets.frame[c.det_idx]
            m0, s0 = divmod(fr.min() / fps, 60)
            m1, s1 = divmod(fr.max() / fps, 60)
            info = f"{int(m0)}:{s0:04.1f} – {int(m1)}:{s1:04.1f}\n" + tr("frames: {n}", n=len(c.det_idx))
            tl = [(int(dets.frame[i]), thumbs[i]) for i in c.samples if i in thumbs]
            row = CardRow(k, c.number, info, tl)
            row.pressed.connect(self._on_pressed)
            row.jump.connect(self.jump)
            self.lay.insertWidget(self.lay.count() - 1, row)
            self.rows.append(row)

    def set_labels(self, labels: list[int], names: list[str]) -> None:
        self.labels, self.names = labels, names
        for r in self.rows:
            r.set_state(labels[r.pos], names, r.pos in self.selection)
        n_un = sum(1 for x in labels if x == UNLABELED)
        total = len(labels)
        if not total:
            self.header.setText(f"<b>{tr('No faces were found in this video.')}</b>")
        elif n_un:
            done = tr("{done} of {total} face cards labeled", done=total - n_un, total=total)
            need = tr("{n} still need a label before rendering.", n=n_un)
            self.header.setText(f"<b style='font-size:12pt'>{done}</b> "
                                f"— <span style='color:#ffb020'>{need}</span>")
        else:
            self.header.setText(f"<b style='font-size:12pt'>{tr('All {total} face cards labeled.', total=total)}</b> "
                                f"<span style='color:#3ddc84'>{tr('Ready to render.')}</span>")
        self._rebuild_buttons()
        self._apply_filter()

    def _rebuild_buttons(self) -> None:
        while self.buttons.count():
            w = self.buttons.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        def add(text, color, fn, tip):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setStyleSheet(f"QPushButton {{ border: 2px solid {color}; padding: 4px 8px; }}")
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.clicked.connect(fn)
            self.buttons.addWidget(b)

        for i, n in enumerate(self.names):
            key = f" ({i + 1})" if i < 9 else ""
            add(f"{n}{key}", person_color(i).name(), lambda _=False, i=i: self._label(i), tr("Mask as {}", n))
        add(tr("New person"), "#ff5cf0", lambda: self._new_person(), tr("Mask the selected cards as a new person"))
        add(tr("Someone else (O)"), "#8a8a8a", lambda: self._label(SOMEONE_ELSE), tr("A real face that is not masked"))
        add(tr("Not a face (N)"), "#5a5a5a", lambda: self._label(NOT_A_FACE), tr("A false detection"))
        add(tr("Clear"), "#444", lambda: self._label(UNLABELED), tr("Remove the label"))

    def _apply_filter(self) -> None:
        mode = self.filter.currentData()
        for r in self.rows:
            lab = self.labels[r.pos] if r.pos < len(self.labels) else UNLABELED
            show = mode == "all" or (mode == "unlabeled" and lab == UNLABELED) or (mode == "people" and lab >= 0)
            r.setVisible(show)

    # ---------------------------------------------------------------- selection
    def _visible_positions(self) -> list[int]:
        return [r.pos for r in self.rows if r.isVisible()]

    def _on_pressed(self, pos: int, mods) -> None:
        self.setFocus()
        if mods & Qt.KeyboardModifier.ShiftModifier and self._anchor is not None:
            vis = self._visible_positions()
            if self._anchor in vis and pos in vis:
                a, b = sorted((vis.index(self._anchor), vis.index(pos)))
                self.selection = set(vis[a:b + 1])
        elif mods & Qt.KeyboardModifier.ControlModifier:
            self.selection ^= {pos}
            self._anchor = pos
        else:
            self.selection = {pos}
            self._anchor = pos
        self._refresh_selection()

    def select(self, pos: int) -> None:
        self.selection = {pos}
        self._anchor = pos
        self._refresh_selection()
        self.area.ensureWidgetVisible(self.rows[pos], 0, 40)

    def _refresh_selection(self) -> None:
        for r in self.rows:
            r.set_state(self.labels[r.pos], self.names, r.pos in self.selection)

    def _label(self, label: int) -> None:
        if not self.selection:
            return
        sel = sorted(self.selection)
        self.labelCards.emit(sel, label)
        self._next_after(sel[-1])

    def _new_person(self) -> None:
        if self.selection:
            sel = sorted(self.selection)
            self.newPersonFor.emit(sel)
            self._next_after(sel[-1])

    def _next_after(self, pos: int) -> None:
        """After labeling, move on to the next unlabeled card (if any)."""
        vis = self._visible_positions()
        order = [p for p in vis if p > pos] + [p for p in vis if p <= pos]
        nxt = next((p for p in order if self.labels[p] == UNLABELED and p not in self.selection), None)
        if nxt is not None:
            self.select(nxt)

    def keyPressEvent(self, e):
        key, text = e.key(), e.text().lower()
        if text and text in "123456789" and int(text) <= len(self.names):
            self._label(int(text) - 1)
        elif text in ("o", "щ"):  # also the same keys on a Russian keyboard layout
            self._label(SOMEONE_ELSE)
        elif text in ("n", "т"):
            self._label(NOT_A_FACE)
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._label(UNLABELED)
        elif key == Qt.Key.Key_Space:
            self._next_after(max(self.selection) if self.selection else -1)
        elif key == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.selection = set(self._visible_positions())
            self._refresh_selection()
        elif key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.rows:
            vis = self._visible_positions()
            if vis:
                cur = max(self.selection) if self.selection else vis[0]
                i = vis.index(cur) if cur in vis else 0
                i = min(len(vis) - 1, i + 1) if key == Qt.Key.Key_Down else max(0, i - 1)
                self.select(vis[i])
        else:
            super().keyPressEvent(e)
