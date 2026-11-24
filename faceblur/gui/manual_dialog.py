"""How long a drawn box blurs: asked right after the box is drawn (and when
its duration is changed later). The box covers its own frame and goes forward
from there; the range is previewed on the timeline while the dialog is open."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout)

from ..identity import next_blurred_frame
from .i18n import tr, trn
from .widgets import Timeline


def seconds_to_frames(seconds: float, fps: float, most: int) -> int:
    """Frames a duration covers on the timeline: at least 1, at most `most`."""
    return int(min(max(round(seconds * fps), 1), max(most, 1)))


class DurationDialog(QDialog):
    """Asks for the duration of a drawn box starting on `frame`. `levels` is
    the blur band without this box (> 0 = blurred), used by "Until the next
    blurred part". After exec(), `frames` holds the chosen length."""

    def __init__(self, parent, frame: int, num_frames: int, fps: float, levels: np.ndarray,
                 frames: int, timeline: Timeline, fmt_time, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.frame, self.fps, self.timeline, self.fmt_time = frame, fps, timeline, fmt_time
        self.most = max(1, num_frames - frame)  # up to the end of the video
        self.frames = min(max(1, frames), self.most)

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Blur for")))
        self.spin = QDoubleSpinBox(decimals=1, singleStep=0.1, minimum=0.1,
                                   maximum=max(0.1, round(self.most / fps, 1) + 0.05))
        self.spin.setSuffix(tr(" s"))
        self.spin.setValue(self.frames / fps)
        self.spin.setKeyboardTracking(False)
        row.addWidget(self.spin)
        row.addStretch(1)
        lay.addLayout(row)
        self.info = QLabel()
        lay.addWidget(self.info)

        quick = QHBoxLayout()
        nxt = next_blurred_frame(levels, frame)
        self.next_btn = QPushButton(tr("Until the next blurred part"))
        if nxt is not None and nxt > frame:
            self.next_btn.setToolTip(tr("Up to {time} (frame {f}), where the blur starts again",
                                        time=fmt_time(nxt / fps), f=nxt))
            self.next_btn.clicked.connect(lambda: self._set_frames(nxt - frame))
        else:
            self.next_btn.setEnabled(False)
            self.next_btn.setToolTip(tr("Nothing is blurred later in the video"))
        quick.addWidget(self.next_btn)
        end_btn = QPushButton(tr("Until the end of the video"))
        end_btn.clicked.connect(lambda: self._set_frames(self.most))
        quick.addWidget(end_btn)
        lay.addLayout(quick)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.spin.valueChanged.connect(self._spin_changed)
        self.finished.connect(lambda _: self.timeline.set_preview(None))
        self._update()

    def _spin_changed(self, seconds: float) -> None:
        self.frames = seconds_to_frames(seconds, self.fps, self.most)
        self._update()

    def _set_frames(self, n: int) -> None:
        self.frames = min(max(1, n), self.most)
        self.spin.blockSignals(True)
        self.spin.setValue(self.frames / self.fps)
        self.spin.blockSignals(False)
        self._update()

    def _update(self) -> None:
        last = self.frame + self.frames - 1
        self.info.setText(tr("{frames}: {start} → {end} (frames {a}–{b})",
                             frames=trn(self.frames, "{n} frame", "{n} frames"),
                             start=self.fmt_time(self.frame / self.fps), end=self.fmt_time((last + 1) / self.fps),
                             a=self.frame, b=last))
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.timeline.set_preview((self.frame, last))
