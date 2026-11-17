"""Background threads: the pipeline never runs on the GUI thread; it reports
through Qt signals."""

from __future__ import annotations

import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..identity import Cancelled


class Task(QThread):
    """Runs fn(progress, cancel) in a worker thread.

    progress(stage, done, total) -> `progress` signal; result -> `done`;
    exceptions -> `failed`; Cancelled -> `cancelled`."""

    progress = Signal(str, int, int)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            result = self._fn(lambda s, d, t: self.progress.emit(s, int(d), int(t)), self._stop.is_set)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # report to the GUI instead of crashing the thread
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
        else:
            self.done.emit(result)


class Player(QThread):
    """Plays from a start frame by streaming frames from FFmpeg at the video's
    frame rate (seeking frame by frame is too slow for playback)."""

    tick = Signal()  # a new frame is available in `latest`

    def __init__(self, info, start: int, parent=None):
        super().__init__(parent)
        self.info = info
        self.start_frame = start
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None

    def stop(self) -> None:
        self._stop.set()

    def latest(self):
        """(frame index, BGR image) of the newest frame; older undisplayed frames
        are dropped, so a slow GUI never falls behind the video."""
        with self._lock:
            out, self._latest = self._latest, None
        return out

    def run(self) -> None:
        from ..video_io import FrameReader

        period = 1.0 / self.info.fps_float
        t0 = time.perf_counter()
        for k, img in enumerate(FrameReader(self.info, hwaccel=False, start_frame=self.start_frame)):
            if self._stop.is_set():
                return
            with self._lock:
                self._latest = (self.start_frame + k, np.array(img))
            self.tick.emit()
            delay = t0 + (k + 1) * period - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
