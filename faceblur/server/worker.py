"""The GPU worker: one thread that runs the heavy steps (face scan, render,
safety check) one job at a time, in the order they were queued.

The steps themselves are in `PipelineOps`, so tests can swap in fakes."""

from __future__ import annotations

import logging
import threading
import traceback
from collections import deque

from ..identity import Cancelled
from .jobs import THUMB, Job, JobStore

log = logging.getLogger("faceblur.server")


class PipelineOps:
    """The real pipeline (same calls as the desktop app makes)."""

    def scan(self, job: Job, progress, cancel) -> None:
        from ..pipeline import Analysis, card_thumbnails

        analysis = Analysis.open(job.video, progress=progress, cancel=cancel, wd=job.work)
        thumbs = card_thumbnails(analysis, THUMB, progress=progress, cancel=cancel)
        job.store.put_analysis(job, analysis, thumbs)

    def render(self, job: Job, progress, cancel) -> dict:
        from ..pipeline import load_target_refs, render, safety_check

        with job.lock:
            project = job.load_project()
        analysis = job.analysis()
        job.update(status="rendering")
        result = render(project, analysis, progress=progress, cancel=cancel, log=log.info)
        info = {"frames": result.frames, "fps": round(result.fps, 1), "unmasked": result.unmasked,
                "sam": result.sam_status, "safety": "skipped", "flagged": None}
        (analysis.wd / "safety_report.json").unlink(missing_ok=True)
        job.update(status="checking", rendered=True, render=info)
        try:
            report = safety_check(result.output, load_target_refs(analysis.wd), analysis.wd,
                                  progress=progress, cancel=cancel)
        except Cancelled:  # the video is done; only the check was cancelled
            return info
        info.update(safety="done", flagged=len(report["flagged"]))
        return info


class Worker(threading.Thread):
    def __init__(self, store: JobStore, ops=None):
        super().__init__(name="faceblur-worker", daemon=True)
        self.store = store
        self.ops = ops or PipelineOps()
        self._queue: deque[tuple[str, str]] = deque()  # (job id, "scan" | "render")
        self._cond = threading.Condition()
        self._current: str | None = None
        self._stopping = False
        for job in store.queued():  # queued before a restart
            self._queue.append((job.state.id, "scan" if job.state.status == "queued_scan" else "render"))

    # ------------------------------------------------------------ queue
    def submit(self, job: Job, kind: str) -> None:
        with self._cond:
            job.cancel_event.clear()
            job.update(status=f"queued_{kind}", stage="", done=0, total=0, error="")
            self._queue.append((job.state.id, kind))
            self._cond.notify()

    def position(self, job: Job) -> int | None:
        """0 = running now, 1 = next, ...; None when not queued."""
        with self._cond:
            if self._current == job.state.id:
                return 0
            for k, (jid, _) in enumerate(self._queue, 1):
                if jid == job.state.id:
                    return k
        return None

    def cancel(self, job: Job) -> bool:
        """Cancel a queued or running step. Returns False if nothing was running."""
        with self._cond:
            for item in list(self._queue):
                if item[0] == job.state.id:
                    self._queue.remove(item)
                    self._after_cancel(job, item[1])
                    return True
            if self._current == job.state.id:
                job.cancel_event.set()
                return True
        return False

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            if self._current:
                job = self.store.get(self._current)
                if job:
                    job.cancel_event.set()
            self._cond.notify()

    # ------------------------------------------------------------ thread
    def run(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                jid, kind = self._queue.popleft()
                self._current = jid
            job = self.store.get(jid)
            try:
                if job is not None:
                    self._run(job, kind)
            finally:
                with self._cond:
                    self._current = None

    def _run(self, job: Job, kind: str) -> None:
        cancel = job.cancel_event.is_set
        log.info("%s %s (%s)", kind, job.state.id, job.state.name)
        try:
            if kind == "scan":
                job.update(status="scanning", stage="Loading face models")
                self.ops.scan(job, job.progress, cancel)
                job.update(status="review", scanned=True, stage="", done=0, total=0)
            else:
                info = self.ops.render(job, job.progress, cancel)
                job.update(status="done", rendered=True, render=info, stage="", done=0, total=0)
        except Cancelled:
            self._after_cancel(job, kind)
        except Exception as e:  # report on the job instead of killing the worker
            log.error("%s %s failed:\n%s", kind, job.state.id, traceback.format_exc())
            job.update(status="failed", error=str(e) or type(e).__name__, stage="")
        finally:
            job.cancel_event.clear()

    @staticmethod
    def _after_cancel(job: Job, kind: str) -> None:
        if kind == "scan":
            job.update(status="cancelled", error="Face detection was cancelled.", stage="", done=0, total=0)
        else:
            job.update(status="done" if job.result.exists() and job.state.rendered else "review",
                       error="Rendering was cancelled.", stage="", done=0, total=0)
