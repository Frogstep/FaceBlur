"""Jobs: one folder per uploaded video, with its state in job.json.

  jobs/<id>/
    job.json                 status, progress, error, video info
    source.<ext>             the uploaded video
    project.faceblur.json    an ordinary project file (also opens in the desktop app)
    work/                    detections, thumbnails, render log, safety report, flagged/
    result.mp4               the rendered video

When a job is deleted, its source, result and small records move to
history/<date>_<name>_<id>/ (see JobStore.delete) for debugging.

Statuses:
  uploading -> queued_scan -> scanning -> review -> queued_render -> rendering
  -> checking -> done; any step can end in failed or cancelled. From review, done
  or a failed render the user can change labels and render again.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..project import Project

log = logging.getLogger("faceblur.server")

QUEUED = {"queued_scan", "queued_render"}
RUNNING = {"scanning", "rendering", "checking"}
BUSY = QUEUED | RUNNING


@dataclass
class JobState:
    id: str
    name: str  # original file name from the client (display only, never a path)
    source: str  # file name of the upload inside the job folder
    created: float
    updated: float
    status: str = "uploading"
    stage: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    scanned: bool = False  # detections and face cards exist
    rendered: bool = False  # result.mp4 exists
    video: dict = field(default_factory=dict)  # width, height, fps, frames, duration, has_audio
    render: dict = field(default_factory=dict)  # last render: frames, fps, unmasked, sam, flagged, safety
    upload_size: int = 0  # expected size of a resumable upload (0 = single-request upload)
    scan_rev: int = 0  # +1 per finished face scan (cache-busting for thumbnails)


class Job:
    def __init__(self, store: "JobStore", state: JobState):
        self.store = store
        self.state = state
        self.dir = store.root / state.id
        self.cancel_event = threading.Event()
        self.lock = threading.RLock()  # project edits
        self._last_save = 0.0

    # ------------------------------------------------------------ paths
    @property
    def video(self) -> Path:
        return self.dir / self.state.source

    @property
    def work(self) -> Path:
        return self.dir / "work"

    @property
    def project_path(self) -> Path:
        return self.dir / "project.faceblur.json"

    @property
    def result(self) -> Path:
        return self.dir / "result.mp4"

    @property
    def part(self) -> Path:
        """The upload while it is incomplete."""
        return self.video.with_name(self.video.name + ".part")

    def uploaded(self) -> int:
        try:
            return self.part.stat().st_size
        except FileNotFoundError:
            return 0

    @property
    def busy(self) -> bool:
        return self.state.status in BUSY

    # ------------------------------------------------------------ state
    def update(self, persist: bool = True, **kw) -> None:
        for k, v in kw.items():
            setattr(self.state, k, v)
        self.state.updated = time.time()
        if persist:
            self.save()

    def progress(self, stage: str, done: int, total: int) -> None:
        """Pipeline progress hook; written to disk at most every 2 s."""
        self.update(persist=False, stage=stage, done=int(done), total=int(total))
        if time.time() - self._last_save > 2.0:
            self.save()

    def save(self) -> None:
        self._last_save = time.time()
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self.state), indent=1), encoding="utf-8")
        tmp.replace(self.dir / "job.json")

    # ------------------------------------------------------------ project
    def load_project(self) -> Project:
        if self.project_path.exists():
            return Project.load(self.project_path)
        p = Project.new(self.video)
        p.output = str(self.result)
        p.path = str(self.project_path)
        return p

    def save_project(self, project: Project) -> None:
        project.output = str(self.result)
        project.save(self.project_path)

    # ------------------------------------------------------------ analysis
    def analysis(self):
        """The face scan's Analysis (cards etc.), cached in memory for a few jobs."""
        return self.store.analysis(self)

    def thumbnails(self) -> dict[int, np.ndarray]:
        return self.store.thumbnails(self)

    def forget_analysis(self) -> None:
        self.store.forget(self.state.id)


THUMB = 160  # thumbnail size in pixels


# kept in the history folder when a job is deleted (the large caches are not:
# detections, thumbnails and flagged frames can all be made again)
HISTORY_FILES = ["job.json", "project.faceblur.json", "result.mp4",
                 "work/render_log.json", "work/safety_report.json"]


class JobStore:
    def __init__(self, root: str | Path, keep_days: float = 7.0, cache_size: int = 3,
                 history: str | Path | None = None):
        """history: folder that receives the source video, the result and the job's
        small records when a job is deleted (None: deleted jobs are gone)."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.history = Path(history).resolve() if history is not None else None
        self.keep_days = keep_days
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # id -> (analysis, thumbs|None)
        self._cache_size = cache_size
        self._cache_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        for d in sorted(self.root.iterdir()):
            f = d / "job.json"
            if not f.is_file():
                continue
            try:
                state = JobState(**json.loads(f.read_text(encoding="utf-8")))
            except Exception as e:  # a broken job folder must not stop the server
                log.warning("skipping %s: %s", d.name, e)
                continue
            job = Job(self, state)
            if state.status == "uploading" and not state.upload_size:
                log.info("removing unfinished upload %s", state.id)
                shutil.rmtree(d, ignore_errors=True)
                continue
            if state.status in RUNNING:
                step = "face scan" if state.status == "scanning" else "render"
                job.update(status="failed", error=f"The server stopped during the {step}. Start it again.")
            self._jobs[state.id] = job

    # ------------------------------------------------------------ jobs
    def create(self, name: str, suffix: str, upload_size: int = 0) -> Job:
        jid = uuid.uuid4().hex
        now = time.time()
        safe_suffix = suffix.lower() if suffix and len(suffix) <= 6 and suffix[1:].isalnum() else ".mp4"
        state = JobState(id=jid, name=Path(name or "video").name[:200], source="source" + safe_suffix,
                         created=now, updated=now, upload_size=upload_size)
        job = Job(self, state)
        job.dir.mkdir(parents=True)
        job.save()
        with self._lock:
            self._jobs[jid] = job
        return job

    def get(self, jid: str) -> Job | None:
        with self._lock:
            return self._jobs.get(jid)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: -j.state.created)

    def queued(self) -> list[Job]:
        """Jobs waiting for the worker after a restart, oldest first."""
        return sorted((j for j in self.all() if j.state.status in QUEUED), key=lambda j: j.state.updated)

    def delete(self, job: Job, reason: str = "deleted by the user", archive: bool = True) -> Path | None:
        """Remove a job. Unless archive is False (or there is no history folder, or
        the upload never completed), its source, result and records are moved to
        the history folder first; returns that folder. If moving fails, the job
        is left as it was and the error is raised."""
        kept = self._archive(job, reason) if archive else None
        with self._lock:
            self._jobs.pop(job.state.id, None)
        self.forget(job.state.id)
        shutil.rmtree(job.dir, ignore_errors=True)
        return kept

    def _archive(self, job: Job, reason: str) -> Path | None:
        if self.history is None or job.state.status == "uploading" or not job.video.is_file():
            return None
        s = job.state
        stem = re.sub(r"[^\w.-]+", "_", Path(s.name).stem, flags=re.UNICODE).strip("._")[:60] or "video"
        dest = self.history / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{stem}_{s.id[:8]}"
        dest.mkdir(parents=True)
        try:
            for rel in [s.source, *HISTORY_FILES]:
                src = job.dir / rel
                if src.is_file():
                    target = dest / rel
                    target.parent.mkdir(exist_ok=True)
                    shutil.move(src, target)  # a rename on the same drive
            record = json.loads((dest / "job.json").read_text(encoding="utf-8")) if (dest / "job.json").is_file() \
                else asdict(s)
            record.update(deleted=time.time(), deleted_reason=reason)
            (dest / "job.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        except BaseException:
            # put back what was moved, so nothing is lost
            for f in sorted(dest.rglob("*"), reverse=True):
                if f.is_file():
                    back = job.dir / f.relative_to(dest)
                    if not back.exists():
                        back.parent.mkdir(exist_ok=True)
                        shutil.move(f, back)
            shutil.rmtree(dest, ignore_errors=True)
            raise
        log.info("job %s (%s) moved to history: %s", s.id, s.name, dest)
        return dest

    def cleanup(self) -> int:
        """Delete jobs untouched for keep_days (never a busy one)."""
        if self.keep_days <= 0:
            return 0
        limit = time.time() - self.keep_days * 86400
        old = [j for j in self.all() if not j.busy and j.state.updated < limit]
        done = 0
        for j in old:
            log.info("deleting old job %s (%s)", j.state.id, j.state.name)
            try:
                self.delete(j, reason=f"not changed for {self.keep_days:g} days")
                done += 1
            except Exception as e:  # e.g. the history disk is full: keep the job, try again later
                log.error("could not move job %s to history, keeping it: %s", j.state.id, e)
        return done

    # ------------------------------------------------------------ analysis cache
    def analysis(self, job: Job):
        from ..pipeline import Analysis

        jid = job.state.id
        with self._cache_lock:
            if jid in self._cache:
                self._cache.move_to_end(jid)
                return self._cache[jid][0]
        if not job.state.scanned:
            raise LookupError("faces have not been detected yet")
        a = Analysis.open(job.video, wd=job.work)  # loads the cached detections only
        with self._cache_lock:
            self._cache[jid] = (a, None)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return a

    def thumbnails(self, job: Job) -> dict[int, np.ndarray]:
        from ..pipeline import card_thumbnails

        a = self.analysis(job)
        jid = job.state.id
        with self._cache_lock:
            entry = self._cache.get(jid)
            if entry and entry[1] is not None:
                return entry[1]
        thumbs = card_thumbnails(a, THUMB)
        with self._cache_lock:
            if jid in self._cache:
                self._cache[jid] = (self._cache[jid][0], thumbs)
        return thumbs

    def put_analysis(self, job: Job, analysis, thumbs=None) -> None:
        with self._cache_lock:
            self._cache[job.state.id] = (analysis, thumbs)
            self._cache.move_to_end(job.state.id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def forget(self, jid: str) -> None:
        with self._cache_lock:
            self._cache.pop(jid, None)
