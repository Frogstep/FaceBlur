"""FastAPI routes (base path /api/v1). Every request needs
`Authorization: Bearer <token>`.

Flow: POST /jobs (upload) -> poll GET /jobs/{id} until status "review" ->
GET /jobs/{id}/cards (+ thumbnails) -> PATCH /jobs/{id}/blur (the cards to blur;
every card starts as not blurred) -> POST render -> poll until "done" ->
GET /jobs/{id}/result.

Browsers (the web app at /) sign in once with POST /api/v1/session and then use
a session cookie; state-changing requests made with the cookie must send the
header `X-FaceBlur: 1` (CSRF protection).

Large uploads can be resumed: POST /uploads {name, size} -> PATCH
/jobs/{id}/upload with `Upload-Offset` and the next chunk as the body, until the
whole file is there (GET /jobs/{id}/upload says how much arrived).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import shutil
import sys
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from .images import draw_cards, jpeg
from .jobs import BUSY, RUNNING, Job, JobStore
from .worker import Worker

log = logging.getLogger("faceblur.server")

API = "/api/v1"
COOKIE = "faceblur_session"
WEB = Path(__file__).resolve().parent / "web"
CHUNK = 16 << 20  # suggested chunk size for resumable uploads


class WebFiles(StaticFiles):
    """The web app's files, revalidated on every load so an updated server is
    picked up at once (they are small, and unchanged files answer 304)."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# ---------------------------------------------------------------- schemas

class Progress(BaseModel):
    stage: str
    done: int
    total: int


class CardCounts(BaseModel):
    total: int
    blurred: int


class ResultInfo(BaseModel):
    url: str
    size: int
    frames: int | None = None
    fps: float | None = None
    unmasked: int | None = None
    safety: str | None = None  # "done", "off" (turned off in the settings) or "skipped" (cancelled)
    flagged: int | None = None  # frames where a masked face may still be recognizable


class UploadProgress(BaseModel):
    offset: int  # bytes received
    size: int  # bytes expected


class JobOut(BaseModel):
    id: str
    name: str
    status: str
    created: float
    updated: float
    progress: Progress
    queue_position: int | None = None
    error: str = ""
    video: dict = {}
    upload: UploadProgress | None = None  # while a resumable upload is incomplete
    scan_rev: int = 0  # changes after every finished face scan (reload the cards)
    cards: CardCounts | None = None
    result: ResultInfo | None = None


class CardOut(BaseModel):
    number: int
    start: float  # seconds
    end: float
    frames: int  # number of frames the face is detected on
    first_frame: int
    track: int  # tracker id, 0 = chain of untracked detections
    blur: bool  # marked to blur (cards start as not blurred)
    thumbs: list[str]
    sample_frames: list[int]  # the frame of each thumbnail


class Settings(BaseModel):
    style: Literal["pixelate", "blur", "fill"] = "pixelate"
    strength: float = Field(1.0, ge=0.2, le=5.0)
    pad: float = Field(0.3, ge=0.0, le=1.0)
    use_sam: bool = False  # SAM 2 face outlines (slow); off: ellipses
    sam_model: Literal["auto", "tiny", "small", "base_plus", "large"] = "auto"  # SAM 2.1 size (tiny = fastest)
    codec: Literal["h265", "h264"] = "h265"  # h265: smaller file; h264: plays in every browser
    quality: Literal["best", "high", "medium", "small"] = "high"
    cq: int | None = Field(None, ge=0, le=51)  # exact NVENC CQ / x264 CRF instead of the level (lower = better)
    safety_check: bool = False  # after rendering, look for blurred faces that are still recognizable


class SettingsPatch(BaseModel):
    style: Literal["pixelate", "blur", "fill"] | None = None
    strength: float | None = Field(None, ge=0.2, le=5.0)
    pad: float | None = Field(None, ge=0.0, le=1.0)
    use_sam: bool | None = None
    sam_model: Literal["auto", "tiny", "small", "base_plus", "large"] | None = None
    codec: Literal["h265", "h264"] | None = None
    quality: Literal["best", "high", "medium", "small"] | None = None  # clears cq unless cq is also given
    cq: int | None = Field(None, ge=0, le=51)
    safety_check: bool | None = None


class ProjectOut(BaseModel):
    blur: list[int]  # numbers of the cards to blur
    settings: Settings


class BlurIn(BaseModel):
    cards: dict[str, bool]  # card number -> blur or not


class BlurSet(BaseModel):
    blur: list[int]  # numbers of the cards to blur; every other card is not blurred


class UploadIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    size: int = Field(..., gt=0)


class ChunkOut(BaseModel):
    id: str
    status: str  # "uploading" until the last chunk, then the job's next status
    offset: int
    size: int


class SessionIn(BaseModel):
    token: str


class ConfigOut(BaseModel):
    version: str
    max_upload: int
    chunk_size: int


# ---------------------------------------------------------------- app

def create_app(store: JobStore, worker: Worker, token: str, max_upload: int = 8 << 30,
               start_worker: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if start_worker and not worker.is_alive():
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="FaceBlur", version="1", lifespan=lifespan,
                  description="Upload a video, choose the faces to blur, render, download.")
    bearer = HTTPBearer(auto_error=False)
    # the cookie holds a value derived from the token, not the token itself; a new
    # token (delete server_token.txt) signs every browser out
    session = hmac.new(token.encode(), b"faceblur-session-v1", hashlib.sha256).hexdigest()

    def auth(request: Request, cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if cred is not None:
            if secrets.compare_digest(cred.credentials.encode(), token.encode()):
                return
        else:
            c = request.cookies.get(COOKIE, "")
            if c and secrets.compare_digest(c.encode(), session.encode()):
                if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("x-faceblur") != "1":
                    raise HTTPException(403, "requests signed in with the cookie need the header X-FaceBlur: 1")
                return
        raise HTTPException(401, "missing or wrong token", headers={"WWW-Authenticate": "Bearer"})

    r = APIRouter(prefix=API, dependencies=[Depends(auth)])
    pub = APIRouter(prefix=API)
    upload_locks: dict[str, asyncio.Lock] = {}

    grabbers: OrderedDict[str, tuple] = OrderedDict()  # job id -> (FrameGrabber, lock)
    grab_lock = threading.Lock()

    def grabber(job: Job):
        from ..video_io import FrameGrabber

        with grab_lock:
            if job.state.id not in grabbers:
                grabbers[job.state.id] = (FrameGrabber(job.analysis().info, cache_size=4), threading.Lock())
                while len(grabbers) > 4:
                    grabbers.popitem(last=False)
            grabbers.move_to_end(job.state.id)
            return grabbers[job.state.id]

    # ------------------------------------------------------------ helpers
    def get_job(job_id: str) -> Job:
        job = store.get(job_id) if job_id.isalnum() else None
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    def scanned(job: Job):
        if not job.state.scanned or job.state.status in ("queued_scan", "scanning"):
            raise HTTPException(409, f"faces are not detected yet (status: {job.state.status})")
        return job.analysis()

    def job_out(job: Job, detail: bool = True) -> JobOut:
        s = job.state
        out = JobOut(id=s.id, name=s.name, status=s.status, created=s.created, updated=s.updated,
                     progress=Progress(stage=s.stage, done=s.done, total=s.total),
                     queue_position=worker.position(job) if s.status in BUSY else None,
                     error=s.error, video=s.video, scan_rev=s.scan_rev)
        if s.status == "uploading" and s.upload_size:
            out.upload = UploadProgress(offset=job.uploaded(), size=s.upload_size)
        if detail and s.scanned and s.status not in ("queued_scan", "scanning"):
            a = job.analysis()
            with job.lock:
                project = job.load_project()
            out.cards = CardCounts(total=len(a.cards), blurred=sum(a.blurred(project)))
        if s.rendered and job.result.exists():
            out.result = ResultInfo(url=f"{API}/jobs/{s.id}/result", size=job.result.stat().st_size,
                                    **{k: s.render.get(k) for k in ("frames", "fps", "unmasked", "safety", "flagged")})
        return out

    def check_space(size: int) -> None:
        if size > max_upload:
            raise HTTPException(413, f"video is larger than {max_upload / 2**30:.1f} GB")
        if shutil.disk_usage(store.root).free < size + (2 << 30):
            raise HTTPException(507, "not enough free disk space on the server")

    async def finish_upload(job: Job) -> None:
        """The whole file is there: check it is a video and queue the face scan."""
        from ..video_io import probe

        size = job.uploaded()
        try:
            job.part.replace(job.video)
            try:
                info = await run_in_threadpool(probe, job.video)
            except Exception as e:
                raise HTTPException(415, f"not a video FFmpeg can read: {e}")
            if info.num_frames <= 0:
                raise HTTPException(415, "the video has no frames")
        except BaseException:
            store.delete(job, archive=False)  # never a complete video
            raise
        job.update(video={"width": info.width, "height": info.height, "fps": round(info.fps_float, 3),
                          "frames": info.num_frames, "duration": round(info.duration, 3),
                          "has_audio": info.has_audio, "vfr": info.vfr, "size": size},
                   stage="", done=0, total=0)
        with job.lock:
            job.save_project(job.load_project())
        worker.submit(job, "scan")
        log.info("uploaded %s (%s, %.1f MB)", job.state.id, job.state.name, size / 1e6)

    # ------------------------------------------------------------ session (no auth needed)
    @pub.post("/session", status_code=204, summary="Browser sign-in: exchange the token for a session cookie")
    def sign_in(request: Request, body: SessionIn):
        if not secrets.compare_digest(body.token.strip().encode(), token.encode()):
            raise HTTPException(401, "wrong token")
        resp = Response(status_code=204)
        resp.set_cookie(COOKIE, session, max_age=180 * 86400, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", path="/")
        return resp

    @pub.delete("/session", status_code=204, summary="Browser sign-out")
    def sign_out():
        resp = Response(status_code=204)
        resp.delete_cookie(COOKIE, path="/")
        return resp

    @r.get("/config", response_model=ConfigOut, summary="Limits the client needs")
    def config():
        return ConfigOut(version=app.version, max_upload=max_upload, chunk_size=CHUNK)

    # ------------------------------------------------------------ resumable uploads
    @r.post("/uploads", status_code=201, response_model=ChunkOut,
            summary="Start a resumable upload; then send the file with PATCH /jobs/{id}/upload")
    def start_upload(body: UploadIn):
        check_space(body.size)
        job = store.create(body.name, Path(body.name).suffix, upload_size=body.size)
        job.update(stage="Uploading", done=0, total=body.size)
        return ChunkOut(id=job.state.id, status=job.state.status, offset=0, size=body.size)

    @r.get("/jobs/{job_id}/upload", response_model=ChunkOut, summary="How much of a resumable upload arrived")
    def upload_status(job_id: str):
        job = get_job(job_id)
        size = job.state.upload_size
        done = job.uploaded() if job.state.status == "uploading" else size
        return ChunkOut(id=job_id, status=job.state.status, offset=done, size=size)

    @r.patch("/jobs/{job_id}/upload", response_model=ChunkOut,
             summary="Append the next chunk (header Upload-Offset = bytes already received)")
    async def upload_chunk(job_id: str, request: Request,
                           upload_offset: int = Header(..., alias="Upload-Offset")):
        job = get_job(job_id)
        size = job.state.upload_size
        if job.state.status != "uploading" or not size:
            raise HTTPException(409, f"not a resumable upload in progress (status: {job.state.status})")
        lock = upload_locks.setdefault(job_id, asyncio.Lock())
        if lock.locked():
            raise HTTPException(409, {"message": "another chunk is being received", "offset": job.uploaded()})
        async with lock:
            have = job.uploaded()
            if upload_offset != have:
                raise HTTPException(409, {"message": f"offset {upload_offset}, but the server has {have} bytes",
                                          "offset": have})
            with open(job.part, "ab") as out:
                try:
                    async for chunk in request.stream():
                        if have + len(chunk) > size:
                            raise HTTPException(413, f"more data than the announced {size} bytes")
                        out.write(chunk)
                        have += len(chunk)
                except ClientDisconnect:  # keep what arrived; the client resumes from GET /upload
                    log.info("upload %s interrupted at %d / %d bytes", job_id, have, size)
                    raise
            job.update(done=have)
            if have == size:
                upload_locks.pop(job_id, None)
                await finish_upload(job)
        return ChunkOut(id=job_id, status=job.state.status, offset=have, size=size)

    # ------------------------------------------------------------ jobs
    @r.post("/jobs", status_code=201, response_model=JobOut,
            summary="Upload a video (raw body, or multipart field 'file'); the face scan starts automatically")
    async def upload(request: Request, name: str | None = Query(None, description="original file name")):
        length = request.headers.get("content-length")
        if length and length.isdigit():
            check_space(int(length))
        ctype = request.headers.get("content-type", "")
        multipart = ctype.startswith("multipart/form-data")
        form = None
        if multipart:
            form = await request.form(max_files=1)
            f = form.get("file")
            if f is None or not hasattr(f, "read"):
                raise HTTPException(422, "multipart upload needs a 'file' field")
            name = name or f.filename
        job = store.create(name or "video.mp4", Path(name or "video.mp4").suffix)
        part = job.part
        size = 0
        try:
            if multipart:
                def copy():
                    with open(part, "wb") as out:
                        shutil.copyfileobj(f.file, out, 1 << 20)
                await run_in_threadpool(copy)
                size = part.stat().st_size
            else:
                with open(part, "wb") as out:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > max_upload:
                            raise HTTPException(413, f"video is larger than {max_upload / 2**30:.1f} GB")
                        out.write(chunk)
            if size == 0:
                raise HTTPException(422, "empty upload")
        except BaseException as e:
            store.delete(job, archive=False)  # upload failed
            if isinstance(e, ClientDisconnect):
                log.info("upload %s aborted by the client", job.state.id)
            raise
        finally:
            if form is not None:
                await form.close()
        await finish_upload(job)
        return job_out(job, detail=False)

    @r.get("/jobs", response_model=list[JobOut], summary="All jobs, newest first")
    def list_jobs():
        return [job_out(j, detail=False) for j in store.all()]

    @r.get("/jobs/{job_id}", response_model=JobOut, summary="Status and progress (poll this)")
    def get_status(job_id: str):
        return job_out(get_job(job_id))

    @r.post("/jobs/{job_id}/cancel", response_model=JobOut, summary="Cancel the queued or running step")
    def cancel(job_id: str):
        job = get_job(job_id)
        if not worker.cancel(job):
            raise HTTPException(409, f"nothing to cancel (status: {job.state.status})")
        return job_out(job, detail=False)

    @r.post("/jobs/{job_id}/scan", response_model=JobOut,
            summary="Detect faces again (clears the faces chosen to blur and the corrections)")
    def rescan(job_id: str):
        job = get_job(job_id)
        if job.busy:
            raise HTTPException(409, f"job is busy (status: {job.state.status})")
        from ..pipeline import clear_scan_cache

        clear_scan_cache(job.work)
        with job.lock:
            project = job.load_project()
            project.labels.clear()
            project.exclusions.clear()
            job.save_project(project)
        job.forget_analysis()
        with grab_lock:
            grabbers.pop(job.state.id, None)
        job.update(scanned=False)
        worker.submit(job, "scan")
        return job_out(job, detail=False)

    @r.delete("/jobs/{job_id}", status_code=204,
              summary="Delete the job; its source and result move to the server's history folder")
    def delete(job_id: str):
        job = get_job(job_id)
        if job.state.status in RUNNING:
            raise HTTPException(409, "the job is running: cancel it first")
        worker.cancel(job)  # if queued
        with grab_lock:
            grabbers.pop(job.state.id, None)
        try:
            store.delete(job)
        except OSError as e:
            log.error("could not move job %s to history: %s", job_id, e)
            raise HTTPException(500, f"could not move the files to the history folder, so nothing was deleted: {e}")
        return Response(status_code=204)

    # ------------------------------------------------------------ face review
    @r.get("/jobs/{job_id}/cards", response_model=list[CardOut],
           summary="Face cards in display order (similar faces next to each other)")
    def cards(job_id: str):
        job = get_job(job_id)
        a = scanned(job)
        with job.lock:
            blurred = a.blurred(job.load_project())
        fps = a.info.fps_float
        rev = job.state.scan_rev
        out = []
        for c, blur in zip(a.cards, blurred):
            fr = a.dets.frame[c.det_idx]
            out.append(CardOut(
                number=c.number, start=round(float(fr.min()) / fps, 3), end=round(float(fr.max()) / fps, 3),
                frames=len(c.det_idx), first_frame=int(fr.min()), track=c.track, blur=blur,
                thumbs=[f"{API}/jobs/{job_id}/cards/{c.number}/thumbs/{k}.jpg?r={rev}"
                        for k in range(len(c.samples))],
                sample_frames=[int(a.dets.frame[i]) for i in c.samples]))
        return out

    @r.get("/jobs/{job_id}/cards/{number}/thumbs/{k}.jpg", response_class=Response,
           responses={200: {"content": {"image/jpeg": {}}}}, summary="One sample thumbnail of a card (160 px)")
    def thumb(job_id: str, number: int, k: int):
        job = get_job(job_id)
        a = scanned(job)
        if not 1 <= number <= len(a.cards):
            raise HTTPException(404, "no such card")
        card = a.cards[number - 1]
        if not 0 <= k < len(card.samples):
            raise HTTPException(404, "no such thumbnail")
        img = job.thumbnails()[card.samples[k]]
        return Response(jpeg(img), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @r.get("/jobs/{job_id}/frames/{frame}.jpg", response_class=Response,
           responses={200: {"content": {"image/jpeg": {}}}},
           summary="A video frame with the face cards boxed and numbered")
    def frame(job_id: str, frame: int, w: int = Query(720, ge=64, le=4096), boxes: bool = True):
        job = get_job(job_id)
        a = scanned(job)
        if not 0 <= frame < a.info.num_frames:
            raise HTTPException(404, "frame out of range")
        g, lock = grabber(job)
        with lock:
            img = g.get(frame)
        if img is None:
            raise HTTPException(500, "could not decode the frame")
        if boxes:
            with job.lock:
                blurred = a.blurred(job.load_project())
            img = draw_cards(img, a, blurred, frame, w)
        else:
            from .images import resize_to_width

            img = resize_to_width(img, w)[0]
        return Response(jpeg(img), media_type="image/jpeg")

    def blur_numbers(a, project) -> list[int]:
        return [c.number for c, b in zip(a.cards, a.blurred(project)) if b]

    def card_positions(a, numbers) -> list[int]:
        bad = [n for n in numbers if not (str(n).isdigit() and 1 <= int(n) <= len(a.cards))]
        if bad:
            raise HTTPException(422, f"no card #{bad[0]} (this video has {len(a.cards)} cards)")
        return [int(n) - 1 for n in numbers]

    @r.get("/jobs/{job_id}/project", response_model=ProjectOut, summary="The cards to blur and the settings")
    def get_project(job_id: str):
        job = get_job(job_id)
        a = scanned(job)
        with job.lock:
            project = job.load_project()
        return ProjectOut(blur=blur_numbers(a, project), settings=Settings(**vars(project.settings)))

    @r.patch("/jobs/{job_id}/blur", response_model=BlurSet,
             summary="Blur cards or not: {'cards': {'3': true, '5': false}}; returns every card to blur")
    def patch_blur(job_id: str, body: BlurIn):
        job = get_job(job_id)
        a = scanned(job)
        with job.lock:
            project = job.load_project()
            changes = dict(zip(card_positions(a, list(body.cards)), body.cards.values()))
            for blur in (True, False):
                a.set_blur(project, [k for k, b in changes.items() if b is blur], blur)
            job.save_project(project)
            return BlurSet(blur=blur_numbers(a, project))

    @r.put("/jobs/{job_id}/blur", response_model=BlurSet,
           summary="Set exactly which cards to blur: {'blur': [1, 4]} (every other card is not blurred)")
    def put_blur(job_id: str, body: BlurSet):
        job = get_job(job_id)
        a = scanned(job)
        with job.lock:
            project = job.load_project()
            want = set(card_positions(a, body.blur))
            a.set_blur(project, [k for k in range(len(a.cards)) if k not in want], False)
            a.set_blur(project, sorted(want), True)
            job.save_project(project)
            return BlurSet(blur=blur_numbers(a, project))

    @r.patch("/jobs/{job_id}/settings", response_model=Settings, summary="Mask style and quality")
    def patch_settings(job_id: str, body: SettingsPatch):
        job = get_job(job_id)
        with job.lock:
            project = job.load_project()
            for k, v in body.model_dump(exclude_none=True).items():
                setattr(project.settings, k, v)
            if body.quality is not None and body.cq is None:
                project.settings.cq = None  # a level replaces an exact number
            job.save_project(project)
            return Settings(**vars(project.settings))

    # ------------------------------------------------------------ render and result
    @r.post("/jobs/{job_id}/render", response_model=JobOut, status_code=202,
            summary="Render (and run the safety check if it is on); at least one card must be marked to blur")
    def start_render(job_id: str):
        job = get_job(job_id)
        if job.busy:
            raise HTTPException(409, f"job is busy (status: {job.state.status})")
        a = scanned(job)
        with job.lock:
            project = job.load_project()
        if not a.plans(project)[1]:
            raise HTTPException(409, "no face is marked to blur")
        worker.submit(job, "render")
        return job_out(job, detail=False)

    @r.get("/jobs/{job_id}/result", response_class=FileResponse,
           responses={200: {"content": {"video/mp4": {}}}},
           summary="Download the rendered video (supports Range requests)")
    def result(job_id: str, download: bool = Query(True, description="false: inline, for a <video> player")):
        job = get_job(job_id)
        if not (job.state.rendered and job.result.exists()):
            raise HTTPException(404, "nothing rendered yet")
        stem = Path(job.state.name).stem or "video"
        return FileResponse(job.result, media_type="video/mp4", filename=f"{stem}_masked.mp4",
                            content_disposition_type="attachment" if download else "inline",
                            headers={"Cache-Control": "private, no-cache"})

    @r.get("/jobs/{job_id}/safety", summary="Safety report of the last render")
    def safety(job_id: str):
        import json

        job = get_job(job_id)
        p = job.work / "safety_report.json"
        if not p.exists():
            raise HTTPException(404, "no safety report (not rendered yet, or the check is off)")
        rep = json.loads(p.read_text(encoding="utf-8"))
        flagged = [dict(x, image=f"{API}/jobs/{job_id}/safety/{x['frame']}.jpg") for x in rep["flagged"]]
        return {"frames": rep["frames"], "threshold": rep["threshold"], "flagged": flagged}

    @r.get("/jobs/{job_id}/safety/{frame}.jpg", response_class=FileResponse,
           responses={200: {"content": {"image/jpeg": {}}}}, summary="A flagged output frame")
    def safety_frame(job_id: str, frame: int):
        job = get_job(job_id)
        p = job.work / "flagged" / f"{frame:06d}.jpg"
        if frame < 0 or not p.exists():
            raise HTTPException(404, "frame not flagged")
        return FileResponse(p, media_type="image/jpeg")

    app.include_router(pub)
    app.include_router(r)

    @app.get("/health", include_in_schema=False)
    def health():
        return {"ok": True}

    if WEB.is_dir():  # the web app; mounted last, so every route above wins
        app.mount("/", WebFiles(directory=WEB, html=True), name="web")

    return app


# ---------------------------------------------------------------- serve

def load_token(path: Path) -> str:
    if path.exists():
        t = path.read_text(encoding="utf-8").strip()
        if t:
            return t
    t = secrets.token_urlsafe(32)
    path.write_text(t + "\n", encoding="utf-8")
    return t


def _lan_addresses() -> list[str]:
    import socket

    out = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            out.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in out if not a.startswith("127."))


def serve(host: str, port: int, jobs_dir: Path, token_file: Path, keep_days: float, max_upload_gb: float,
          history_dir: Path | None) -> int:
    import uvicorn

    from .. import device

    # names and file names can be Cyrillic etc.; when the console output goes to a
    # file or pipe on Windows its encoding cannot show them, which must not break logging
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # browsers drop connections all the time (video seeking, closed tabs); on Windows
    # asyncio logs each one as an error with a traceback
    logging.getLogger("asyncio").addFilter(
        lambda rec: not (rec.exc_info and isinstance(rec.exc_info[1], ConnectionResetError)))
    store = JobStore(jobs_dir, keep_days=keep_days, history=history_dir)
    removed = store.cleanup()
    worker = Worker(store)
    token = load_token(token_file)
    app = create_app(store, worker, token, max_upload=int(max_upload_gb * 2**30))

    def cleanup_loop():
        while True:
            threading.Event().wait(6 * 3600)
            store.cleanup()

    threading.Thread(target=cleanup_loop, name="faceblur-cleanup", daemon=True).start()

    print(device.summary())
    print(f"jobs: {store.root} ({len(store.all())} kept, {removed} old removed)")
    print(f"deleted jobs' source and result: {store.history}" if store.history
          else "deleted jobs are not kept (--no-history)")
    addrs = ["127.0.0.1"] + (_lan_addresses() if host in ("0.0.0.0", "::") else [])
    for a in addrs if host in ("0.0.0.0", "::") else [host]:
        print(f"  http://{a}:{port}{API}   (API docs: http://{a}:{port}/docs)")
    print(f"token ({token_file}):\n  {token}")
    print("Web app: open one of these links once in the browser to sign in:")
    for a in addrs if host in ("0.0.0.0", "::") else [host]:
        print(f"  http://{a}:{port}/#token={token}")
    print("Keep this on your home network or a VPN such as Tailscale; do not expose it to the internet "
          "without HTTPS.")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
