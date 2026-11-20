"""Web API tests. The face scan and the render are replaced by fakes (no models,
no GPU); everything else is real: uploads, ffprobe, face cards, blur choice,
thumbnails and frames (FFmpeg), the job store and the worker.

The video is the first 20 frames of the test video (_DSG0777.mp4, see
conftest.py), re-encoded at 640x360 so the tests stay fast; without the test
video an FFmpeg test pattern is used instead.

  .venv/Scripts/python -m pytest tests/server -q
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faceblur.identity import Cancelled, Detections
from faceblur.server.app import API, create_app
from faceblur.server.jobs import JobStore
from faceblur.server.worker import Worker
from faceblur.video_io import _ffbin

TOKEN = "test-token"
H = {"Authorization": f"Bearer {TOKEN}"}
FRAMES = 20


WIDTH, HEIGHT = 640, 360


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    from tests.conftest import TEST_VIDEO

    out = tmp_path_factory.mktemp("video") / "clip.mp4"
    if TEST_VIDEO.is_file():
        src = ["-i", str(TEST_VIDEO), "-frames:v", str(FRAMES), "-vf", f"scale={WIDTH}:{HEIGHT}",
               "-c:a", "aac", "-shortest"]
    else:
        src = ["-f", "lavfi", "-i", f"testsrc=size={WIDTH}x{HEIGHT}:rate=10:duration={FRAMES / 10}",
               "-frames:v", str(FRAMES)]
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", *src, "-pix_fmt", "yuv420p", "-c:v", "libx264",
                    "-preset", "veryfast", str(out)], check=True)
    return out


def fake_detections() -> Detections:
    """Two tracked faces on every frame and one short untracked chain: 3 cards."""
    rng = np.random.default_rng(0)
    fr, bx, tr = [], [], []
    for f in range(FRAMES):
        fr += [f, f]
        bx += [[20, 20, 90, 90], [200, 100, 270, 170]]
        tr += [1, 2]
    for f in range(5, 9):
        fr.append(f)
        bx.append([120, 150, 170, 200])
        tr.append(0)
    n = len(fr)
    emb = rng.normal(size=(n, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return Detections(np.array(fr), np.array(bx, np.float32), np.full(n, 0.9, np.float32),
                      np.zeros((n, 5, 2), np.float32), emb, np.array(tr), FRAMES)


class FakeOps:
    def __init__(self):
        self.render_delay = 0.0

    def scan(self, job, progress, cancel):
        from faceblur.pipeline import Analysis, card_thumbnails

        job.work.mkdir(parents=True, exist_ok=True)
        fake_detections().save(job.work / "detections.npz")
        progress("Detecting faces", FRAMES, FRAMES)
        a = Analysis.open(job.video, wd=job.work)
        job.store.put_analysis(job, a, card_thumbnails(a, 160))

    def render(self, job, progress, cancel):
        job.update(status="rendering")
        t0 = time.time()
        while time.time() - t0 < self.render_delay:
            if cancel():
                raise Cancelled()
            time.sleep(0.02)
        shutil.copy(job.video, job.result)
        (job.work / "flagged").mkdir(exist_ok=True)
        (job.work / "flagged" / "000003.jpg").write_bytes(b"\xff\xd8fake")
        (job.work / "safety_report.json").write_text(json.dumps(
            {"video": str(job.result), "threshold": 0.28, "frames": FRAMES,
             "flagged": [{"frame": 3, "time": 0.3, "sim": [0.5], "box": [[1, 2, 3, 4]]}]}))
        return {"frames": FRAMES, "fps": 100.0, "unmasked": 0, "sam": {}, "safety": "done", "flagged": 1}


@pytest.fixture
def env(tmp_path):
    store = JobStore(tmp_path / "jobs")
    ops = FakeOps()
    worker = Worker(store, ops)
    app = create_app(store, worker, TOKEN, max_upload=50 << 20)
    with TestClient(app) as client:
        yield client, store, ops


def wait(client, jid, *statuses, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"{API}/jobs/{jid}", headers=H).json()
        if s["status"] in statuses:
            return s
        time.sleep(0.05)
    raise AssertionError(f"job stayed {s['status']}, wanted {statuses}")


def upload(client, video: Path) -> str:
    r = client.post(f"{API}/jobs", params={"name": "VID_1.mp4"}, content=video.read_bytes(),
                    headers={**H, "Content-Type": "video/mp4"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- tests

def test_auth_required(env):
    client, *_ = env
    assert client.get(f"{API}/jobs").status_code == 401
    assert client.get(f"{API}/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get(f"{API}/jobs", headers=H).json() == []


def test_full_flow(env, sample_video):
    client, store, _ = env
    jid = upload(client, sample_video)
    s = wait(client, jid, "review")
    assert s["video"]["frames"] == FRAMES and s["video"]["width"] == WIDTH
    assert s["cards"] == {"total": 3, "blurred": 0}

    cards = client.get(f"{API}/jobs/{jid}/cards", headers=H).json()
    assert [c["number"] for c in cards] == [1, 2, 3]
    assert not any(c["blur"] for c in cards)  # every face starts as not blurred
    big = sorted(cards, key=lambda c: -c["frames"])
    r = client.get(cards[0]["thumbs"][0], headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    r = client.get(f"{API}/jobs/{jid}/frames/6.jpg", params={"w": 200}, headers=H)
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert client.get(f"{API}/jobs/{jid}/frames/999.jpg", headers=H).status_code == 404

    # nothing to blur yet: render is refused
    r = client.post(f"{API}/jobs/{jid}/render", headers=H)
    assert r.status_code == 409 and "no face is marked to blur" in r.json()["detail"]

    # click, click, click: blur two faces, then change the mind about one
    b0, b1 = str(big[0]["number"]), str(big[1]["number"])
    r = client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {b0: True, b1: True}}, headers=H)
    assert r.status_code == 200 and r.json()["blur"] == sorted([int(b0), int(b1)])
    r = client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {b1: False}}, headers=H)
    assert r.json()["blur"] == [int(b0)]
    assert client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {"99": True}}, headers=H).status_code == 422
    cards = client.get(f"{API}/jobs/{jid}/cards", headers=H).json()
    assert [c["number"] for c in cards if c["blur"]] == [int(b0)]
    assert client.get(f"{API}/jobs/{jid}", headers=H).json()["cards"] == {"total": 3, "blurred": 1}

    # PUT sets exactly the given cards
    assert client.put(f"{API}/jobs/{jid}/blur", json={"blur": [1, 2, 3]}, headers=H).json()["blur"] == [1, 2, 3]
    assert client.put(f"{API}/jobs/{jid}/blur", json={"blur": [int(b0)]}, headers=H).json()["blur"] == [int(b0)]

    proj = client.get(f"{API}/jobs/{jid}/project", headers=H).json()
    assert proj["blur"] == [int(b0)] and proj["settings"]["style"] == "pixelate"

    r = client.patch(f"{API}/jobs/{jid}/settings", json={"style": "blur", "pad": 0.4}, headers=H)
    assert r.json()["style"] == "blur" and r.json()["pad"] == 0.4 and r.json()["strength"] == 1.0
    assert client.patch(f"{API}/jobs/{jid}/settings", json={"style": "bad"}, headers=H).status_code == 422

    # the project file is an ordinary desktop-app project
    from faceblur.project import Project

    p = Project.load(store.get(jid).project_path)
    assert p.settings.style == "blur" and [lab.label for lab in p.labels] == [0]

    assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 202
    s = wait(client, jid, "done")
    assert s["result"]["flagged"] == 1 and s["result"]["size"] == sample_video.stat().st_size

    r = client.get(f"{API}/jobs/{jid}/result", headers=H)
    assert r.status_code == 200 and r.content == sample_video.read_bytes()
    assert "VID_1_masked.mp4" in r.headers["content-disposition"]
    r = client.get(f"{API}/jobs/{jid}/result", headers={**H, "Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == sample_video.read_bytes()[10:20]

    rep = client.get(f"{API}/jobs/{jid}/safety", headers=H).json()
    assert rep["flagged"][0]["image"].endswith("/safety/3.jpg")
    assert client.get(rep["flagged"][0]["image"], headers=H).status_code == 200

    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 204
    assert client.get(f"{API}/jobs/{jid}", headers=H).status_code == 404
    assert not store.root.joinpath(jid).exists()


def test_multipart_upload(env, sample_video):
    client, *_ = env
    with open(sample_video, "rb") as f:
        r = client.post(f"{API}/jobs", files={"file": ("phone.mov", f, "video/quicktime")}, headers=H)
    assert r.status_code == 201, r.text
    s = wait(client, r.json()["id"], "review")
    assert s["name"] == "phone.mov"


def test_not_a_video(env):
    client, store, _ = env
    r = client.post(f"{API}/jobs", params={"name": "x.mp4"}, content=b"not a video" * 100, headers=H)
    assert r.status_code == 415
    assert store.all() == [] and list(store.root.iterdir()) == []


def test_upload_too_large(env):
    client, store, _ = env
    r = client.post(f"{API}/jobs", content=b"0" * (51 << 20), headers=H)
    assert r.status_code == 413
    assert store.all() == []


def test_cancel_render_and_rescan(env, sample_video):
    client, store, ops = env
    jid = upload(client, sample_video)
    wait(client, jid, "review")
    client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {"1": True}}, headers=H)
    ops.render_delay = 10
    assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 202
    wait(client, jid, "rendering")
    assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 409  # busy
    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 409  # running
    assert client.post(f"{API}/jobs/{jid}/cancel", headers=H).status_code == 200
    s = wait(client, jid, "review")
    assert "cancelled" in s["error"]
    assert s["result"] is None

    # detect again: the choice of faces to blur is cleared
    assert client.post(f"{API}/jobs/{jid}/scan", headers=H).status_code == 200
    s = wait(client, jid, "review")
    assert s["cards"] == {"total": 3, "blurred": 0}


def test_restart_marks_running_jobs_failed(tmp_path, sample_video):
    store = JobStore(tmp_path / "jobs")
    job = store.create("a.mp4", ".mp4")
    shutil.copy(sample_video, job.video)
    job.update(status="scanning")
    queued = store.create("b.mp4", ".mp4")
    shutil.copy(sample_video, queued.video)
    queued.update(status="queued_scan")
    uploading = store.create("c.mp4", ".mp4")

    again = JobStore(tmp_path / "jobs")
    assert again.get(job.state.id).state.status == "failed"
    assert again.get(uploading.state.id) is None and not uploading.dir.exists()
    worker = Worker(again, FakeOps())
    assert worker.position(again.get(queued.state.id)) == 1


def test_cleanup_old_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs", keep_days=1)
    old = store.create("a.mp4", ".mp4")
    old.update(status="review")
    old.state.updated = time.time() - 2 * 86400
    old.save()
    new = store.create("b.mp4", ".mp4")
    new.update(status="review")
    assert store.cleanup() == 1
    assert store.get(old.state.id) is None and store.get(new.state.id) is not None


# ---------------------------------------------------------------- phase 3-4

def test_cards_have_sample_frames_and_fresh_thumb_urls(env, sample_video):
    client, *_ = env
    jid = upload(client, sample_video)
    s = wait(client, jid, "review")
    assert s["scan_rev"] == 1
    cards = client.get(f"{API}/jobs/{jid}/cards", headers=H).json()
    for c in cards:
        assert len(c["sample_frames"]) == len(c["thumbs"])
        assert all(c["first_frame"] <= f < FRAMES for f in c["sample_frames"])
        assert all(t.endswith("?r=1") for t in c["thumbs"])
    client.post(f"{API}/jobs/{jid}/scan", headers=H)
    s = wait(client, jid, "review")
    assert s["scan_rev"] == 2
    assert client.get(f"{API}/jobs/{jid}/cards", headers=H).json()[0]["thumbs"][0].endswith("?r=2")


def test_browser_session_cookie(env):
    client, *_ = env
    assert client.post(f"{API}/session", json={"token": "wrong"}).status_code == 401
    r = client.post(f"{API}/session", json={"token": TOKEN})
    assert r.status_code == 204
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and TOKEN not in cookie
    # GET works with the cookie alone
    assert client.get(f"{API}/jobs").status_code == 200
    assert client.get(f"{API}/config").json()["max_upload"] == 50 << 20
    # changes need the CSRF header
    assert client.post(f"{API}/uploads", json={"name": "a.mp4", "size": 10}).status_code == 403
    r = client.post(f"{API}/uploads", json={"name": "a.mp4", "size": 10}, headers={"X-FaceBlur": "1"})
    assert r.status_code == 201
    # a wrong bearer token is refused even with a good cookie
    assert client.get(f"{API}/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.delete(f"{API}/session").status_code == 204
    assert client.get(f"{API}/jobs").status_code == 401


def test_web_app_is_served(env):
    client, *_ = env
    r = client.get("/")
    assert r.status_code == 200 and "<title>FaceBlur</title>" in r.text
    for f in ("app.js", "style.css"):
        assert client.get(f"/{f}").status_code == 200
    assert client.get("/app.js").headers["cache-control"] == "no-cache"  # updates show at once
    assert client.get("/docs").status_code == 200  # API docs still reachable
    assert client.get(f"{API}/config").status_code == 401  # the API is not public


def test_resumable_upload(tmp_path, sample_video):
    data = sample_video.read_bytes()
    third = len(data) // 3
    store = JobStore(tmp_path / "jobs")
    app = create_app(store, Worker(store, FakeOps()), TOKEN, max_upload=50 << 20)
    with TestClient(app) as client:
        r = client.post(f"{API}/uploads", json={"name": "VID_2.mp4", "size": len(data)}, headers=H)
        assert r.status_code == 201
        jid = r.json()["id"]
        r = client.patch(f"{API}/jobs/{jid}/upload", content=data[:third], headers={**H, "Upload-Offset": "0"})
        assert r.json() == {"id": jid, "status": "uploading", "offset": third, "size": len(data)}
        # a chunk sent twice (e.g. the response was lost) is refused with the real offset
        r = client.patch(f"{API}/jobs/{jid}/upload", content=data[:third], headers={**H, "Upload-Offset": "0"})
        assert r.status_code == 409 and r.json()["detail"]["offset"] == third
        s = client.get(f"{API}/jobs/{jid}", headers=H).json()
        assert s["status"] == "uploading" and s["upload"] == {"offset": third, "size": len(data)}

    # the server restarts in the middle of the upload: the upload is kept
    store = JobStore(tmp_path / "jobs")
    app = create_app(store, Worker(store, FakeOps()), TOKEN, max_upload=50 << 20)
    with TestClient(app) as client:
        assert client.get(f"{API}/jobs/{jid}/upload", headers=H).json()["offset"] == third
        r = client.patch(f"{API}/jobs/{jid}/upload", content=data[third:2 * third], headers={**H, "Upload-Offset": str(third)})
        assert r.json()["offset"] == 2 * third
        # more than announced is refused
        r = client.patch(f"{API}/jobs/{jid}/upload", content=data[2 * third:] + b"x",
                         headers={**H, "Upload-Offset": str(2 * third)})
        assert r.status_code == 413
        have = client.get(f"{API}/jobs/{jid}/upload", headers=H).json()["offset"]
        r = client.patch(f"{API}/jobs/{jid}/upload", content=data[have:], headers={**H, "Upload-Offset": str(have)})
        assert r.status_code == 200 and r.json()["status"] in ("queued_scan", "scanning", "review")
        s = wait(client, jid, "review")
        assert s["video"]["size"] == len(data) and s["upload"] is None
        assert store.get(jid).video.read_bytes() == data
        # finished: no more chunks
        r = client.patch(f"{API}/jobs/{jid}/upload", content=b"x", headers={**H, "Upload-Offset": str(len(data))})
        assert r.status_code == 409


def test_resumable_upload_limits(env):
    client, store, _ = env
    r = client.post(f"{API}/uploads", json={"name": "big.mp4", "size": 51 << 20}, headers=H)
    assert r.status_code == 413
    r = client.post(f"{API}/uploads", json={"name": "x.mp4", "size": 100}, headers=H)
    jid = r.json()["id"]
    r = client.patch(f"{API}/jobs/{jid}/upload", content=b"y" * 100, headers={**H, "Upload-Offset": "0"})
    assert r.status_code == 415  # complete, but not a video: the job is removed
    assert store.get(jid) is None


def test_result_inline_for_video_player(env, sample_video):
    client, *_ = env
    jid = upload(client, sample_video)
    wait(client, jid, "review")
    client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {"1": True}}, headers=H)
    client.post(f"{API}/jobs/{jid}/render", headers=H)
    wait(client, jid, "done")
    r = client.get(f"{API}/jobs/{jid}/result", params={"download": "false"}, headers=H)
    assert r.headers["content-disposition"].startswith("inline")
    r = client.get(f"{API}/jobs/{jid}/result", headers=H)
    assert r.headers["content-disposition"].startswith("attachment")


# ---------------------------------------------------------------- history

@pytest.fixture
def env_history(tmp_path):
    store = JobStore(tmp_path / "jobs", history=tmp_path / "history")
    app = create_app(store, Worker(store, FakeOps()), TOKEN, max_upload=50 << 20)
    with TestClient(app) as client:
        yield client, store, tmp_path / "history"


def rendered_job(client, video) -> str:
    jid = upload(client, video)
    wait(client, jid, "review")
    client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {"1": True}}, headers=H)
    client.post(f"{API}/jobs/{jid}/render", headers=H)
    wait(client, jid, "done")
    return jid


def test_deleted_job_moves_to_history(env_history, sample_video):
    client, store, history = env_history
    jid = rendered_job(client, sample_video)
    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 204
    assert not (store.root / jid).exists()
    [kept] = list(history.iterdir())
    assert kept.name.endswith(f"_VID_1_{jid[:8]}")
    files = sorted(str(p.relative_to(kept)).replace("\\", "/") for p in kept.rglob("*") if p.is_file())
    assert files == ["job.json", "project.faceblur.json", "result.mp4", "source.mp4", "work/safety_report.json"]
    assert (kept / "source.mp4").read_bytes() == sample_video.read_bytes()
    record = json.loads((kept / "job.json").read_text(encoding="utf-8"))
    assert record["name"] == "VID_1.mp4" and record["deleted_reason"] == "deleted by the user"
    # the kept project still points at the kept video
    from faceblur.project import Project

    assert Path(Project.load(kept / "project.faceblur.json").video) == (kept / "source.mp4").resolve()


def test_failed_uploads_are_not_kept(env_history):
    client, store, history = env_history
    r = client.post(f"{API}/jobs", params={"name": "x.mp4"}, content=b"not a video" * 100, headers=H)
    assert r.status_code == 415
    jid = client.post(f"{API}/uploads", json={"name": "half.mp4", "size": 1000}, headers=H).json()["id"]
    client.patch(f"{API}/jobs/{jid}/upload", content=b"z" * 400, headers={**H, "Upload-Offset": "0"})
    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 204
    assert not history.exists() or not any(history.iterdir())


def test_cleanup_moves_old_jobs_to_history(env_history, sample_video):
    client, store, history = env_history
    jid = rendered_job(client, sample_video)
    job = store.get(jid)
    job.state.updated = time.time() - 30 * 86400
    job.save()
    assert store.cleanup() == 1
    [kept] = list(history.iterdir())
    assert json.loads((kept / "job.json").read_text(encoding="utf-8"))["deleted_reason"] == "not changed for 7 days"


def test_job_survives_when_history_cannot_be_written(env_history, sample_video):
    client, store, history = env_history
    jid = rendered_job(client, sample_video)
    history.write_text("in the way")  # a file where the history folder should be
    r = client.delete(f"{API}/jobs/{jid}", headers=H)
    assert r.status_code == 500 and "nothing was deleted" in r.json()["detail"]
    job = store.get(jid)
    assert job is not None and job.video.is_file() and job.result.is_file()
    assert client.get(f"{API}/jobs/{jid}", headers=H).json()["status"] == "done"


def test_history_move_failing_halfway_puts_files_back(env_history, sample_video, monkeypatch):
    client, store, history = env_history
    jid = rendered_job(client, sample_video)
    import faceblur.server.jobs as jobs_mod

    real_move = jobs_mod.shutil.move

    def flaky(src, dst):
        if Path(src).name == "result.mp4" and "history" in str(dst):
            raise OSError("disk full")
        return real_move(src, dst)

    monkeypatch.setattr(jobs_mod.shutil, "move", flaky)
    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 500
    job = store.get(jid)
    assert job.video.read_bytes() == sample_video.read_bytes()  # moved first, then put back
    assert job.result.is_file() and (job.dir / "job.json").is_file()
    assert not any(history.iterdir())


# ---------------------------------------------------------------- defaults

def test_sam_and_safety_check_are_off_by_default(env, sample_video):
    client, *_ = env
    jid = upload(client, sample_video)
    wait(client, jid, "review")
    s = client.get(f"{API}/jobs/{jid}/project", headers=H).json()["settings"]
    assert s["use_sam"] is False and s["safety_check"] is False
    r = client.patch(f"{API}/jobs/{jid}/settings", json={"use_sam": True, "safety_check": True}, headers=H)
    assert r.json()["use_sam"] is True and r.json()["safety_check"] is True


def test_output_format_and_quality(env, sample_video):
    client, *_ = env
    jid = upload(client, sample_video)
    wait(client, jid, "review")
    s = client.get(f"{API}/jobs/{jid}/project", headers=H).json()["settings"]
    assert (s["codec"], s["quality"], s["cq"]) == ("h265", "high", None)
    url = f"{API}/jobs/{jid}/settings"
    r = client.patch(url, json={"codec": "h264", "cq": 21}, headers=H).json()
    assert (r["codec"], r["cq"]) == ("h264", 21)
    r = client.patch(url, json={"quality": "small"}, headers=H).json()
    assert (r["quality"], r["cq"]) == ("small", None)  # a level replaces the exact number
    assert client.patch(url, json={"codec": "vp9"}, headers=H).status_code == 422
    assert client.patch(url, json={"quality": "ultra"}, headers=H).status_code == 422


def test_worker_runs_the_safety_check_only_when_it_is_on(tmp_path, sample_video, monkeypatch):
    """The real worker step, with the slow render and check replaced."""
    import faceblur.pipeline as pl
    from faceblur.pipeline import RenderResult
    from faceblur.server.worker import PipelineOps

    calls = []

    def fake_render(project, analysis, **kw):
        Path(project.output).write_bytes(b"video")
        return RenderResult(Path(project.output), FRAMES, 0, {}, 50.0, analysis.wd / "render_log.json")

    def fake_check(masked, refs, wd, **kw):
        calls.append(masked)
        return {"flagged": [{"frame": 1}]}

    monkeypatch.setattr(pl, "render", fake_render)
    monkeypatch.setattr(pl, "safety_check", fake_check)
    store = JobStore(tmp_path / "jobs")
    job = store.create("a.mp4", ".mp4")
    shutil.copy(sample_video, job.video)
    FakeOps().scan(job, lambda *a: None, lambda: False)
    job.update(status="review", scanned=True)
    with job.lock:
        p = job.load_project()
        a = job.analysis()
        a.set_blur(p, [0], True)
        job.save_project(p)
    info = PipelineOps().render(job, lambda *a: None, lambda: False)
    assert info["safety"] == "off" and calls == []
    with job.lock:
        p = job.load_project()
        p.settings.safety_check = True
        job.save_project(p)
    info = PipelineOps().render(job, lambda *a: None, lambda: False)
    assert info["safety"] == "done" and info["flagged"] == 1 and len(calls) == 1
