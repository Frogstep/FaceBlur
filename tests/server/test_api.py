"""Web API tests. The face scan and the render are replaced by fakes (no models,
no GPU); everything else is real: uploads, ffprobe, face cards, labels,
thumbnails and frames (FFmpeg), the job store and the worker.

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


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run([_ffbin("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=320x240:rate=10:duration={FRAMES / 10}", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", str(out)], check=True)
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
    assert s["video"]["frames"] == FRAMES and s["video"]["width"] == 320
    assert s["cards"] == {"total": 3, "unlabeled": 3}

    cards = client.get(f"{API}/jobs/{jid}/cards", headers=H).json()
    assert [c["number"] for c in cards] == [1, 2, 3]
    assert all(c["label"] is None for c in cards)
    big = sorted(cards, key=lambda c: -c["frames"])
    r = client.get(cards[0]["thumbs"][0], headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    r = client.get(f"{API}/jobs/{jid}/frames/6.jpg", params={"w": 200}, headers=H)
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert client.get(f"{API}/jobs/{jid}/frames/999.jpg", headers=H).status_code == 404

    # render is refused while cards are unlabeled
    r = client.post(f"{API}/jobs/{jid}/render", headers=H)
    assert r.status_code == 409 and r.json()["detail"]["unlabeled"] == [1, 2, 3]

    # a person must exist before a card is labeled with them
    r = client.put(f"{API}/jobs/{jid}/labels", json={"labels": {str(big[0]["number"]): 0}}, headers=H)
    assert r.status_code == 422
    assert client.put(f"{API}/jobs/{jid}/people", json=[{"name": "Anna"}], headers=H).status_code == 200
    r = client.put(f"{API}/jobs/{jid}/labels", json={"labels": {str(big[0]["number"]): 0}}, headers=H)
    assert r.json()["unlabeled"] == sorted(c["number"] for c in big[1:])
    r = client.put(f"{API}/jobs/{jid}/labels", json={"labels": {"99": 0}}, headers=H)
    assert r.status_code == 422
    r = client.put(f"{API}/jobs/{jid}/labels", json={"rest": "other"}, headers=H)
    assert r.json()["unlabeled"] == []

    # removing a person who still has cards is refused
    assert client.put(f"{API}/jobs/{jid}/people", json=[], headers=H).status_code == 409

    proj = client.get(f"{API}/jobs/{jid}/project", headers=H).json()
    assert proj["people"] == [{"name": "Anna"}]
    assert proj["labels"][str(big[0]["number"])] == 0
    assert sorted(v for v in proj["labels"].values() if v != 0) == ["other", "other"]

    r = client.patch(f"{API}/jobs/{jid}/settings", json={"style": "blur", "pad": 0.4}, headers=H)
    assert r.json()["style"] == "blur" and r.json()["pad"] == 0.4 and r.json()["strength"] == 1.0
    assert client.patch(f"{API}/jobs/{jid}/settings", json={"style": "bad"}, headers=H).status_code == 422

    # the project file is an ordinary desktop-app project
    from faceblur.project import Project

    p = Project.load(store.get(jid).project_path)
    assert p.settings.style == "blur" and len(p.labels) == 3

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
    client.put(f"{API}/jobs/{jid}/people", json=[{"name": "A"}], headers=H)
    client.put(f"{API}/jobs/{jid}/labels", json={"labels": {"1": 0}, "rest": "not_face"}, headers=H)
    ops.render_delay = 10
    assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 202
    wait(client, jid, "rendering")
    assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 409  # busy
    assert client.delete(f"{API}/jobs/{jid}", headers=H).status_code == 409  # running
    assert client.post(f"{API}/jobs/{jid}/cancel", headers=H).status_code == 200
    s = wait(client, jid, "review")
    assert "cancelled" in s["error"]
    assert s["result"] is None

    # detect again: labels are cleared
    assert client.post(f"{API}/jobs/{jid}/scan", headers=H).status_code == 200
    s = wait(client, jid, "review")
    assert s["cards"]["unlabeled"] == 3


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
