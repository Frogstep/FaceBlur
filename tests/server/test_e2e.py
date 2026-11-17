"""End-to-end through the API with the real pipeline (face models, SAM 2, NVENC)
on the whole test video. GPU only; takes a few minutes:

  .venv/Scripts/python -m pytest -m e2e tests/server -s
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from faceblur.server.app import API, create_app
from faceblur.server.jobs import JobStore
from faceblur.server.worker import Worker
from faceblur.video_io import probe

pytestmark = pytest.mark.e2e

TOKEN = "e2e-token"
H = {"Authorization": f"Bearer {TOKEN}"}


def wait(client, jid, want, timeout):
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        s = client.get(f"{API}/jobs/{jid}", headers=H).json()
        if (s["status"], s["progress"]["stage"]) != last:
            last = (s["status"], s["progress"]["stage"])
            print(f"  {time.time() - t0:6.1f}s {s['status']:13} {s['progress']['stage']}")
        if s["status"] == want:
            return s
        assert s["status"] not in ("failed", "cancelled"), s["error"]
        time.sleep(1)
    raise AssertionError(f"still {s['status']} after {timeout} s")


def test_whole_video_on_gpu(tmp_path, test_video):
    from faceblur import device

    if not (device.faces_on_gpu() and device.sam_device() == "cuda"):
        pytest.skip("needs a working NVIDIA GPU (on the CPU this takes far too long)")
    src = probe(test_video)
    store = JobStore(tmp_path / "jobs")
    app = create_app(store, Worker(store), TOKEN, max_upload=1 << 30)
    with TestClient(app) as client:
        r = client.post(f"{API}/jobs", params={"name": test_video.name}, content=test_video.read_bytes(),
                        headers=H)
        assert r.status_code == 201, r.text
        jid = r.json()["id"]

        s = wait(client, jid, "review", timeout=900)
        assert s["video"]["frames"] == src.num_frames  # from the container, like probe()
        assert s["cards"]["total"] > 0, "no faces found in the test video"
        cards = client.get(f"{API}/jobs/{jid}/cards", headers=H).json()
        print(f"  {len(cards)} cards: " + ", ".join(f"#{c['number']} ({c['frames']} f)" for c in cards))
        assert client.get(cards[0]["thumbs"][0], headers=H).status_code == 200
        assert client.get(f"{API}/jobs/{jid}/frames/{cards[0]['first_frame']}.jpg", headers=H).status_code == 200

        # blur the face seen on the most frames; every other face stays visible
        biggest = max(cards, key=lambda c: c["frames"])
        r = client.patch(f"{API}/jobs/{jid}/blur", json={"cards": {str(biggest["number"]): True}}, headers=H)
        assert r.json()["blur"] == [biggest["number"]]
        # both are off by default; this test covers the full pipeline
        r = client.patch(f"{API}/jobs/{jid}/settings", json={"use_sam": True, "safety_check": True}, headers=H)
        assert r.json()["use_sam"] and r.json()["safety_check"]
        assert client.post(f"{API}/jobs/{jid}/render", headers=H).status_code == 202

        s = wait(client, jid, "done", timeout=1800)
        res = s["result"]
        print(f"  result: {res}")
        # the container's frame count can be one more than the decodable frames
        # (_DSG0777.mp4 says 498, FFmpeg decodes 497), so allow that much
        assert abs(res["frames"] - src.num_frames) <= 1 and res["safety"] == "done"

        out = tmp_path / "result.mp4"
        r = client.get(f"{API}/jobs/{jid}/result", headers=H)
        assert r.status_code == 200 and len(r.content) == res["size"]
        out.write_bytes(r.content)
        info = probe(out)
        assert (info.width, info.height) == (src.width, src.height)
        assert info.num_frames == res["frames"]
        assert info.has_audio == src.has_audio

        rep = client.get(f"{API}/jobs/{jid}/safety", headers=H).json()
        assert rep["frames"] == res["frames"] and len(rep["flagged"]) == res["flagged"]
