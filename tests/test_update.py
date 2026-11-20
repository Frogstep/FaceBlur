"""install/update.py end to end: release zips from make_dist, a fake install folder, a
file:// URL in place of GitHub."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "install"))
import make_dist  # noqa: E402
import update  # noqa: E402


def test_parse_version():
    assert update.parse_version("2026.11.20.2") > update.parse_version("2026.11.20")
    assert update.parse_version("2026.12.1") > update.parse_version("2026.11.30")
    assert update.parse_version("dev") == update.parse_version("") == ()


def _release(tmp: Path, version: str) -> Path:
    """Build a release zip for version into tmp/releases/<version>/ with its version.json."""
    out = tmp / "releases" / version
    dest = make_dist.build(version, out)
    (out / "version.json").write_text(json.dumps({
        "version": version, "zip": dest.name, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}))
    return dest


def _install(tmp: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp / "pc")
    app = tmp / "pc" / "FaceBlur"
    (app / ".venv").mkdir()
    (app / "models").mkdir()
    (app / "models" / "keep.onnx").write_text("weights")
    (app / "work").mkdir()
    (app / "server_token.txt").write_text("secret")
    return app


def _run(app: Path, *args: str, url: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if url:
        env["FACEBLUR_UPDATE_URL"] = url.as_uri()
    return subprocess.run([sys.executable, str(app / "install" / "update.py"), *args],
                          capture_output=True, text=True, env=env, cwd=app)


@pytest.fixture
def releases(tmp_path):
    old = _release(tmp_path, "2026.11.19")
    new = _release(tmp_path, "2026.11.20")
    return tmp_path, old, new


def test_update_rollback_and_offline(releases):
    tmp, old, new = releases
    app = _install(tmp, old)
    url = new.with_name("version.json")
    assert update.installed_version(app) == "2026.11.19"
    (app / "faceblur" / "obsolete.py").write_text("")  # a module the new version dropped

    r = _run(app, "--check", url=url)
    assert "update available" in r.stdout, r.stdout + r.stderr

    r = _run(app, url=url)
    assert r.returncode == 0, r.stdout + r.stderr
    assert update.installed_version(app) == "2026.11.20"
    assert not (app / "faceblur" / "obsolete.py").exists()
    assert (app / "models" / "keep.onnx").read_text() == "weights"
    assert (app / "server_token.txt").read_text() == "secret"
    assert update.installed_version(app / "update" / "backup") == "2026.11.19"
    assert not list((app / "update").glob("*.zip"))

    r = _run(app, url=url)
    assert "is the newest version" in r.stdout

    r = _run(app, "--rollback")
    assert r.returncode == 0, r.stdout + r.stderr
    assert update.installed_version(app) == "2026.11.19"
    assert (app / "faceblur" / "obsolete.py").exists()

    r = _run(app, str(new))  # offline, from the zip file
    assert r.returncode == 0, r.stdout + r.stderr
    assert update.installed_version(app) == "2026.11.20"

    r = _run(app, str(old))  # older zip: refused without --force
    assert "Nothing to do" in r.stdout and update.installed_version(app) == "2026.11.20"


def test_corrupt_download_changes_nothing(releases):
    tmp, old, new = releases
    app = _install(tmp, old)
    url = new.with_name("version.json")
    info = json.loads(url.read_text())
    url.write_text(json.dumps(dict(info, sha256="0" * 64)))
    r = _run(app, url=url)
    assert r.returncode == 1 and "corrupted" in r.stdout
    assert update.installed_version(app) == "2026.11.19"


def test_refuses_git_checkout_and_missing_install(releases):
    tmp, old, new = releases
    app = _install(tmp, old)
    (app / ".git").mkdir()
    assert "git checkout" in _run(app, url=new.with_name("version.json")).stdout
    (app / ".git").rmdir()
    (app / ".venv").rmdir()
    assert "not installed" in _run(app, url=new.with_name("version.json")).stdout


def test_refresh_runs_installer_only_when_needed(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    lock = f"install/requirements-{'windows' if update.WINDOWS else 'linux'}-cpu.txt"
    for d in (old, new):
        for f in ("install.bat", "install.sh", lock, "faceblur/setup_models.py"):
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text("same")
    calls = []
    monkeypatch.setattr(update, "ROOT", new)
    monkeypatch.setattr(update.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert update.refresh_environment(old, "cpu") == 0 and calls == []
    (new / "faceblur/setup_models.py").write_text("new models")
    update.refresh_environment(old, "cpu")
    assert "faceblur.setup_models" in calls[-1]
    (new / lock).write_text("torch==9")
    update.refresh_environment(old, "cpu")
    assert calls[-1][-1] == "cpu" and any("install" in c for c in calls[-1])


def test_locks_current_follows_requirements(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    for f in make_dist.LOCKS:
        (tmp_path / f).write_text("pinned==1\n")
    (install / "requirements.in").write_text("numpy\n")
    monkeypatch.setattr(make_dist, "ROOT", tmp_path)
    monkeypatch.setattr(make_dist, "INSTALL", install)
    monkeypatch.setattr(make_dist, "LOCK_STAMP", install / "locks.sha256")
    assert not make_dist.locks_current()  # never locked
    (install / "locks.sha256").write_text(make_dist.lock_inputs_hash() + "  comment\n")
    assert make_dist.locks_current()
    (install / "requirements.in").write_bytes(b"numpy\r\n")  # only line endings: still current
    assert make_dist.locks_current()
    (install / "requirements.in").write_text("numpy\nscipy\n")
    assert not make_dist.locks_current()


def test_repo_locks_are_current():
    assert make_dist.locks_current(), "run make_dist.py --lock"
