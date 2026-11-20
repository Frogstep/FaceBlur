"""Update FaceBlur to the newest release on GitHub (or from a zip file), keeping models,
projects, jobs and settings. Run through update.bat (Windows) or ./update.sh (Linux).

  update.bat                    install the newest release if it is newer than this one
  update.bat --check            only say whether there is a newer release
  update.bat FaceBlur-X.zip     install from a zip file (no internet needed for the code)
  update.bat --rollback         go back to the version before the last update

Only the app's own files are replaced: faceblur/, the scripts, README and the package
locks. The previous ones go to update/backup/. If the new version needs other Python
packages or models, the installer runs again and downloads only what changed.

Uses only the standard library and never imports the faceblur package, so it works
whatever version of the app is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

RELEASES_REPO = "Frogstep/FaceBlur-releases"  # public repo with only the release zips
LATEST_URL = f"https://github.com/{RELEASES_REPO}/releases/latest/download/version.json"

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = os.name == "nt"
UPDATE = ROOT / "update"
BACKUP = UPDATE / "backup"
TOP = "FaceBlur"  # folder at the top of every zip
CODE_DIR = "faceblur"  # replaced as a whole, so deleted modules go too


class UpdateError(Exception):
    pass


# ---------------------------------------------------------------- versions

def parse_version(text: str) -> tuple[int, ...]:
    """'2026.11.20.2' -> (2026, 11, 20, 2); anything else (no version file, 'dev') -> ()."""
    return tuple(int(p) for p in text.split(".")) if re.fullmatch(r"\d+(\.\d+)*", text or "") else ()


def read_version_file(text: str) -> str:
    m = re.search(r'__version__\s*=\s*"([^"]*)"', text)
    return m.group(1) if m else ""


def installed_version(root: Path = ROOT) -> str:
    """Version of the installed app, "" when unknown (installed from a zip made before
    versions existed)."""
    f = root / CODE_DIR / "_version.py"
    return read_version_file(f.read_text(encoding="utf-8")) if f.exists() else ""


def show(version: str) -> str:
    return version or "unknown (older than the updater)"


# ---------------------------------------------------------------- download

def fetch(url: str, dest: Path | None = None) -> bytes | None:
    """Download url into dest (via a .part file, with progress), or return its bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "FaceBlur-updater"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if dest is None:
            return r.read()
        total = int(r.headers.get("Content-Length") or 0)
        part = dest.with_name(dest.name + ".part")
        done = 0
        with open(part, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e3:,.0f} of {total / 1e3:,.0f} KB", end="", flush=True)
        if total:
            print()
        part.replace(dest)
    return None


def latest_release(url: str) -> dict:
    try:
        info = json.loads(fetch(url))
    except Exception as e:  # noqa: BLE001  (no internet, GitHub down, no release yet)
        raise UpdateError(f"could not read the newest version from {url}: {e}") from e
    if not parse_version(info.get("version", "")) or not info.get("zip") or not info.get("sha256"):
        raise UpdateError(f"{url} is not a FaceBlur version file")
    return info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_release(info: dict, version_url: str) -> Path:
    name = Path(info["zip"]).name
    url = urllib.parse.urljoin(version_url, name)  # the zip sits next to version.json
    UPDATE.mkdir(exist_ok=True)
    dest = UPDATE / name
    print(f"Downloading {name}")
    try:
        fetch(url, dest)
    except Exception as e:  # noqa: BLE001
        raise UpdateError(f"download failed: {e}") from e
    if sha256(dest) != info["sha256"].lower():
        dest.unlink()
        raise UpdateError("the download is corrupted (SHA-256 does not match); run the update again")
    return dest


# ---------------------------------------------------------------- unpack

def unpack(zip_path: Path) -> tuple[Path, str]:
    """Extract a release zip to update/new/ and return (its FaceBlur folder, version)."""
    new = UPDATE / "new"
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir(parents=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for n in names:
                parts = Path(n).parts
                if not parts or parts[0] != TOP or ".." in parts or Path(n).is_absolute():
                    raise UpdateError(f"{zip_path.name} is not a FaceBlur zip (unexpected entry {n})")
            if f"{TOP}/{CODE_DIR}/__init__.py" not in names:
                raise UpdateError(f"{zip_path.name} is not a FaceBlur zip")
            z.extractall(new)
    except zipfile.BadZipFile as e:
        raise UpdateError(f"{zip_path.name} is not a valid zip: {e}") from e
    src = new / TOP
    if not WINDOWS:  # keep the scripts executable (zipfile does not restore modes)
        for p in src.glob("*.sh"):
            p.chmod(0o755)
    return src, installed_version(src)


# ---------------------------------------------------------------- checks

def check_folder() -> None:
    if (ROOT / ".git").exists():
        raise UpdateError(f"{ROOT} is a git checkout (the development copy): update it with git instead")
    if not (ROOT / ".venv").exists():
        raise UpdateError("FaceBlur is not installed in this folder yet: run the installer first")


def running_instances() -> list[str]:
    """Command lines of FaceBlur processes started from this folder (app or server)."""
    root = str(ROOT).lower()
    me = {os.getpid(), os.getppid()}
    found = []
    if WINDOWS:
        ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*-m faceblur*' } | "
              "ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId, $_.ExecutablePath, $_.CommandLine }")
        env = dict(os.environ, PSModulePath="")  # Windows PowerShell's own modules, even from pwsh
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                                 text=True, timeout=60, env=env).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        for line in out.splitlines():
            pid, exe, cmd = (line.split("|", 2) + ["", ""])[:3]
            if pid.isdigit() and int(pid) not in me and exe.lower().startswith(root):
                found.append(cmd.strip())
    else:
        for d in Path("/proc").glob("[0-9]*"):
            try:
                if int(d.name) in me:
                    continue
                args = (d / "cmdline").read_bytes().split(b"\0")
                exe = os.readlink(d / "exe")
            except OSError:
                continue
            if b"-m" in args and b"faceblur" in args and exe.lower().startswith(root):
                found.append(" ".join(a.decode(errors="replace") for a in args if a))
    return found


def check_not_running() -> None:
    running = running_instances()
    if running:
        lines = "\n".join(f"  {c}" for c in running)
        raise UpdateError(f"FaceBlur is running; close the app and the server first:\n{lines}")


# ---------------------------------------------------------------- swap

def shipped_files(src: Path) -> list[str]:
    """Paths (relative, '/'-separated) the update replaces: the code folder as a whole,
    and every other file in the zip one by one."""
    files = [CODE_DIR]
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and not rel.startswith(CODE_DIR + "/"):
            files.append(rel)
    return files


def _move(a: Path, b: Path) -> None:
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(a), str(b))


def swap(src: Path, files: list[str]) -> None:
    """Move the installed copies of files to update/backup/ and the new ones into place.
    On any error, put everything back."""
    shutil.rmtree(BACKUP, ignore_errors=True)
    BACKUP.mkdir(parents=True)
    moved: list[str] = []  # backed up
    placed: list[str] = []  # new copy in place
    try:
        for rel in files:
            if (ROOT / rel).exists():
                _move(ROOT / rel, BACKUP / rel)
                moved.append(rel)
            _move(src / rel, ROOT / rel)
            placed.append(rel)
    except OSError as e:
        for rel in reversed(placed):
            _move(ROOT / rel, src / rel)
        for rel in reversed(moved):
            _move(BACKUP / rel, ROOT / rel)
        raise UpdateError(f"could not replace the files ({e}); nothing was changed. "
                          "Close every FaceBlur window and folder view and try again") from e
    (BACKUP / "files.json").write_text(json.dumps({"files": files}), encoding="utf-8")


def _text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None


# ---------------------------------------------------------------- packages and models

def variant() -> str:
    """'gpu' or 'cpu': which build the installer put into .venv."""
    pattern = "Lib/site-packages" if WINDOWS else "lib/python*/site-packages"
    return "gpu" if any((ROOT / ".venv").glob(f"{pattern}/onnxruntime_gpu-*.dist-info")) else "cpu"


def refresh_environment(old: Path, var: str) -> int:
    """After the files changed: rerun the installer if the package locks or the installer
    changed, or only the model download if the model list changed. old holds the
    previous files. Returns the exit code (0 = fine or nothing to do)."""
    platform = "windows" if WINDOWS else "linux"
    installer = "install.bat" if WINDOWS else "install.sh"
    lock = f"install/requirements-{platform}-{var}.txt"
    if any(_text(old / f) != _text(ROOT / f) for f in (installer, lock)):
        print(f"\nThis version needs other Python packages: running the installer ({var} build).")
        cmd = ["cmd", "/c", installer, var] if WINDOWS else ["bash", installer, var]
        return subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL).returncode  # no "press a key"
    models = f"{CODE_DIR}/setup_models.py"
    if _text(old / models) != _text(ROOT / models):
        print("\nChecking the model weights and FFmpeg (downloads only what is new).")
        python = ROOT / ".venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
        return subprocess.run([str(python), "-m", "faceblur.setup_models", f"--{var}"], cwd=ROOT).returncode
    return 0


# ---------------------------------------------------------------- commands

def install_zip(zip_path: Path, force: bool, expect: str = "") -> int:
    current = installed_version()
    src, new = unpack(zip_path)
    if expect and new != expect:
        raise UpdateError(f"{zip_path.name} contains version {show(new)}, not {expect}")
    if not force and parse_version(new) <= parse_version(current) and parse_version(current):
        print(f"{zip_path.name} has version {show(new)}; installed is {current}. Nothing to do "
              "(--force installs it anyway).")
        return 0
    var = variant()
    print(f"Updating {show(current)} -> {show(new)}")
    swap(src, shipped_files(src))
    shutil.rmtree(UPDATE / "new", ignore_errors=True)
    code = refresh_environment(BACKUP, var)
    if code:
        print(f"\nThe app files are updated to {show(new)}, but the installer step failed (code {code}).\n"
              f"Run {'install.bat' if WINDOWS else './install.sh'} again (it needs the internet),\n"
              f"or go back with {'update.bat' if WINDOWS else './update.sh'} --rollback.")
        return code
    print(f"\nFaceBlur is updated to {show(new)}. The previous version is kept in update/backup/.")
    return 0


def rollback() -> int:
    meta = BACKUP / "files.json"
    if not meta.exists():
        raise UpdateError("there is no previous version to go back to")
    files = json.loads(meta.read_text(encoding="utf-8"))["files"]
    current, old = installed_version(), installed_version(BACKUP)
    var = variant()
    # the current files become the backup, so a second rollback undoes the first
    hold = UPDATE / "rollback"
    shutil.rmtree(hold, ignore_errors=True)
    shutil.move(str(BACKUP), str(hold))
    swap(hold, [f for f in files if (hold / f).exists()])
    shutil.rmtree(hold, ignore_errors=True)
    print(f"Went back from {show(current)} to {show(old)}.")
    print("Projects saved by the newer version may not open in the older one.")
    return refresh_environment(BACKUP, var)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="update", description=__doc__.split("\n\n")[0])
    ap.add_argument("zip", nargs="?", type=Path, help="install from this release zip instead of downloading")
    ap.add_argument("--check", action="store_true", help="only say whether a newer version exists")
    ap.add_argument("--rollback", action="store_true", help="go back to the version before the last update")
    ap.add_argument("--force", action="store_true", help="install even if it is not newer")
    a = ap.parse_args(argv)
    url = os.environ.get("FACEBLUR_UPDATE_URL", LATEST_URL)  # tests point this at a local folder
    try:
        if a.check:
            info, current = latest_release(url), installed_version()
            newer = parse_version(info["version"]) > parse_version(current)
            print(f"Installed: {show(current)}. Newest: {info['version']}"
                  + (" (update available)." if newer else " (up to date)."))
            if newer and info.get("notes"):
                print(info["notes"])
            return 0
        check_folder()
        check_not_running()
        if a.rollback:
            return rollback()
        if a.zip:
            if not a.zip.is_file():
                raise UpdateError(f"{a.zip} does not exist")
            return install_zip(a.zip.resolve(), a.force)
        current = installed_version()
        info = latest_release(url)
        if not a.force and parse_version(info["version"]) <= parse_version(current) and parse_version(current):
            print(f"FaceBlur {current} is the newest version.")
            return 0
        print(f"New version {info['version']} (installed: {show(current)}).")
        if info.get("notes"):
            print(info["notes"])
        zip_path = download_release(info, url)
        try:
            return install_zip(zip_path, force=True, expect=info["version"])
        finally:
            zip_path.unlink(missing_ok=True)
    except UpdateError as e:
        print(f"\nUpdate failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
