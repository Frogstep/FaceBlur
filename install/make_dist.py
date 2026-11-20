"""Build the zip to copy to another PC: code, launcher and installer, no models or Python.

  .venv\\Scripts\\python install\\make_dist.py          -> dist\\FaceBlur-<date>.zip
  .venv\\Scripts\\python install\\make_dist.py --lock   regenerate install\\requirements-*.txt first

On the other PC: unzip, run install.bat (Windows) or ./install.sh (Linux), then
FaceBlur.bat or ./FaceBlur.sh.

publish.bat (install/publish.py) builds the same zip with a version number written into
faceblur/_version.py and uploads it as a GitHub release; update.bat / ./update.sh on the
other PC installs the newest release (install/update.py).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install"
PLATFORMS = {"windows": "windows", "linux": "x86_64-manylinux_2_34"}  # uv --python-platform
ONNXRUNTIME = {"gpu": "onnxruntime-gpu==1.32.0", "cpu": "onnxruntime==1.32.0"}
LOCKS = [f"install/requirements-{p}-{v}.txt" for p in PLATFORMS for v in ONNXRUNTIME]
SCRIPTS = ["install.bat", "FaceBlur.bat", "FaceBlurServer.bat", "update.bat",
           "install.sh", "FaceBlur.sh", "update.sh"]
FILES = SCRIPTS + ["README.md", "install/update.py"] + LOCKS
VERSION_FILE = ROOT / "faceblur" / "_version.py"  # only inside published zips
# SHA-256 of what the locks were made from, so publish.py notices requirements.in
# changes that were not locked (git checkouts reset file dates, so dates cannot tell)
LOCK_STAMP = INSTALL / "locks.sha256"


def lock_inputs_hash() -> str:
    base = (INSTALL / "requirements.in").read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(json.dumps([base, PLATFORMS, ONNXRUNTIME], sort_keys=True).encode()).hexdigest()


def locks_current() -> bool:
    """True when the locks were made from the current requirements.in (by lock())."""
    stamp = LOCK_STAMP.read_text(encoding="utf-8").split() if LOCK_STAMP.exists() else []
    return bool(stamp) and stamp[0] == lock_inputs_hash() and all((ROOT / f).exists() for f in LOCKS)


def lock() -> None:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("uv is needed for --lock (https://docs.astral.sh/uv/)")
    base = (INSTALL / "requirements.in").read_text(encoding="utf-8")
    for platform, uv_platform in PLATFORMS.items():
        for variant, ort in ONNXRUNTIME.items():
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "requirements.in"
                src.write_text(f"{base}\n{ort}\n", encoding="utf-8")
                out = INSTALL / f"requirements-{platform}-{variant}.txt"
                subprocess.run([uv, "pip", "compile", str(src), "--python-version", "3.12",
                                "--python-platform", uv_platform, "--no-header", "--no-annotate",
                                "-q", "-o", str(out)], check=True)
                print(f"locked {out.relative_to(ROOT)}")
    LOCK_STAMP.write_text(f"{lock_inputs_hash()}  written by make_dist.py --lock; commit it with the locks\n",
                          encoding="utf-8")


def _entry(z: zipfile.ZipFile, path: Path, data: bytes | None = None) -> None:
    """Add one file with Unix permissions (scripts executable) and the line endings
    its shell needs (CRLF for .bat, LF for everything else), whatever git checked out.
    With data, that is written instead of the file's content (the file need not exist)."""
    if data is None:
        data = path.read_bytes()
    if path.suffix in (".bat", ".sh", ".py", ".txt", ".md", ".in"):
        data = data.replace(b"\r\n", b"\n")
        if path.suffix == ".bat":
            data = data.replace(b"\n", b"\r\n")
    info = zipfile.ZipInfo(str(Path("FaceBlur") / path.relative_to(ROOT)).replace("\\", "/"),
                           date_time=time.localtime(path.stat().st_mtime if path.exists() else None)[:6])
    info.create_system = 3  # Unix, so unzip applies the mode below
    info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def build(version: str | None = None, out_dir: Path | None = None) -> Path:
    """Build the zip in out_dir (default dist/). With a version, the zip is named after it
    and the version is written into faceblur/_version.py inside the zip (the working tree
    is not changed)."""
    out = (out_dir or ROOT / "dist") / f"FaceBlur-{version or f'{datetime.date.today():%Y-%m-%d}'}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    code = [p for p in (ROOT / "faceblur").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p != VERSION_FILE]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in code + [ROOT / f for f in FILES]:
            _entry(z, p)
        if version:
            _entry(z, VERSION_FILE, f'__version__ = "{version}"\n'.encode())
    count = len(code) + len(FILES) + bool(version)
    print(f"{out}: {count} files, {out.stat().st_size / 1e3:,.0f} KB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", action="store_true", help="regenerate the requirement locks first")
    if ap.parse_args().lock:
        lock()
    build()


if __name__ == "__main__":
    main()
