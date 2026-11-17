"""Build the zip to copy to another PC: code, launcher and installer, no models or Python.

  .venv\\Scripts\\python install\\make_dist.py          -> dist\\FaceBlur-<date>.zip
  .venv\\Scripts\\python install\\make_dist.py --lock   regenerate install\\requirements-*.txt first

On the other PC: unzip, run install.bat (Windows) or ./install.sh (Linux), then
FaceBlur.bat or ./FaceBlur.sh.
"""

from __future__ import annotations

import argparse
import datetime
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
SCRIPTS = ["install.bat", "FaceBlur.bat", "FaceBlurServer.bat", "install.sh", "FaceBlur.sh"]
FILES = SCRIPTS + ["README.md"] + LOCKS


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


def _entry(z: zipfile.ZipFile, path: Path) -> None:
    """Add one file with Unix permissions (scripts executable) and the line endings
    its shell needs (CRLF for .bat, LF for everything else), whatever git checked out."""
    data = path.read_bytes()
    if path.suffix in (".bat", ".sh", ".py", ".txt", ".md", ".in"):
        data = data.replace(b"\r\n", b"\n")
        if path.suffix == ".bat":
            data = data.replace(b"\n", b"\r\n")
    info = zipfile.ZipInfo(str(Path("FaceBlur") / path.relative_to(ROOT)).replace("\\", "/"),
                           date_time=time.localtime(path.stat().st_mtime)[:6])
    info.create_system = 3  # Unix, so unzip applies the mode below
    info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def build() -> Path:
    out = ROOT / "dist" / f"FaceBlur-{datetime.date.today():%Y-%m-%d}.zip"
    out.parent.mkdir(exist_ok=True)
    code = [p for p in (ROOT / "faceblur").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in code + [ROOT / f for f in FILES]:
            _entry(z, p)
    print(f"{out.relative_to(ROOT)}: {len(code) + len(FILES)} files, {out.stat().st_size / 1e3:,.0f} KB")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", action="store_true", help="regenerate the requirement locks first")
    if ap.parse_args().lock:
        lock()
    build()


if __name__ == "__main__":
    main()
