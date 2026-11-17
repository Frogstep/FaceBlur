"""Download the model weights and FFmpeg, and check an installation (Windows and Linux).

  python -m faceblur.setup_models --gpu     SAM 2.1 large (+ face models, FFmpeg)
  python -m faceblur.setup_models --cpu     SAM 2.1 tiny  (+ face models, FFmpeg)
  python -m faceblur.setup_models --all     both SAM 2.1 sizes
  python -m faceblur.setup_models --check   report what is installed and what runs on the GPU

Uses only the standard library, so it runs before anything else works. Files that
are already present with the right SHA-256 are skipped, so it is safe to run again
after a failed download, or after copying models/ from another PC.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FFMPEG_DIR = ROOT / "tools" / "ffmpeg"
DOWNLOADS = ROOT / "tools" / "downloads"
WINDOWS = sys.platform == "win32"


@dataclass
class Item:
    name: str                 # file name in its target folder
    sha256: str               # "" = not checked (it comes from a checked archive)
    url: str                  # the file itself, or the archive that contains it
    member: str = ""          # path inside the archive ("" = url is the file)
    archive_sha256: str = ""  # SHA-256 of the archive...
    sha_url: str = ""         # ...or a checksums file that lists it
    folder: Path = MODELS_DIR
    executable: bool = False
    optional: bool = False    # a missing member is not an error


# InsightFace buffalo_l (non-commercial research licence): detection + identity.
_BUFFALO = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_BUFFALO_SHA = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
FACE_MODELS = [
    Item("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
         _BUFFALO, "det_10g.onnx", _BUFFALO_SHA),
    Item("w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
         _BUFFALO, "w600k_r50.onnx", _BUFFALO_SHA),
]

# SAM 2.1 from Meta's Hugging Face repos (dl.fbaipublicfiles.com refuses direct downloads).
SAM_MODELS = {
    "large": Item("sam2.1_hiera_large.pt", "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
                  "https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt"),
    "tiny": Item("sam2.1_hiera_tiny.pt", "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
                 "https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt"),
}

# FFmpeg 9.0 GPL builds with NVENC/NVDEC and libx265.
if WINDOWS:
    # Gyan's "essentials" build, pinned
    _FF = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
    _FF_DIR = "ffmpeg-9.0.2-essentials_build"
    FFMPEG = [Item(n, "", _FF, f"{_FF_DIR}/{m}", "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
                   folder=FFMPEG_DIR)
              for n, m in (("ffmpeg.exe", "bin/ffmpeg.exe"), ("ffprobe.exe", "bin/ffprobe.exe"), ("LICENSE", "LICENSE"))]
else:
    # BtbN's static build of the latest 9.0.x; BtbN keeps no fixed versions, so the
    # archive is checked against the checksums published with the same release
    _REL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
    _FF_DIR = "ffmpeg-n9.0-latest-linux64-gpl-9.0"
    FFMPEG = [Item(n, "", f"{_REL}/{_FF_DIR}.tar.xz", f"{_FF_DIR}/{m}", sha_url=f"{_REL}/checksums.sha256",
                   folder=FFMPEG_DIR, executable=x, optional=not x)
              for n, m, x in (("ffmpeg", "bin/ffmpeg", True), ("ffprobe", "bin/ffprobe", True),
                              ("LICENSE.txt", "LICENSE.txt", False))]


# ---------------------------------------------------------------- helpers

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _ok(item: Item) -> bool:
    path = item.folder / item.name
    if not path.exists():
        return False
    return not item.sha256 or sha256(path) == item.sha256


def _mb(n: float) -> str:
    return f"{n / 1e6:,.0f} MB"


def _open(url: str, **headers):
    req = urllib.request.Request(url, headers={"User-Agent": "faceblur-setup", **headers})
    return urllib.request.urlopen(req, timeout=60)


def published_sha(sha_url: str, filename: str) -> str:
    """Look filename up in a sha256sum-style checksums file."""
    with _open(sha_url) as r:
        for line in r.read().decode().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*").rsplit("/", 1)[-1] == filename:
                return parts[0].lower()
    raise RuntimeError(f"{filename} is not listed in {sha_url}")


def download(url: str, dest: Path, expected_sha: str, tries: int = 4) -> None:
    """Download url to dest (resuming a partial .part file) and check its SHA-256."""
    if dest.exists() and sha256(dest) == expected_sha:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            with _open(url, **({"Range": f"bytes={have}-"} if have else {})) as r:
                if have and r.status != 206:  # server ignored the range: start over
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                done, last = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last > 0.5:
                            last = time.monotonic()
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(f"\r  {dest.name}: {_mb(done)} / {_mb(total)} {pct}   ", end="", flush=True)
            print(f"\r  {dest.name}: {_mb(done)} downloaded{' ' * 20}")
            got = sha256(part)
            if got != expected_sha:
                part.unlink()
                raise RuntimeError(f"checksum mismatch for {dest.name} (got {got[:12]}...)")
            part.replace(dest)
            return
        except Exception as e:  # network hiccup, timeout, bad checksum
            print(f"\n  attempt {attempt}/{tries} failed: {e}")
            if attempt == tries:
                raise
            time.sleep(3 * attempt)


def _extract(archive: Path, member: str, dest: Path) -> bool:
    """Copy one member of a .zip or .tar.* to dest. False if it is not in the archive."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            if member not in z.namelist():
                return False
            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
        return True
    with tarfile.open(archive) as t:
        try:
            src = t.extractfile(member)
        except KeyError:
            return False
        with src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    return True


def fetch(items: list[Item]) -> None:
    todo = [it for it in items if not _ok(it)]
    for it in items:
        if it not in todo and (it.folder / it.name).exists():
            print(f"  {it.name}: already present")
    by_archive: dict[str, list[Item]] = {}
    for it in todo:
        if it.member:
            by_archive.setdefault(it.url, []).append(it)
        else:
            print(f"Downloading {it.name}")
            download(it.url, it.folder / it.name, it.sha256)
    for url, members in by_archive.items():
        if all(it.optional for it in members):  # only optional files (a licence text) missing
            continue
        archive = DOWNLOADS / url.rsplit("/", 1)[1]
        first = members[0]
        expected = first.archive_sha256 or published_sha(first.sha_url, archive.name)
        print(f"Downloading {archive.name}")
        download(url, archive, expected)
        for it in members:
            it.folder.mkdir(parents=True, exist_ok=True)
            dest = it.folder / it.name
            if not _extract(archive, it.member, dest):
                if it.optional:
                    continue
                raise RuntimeError(f"{it.member} is missing from {archive.name}")
            if it.sha256 and sha256(dest) != it.sha256:
                raise RuntimeError(f"{it.name} inside {archive.name} has an unexpected checksum")
            if it.executable:
                dest.chmod(0o755)
            print(f"  {it.name}: extracted")
        archive.unlink()
    shutil.rmtree(DOWNLOADS, ignore_errors=True)


# ---------------------------------------------------------------- Linux system libraries

# Qt's X11 plugin needs a few system libraries that minimal desktops may lack.
# Package names per distribution family (from ID / ID_LIKE in /etc/os-release).
QT_PACKAGES = {
    "debian": "sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 "
              "libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libegl1 libgl1 libfontconfig1 libdbus-1-3",
    "fedora": "sudo dnf install xcb-util-cursor xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil "
              "libxkbcommon-x11 mesa-libEGL mesa-libGL fontconfig dbus-libs",
    "arch": "sudo pacman -S --needed xcb-util-cursor xcb-util-wm xcb-util-image xcb-util-keysyms "
            "xcb-util-renderutil libxkbcommon-x11 libglvnd fontconfig",
    "suse": "sudo zypper install libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 "
            "libxcb-render-util0 libxcb-xkb1 libEGL1 libGL1 fontconfig",
}
_FAMILY = {"ubuntu": "debian", "linuxmint": "debian", "pop": "debian", "rhel": "fedora", "centos": "fedora",
           "rocky": "fedora", "almalinux": "fedora", "manjaro": "arch", "endeavouros": "arch",
           "opensuse": "suse", "opensuse-leap": "suse", "opensuse-tumbleweed": "suse", "sles": "suse"}


def distro_family() -> str | None:
    try:
        info = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    except OSError:
        return None
    for d in [info.get("ID", "")] + info.get("ID_LIKE", "").split():
        d = d.strip('"').lower()
        if d in QT_PACKAGES:
            return d
        if d in _FAMILY:
            return _FAMILY[d]
    return None


def qt_missing_libs() -> list[str]:
    """System libraries Qt's X11 platform plugin cannot find (Linux only)."""
    try:
        import PySide6
    except ImportError:
        return []
    plugin = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms" / "libqxcb.so"
    if not plugin.exists() or not shutil.which("ldd"):
        return []
    out = subprocess.run(["ldd", str(plugin)], capture_output=True, text=True).stdout
    return sorted({ln.split()[0] for ln in out.splitlines() if "not found" in ln})


# ---------------------------------------------------------------- check

def check() -> int:
    """Print what is installed and where things will run. Returns 0 when usable."""
    problems = 0
    print("Models:")
    for it in FACE_MODELS + list(SAM_MODELS.values()):
        path = it.folder / it.name
        state = "ok" if path.exists() else "missing" if it in FACE_MODELS else "not downloaded (optional)"
        print(f"  {it.name:24s} {state}")
        if not path.exists() and it in FACE_MODELS:
            problems += 1
    if not any((MODELS_DIR / it.name).exists() for it in SAM_MODELS.values()):
        print("  no SAM 2.1 model: run the installer again")
        problems += 1

    from .video_io import _ffbin

    print("FFmpeg:")
    for name in ("ffmpeg", "ffprobe"):
        try:
            print(f"  {name:8s} {_ffbin(name)}")
        except RuntimeError as e:
            print(f"  {e}")
            problems += 1

    if not WINDOWS:
        missing = qt_missing_libs()
        if missing:
            problems += 1
            print("System libraries for the app window are missing:\n  " + " ".join(missing))
            fam = distro_family()
            if fam:
                print(f"Install them with:\n  {QT_PACKAGES[fam]}")
            else:
                print("Install your distribution's packages for these libraries (xcb-util-cursor and friends).")
            print("(The command line works without them.)")

    print("Python packages:")
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu = f" - {torch.cuda.get_device_name(0)}" if cuda else ""
        print(f"  PyTorch {torch.__version__}, CUDA {'available' + gpu if cuda else 'not available'}")
        if torch.version.cuda and not cuda:
            rerun = "install.bat cpu" if WINDOWS else "./install.sh cpu"
            print("  This is the GPU build but CUDA does not work: update the NVIDIA driver\n"
                  f"  (version 580 or newer), or reinstall with: {rerun}")
        import cv2
        import onnxruntime
        import PySide6
        import sam2  # noqa: F401

        print(f"  onnxruntime {onnxruntime.__version__}, OpenCV {cv2.__version__}, PySide6 {PySide6.__version__}")
    except Exception as e:
        print(f"  import failed: {e}")
        return problems + 1

    from . import device

    print("Will run on:")
    print(f"  faces   {'GPU' if device.faces_on_gpu() else 'CPU'}")
    print(f"  SAM 2   {'GPU' if device.sam_device() == 'cuda' else 'CPU'}")
    print(f"  decode  {'GPU (NVDEC)' if device.nvdec_works() else 'CPU'}")
    print(f"  encode  {'GPU (NVENC)' if device.nvenc_works() else 'CPU (libx265)'}")
    print("Everything is in place." if not problems else f"{problems} problem(s) found.")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m faceblur.setup_models", description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--gpu", action="store_true", help="SAM 2.1 large (default)")
    which.add_argument("--cpu", action="store_true", help="SAM 2.1 tiny only (CPU-only PC)")
    which.add_argument("--all", action="store_true", help="both SAM 2.1 sizes")
    ap.add_argument("--no-ffmpeg", action="store_true", help="use FFmpeg from PATH instead of downloading it")
    ap.add_argument("--check", action="store_true", help="only report the installation")
    a = ap.parse_args(argv)
    if a.check:
        return check()

    sizes = ["tiny"] if a.cpu else ["large", "tiny"] if a.all else ["large"]
    items = FACE_MODELS + [SAM_MODELS[s] for s in sizes] + ([] if a.no_ffmpeg else FFMPEG)
    print("Model weights and FFmpeg:")
    try:
        fetch(items)
    except Exception as e:
        print(f"\nDownload failed: {e}\nCheck the internet connection and run the installer again;"
              " finished files are kept.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
