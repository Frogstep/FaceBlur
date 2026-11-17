#!/usr/bin/env bash
# FaceBlur installer for Linux (x86-64, glibc 2.34+: Ubuntu 22.04+, Debian 12+,
# Fedora 35+, RHEL 9+, Arch, openSUSE Tumbleweed, ...). Downloads Python, the Python
# packages, the model weights and FFmpeg into this folder; nothing is installed
# system-wide and no root is needed. Safe to run again: it skips what is already
# there and resumes interrupted model downloads.
#
#   ./install.sh        GPU build if an NVIDIA driver is found, else CPU build
#   ./install.sh gpu    force the NVIDIA (CUDA 13) build
#   ./install.sh cpu    force the CPU-only build (much smaller download)
#
# To uninstall, delete this folder.

set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"

UV_VERSION=0.11.15
UV_SHA256=b03e572f010bea94a4a52d42671ba72981e12894f71576181a1d26ff68546da7
SAM2="SAM-2 @ https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.zip"

# keep everything inside this folder
UV="$ROOT/tools/uv/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/tools/python"
export UV_CACHE_DIR="$ROOT/tools/uv-cache"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300

fail() {
    echo
    echo "Installation failed: $*"
    echo "Fix the problem (e.g. the internet connection) and run ./install.sh again:"
    echo "finished steps are skipped."
    exit 1
}
trap 'fail "the command above returned an error"' ERR

# ------------------------------------------------------------ system checks
[ "$(uname -s)" = Linux ] || fail "this script is for Linux (use install.bat on Windows)"
[ "$(uname -m)" = x86_64 ] || fail "only x86-64 PCs are supported (this is $(uname -m))"
glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')" || true
[ -n "$glibc" ] || fail "glibc not found (musl-based distributions such as Alpine are not supported)"
if [ "$(printf '%s\n' 2.34 "$glibc" | sort -V | head -1)" != 2.34 ]; then
    fail "glibc $glibc is too old; 2.34 or newer is needed (Ubuntu 22.04+, Debian 12+, Fedora 35+)"
fi
if command -v curl >/dev/null; then
    fetch() { curl -fL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null; then
    fetch() { wget -q --show-progress -O "$2" "$1"; }
else
    fail "neither curl nor wget is installed (e.g. sudo apt install curl)"
fi
for tool in tar sha256sum; do
    command -v $tool >/dev/null || fail "$tool is not installed"
done

# ------------------------------------------------------------ GPU or CPU
VARIANT="${1:-}"
if [ -z "$VARIANT" ]; then
    VARIANT=cpu
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then VARIANT=gpu; fi
fi
case "$VARIANT" in
    gpu) TORCH="torch==2.11.0+cu130 torchvision==0.26.0+cu130"
         TORCH_INDEX=https://download.pytorch.org/whl/cu130
         OTHER_ORT=onnxruntime
         echo "Installing the NVIDIA GPU build: about 5 GB to download." ;;
    cpu) TORCH="torch==2.11.0+cpu torchvision==0.26.0+cpu"
         TORCH_INDEX=https://download.pytorch.org/whl/cpu
         OTHER_ORT=onnxruntime-gpu
         echo "Installing the CPU-only build: about 1 GB to download. Processing will be slow." ;;
    *) echo "Usage: ./install.sh [gpu|cpu]"; exit 2 ;;
esac
echo

# ------------------------------------------------------------ uv
if [ -x "$UV" ]; then
    echo "[1/5] uv already present"
else
    echo "[1/5] Downloading uv $UV_VERSION (Python installer)"
    mkdir -p "$ROOT/tools/uv"
    archive="$ROOT/tools/uv.tar.gz"
    fetch "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" "$archive"
    if ! echo "$UV_SHA256  $archive" | sha256sum -c --status; then
        rm -f "$archive"
        fail "uv download is corrupted"
    fi
    tar -xzf "$archive" -C "$ROOT/tools/uv" --strip-components=1
    rm -f "$archive"
fi

# ------------------------------------------------------------ Python + venv
echo "[2/5] Python 3.12 and virtual environment"
if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.12 --python-preference only-managed .venv
fi

# ------------------------------------------------------------ packages
echo "[3/5] Python packages (PyTorch is large; this takes a while)"
# onnxruntime and onnxruntime-gpu share one module: remove the other build if present
"$UV" pip uninstall --python .venv "$OTHER_ORT" >/dev/null 2>&1 || true
# exact local versions (+cu130 / +cpu) so switching builds really replaces PyTorch
# shellcheck disable=SC2086
"$UV" pip install --python .venv $TORCH --index-url "$TORCH_INDEX"
"$UV" pip install --python .venv -r "install/requirements-linux-$VARIANT.txt"
# SAM 2 from source: no CUDA extension (optional, needs a compiler), and no build
# isolation so it does not download a second PyTorch just to build
SAM2_BUILD_CUDA=0 "$UV" pip install --python .venv --no-deps --no-build-isolation "$SAM2"

# ------------------------------------------------------------ models + FFmpeg
echo "[4/5] Model weights and FFmpeg"
.venv/bin/python -m faceblur.setup_models "--$VARIANT"

# the package cache is only needed during install
rm -rf "$UV_CACHE_DIR"

# ------------------------------------------------------------ check
trap - ERR

# ------------------------------------------------------------ system libraries
# The app window needs a few system libraries that minimal installs lack. Installing
# them is the only step that needs root: show the command and ask first.
deps="$(.venv/bin/python -m faceblur.setup_models --system-deps 2>/dev/null || true)"
if [ -n "$deps" ]; then
    echo
    echo "The app window needs some system libraries that are not installed. Command:"
    echo "  sudo sh -c '$deps'"
    if [ "$(id -u)" = 0 ]; then
        sh -c "$deps" || echo "Installing the libraries failed; the check below lists what is missing."
    elif command -v sudo >/dev/null && [ -t 0 ]; then
        read -r -p "Install them now? sudo will ask for your password. [Y/n] " answer
        case "$answer" in
            [nN]*) echo "Skipped. The command line works without them." ;;
            *) sudo sh -c "$deps" || echo "Installing the libraries failed; the check below lists what is missing." ;;
        esac
    else
        echo "Run it as root, then start ./FaceBlur.sh. The command line works without them."
    fi
    echo
fi

echo "[5/5] Checking the installation"
echo
if .venv/bin/python -m faceblur.setup_models --check; then
    echo
    echo "Done. Start FaceBlur with ./FaceBlur.sh (optionally followed by a video)."
else
    echo
    echo "Installed, but fix the problems listed above before starting ./FaceBlur.sh."
    exit 1
fi
