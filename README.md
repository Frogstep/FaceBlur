# FaceBlur

Masks chosen people's faces through a whole video. Pipeline: FFmpeg decode → low-light
enhancement (detection copy only) → SCRFD face detection + ByteTrack face tracks → **you
label every face card** → masks from your labels (gap filling, continuity, SAM 2.1
segmentation) → pixelate/blur → H.265 encode with audio copied → safety pass.

Nothing is joined by appearance automatically: a **face card** is one face followed
continuously by the tracker (by position, frame to frame), and you say who each card is.

## Desktop app

Double-click `FaceBlur.bat` (or drag a video onto it), or:

```bash
.venv/Scripts/python -m faceblur gui
```

1. **File › Open** a video. Faces are detected once and cached in `work/<video>/`.
2. **Review faces tab:** every face card shows thumbnails across it. Label each card:
   a person to mask (**1–9**, or *New person*), **O** someone else (a real face, not
   masked) or **N** not a face. Select several cards with Ctrl/Shift+click. One person
   usually has several cards (they leave the frame, turn away, the scene changes).
   Similar faces are listed next to each other; *Show: Not labeled* lists what is left.
   **Render stays disabled until every card has a label.**
3. **Check the timeline:** coloured bands show where each person is masked (bright = face
   detected, dim = bridged gap, nearby weak detection or drawn box). Tick *Preview mask*
   to see it on the frame.
4. **Fix mistakes on the video:**
   - click a face box → relabel its whole card, or *Not this person* (this frame or the
     whole track)
   - drag a rectangle → *Mask this area* for a person; SAM 2 tracks it from there
5. **Render** (Ctrl+R). The safety check runs afterwards; frames where a masked face may
   still be recognizable are listed and marked red on the timeline. Fix them and render again.

Weak detections that belong to no card (very dark, blurred, turned away) are masked only
where a labeled person's face was within the last or next second.

Keys on the video (click it first): Space play/pause, ←/→ one frame, Shift+←/→ one second,
Home/End.

Everything you choose is saved automatically (about a second after each change) next to the
video as `<video>.faceblur.json`. Reopening the video or project restores it, with the last
output and its safety report. Projects from before the face review keep their settings and
corrections, but their faces need labeling again.

## Command line

```bash
.venv/Scripts/python -m faceblur pick input.mp4
```

Open `work/input/faces.jpg` (one numbered row per face card), then label **every** card:

```bash
.venv/Scripts/python -m faceblur render input.mp4 --face 1,2,5 --face 3,4 --other 6 --not-face 7
```

`--face` once per person to mask; `--other` real faces that stay visible; `--not-face`
false detections; `--rest other` (or `--rest not-face`) labels every card you did not list.
Rendering is refused while any card has no label.

A project saved from the app (including its corrections) renders with:

```bash
.venv/Scripts/python -m faceblur render --project input.faceblur.json
```

Output: `input_masked.mp4`. Options: `--style pixelate|blur|fill`, `--strength 1.5`
(coarser), `--pad 0.4` (bigger margin), `--debug` (overlay video in `work/input/debug.mp4`),
`--no-sam` (ellipse masks only), `--no-check`.

The safety pass runs automatically after rendering (exit code 2 if something is flagged).
To run it again on its own:

```bash
.venv/Scripts/python -m faceblur check input_masked.mp4 --video input.mp4
```

Detections are cached in `work/<name>/detections.npz`; delete it (or use *File › Detect
faces again*) to force a rescan.

## Web API (for a phone app)

```bash
.venv/Scripts/python -m faceblur serve
```

(or double-click `FaceBlurServer.bat`; `./FaceBlur.sh serve` on Linux). It listens on port 8765
on all network addresses (`--host 127.0.0.1` for this PC only) and prints the addresses and an
access token (kept in `server_token.txt`). Every request needs `Authorization: Bearer <token>`.
The interactive API documentation is at `http://<address>:8765/docs` (*Authorize* with the token).

Flow: `POST /api/v1/jobs` with the video as the request body (or multipart field `file`) →
poll `GET /api/v1/jobs/{id}` until `status` is `review` → `GET …/cards` (thumbnails, and
`…/frames/{n}.jpg` for whole frames with numbered boxes) → `PUT …/people` and `PUT …/labels`
(`{"labels": {"1": 0, "4": "other", "7": "not_face"}, "rest": "other"}`) → `POST …/render`
→ poll until `done` → `GET …/result` (resumable download) and `GET …/safety`.

Each upload gets its own folder in `jobs/` holding an ordinary project file, so a job can also
be opened in the desktop app (`faceblur gui jobs/<id>/project.faceblur.json`). Jobs untouched
for 7 days are deleted (`--keep-days`). The GPU runs one job at a time; others queue.
Keep the server on your home network or a VPN such as Tailscale: it has no HTTPS of its own.
Tests: `.venv/Scripts/python -m pytest tests/server`.

## GPU or CPU

Everything runs on an NVIDIA GPU when one works, and falls back to the CPU otherwise, so
the app also runs on a PC without a dedicated GPU (much slower, same algorithm). Choose
with **Device** in the app's Render box, `--device auto|gpu|cpu` on the command line, or the
`FACEBLUR_DEVICE` environment variable.

| Stage | GPU | CPU fallback |
| --- | --- | --- |
| Decode | FFmpeg NVDEC | FFmpeg software |
| Face detection + matching | onnxruntime CUDA | onnxruntime CPU |
| Noise reduction (dark frames) | CPU, in parallel worker threads | same |
| SAM 2 | large model, CUDA | tiny model (`models/sam2.1_hiera_tiny.pt`), CPU |
| Encode | `hevc_nvenc` | `libx265` |

On the CPU, SAM 2 is slow even with the tiny model; turning it off (ellipse masks) is much
faster. Detections cached on one machine are valid on another.

## Install on another PC

Nothing heavy travels with the app: the installer downloads it on the target PC.

1. On this PC, build the zip (under 100 KB):

   ```bash
   .venv/Scripts/python install/make_dist.py
   ```

2. Copy `dist/FaceBlur-<date>.zip` to the other PC and unzip it where it should live
   (moving the folder later breaks the Python environment; run the installer again if
   you do). Then run the installer:
   - **Windows 10/11:** double-click **`install.bat`**
   - **Linux** (x86-64 with glibc 2.34+: Ubuntu 22.04+, Debian 12+, Fedora 35+, RHEL 9+,
     Arch, openSUSE Tumbleweed; any distribution, no root needed): **`./install.sh`**

   It needs only an internet connection, and puts everything inside the folder:
   - Python 3.12 (via [uv](https://docs.astral.sh/uv/)) and a `.venv`
   - PyTorch, the CUDA 13 build if an NVIDIA driver is found, else the CPU build
     (`install.bat gpu|cpu` or `./install.sh gpu|cpu` to choose)
   - the models: face models and SAM 2.1 large on a GPU PC, SAM 2.1 tiny on a CPU PC
   - FFmpeg 9.0 in `tools/ffmpeg/` (Gyan's build on Windows, BtbN's static build on Linux)

   Downloads are about 4-5 GB for the GPU build and 1 GB for the CPU build. Every file is
   checked against a SHA-256; running the installer again skips what is done and resumes
   unfinished model downloads. To skip the model download, copy `models/` from this PC
   first.
3. Start with **`FaceBlur.bat`** (Windows) or **`./FaceBlur.sh`** (Linux, also takes the
   command-line commands: `./FaceBlur.sh render input.mp4 --face 1`). To uninstall, delete
   the folder.

A GPU PC needs an NVIDIA driver recent enough for CUDA 13 (580 or newer); the check at
the end of the install says if CUDA does not work. On Linux it also lists any system
libraries the app window needs and prints the install command for your distribution
(`apt`, `dnf`, `pacman` or `zypper`); the command line works without them.
Check an installation at any time with `.venv/Scripts/python -m faceblur.setup_models --check`
(`.venv/bin/python` on Linux).
Package versions are pinned in `install/requirements-<windows|linux>-<gpu|cpu>.txt`; after
changing `install/requirements.in`, regenerate them with `make_dist.py --lock`.

## Development setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m faceblur.setup_models --all --no-ffmpeg
```

Models in `models/`: `det_10g.onnx` and `w600k_r50.onnx` (InsightFace buffalo_l,
non-commercial license), `sam2.1_hiera_large.pt` and `sam2.1_hiera_tiny.pt` (Meta,
Apache 2.0).
