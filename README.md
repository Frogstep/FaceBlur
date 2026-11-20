# FaceBlur

Blurs the faces you choose through a whole video. Pipeline: FFmpeg decode → low-light
enhancement (detection copy only) → SCRFD face detection + ByteTrack face tracks → **you
click the faces to blur** → masks (gap filling, continuity; ellipses, or exact outlines with
SAM 2.1 when you turn it on) → pixelate/blur → H.265 encode with audio copied → optional
safety pass.

Both slow parts are **off by default** in the desktop app and the web app: *Precise outline
with SAM 2* (otherwise each face gets a padded ellipse, much faster) and the *safety check*.

A **face card** is one face followed continuously by the tracker (by position, frame to
frame). Every face starts as **not blurred**; you switch on the ones to blur. One person can
have several cards (they leave the frame, turn away, the scene changes): switch on each.
Nothing is ever matched by appearance. For masking, blurred cards that follow each other
in the same place within a second are joined into one track, so short gaps stay covered.

## Desktop app

Double-click `FaceBlur.bat` (or drag a video onto it), or:

```bash
.venv/Scripts/python -m faceblur gui
```

1. **File › Open** a video. Faces are detected once and cached in `work/<video>/`.
2. **Review faces tab:** every face card shows thumbnails across it, all marked *Not
   blurred*. **Click a face to blur it**; click it again to leave it visible. Blurred faces
   get a red frame and a red *BLURRED* tag. *Blur all* / *Blur none* switch every card;
   *Show* lists only the blurred or the visible ones. Keys: ↑/↓ move, Space switch, Enter
   show in the video.
3. **Check the timeline:** the red band shows where faces are blurred (bright = face
   detected, dim = bridged gap, nearby weak detection or drawn area). On the video,
   blurred faces have red boxes and visible faces green ones. Tick *Preview mask* to see it.
4. **Fix mistakes on the video:**
   - click a face box → blur that face or not, or *Don't blur here* (this frame, or while
     the face is followed continuously)
   - drag a rectangle → *Blur this area* (with SAM 2 on, it tracks the area from there)
5. **Render** (Ctrl+R). If *Run safety check after rendering* is ticked, the check runs
   afterwards; frames where a blurred face may still be recognizable are listed and marked
   red on the timeline. Fix them and render again.

Weak detections that belong to no card (very dark, blurred, turned away) are blurred only
where a blurred face was within the last or next second.

Keys on the video (click it first): Space play/pause, ←/→ one frame, Shift+←/→ one second,
Home/End.

Everything you choose is saved automatically (about a second after each change) next to the
video as `<video>.faceblur.json`. Reopening the video or project restores it, with the last
output and its safety report. In projects from earlier versions, faces labeled as a person
are blurred and every other face (someone else, not a face, no label) is not.

## Command line

```bash
.venv/Scripts/python -m faceblur pick input.mp4
```

Open `work/input/faces.jpg` (one numbered row per face card), then list the cards to blur;
every other face stays visible:

```bash
.venv/Scripts/python -m faceblur render input.mp4 --blur 1,2,5
```

A project saved from the app (including its corrections) renders with:

```bash
.venv/Scripts/python -m faceblur render --project input.faceblur.json
```

Output: `input_masked.mp4`. Options: `--style pixelate|blur|fill`, `--strength 1.5`
(coarser), `--pad 0.4` (bigger margin), `--debug` (overlay video in `work/input/debug.mp4`),
`--sam` (exact outlines with SAM 2; new projects use ellipses), `--no-sam`, `--no-check`.

The safety pass runs automatically after rendering (exit code 2 if something is flagged).
To run it again on its own:

```bash
.venv/Scripts/python -m faceblur check input_masked.mp4 --video input.mp4
```

Detections are cached in `work/<name>/detections.npz`; delete it (or use *File › Detect
faces again*) to force a rescan.

## Web app and API (use it from another device)

```bash
.venv/Scripts/python -m faceblur serve
```

(or double-click `FaceBlurServer.bat`; `./FaceBlur.sh serve` on Linux). It listens on port 8765
on all network addresses (`--host 127.0.0.1` for this PC only) and prints the addresses, an
access token (kept in `server_token.txt`) and a sign-in link per address.

**Web app:** open the printed sign-in link (`http://<address>:8765/#token=…`) once in any
browser on your network, a phone's included; it stays signed in. Then: drop or choose a video
(large files upload in parts; if the connection drops, choose the same file again and it
continues) → wait for the face detection → **click the faces to blur** (click again to leave
one visible; *View* shows a face on the video) → set the mask style → **Render** → play or
download the result. Under *Settings* you can turn on SAM 2 and the safety check (both off by
default, per video); with the check on, the result page lists the flagged frames.
Chrome and Edge play the H.265 result in the page; other browsers may only download it.
The web app is in English and Russian: it follows the browser's language, and the switch in
the top bar changes it (remembered per browser).

**API** (for scripts or a native app): every request needs `Authorization: Bearer <token>`;
interactive documentation at `http://<address>:8765/docs` (*Authorize* with the token).
`POST /api/v1/jobs` with the video as the request body (or multipart field `file`), or a
resumable upload (`POST /api/v1/uploads`, then `PATCH /api/v1/jobs/{id}/upload` chunks with
`Upload-Offset`) → poll `GET /api/v1/jobs/{id}` until `status` is `review` → `GET …/cards`
(each with `blur: false` at first; `…/frames/{n}.jpg` for whole frames with numbered boxes)
→ `PATCH …/blur` (`{"cards": {"1": true, "4": true}}`; or `PUT …/blur` with `{"blur": [1, 4]}`
to set exactly those) → `POST …/render` → poll until `done` → `GET …/result` (resumable download) and `GET …/safety`.

Each upload gets its own folder in `jobs/` holding an ordinary project file, so a job can also
be opened in the desktop app (`faceblur gui jobs/<id>/project.faceblur.json`). Jobs untouched
for 7 days are deleted (`--keep-days`). The GPU runs one job at a time; others queue.
A deleted job (by you or by that cleanup) is not lost: its source video, result, project file,
`job.json` (with when and why it was deleted), render log and safety report move to
`history/<date>_<name>_<id>/` next to the jobs folder, for debugging. The history is never
cleaned up automatically; delete old folders by hand. `--history-dir` puts it elsewhere,
`--no-history` deletes jobs completely. Uploads that never completed are not kept.
Keep the server on your home network or a VPN such as Tailscale: it has no HTTPS of its own.
Deleting `server_token.txt` (and restarting) makes a new token and signs every browser out.

Tests: `.venv/Scripts/python -m pytest tests/server` (seconds, no models). The end-to-end
test runs the whole test video `_DSG0777.mp4` (or `FACEBLUR_TEST_VIDEO`) through the real
pipeline on the GPU: `.venv/Scripts/python -m pytest -m e2e tests/server -s` (a few minutes).

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

1. On this PC, build the zip (about 150 KB), or download the newest release from
   <https://github.com/Frogstep/FaceBlur-releases/releases/latest>:

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

## Updates

New versions are published as releases of the public repo
[Frogstep/FaceBlur-releases](https://github.com/Frogstep/FaceBlur-releases), which holds
only the release zips (this source repo stays private).

**Publish (this PC):** commit your changes, then run **`publish.bat`**. It needs the
[GitHub CLI](https://cli.github.com/) (`gh auth login` once). It:
- numbers the version by date (`2026.11.20`, then `2026.11.20.2` for a second release
  that day) and writes it into `faceblur/_version.py` inside the zip only
- builds the zip and a `version.json` (version, SHA-256, notes) in `dist/`
- asks for a one-line note and a confirmation, then uploads both as a release and tags
  the commit `v<version>` locally
- the first time, offers to create the releases repo

`publish.bat --dry-run` only builds the files. Uncommitted changes to the shipped files
(`faceblur/`, `install/`, the scripts, README) stop it, so each release matches a commit.
It also stops when `install/requirements.in` changed after the locks were made (other PCs
would not get the new packages): run `make_dist.bat --lock`, which regenerates
`install/requirements-*.txt` and `install/locks.sha256`, and commit them.

**Update (the other PC):** close FaceBlur and its server, then run **`update.bat`**
(`./update.sh` on Linux). It reads `version.json` of the newest release; if it is newer
than the installed version (shown under **Help** in the app), it downloads the zip,
checks its SHA-256 and replaces the app's own files: `faceblur/`, the scripts, README and
the package locks. Models, `.venv`, projects, `work/`, `jobs/`, `history/`, `out/`,
`server_token.txt` and the app settings are not touched. If the new version needs other
Python packages, the installer runs again by itself (it downloads only what changed); if
only the model list changed, just the model download runs.

- `update.bat --check`: only say whether a newer version exists
- `update.bat FaceBlur-<version>.zip` (or drag the zip onto it): install from a zip
  copied by hand, e.g. on a PC without internet (it still needs the internet if the
  version needs new packages)
- `update.bat --rollback`: go back to the previous version, kept in `update/backup/`
  (projects saved by the newer version may not open in the older one)

An install from a zip made before the updater existed has no `update.bat`: unzip the
new release over its folder once, replacing the files. The updater refuses to run in a
git checkout.

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

## License

FaceBlur's own code is under the [MIT License](LICENSE). The installer downloads
third-party parts that keep their own licenses: the InsightFace face models
(non-commercial research use only), SAM 2.1 (Apache 2.0), FFmpeg (GPL builds) and the
Python packages.
