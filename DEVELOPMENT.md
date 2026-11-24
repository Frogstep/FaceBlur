# FaceBlur development

Notes for working on FaceBlur itself. For using the app, see the [README](README.md).

## How it works

FFmpeg decode → low-light enhancement (on the detection copy only) → SCRFD face detection +
ByteTrack face tracks → the user picks the face cards to blur → masks (gap filling,
continuity; padded ellipses, or exact outlines with SAM 2.1) → pixelate / blur / fill →
H.265 or H.264 encode with the audio copied → optional safety pass (looks for faces that
are still recognizable in the result).

A **face card** is one face followed continuously by the tracker, by position from frame to
frame. Nothing is matched by appearance. For masking, blurred cards that follow each other
in the same place within a second are joined into one track, so short gaps stay covered.
Weak detections that belong to no card (very dark, blurred, turned away) are blurred only
where a blurred face was within the last or next second.

Projects from earlier versions, which labeled faces as a person, load with the faces
labeled as a person blurred and every other face (someone else, not a face, no label) not.

## Setup

Windows, Python 3.12, NVIDIA GPU:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m faceblur.setup_models --all --no-ffmpeg
```

Models in `models/`: `det_10g.onnx` and `w600k_r50.onnx` (InsightFace buffalo_l,
non-commercial license), `sam2.1_hiera_large.pt`, `sam2.1_hiera_tiny.pt` and, when chosen,
`sam2.1_hiera_small.pt` / `sam2.1_hiera_base_plus.pt` (Meta, Apache 2.0).

Check an installation with `.venv/Scripts/python -m faceblur.setup_models --check`
(`.venv/bin/python` on Linux).

## Tests

```bash
.venv/Scripts/python -m pytest
```

`tests/server` alone takes seconds and needs no models. Tests that need a real video use
`_DSG0777.mp4` in the project folder (git-ignored) or the file named by
`FACEBLUR_TEST_VIDEO`, and are skipped without one. The end-to-end test runs the whole
video through the real pipeline on the GPU (a few minutes):

```bash
.venv/Scripts/python -m pytest -m e2e tests/server -s
```

CI (`.github/workflows/tests.yml`) runs the tests on Linux with the CPU packages.

## Package locks

Package versions are pinned in `install/requirements-<windows|linux>-<gpu|cpu>.txt`. After
changing `install/requirements.in`, regenerate them and `install/locks.sha256`, and commit
them:

```bash
.venv/Scripts/python install/make_dist.py --lock
```

(or `make_dist.bat --lock`).

## Building and publishing a release

`install/make_dist.py` (or `make_dist.bat`) builds `dist/FaceBlur-<date>.zip` (about 150 KB):
the app's code, the scripts, README, LICENSE and the package locks. Nothing heavy is in it;
the installer downloads Python, PyTorch, the models and FFmpeg on the target PC.

Releases go to the public repo
[Frogstep/FaceBlur-releases](https://github.com/Frogstep/FaceBlur-releases)
(`RELEASES_REPO` in `install/update.py`), which holds only the release zips. To publish,
commit your changes, then run **`publish.bat`**. It needs the
[GitHub CLI](https://cli.github.com/) (`gh auth login` once). It:

- numbers the version by date (`2026.11.20`, then `2026.11.20.2` for a second release
  that day) and writes it into `faceblur/_version.py` inside the zip only
- builds the zip and a `version.json` (version, SHA-256, notes) in `dist/`
- asks for a one-line note and a confirmation, then uploads both as a release and tags
  the commit `v<version>` locally
- the first time, offers to create the releases repo

`publish.bat --dry-run` only builds the files. Uncommitted changes to the shipped files
(`faceblur/`, `install/`, the scripts, README, DEVELOPMENT.md, LICENSE) stop it, so each release matches a
commit. It also stops when `install/requirements.in` changed after the locks were made.

The updater (`update.bat` / `update.sh`) refuses to run in a git checkout. An install from a
zip made before the updater existed has no `update.bat`: unzip a new release over its
folder once.

## Web API

`faceblur serve` starts the web app and an HTTP API on port 8765. Every request needs
`Authorization: Bearer <token>` (the token is in `server_token.txt`); interactive
documentation is at `http://<address>:8765/docs` (*Authorize* with the token).

1. Upload: `POST /api/v1/jobs` with the video as the request body (or multipart field
   `file`), or a resumable upload: `POST /api/v1/uploads`, then
   `PATCH /api/v1/jobs/{id}/upload` chunks with `Upload-Offset`.
2. Poll `GET /api/v1/jobs/{id}` until `status` is `review`.
3. `GET …/cards` lists the face cards, each with `blur: false` at first; `…/frames/{n}.jpg`
   gives whole frames with numbered boxes.
4. Choose: `PATCH …/blur` with `{"cards": {"1": true, "4": true}}`, or `PUT …/blur` with
   `{"blur": [1, 4]}` to set exactly those.
5. `PATCH …/settings` sets the mask style, SAM 2, the safety check and the output `codec`
   (`h265`/`h264`), `quality` (`best`/`high`/`medium`/`small`) or an exact `cq`. The web
   app does not offer the output settings yet, so its results are H.265 at *High*.
6. `POST …/render`, poll until `done`, then `GET …/result` (resumable download) and
   `GET …/safety`.

Each upload gets its own folder in `jobs/` holding an ordinary project file, so a job can
also be opened in the desktop app (`faceblur gui jobs/<id>/project.faceblur.json`). The GPU
runs one job at a time; others queue.

Jobs untouched for 7 days are deleted (`--keep-days`). A deleted job is not lost: its source
video, result, project file, `job.json` (with when and why it was deleted), render log and
safety report move to `history/<date>_<name>_<id>/` next to the jobs folder, for debugging.
The history is never cleaned up automatically. `--history-dir` puts it elsewhere,
`--no-history` deletes jobs completely. Uploads that never completed are not kept.

## Performance notes

On an RTX 5060 Ti, SAM 2.1 tiny, small and base+ render about 1.5–1.7× faster than large;
small is a good everyday choice. On the CPU, SAM 2 is slow even with the tiny model.
Detections cached on one machine are valid on another.

| Stage | GPU | CPU fallback |
| --- | --- | --- |
| Decode | FFmpeg NVDEC | FFmpeg software |
| Face detection + matching | onnxruntime CUDA | onnxruntime CPU |
| Noise reduction (dark frames) | CPU, in parallel worker threads | same |
| SAM 2 | large model, CUDA | tiny model, CPU |
| Encode | `hevc_nvenc` / `h264_nvenc` | `libx265` / `libx264` |
