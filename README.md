# FaceBlur

Blur the faces you choose in a video, and only those. FaceBlur finds every face, shows each
one as a card, and you click the ones to hide. It follows those faces through the whole video
and writes a new file with them pixelated or blurred, keeping the original sound.

- **You decide who is blurred.** Every face starts visible; nothing is blurred that you did
  not pick.
- **Follows faces through the video**, including short moments where a face is turned away
  or hard to see.
- **Runs on your own computer.** Your videos are never uploaded anywhere.
- **Works with or without a graphics card.** An NVIDIA GPU makes it much faster, but any
  modern PC works.
- **Desktop app, web app for your phone, and command line.** English and Russian.

## Requirements

- Windows 10/11, or 64-bit Linux (Ubuntu 22.04+, Debian 12+, Fedora 35+, RHEL 9+, Arch,
  openSUSE Tumbleweed, or another distribution with glibc 2.34+)
- An internet connection during installation. The installer downloads about 4–5 GB on a PC
  with an NVIDIA GPU, or about 1 GB without one.
- Optional: an NVIDIA graphics card with driver 580 or newer, for GPU speed

## Install

1. Download the newest `FaceBlur-<version>.zip` from
   [the releases page](https://github.com/Frogstep/FaceBlur-releases/releases/latest).
2. Unzip it into the folder where it should stay. Moving the folder later breaks the
   installation; if you do move it, run the installer again.
3. Run the installer:
   - **Windows:** double-click **`install.bat`**
   - **Linux:** run **`./install.sh`** (no root needed)

The installer puts everything inside that folder: Python, the AI models and FFmpeg. It
chooses the GPU or CPU version by itself (`install.bat gpu` or `install.bat cpu` to choose
yourself). At the end it checks that everything works and tells you if the GPU can't be used.
On Linux, it also prints the command to install any system libraries the app window needs.
Running it again skips what is already done and continues unfinished downloads.

FaceBlur's tools (Python, FFmpeg and the installer's helper) are open source and not
code-signed. If Windows' **Smart App Control** is on, it blocks them and the installer says
so. FaceBlur then needs Smart App Control turned off (Windows Security → App & browser
control → Smart App Control settings).

To uninstall, delete the folder.

## Quick start

1. Start **`FaceBlur.bat`** (Windows) or **`./FaceBlur.sh`** (Linux). You can also drag a
   video onto `FaceBlur.bat`.
2. **File › Open** a video. If you only need part of it, cut and crop it first: faces are
   only looked for in what you keep. Then let it detect the faces.
3. In the **Review faces** tab, **click each face you want to blur.** Blurred faces get a red
   frame.
4. Play through the video and fix anything missed (see below).
5. Click **Render** (Ctrl+R). The new video is saved next to the original as
   `<video>_masked.mp4`.

Your choices are saved automatically next to the video as `<video>.faceblur.json`, so you
can close the app and continue later.

## Using the desktop app

**Face cards.** A card is one face that the app followed continuously. The same person can
get several cards, for example after leaving the frame or when the scene changes, so switch
on each of them. *Blur all* and *Blur none* switch every card at once; *Show* lists only the
blurred or only the visible ones. Keys: ↑/↓ move, Space switch, Enter show in the video.

**Checking the video.** Blurred faces have red boxes, visible faces green ones. The red band
on the timeline shows where blurring happens. Tick *Preview mask* to see the result before
rendering.

**Fixing mistakes.** On the video:
- click a face box to blur it or not, or choose *Don't blur here*
- drag a rectangle over a face the app missed and choose *Blur this area*; it asks how long
  to keep blurring it

Keys on the video (click it first): Space play/pause, ←/→ one frame, Shift+←/→ one second,
Home/End.

**Render settings.**
- **Format:** *H.265* (default, about half the file size) or *H.264* (plays on every device
  and in every browser)
- **Quality:** *Best*, *High* (default), *Medium* or *Small file*
- **Style:** *Pixelate*, *Blur* or *Solid black*, how strong, and how much margin around
  each face
- **Precise outline with SAM 2** (off by default): follows the exact shape of the head instead
  of an oval. It looks better but is much slower, especially without a GPU.
- **Safety check** (off by default): after rendering, looks for blurred faces that might still
  be recognizable and marks those frames in red on the timeline, so you can fix them and
  render again.
- **Device:** GPU or CPU (automatic by default)

The language is under **Language**, and the installed version under **Help**.

## Using it from your phone (web app)

Start **`FaceBlurServer.bat`** (Windows) or **`./FaceBlur.sh serve`** (Linux) on the PC. It
prints a sign-in link for each of the PC's network addresses. Open the link once in a
browser on any device in your home network, a phone included; it stays signed in.

Then upload a video, click the faces to blur, and render. You can play or download the
result in the browser. Large uploads continue where they stopped if the connection drops:
choose the same file again. Chrome and Edge can play H.265 videos in the page; other browsers
may only download them.

Keep the server inside your home network or a VPN such as Tailscale: it does not use HTTPS.
To sign every device out, delete `server_token.txt` and restart the server.

Server options: `--port 9000`, `--host 127.0.0.1` (this PC only), `--keep-days 7` (how long
unused uploads are kept). For the HTTP API, see [DEVELOPMENT.md](DEVELOPMENT.md#web-api).

## Command line

On Linux, use `./FaceBlur.sh <command>`. On Windows, use `.venv\Scripts\python -m faceblur
<command>` in the FaceBlur folder.

```bash
./FaceBlur.sh pick input.mp4                # find the faces
./FaceBlur.sh render input.mp4 --blur 1,2,5 # blur cards 1, 2 and 5
```

`pick` writes `work/input/faces.jpg` with one numbered row per face card. Every face you don't
list in `--blur` stays visible. A project saved from the desktop app renders with
`render --project input.faceblur.json`.

| Option | Meaning |
| --- | --- |
| `--style pixelate\|blur\|fill` | how faces are hidden (`fill` = solid black) |
| `--strength 1.5` | stronger pixelation (default 1.0) |
| `--pad 0.4` | bigger margin around faces (default 0.3) |
| `--codec h265\|h264` | output format (default h265) |
| `--quality best\|high\|medium\|small` | output quality (default high) |
| `--sam`, `--sam-model tiny\|small\|base_plus\|large` | precise outlines with SAM 2 |
| `--keep 0:12-1:30,2:05-` | keep only these parts of the video |
| `--crop X,Y,W,H`, `--turn 90` | crop, or turn a video filmed sideways |
| `--any-angle` | also find sideways and upside-down faces (slower) |
| `--no-check` | skip the safety check |
| `--device auto\|gpu\|cpu` | where to process |
| `-o out.mp4` | output file name |

`check input_masked.mp4 --video input.mp4` runs the safety check on its own. `--help` after
any command lists all options.

## Updating

Close FaceBlur and its server, then run **`update.bat`** (`./update.sh` on Linux). It
downloads the newest version and replaces only the app itself: your projects, settings and
models are kept. If the new version needs new packages, it installs them by itself.

- `update.bat --check`: only check whether a newer version exists
- `update.bat FaceBlur-<version>.zip`: install from a zip you downloaded yourself
- `update.bat --rollback`: go back to the previous version

## License

FaceBlur's own code is under the [MIT License](LICENSE). The installer downloads parts that
keep their own licenses:

- **InsightFace face models: for non-commercial research use only.** This means FaceBlur as
  installed is not for commercial use.
- SAM 2.1 (Apache 2.0), FFmpeg (GPL builds) and the Python packages.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setting up a development copy, running the tests,
publishing releases and the HTTP API.
