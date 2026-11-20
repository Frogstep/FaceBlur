"""Command line interface.

  python -m faceblur gui    [VIDEO | PROJECT]   open the desktop app
  python -m faceblur pick   VIDEO               scan faces, write the numbered face-card sheet
  python -m faceblur render VIDEO --blur 1,3    blur the faces on cards 1 and 3 (the rest stay visible)
  python -m faceblur render --project P.faceblur.json   render a project saved from the app
  pick and render also take --keep 0:12-1:30,2:05- and --crop X,Y,W,H (cut & crop first)
  python -m faceblur check  MASKED --video SRC  safety pass on an already rendered file
  python -m faceblur serve  [--port 8765]       HTTP API for other devices (e.g. a phone app)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class _Bar:
    """Console progress for pipeline progress(stage, done, total) callbacks."""

    def __init__(self):
        self.bar = None
        self.stage = None

    def __call__(self, stage: str, done: int, total: int) -> None:
        if stage != self.stage:
            self.close()
            self.stage = stage
            if total:
                self.bar = tqdm(total=total, desc=stage, unit="f")
            else:
                print(f"{stage}...")
        if self.bar is not None:
            self.bar.update(done - self.bar.n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.update(self.bar.total - self.bar.n)
            self.bar.close()
            self.bar = None


def _open(video: Path, edit=None):
    from .pipeline import Analysis, detections_cache, get_analyzer, open_video, workdir

    from .device import summary

    get_analyzer()
    print(summary())
    info = open_video(video, edit)
    if info.edited:
        print(f"edited video: {info.num_frames} frames ({info.duration:.1f}s), {info.width}x{info.height}")
    if detections_cache(workdir(video), info).exists():
        print(f"using cached detections in work/{video.stem}/")
    bar = _Bar()
    analysis = Analysis.open(video, progress=bar, edit=edit)
    bar.close()
    if analysis.info.vfr:
        print(f"  warning: variable frame rate source; output uses a constant "
              f"{analysis.info.fps_float:.3f} fps")
    return analysis


# ---------------------------------------------------------------- cut & crop

def parse_time(text: str) -> float:
    """Seconds from "90", "1:30", "1:30.5" or "1:02:03"."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(not x for x in parts):
        raise ValueError(f"not a time: {text!r}")
    secs = 0.0
    for x in parts:
        secs = secs * 60 + float(x)
    if secs < 0:
        raise ValueError(f"not a time: {text!r}")
    return secs


def parse_keep(spec: str, fps: float, frames: int) -> list[list[int]]:
    """"0:12-1:30,2:05-" -> [start, end) source frame ranges (an empty start is
    the beginning of the video, an empty end its end)."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if part.count("-") != 1:
            raise ValueError(f"a kept part is START-END, e.g. 0:12-1:30 (got {part!r})")
        a, b = part.split("-")
        start = round(parse_time(a) * fps) if a else 0
        end = round(parse_time(b) * fps) if b else frames
        if end <= start:
            raise ValueError(f"the part {part!r} ends before it starts")
        out.append([start, end])
    if not out:
        raise ValueError("--keep lists no parts")
    return out


def parse_crop(spec: str) -> list[int]:
    try:
        v = [int(round(float(x))) for x in spec.replace(" ", "").split(",")]
    except ValueError:
        v = []
    if len(v) != 4 or v[2] <= 0 or v[3] <= 0:
        raise ValueError(f"--crop is X,Y,WIDTH,HEIGHT in pixels, e.g. 0,420,1080,1080 (got {spec!r})")
    return v


def _edit_from_args(args, video: Path):
    """The cut & crop given with --keep / --crop: (Edit, None) or (None, error)."""
    from .project import Edit
    from .video_io import probe

    if not (args.keep or args.crop):
        return Edit(), None
    info = probe(video)
    try:
        edit = Edit(keep=parse_keep(args.keep, info.fps_float, info.num_frames) if args.keep else [],
                    crop=parse_crop(args.crop) if args.crop else None)
        info.with_edit(edit.keep, edit.crop)  # checks it leaves something
    except ValueError as e:
        return None, str(e)
    return edit, None


def _edit_flags(args) -> str:
    out = ""
    if args.keep:
        out += f" --keep {args.keep}"
    if args.crop:
        out += f" --crop {args.crop}"
    return out


# ---------------------------------------------------------------------- pick

def contact_sheet(analysis, out: Path, thumb: int = 96) -> None:
    """One row per face card: number, time range and 5 thumbnails across the card."""
    from .pipeline import card_thumbnails

    thumbs = card_thumbnails(analysis, thumb)
    dets, fps = analysis.dets, analysis.info.fps_float
    label_w, k = 190, 5
    rows = []
    for c in analysis.cards:
        row = np.full((thumb, label_w + k * (thumb + 4), 3), 32, np.uint8)
        fr = dets.frame[c.det_idx]
        cv2.putText(row, f"#{c.number}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(row, f"{fr.min() / fps:.1f}-{fr.max() / fps:.1f}s", (8, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(row, f"{len(c.det_idx)} frames", (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        for j, i in enumerate(c.samples):
            x = label_w + j * (thumb + 4)
            row[:, x:x + thumb] = thumbs[i]
        rows.append(np.vstack([row, np.full((4, row.shape[1], 3), 20, np.uint8)]))
    cv2.imwrite(str(out), np.vstack(rows) if rows else np.zeros((10, 10, 3), np.uint8))


def cmd_pick(args) -> int:
    video = Path(args.video).resolve()
    edit, err = _edit_from_args(args, video)
    if err:
        print(err)
        return 1
    analysis = _open(video, edit)
    if not analysis.cards:
        print("no faces found")
        return 1
    sheet = analysis.wd / "faces.jpg"
    contact_sheet(analysis, sheet)
    fps = analysis.info.fps_float
    print(f"\n{len(analysis.cards)} face cards. Sheet: {sheet}")
    print("Each card is one face followed continuously; similar faces are listed next to each other.")
    print("Every face stays visible unless you list its card to blur.")
    for c in analysis.cards:
        fr = analysis.dets.frame[c.det_idx]
        print(f"  #{c.number}: {fr.min() / fps:.1f}s - {fr.max() / fps:.1f}s, {len(c.det_idx)} frames")
    print(f"\nNext, choose the cards to blur, e.g.:")
    print(f"  python -m faceblur render {args.video}{_edit_flags(args)} --blur 1,2,5")
    if edit.active:
        print("(give render the same --keep / --crop: card numbers belong to this cut)")
    return 0


def _card_list(spec: str) -> list[int]:
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def _blur_from_args(project, analysis, args) -> str | None:
    """Mark the cards given with --blur. Returns an error or None."""
    by_number = {c.number: k for k, c in enumerate(analysis.cards)}
    try:
        numbers = {n for spec in args.blur or [] for n in _card_list(spec)}
    except ValueError:
        return "card lists are numbers separated by commas, e.g. --blur 1,2,5"
    unknown = sorted(n for n in numbers if n not in by_number)
    if unknown:
        return f"no card #{', #'.join(map(str, unknown))} (this video has {len(analysis.cards)} cards)"
    analysis.set_blur(project, [by_number[n] for n in sorted(numbers)], True)
    return None


# -------------------------------------------------------------------- render

def cmd_render(args) -> int:
    from .pipeline import render
    from .project import Project

    if args.project:
        if not Path(args.project).is_file():
            print(f"project file not found: {args.project}")
            print("A project is created when you open the video in the app and choose faces to blur")
            print("(it is saved automatically next to the video). Or choose them here: run")
            print("`python -m faceblur pick VIDEO`, then `render VIDEO --blur ...`.")
            return 1
        if args.blur:
            print("use either --project or --blur, not both")
            return 1
        if args.keep or args.crop:
            print("--keep / --crop apply to a new project; a saved project keeps its own cut & crop")
            return 1
        project = Project.load(args.project)
        print(f"project: {args.project} ({len(project.exclusions)} exclusions, {len(project.manual)} drawn boxes)")
    elif args.video and args.blur:
        project = Project.new(args.video)
        edit, err = _edit_from_args(args, Path(project.video))
        if err:
            print(err)
            return 1
        project.edit = edit
    else:
        print("give a VIDEO with --blur CARDS (see `pick`), or --project FILE")
        return 1
    if args.output:
        project.output = str(Path(args.output).resolve())
    s = project.settings
    for key in ("style", "strength", "pad", "codec", "cq"):
        if getattr(args, key) is not None:
            setattr(s, key, getattr(args, key))
    if args.quality is not None:
        s.quality = args.quality
        if args.cq is None:
            s.cq = None  # a level replaces an exact number saved in the project
    if args.sam:
        s.use_sam = True
    if args.no_sam:
        s.use_sam = False

    analysis = _open(Path(project.video), project.edit)
    if not args.project:
        err = _blur_from_args(project, analysis, args)
        if err:
            print(err)
            return 1
    blurred = [c.number for c, b in zip(analysis.cards, analysis.blurred(project)) if b]
    print(f"blurring face cards: {', '.join('#' + str(n) for n in blurred) or 'none'}")
    if not blurred and not project.manual:
        print("no face is marked to blur")
        return 1

    bar = _Bar()
    result = render(project, analysis, progress=bar, debug=args.debug, log=print)
    bar.close()
    print(f"\nwrote {result.output}  ({result.frames} frames, {result.fps:.1f} fps)")
    print(f"SAM 2 status: {result.sam_status}")
    print(f"frames with no mask at all: {result.unmasked}")
    if args.debug:
        print(f"debug overlay: {analysis.wd / 'debug.mp4'}")
    if args.no_check:
        return 0
    from .pipeline import load_target_refs

    return _check(result.output, load_target_refs(analysis.wd), analysis.wd)


# --------------------------------------------------------------------- check

def _check(masked: Path, refs_list, wd: Path) -> int:
    from .pipeline import safety_check

    bar = _Bar()
    report = safety_check(masked, refs_list, wd, progress=bar)
    bar.close()
    flagged = report["flagged"]
    if flagged:
        frames = [x["frame"] for x in flagged]
        print(f"\nSAFETY: target face still recognizable on {len(flagged)} frame(s): "
              f"{frames[:20]}{' ...' if len(frames) > 20 else ''}")
        print(f"  see {wd / 'flagged'} and {wd / 'safety_report.json'}")
        return 2
    print("\nSAFETY: target face not recognizable on any output frame")
    return 0


def cmd_check(args) -> int:
    from .pipeline import load_target_refs, workdir

    wd = workdir(Path(args.video).resolve())
    refs = load_target_refs(wd)
    if not refs:
        print("no target recorded for this video; run render first")
        return 1
    return _check(Path(args.masked).resolve(), refs, wd)


def cmd_serve(args) -> int:
    try:
        from .server.app import serve
    except ImportError as e:
        print(f"the server needs FastAPI and uvicorn ({e.name} is missing): run the installer again, or\n"
              f"  pip install fastapi uvicorn python-multipart")
        return 1
    root = Path(__file__).resolve().parent.parent
    jobs = Path(args.jobs_dir or root / "jobs").resolve()
    # next to the jobs folder by default, so moving a deleted job there is a quick rename
    history = None if args.no_history else Path(args.history_dir or jobs.parent / "history")
    return serve(args.host, args.port, jobs, Path(args.token_file or root / "server_token.txt"),
                 args.keep_days, args.max_upload_gb, history)


def cmd_gui(args) -> int:
    from .gui.app import run

    return run(args.path)


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: a subcommand must not reset a --device given before it
    common.add_argument("--device", choices=["auto", "gpu", "cpu"], default=argparse.SUPPRESS,
                        help="where to process: auto (NVIDIA GPU if it works, else CPU; default), gpu, cpu")
    ap = argparse.ArgumentParser(prog="faceblur", description="Mask chosen faces through a video.",
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", parents=[common], help="open the desktop app")
    g.add_argument("path", nargs="?", help="video or .faceblur.json project to open")
    g.set_defaults(fn=cmd_gui)

    edit = argparse.ArgumentParser(add_help=False)
    edit.add_argument("--keep", metavar="RANGES",
                      help="cut: keep only these parts, e.g. 0:12-1:30,2:05- (times as [h:]m:s or seconds; "
                           "an open end means the end of the video). Faces are only looked for there.")
    edit.add_argument("--crop", metavar="X,Y,W,H",
                      help="crop to this rectangle in pixels (rounded to even numbers), e.g. 0,420,1080,1080")

    p = sub.add_parser("pick", parents=[common, edit], help="find faces and write the numbered face-card sheet")
    p.add_argument("video")
    p.set_defaults(fn=cmd_pick)

    r = sub.add_parser("render", parents=[common, edit], help="blur the chosen faces and encode H.265 or H.264")
    r.add_argument("video", nargs="?")
    r.add_argument("--blur", "--face", dest="blur", action="append", metavar="N[,N...]",
                   help="card numbers (from `pick`) to blur, e.g. --blur 1,2,5; every other face stays visible")
    r.add_argument("--project", help="render a .faceblur.json project saved from the app")
    r.add_argument("-o", "--output")
    r.add_argument("--style", choices=["pixelate", "blur", "fill"])
    r.add_argument("--strength", type=float, help="higher = coarser pixelation (default 1.0)")
    r.add_argument("--pad", type=float, help="extra margin around the face box (default 0.3)")
    r.add_argument("--codec", choices=["h265", "h264"],
                   help="output format: h265 = smaller file (default), h264 = plays everywhere")
    r.add_argument("--quality", choices=["best", "high", "medium", "small"],
                   help="best = biggest file, small = smallest (default high)")
    r.add_argument("--cq", type=int, choices=range(0, 52), metavar="0-51",
                   help="exact encoder quality instead of --quality (NVENC CQ / x264 CRF; lower = better)")
    r.add_argument("--sam", action="store_true", help="precise face outlines with SAM 2 (slower; off by default)")
    r.add_argument("--no-sam", action="store_true", help="box/ellipse masking only (the default for new projects)")
    r.add_argument("--debug", action="store_true", help="also write work/<name>/debug.mp4 overlay")
    r.add_argument("--no-check", action="store_true", help="skip the safety pass")
    r.set_defaults(fn=cmd_render)

    c = sub.add_parser("check", parents=[common], help="safety pass on a rendered file")
    c.add_argument("masked")
    c.add_argument("--video", required=True, help="the original video (to find the target)")
    c.set_defaults(fn=cmd_check)

    sv = sub.add_parser("serve", parents=[common], help="HTTP API for other devices (e.g. a phone app)")
    sv.add_argument("--host", default="0.0.0.0",
                    help="address to listen on (default 0.0.0.0 = all; 127.0.0.1 = this PC only)")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--jobs-dir", help="where uploads and results are kept (default jobs/)")
    sv.add_argument("--token-file", help="access token file, created on first start (default server_token.txt)")
    sv.add_argument("--keep-days", type=float, default=7.0,
                    help="delete jobs untouched for this many days (0 = never; default 7)")
    sv.add_argument("--max-upload-gb", type=float, default=8.0, help="largest accepted video (default 8 GB)")
    sv.add_argument("--history-dir",
                    help="where a deleted job's source video, result and records are moved, for debugging "
                         "(default: history/ next to the jobs folder; never cleaned up automatically)")
    sv.add_argument("--no-history", action="store_true", help="delete jobs completely instead")
    sv.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    if getattr(args, "device", None):
        from .device import set_device

        set_device(args.device)
    if args.cmd is None:  # no arguments: open the app
        return cmd_gui(argparse.Namespace(path=None))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
