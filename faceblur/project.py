"""Project file (<video>.faceblur.json, next to the video).

Stores which faces to blur, the mask settings and the user's manual corrections, so a
project can be reopened, a few frames fixed and the video rendered again
without re-detecting faces (detections are cached in work/<video>/).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

VERSION = 4  # 2: face cards labeled by the user (replaces automatic face groups); 3: cut & crop;
# 4: output format and quality level (cq became an optional override)


@dataclass
class Person:
    """Older projects only (people are no longer used; kept so their files load)."""
    name: str


@dataclass
class Label:
    """A face card the user marked to blur, anchored to the card's first
    detection (frame + box) so it survives reopening the project. Cards without
    a label are not blurred."""
    frame: int
    box: list[float]
    label: int  # 0 = blur (older projects: any person index >= 0 = blur; -1/-2 = not blurred)


@dataclass
class Exclusion:
    """'Don't blur here': a detection of a blurred face that must stay visible."""
    frame: int
    box: list[float]
    scope: str = "track"  # "frame" = only this detection, "track" = its whole track


@dataclass
class Manual:
    """'Blur this area': a user-drawn box, masked for span frames around its frame."""
    frame: int
    box: list[float]
    person: int = 0  # older projects only (ignored)
    span: int = 15  # frames before and after


@dataclass
class Edit:
    """Cut and crop, applied when frames are read (the source file is never
    changed). Face detection and everything after it see only the edited
    video, so frame numbers elsewhere in the project count edited frames."""
    keep: list[list[int]] = field(default_factory=list)  # [start, end) source frames; empty = whole video
    crop: list[int] | None = None  # [x, y, w, h] in display pixels; None = full frame

    @property
    def active(self) -> bool:
        return bool(self.keep or self.crop)


@dataclass
class Settings:
    style: str = "pixelate"  # pixelate / blur / fill
    strength: float = 1.0
    pad: float = 0.3
    use_sam: bool = False  # SAM 2 face outlines (slow); off: ellipses
    codec: str = "h265"  # output format: h265 (smaller files) / h264 (plays everywhere)
    quality: str = "high"  # best / high / medium / small (see video_io.QUALITY)
    cq: int | None = None  # exact CQ/CRF number instead of the level (lower = better)
    safety_check: bool = False  # web server: run the safety check after rendering


def _known(cls, d: dict) -> dict:
    """Keep only the fields cls knows, so files saved by a newer version (or
    another branch) still open; their extra fields are dropped."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


def _settings(d: dict, version: int) -> Settings:
    d = _known(Settings, d)
    if version < 4 and d.get("cq") == 19:
        d.pop("cq")  # the old fixed default, now the "high" level: lets the level follow the format
    return Settings(**d)


@dataclass
class Project:
    video: str
    people: list[Person] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    manual: list[Manual] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
    edit: Edit = field(default_factory=Edit)
    output: str = ""
    path: str = ""  # where this project is saved (not stored in the file)

    @staticmethod
    def default_path(video: str | Path) -> Path:
        v = Path(video)
        return v.with_name(v.stem + ".faceblur.json")

    @classmethod
    def new(cls, video: str | Path) -> "Project":
        v = Path(video).resolve()
        return cls(video=str(v), output=str(v.with_name(v.stem + "_masked.mp4")),
                   path=str(cls.default_path(v)))

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path or self.default_path(self.video))
        self.path = str(p)
        data = asdict(self)
        data.pop("path")
        data["version"] = VERSION
        # store paths relative to the project file when possible, so the
        # project survives moving the folder
        for key in ("video", "output"):
            try:
                data[key] = str(Path(data[key]).resolve().relative_to(p.parent.resolve()))
            except ValueError:
                pass
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        p = Path(path).resolve()
        d = json.loads(p.read_text(encoding="utf-8"))

        def absolute(s: str) -> str:
            q = Path(s)
            return str(q if q.is_absolute() else (p.parent / q).resolve())

        return cls(
            video=absolute(d["video"]),
            people=[Person(name=x["name"]) for x in d.get("people", [])],
            labels=[Label(**_known(Label, x)) for x in d.get("labels", [])],
            exclusions=[Exclusion(**_known(Exclusion, x)) for x in d.get("exclusions", [])],
            manual=[Manual(**_known(Manual, x)) for x in d.get("manual", [])],
            settings=_settings(d.get("settings", {}), d.get("version", 1)),
            edit=Edit(**_known(Edit, d.get("edit", {}))),
            output=absolute(d["output"]) if d.get("output") else "",
            path=str(p),
        )
