"""Project file (<video>.faceblur.json, next to the video).

Stores who to mask, the mask settings and the user's manual corrections, so a
project can be reopened, a few frames fixed and the video rendered again
without re-detecting faces (detections are cached in work/<video>/).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

VERSION = 2  # 2: face cards labeled by the user (replaces automatic face groups)


@dataclass
class Person:
    name: str


@dataclass
class Label:
    """The user's label for one face card, anchored to the card's first
    detection (frame + box) so it survives reopening the project."""
    frame: int
    box: list[float]
    label: int  # person index, -1 = someone else (not masked), -2 = not a face


@dataclass
class Exclusion:
    """'Not this person': a detection that must not be masked as a target."""
    frame: int
    box: list[float]
    scope: str = "track"  # "frame" = only this detection, "track" = its whole track


@dataclass
class Manual:
    """'Mask here too': a user-drawn box for a person."""
    frame: int
    box: list[float]
    person: int = 0
    span: int = 15  # frames before and after


@dataclass
class Settings:
    style: str = "pixelate"  # pixelate / blur / fill
    strength: float = 1.0
    pad: float = 0.3
    use_sam: bool = True
    cq: int = 19  # NVENC quality (lower = better)


@dataclass
class Project:
    video: str
    people: list[Person] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    exclusions: list[Exclusion] = field(default_factory=list)
    manual: list[Manual] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)
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
            labels=[Label(**x) for x in d.get("labels", [])],
            exclusions=[Exclusion(**x) for x in d.get("exclusions", [])],
            manual=[Manual(**x) for x in d.get("manual", [])],
            settings=Settings(**d.get("settings", {})),
            output=absolute(d["output"]) if d.get("output") else "",
            path=str(p),
        )
