"""Cutting a video into parts, as in a video editor: split points divide the
source frames [0, n) into parts, and each part is kept or removed. The result is
the project's Edit.keep (the kept ranges). No Qt here: the desktop app and the
web app share it."""

from __future__ import annotations


class CutList:
    def __init__(self, n: int, keep=None):
        self.n = max(1, int(n))
        self.parts: list[list] = [[0, self.n, False]]  # [start, end, removed]
        if keep:
            parts, pos = [], 0
            for a, b in sorted((max(0, int(a)), min(self.n, int(b))) for a, b in keep):
                if b <= max(a, pos):
                    continue
                a = max(a, pos)
                if a > pos:
                    parts.append([pos, a, True])
                parts.append([a, b, False])
                pos = b
            if pos < self.n:
                parts.append([pos, self.n, True])
            if any(not r for _, _, r in parts):
                self.parts = parts

    # ------------------------------------------------------------ state (undo)
    def state(self) -> tuple:
        return tuple(tuple(p) for p in self.parts)

    def restore(self, state: tuple) -> None:
        self.parts = [list(p) for p in state]

    # ------------------------------------------------------------ queries
    def index_at(self, frame: int) -> int:
        frame = min(max(int(frame), 0), self.n - 1)
        for i, (a, b, _) in enumerate(self.parts):
            if a <= frame < b:
                return i
        return len(self.parts) - 1

    def removed_at(self, frame: int) -> bool:
        return self.parts[self.index_at(frame)][2]

    def kept_frames(self) -> int:
        return sum(b - a for a, b, r in self.parts if not r)

    def kept_parts(self) -> int:
        return len(self.keep()) or 1

    def keep(self) -> list[list[int]]:
        """Kept [start, end) ranges with neighbours joined; [] when nothing is cut."""
        out: list[list[int]] = []
        for a, b, r in self.parts:
            if r:
                continue
            if out and out[-1][1] == a:
                out[-1][1] = b
            else:
                out.append([a, b])
        return [] if out == [[0, self.n]] else out

    def split_points(self) -> list[int]:
        return [a for a, _, _ in self.parts[1:]]

    # ------------------------------------------------------------ edits (True when something changed)
    def split(self, frame: int) -> bool:
        """Split the part at `frame` so a new part starts there."""
        frame = int(frame)
        if not 0 < frame < self.n:
            return False
        i = self.index_at(frame)
        a, b, r = self.parts[i]
        if frame == a:
            return False
        self.parts[i:i + 1] = [[a, frame, r], [frame, b, r]]
        return True

    def set_removed(self, i: int, removed: bool) -> bool:
        """Remove or restore part i. Refuses to remove the last kept frames."""
        a, b, r = self.parts[i]
        if r == removed:
            return False
        if removed and self.kept_frames() - (b - a) <= 0:
            return False
        self.parts[i][2] = removed
        return True

    def toggle_at(self, frame: int) -> bool:
        i = self.index_at(frame)
        return self.set_removed(i, not self.parts[i][2])

    def set_in(self, frame: int) -> bool:
        """Start the video at `frame`: everything before it is removed."""
        return self._trim(int(frame), before=True)

    def set_out(self, frame: int) -> bool:
        """End the video with `frame`: everything after it is removed."""
        return self._trim(int(frame) + 1, before=False)

    def _trim(self, at: int, before: bool) -> bool:
        if not 0 <= at <= self.n:
            return False
        old = self.state()
        self.split(at)
        if before:
            outside = [p for p in self.parts if p[1] <= at]
            inside = next((p for p in self.parts if p[0] == at), None)  # the part that now starts the video
        else:
            outside = [p for p in self.parts if p[0] >= at]
            inside = next((p for p in self.parts if p[1] == at), None)  # the part that now ends it
        if inside is None:  # in at the very end / out before the start: nothing would be left
            self.restore(old)
            return False
        inside[2] = False
        for p in outside:
            p[2] = True
        if self.kept_frames() <= 0:
            self.restore(old)
            return False
        return self.state() != old
