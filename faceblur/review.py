"""Manual face review: the video's faces as cards that the user labels.

A card is one face followed continuously by the tracker (a ByteTrack track;
position only, no face matching), or a short run of confident detections the
tracker did not link, chained by position. Nothing is merged by appearance:
the user decides who each card is. Cards are only *ordered* by appearance so
that similar faces sit next to each other.

The user's only choice per card is blur or not: a card is blurred when its
label is BLUR (any label >= 0; older projects stored person indices there),
and every other card, labeled or not, stays visible. Weak detections outside
any card are never labeled; they are masked only by spatial continuity with a
blurred face (see identity.plan_labeled).

For masking, blurred cards are linked into tracks (chain_cards) by position
and time only, like the tracker itself: a card that starts where and shortly
after another one ended continues it, so gaps between them are bridged.
Nothing is ever joined by appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identity import Detections, _box_iou, _norm

UNLABELED = -3
NOT_A_FACE = -2  # older projects only: now simply "not blurred"
SOMEONE_ELSE = -1  # older projects only: now simply "not blurred"
BLUR = 0


def is_blur(label: int) -> bool:
    return label >= 0

CONFIDENT = 0.5  # untracked detections at or above this score get a card
CHAIN_GAP = 15  # frames: untracked detections this close in time and overlapping are one card


@dataclass
class Card:
    number: int  # 1-based, shown to the user
    det_idx: np.ndarray  # detection indices, sorted by frame
    track: int  # tracker id, 0 for a chain of untracked detections
    samples: list[int]  # detection indices to show as thumbnails
    emb: np.ndarray  # quality-weighted mean embedding (for ordering only)

    @property
    def anchor(self) -> int:
        """Detection used to find this card again (its first one)."""
        return int(self.det_idx[0])


def _samples(dets: Detections, idx: np.ndarray, k: int = 5) -> list[int]:
    """k detections spread over the card; the best-quality one in each segment."""
    q = dets.quality()
    if len(idx) <= k:
        return [int(i) for i in idx]
    out = []
    for seg in np.array_split(idx, k):
        out.append(int(seg[np.argmax(q[seg])]))
    return out


def _order(cards: list[Card]) -> list[Card]:
    """Similar-looking cards next to each other (hierarchical clustering leaf
    order), starting with the biggest group."""
    if len(cards) < 3:
        return sorted(cards, key=lambda c: -len(c.det_idx))
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    e = np.array([c.emb for c in cards])
    d = np.clip(pdist(e, "cosine"), 0, 2)
    z = optimal_leaf_ordering(linkage(d, "average"), d)
    return [cards[i] for i in leaves_list(z)]


def build_cards(dets: Detections) -> list[Card]:
    q = dets.quality()
    groups: list[tuple[int, np.ndarray]] = []
    for t in np.unique(dets.track):
        if t != 0:
            idx = np.where(dets.track == t)[0]
            groups.append((int(t), idx[np.argsort(dets.frame[idx], kind="stable")]))

    # chains of confident untracked detections
    loose = np.where((dets.track == 0) & (dets.score >= CONFIDENT))[0]
    loose = loose[np.argsort(dets.frame[loose], kind="stable")]
    chains: list[list[int]] = []
    for i in loose:
        for ch in chains:
            last = ch[-1]
            if 0 < dets.frame[i] - dets.frame[last] <= CHAIN_GAP and _box_iou(dets.box[i], dets.box[last]) > 0.3:
                ch.append(int(i))
                break
        else:
            chains.append([int(i)])
    groups += [(0, np.array(ch)) for ch in chains]

    cards = []
    for t, idx in groups:
        emb = _norm((dets.emb[idx] * q[idx, None]).sum(0))
        cards.append(Card(0, idx, t, _samples(dets, idx), emb))
    cards = _order(cards)
    for n, c in enumerate(cards, 1):
        c.number = n
    return cards


def card_of_detection(dets: Detections, cards: list[Card]) -> np.ndarray:
    """Card position (index into cards) per detection, -1 if in no card."""
    out = np.full(len(dets.frame), -1, np.int64)
    for k, c in enumerate(cards):
        out[c.det_idx] = k
    return out


def detection_labels(dets: Detections, cards: list[Card], card_labels: list[int]) -> np.ndarray:
    """Label per detection: the card's label, or UNLABELED for detections in no card."""
    out = np.full(len(dets.frame), UNLABELED, np.int64)
    for c, lab in zip(cards, card_labels):
        out[c.det_idx] = lab
    return out


def chain_cards(dets: Detections, cards: list[Card], positions: list[int], fps: float,
                max_gap_s: float = 1.0) -> list[list[int]]:
    """Link cards (positions in `cards`) into masking tracks: a card joins the
    track whose last face ended at most max_gap_s before it starts, in about the
    same place (the tracker lost the face briefly). Cards that overlap in time
    are never linked, so two faces on screen together are always two tracks."""
    max_gap = max(1, int(round(max_gap_s * fps)))
    tracks: list[list[int]] = []
    ends: list[int] = []  # last detection of each track
    for k in sorted(positions, key=lambda k: int(dets.frame[cards[k].det_idx[0]])):
        first = int(cards[k].det_idx[0])
        f0 = int(dets.frame[first])
        best, best_gap = None, None
        for t, last in enumerate(ends):
            gap = f0 - int(dets.frame[last])
            if 0 < gap <= max_gap and _near(dets.box[last], dets.box[first]):
                if best_gap is None or gap < best_gap:
                    best, best_gap = t, gap
        if best is None:
            tracks.append([k])
            ends.append(int(cards[k].det_idx[-1]))
        else:
            tracks[best].append(k)
            ends[best] = int(cards[k].det_idx[-1])
    return tracks


def _near(a, b) -> bool:
    """Same place: overlapping boxes, or centres less than a face size apart."""
    if _box_iou(a, b) > 0.1:
        return True
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    d = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    return bool(d < size)


def person_refs(dets: Detections, det_idx: np.ndarray) -> np.ndarray | None:
    """Reference embeddings of a labeled person (for the safety check only)."""
    from .identity import select_refs

    if not len(det_idx):
        return None
    return select_refs(dets, det_idx)
