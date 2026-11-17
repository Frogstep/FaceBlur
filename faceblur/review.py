"""Manual face review: the video's faces as cards that the user labels.

A card is one face followed continuously by the tracker (a ByteTrack track;
position only, no face matching), or a short run of confident detections the
tracker did not link, chained by position. Nothing is merged by appearance:
the user decides who each card is. Cards are only *ordered* by appearance so
that similar faces sit next to each other.

Labels: a person index (0, 1, ...), SOMEONE_ELSE (a real face that must not be
masked) or NOT_A_FACE (a false detection). Weak detections outside any card
are never labeled; they are masked only by spatial continuity with a labeled
person (see identity.plan_labeled).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identity import Detections, _box_iou, _norm

UNLABELED = -3
NOT_A_FACE = -2
SOMEONE_ELSE = -1

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


def person_refs(dets: Detections, det_idx: np.ndarray) -> np.ndarray | None:
    """Reference embeddings of a labeled person (for the safety check only)."""
    from .identity import select_refs

    if not len(det_idx):
        return None
    return select_refs(dets, det_idx)
