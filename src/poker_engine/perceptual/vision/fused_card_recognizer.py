"""Fused-glyph card recognizer (capture-card platform, stage I v3).

Single-frame corner-glyph matching cannot separate same-colour suits at
53x78 on this platform, and naive temporal averaging blurs shapes (an
aspect-fit letterbox jitters every glyph a few px per frame, so a club's
three lobes smear into a spade). This recognizer is the measured fix:

1. **Glyph normalization** — the tight glyph crop is resized to a fixed
   height (aspect preserved) and anchored by its ink centroid, so every
   frame's glyph is registered before averaging;
2. **Temporal fusion** — per-slot glyphs accumulate while BOTH gates
   pass: the street gate (board strip signature, same street) and the
   slot gate (the slot's own card region — at PRE_FLOP an empty board is
   indistinguishable between two hands, so only the slot signature stops
   the next hand's glyphs bleeding in; measured: an 8S fused into a K
   ghost without it). Phase correlation removes residual sub-pixel
   jitter before the float mean;
3. **Colour router** — suit colour family from the BGR (R-B) ink
   statistic (HSV hue is undefined for black ink: its noise lands in the
   red range). The corpus distribution is perfectly bimodal
   (black = 0, red >= 35), so a single threshold at 30 is exact;
4. **MLP heads** — tiny single-hidden-layer networks (rank 13-class
   softmax, red/black family 2-class logistic) exported to
   ``card_heads.npz`` (see ``card_heads.json`` meta) and evaluated in
   numpy (no sklearn at runtime).

Failure-closed: a card is only reported when the fusion has enough
gated glyphs AND both margins clear the calibrated floors
(``suit_floor``/``rank_floor`` from the locked-split calibration, see the
platform ``calibration.json``); anything else is UNKNOWN. Evidence
(private dataset ``evidence/field_metrics.json``): calibration 62/62,
locked validation 116/116, zero false VALID for the full card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card

from .corner_glyph_recognizer import (
    DEFAULT_GEOMETRY,
    CornerGlyphGeometry,
    isolate_glyph,
    locate_card_face,
)
from .protocols import CardRecognition, CardSlotResult

__all__ = [
    "FusedCardRecognizer",
    "FusedSlotBuffer",
    "GlyphNormalizer",
    "MlpHead",
    "ink_red_stat",
    "load_card_heads",
]

NORM = (40, 40)
GLYPH_H = 26
RED_FLOOR = 30.0
_RANK_MAP = {
    "A": Rank.ACE, "K": Rank.KING, "Q": Rank.QUEEN, "J": Rank.JACK,
    "T": Rank.TEN, "9": Rank.NINE, "8": Rank.EIGHT, "7": Rank.SEVEN,
    "6": Rank.SIX, "5": Rank.FIVE, "4": Rank.FOUR, "3": Rank.THREE,
    "2": Rank.TWO,
}
_SUIT_MAP = {
    "S": Suit.SPADES, "H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS,
}
RED_SUITS = frozenset(("H", "D"))
BLACK_SUITS = frozenset(("S", "C"))


class GlyphNormalizer:
    """Fixed-height + ink-centroid glyph normalization (deterministic)."""

    @staticmethod
    def normalize(
        glyph: np.ndarray,
        geometry: CornerGlyphGeometry = DEFAULT_GEOMETRY,
    ) -> np.ndarray | None:
        del geometry  # normalization is geometry-independent
        import cv2

        if glyph is None or glyph.size == 0:
            return None
        gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY) if glyph.ndim == 3 else glyph
        h, w = gray.shape
        if h < 3 or w < 2:
            return None
        scale = GLYPH_H / h
        nw = max(1, int(round(w * scale)))
        if nw > NORM[0] - 4:
            return None  # absurdly wide component: not a glyph
        gray = cv2.resize(gray, (nw, GLYPH_H), interpolation=cv2.INTER_AREA)
        h, w = gray.shape
        mass = 255.0 - gray.astype(np.float32)
        total = float(mass.sum())
        if total < 50.0:
            return None
        cx = float((mass * np.arange(w, dtype=np.float32)).sum() / total)
        cy = float((mass * np.arange(h, dtype=np.float32)[:, None]).sum() / total)
        canvas = np.full(NORM, 255, dtype=np.float32)
        ox = int(round(NORM[0] / 2 - cx))
        oy = int(round(NORM[1] / 2 - cy))
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        cw = min(w - sx0, NORM[0] - dx0)
        ch = min(h - sy0, NORM[1] - dy0)
        if cw <= 0 or ch <= 0:
            return None
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = (
            gray[sy0:sy0 + ch, sx0:sx0 + cw].astype(np.float32)
        )
        return canvas

    @staticmethod
    def from_card(
        card_image: np.ndarray,
        kind: str,
        geometry: CornerGlyphGeometry = DEFAULT_GEOMETRY,
    ) -> np.ndarray | None:
        """Normalize the rank/suit corner glyph of one located card face."""
        band = geometry.rank_band if kind == "rank" else geometry.suit_band
        glyph = isolate_glyph(card_image, band, geometry)
        if glyph is None:
            return None
        return GlyphNormalizer.normalize(glyph, geometry)


def ink_red_stat(
    card_image: np.ndarray,
    geometry: CornerGlyphGeometry = DEFAULT_GEOMETRY,
) -> float | None:
    """Median (R-B) over the suit-band ink pixels; None when unreadable.

    Red ink has R >> B, black ink R ~= B. Corpus: black = 0, red >= 35.
    """
    import cv2

    glyph = isolate_glyph(card_image, geometry.suit_band, geometry)
    if glyph is None or glyph.ndim != 3:
        return None
    gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    ink = gray < 160
    if int(ink.sum()) < 5:
        return None
    b = glyph[..., 0].astype(np.int16)
    r = glyph[..., 2].astype(np.int16)
    return float(np.median((r - b)[ink]))


@dataclass(frozen=True)
class MlpHead:
    """Single-hidden-layer MLP (relu hidden; softmax or logistic output)."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    classes: tuple[str, ...]
    out_activation: str

    def predict(self, canvas: np.ndarray) -> tuple[str, float]:
        """Return (label, margin=top1-top2 probability) for a NORM canvas."""
        x = (canvas.astype(np.float64) / 255.0).ravel()
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        z = h @ self.w2 + self.b2
        if self.out_activation == "logistic":
            p1 = 1.0 / (1.0 + float(np.exp(-z[0])))
            proba = np.array([1.0 - p1, p1])
        else:
            zz = z - float(z.max())
            e = np.exp(zz)
            proba = e / e.sum()
        order = np.argsort(proba)[::-1]
        margin = float(proba[order[0]] - (proba[order[1]] if len(order) > 1 else 0.0))
        return self.classes[int(order[0])], margin


def load_card_heads(path: Path | str) -> dict[str, MlpHead]:
    """Load rank + suit_red + suit_black heads from an npz (fail closed)."""
    source = Path(path)
    meta_path = source.with_suffix(".json")
    if not source.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"card heads missing at {source}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != "mlp-v1":
        raise ValueError("unsupported card heads format")
    data = np.load(source)
    heads: dict[str, MlpHead] = {}
    for name, info in meta["heads"].items():
        heads[name] = MlpHead(
            w1=data[f"{name}__w1"],
            b1=data[f"{name}__b1"],
            w2=data[f"{name}__w2"],
            b2=data[f"{name}__b2"],
            classes=tuple(str(c) for c in data[f"{name}__classes"]),
            out_activation=str(info["out_activation"]),
        )
    missing = {"rank", "suit_red", "suit_black"} - set(heads)
    if missing:
        raise ValueError(f"card heads incomplete: {sorted(missing)}")
    return heads


class FusedSlotBuffer:
    """Temporal glyph accumulator for ONE card slot (street+slot gated).

    ``ingest`` returns True when the frame's slot region matches the
    buffer's anchor (same physical card); a mismatch resets the buffer
    (the card changed — new hand or the slot emptied). ``fused`` returns
    the phase-registered float mean once at least ``min_glyphs`` gated
    glyphs have accumulated, else None (fail closed).
    """

    def __init__(
        self,
        box: tuple[int, int, int, int],
        *,
        min_glyphs: int = 3,
        slot_gate: float = 10.0,
        geometry: CornerGlyphGeometry = DEFAULT_GEOMETRY,
    ) -> None:
        self._box = box
        self._min = min_glyphs
        self._gate = slot_gate
        self._geometry = geometry
        self._anchor: np.ndarray | None = None
        self._rank_glyphs: list[np.ndarray] = []
        self._suit_glyphs: list[np.ndarray] = []
        self._red_stat: float | None = None

    @staticmethod
    def _sig(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        import cv2

        x0, y0, x1, y1 = box
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (20, 20))

    @property
    def glyph_count(self) -> int:
        return len(self._suit_glyphs)

    def ingest(self, roi_image: np.ndarray) -> bool:
        import cv2

        sig = self._sig(roi_image, self._box)
        if self._anchor is not None:
            diff = float(np.mean(cv2.absdiff(sig, self._anchor)))
            if diff > self._gate:
                self._anchor = None
                self._rank_glyphs = []
                self._suit_glyphs = []
                self._red_stat = None
        if self._anchor is None:
            self._anchor = sig
        x0, y0, x1, y1 = self._box
        card = locate_card_face(roi_image[y0:y1, x0:x1])
        rank = GlyphNormalizer.from_card(card, "rank", self._geometry)
        suit = GlyphNormalizer.from_card(card, "suit", self._geometry)
        if rank is None or suit is None:
            return False
        self._rank_glyphs.append(rank)
        self._suit_glyphs.append(suit)
        stat = ink_red_stat(card, self._geometry)
        if stat is not None:
            self._red_stat = stat
        return True

    @staticmethod
    def _registered_mean(glyphs: Sequence[np.ndarray]) -> np.ndarray | None:
        import cv2

        if len(glyphs) < 2:
            return glyphs[0] if glyphs else None
        win = cv2.createHanningWindow(NORM, cv2.CV_32F)
        anchor = glyphs[0]
        acc = anchor.astype(np.float64)
        used = 1
        for g in glyphs[1:]:
            try:
                (dx, dy), _ = cv2.phaseCorrelate(anchor, g, win)
            except cv2.error:
                continue
            if abs(dx) > 6 or abs(dy) > 6:
                continue
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            acc += cv2.warpAffine(g, m, NORM, borderValue=255.0)
            used += 1
        return (acc / used).astype(np.float32) if used >= 2 else anchor

    def fused(self) -> tuple[np.ndarray, np.ndarray, bool | None] | None:
        """(fused_rank, fused_suit, is_red) or None when under-sampled."""
        if len(self._suit_glyphs) < self._min:
            return None
        rank = self._registered_mean(self._rank_glyphs)
        suit = self._registered_mean(self._suit_glyphs)
        if rank is None or suit is None:
            return None
        is_red = (self._red_stat > RED_FLOOR) if self._red_stat is not None else None
        return rank, suit, is_red


class FusedCardRecognizer:
    """Rank+suit classification over fused glyphs with calibrated floors."""

    def __init__(
        self,
        heads: Mapping[str, MlpHead],
        *,
        rank_floor: float = 0.0,
        suit_floor: float = 0.3,
    ) -> None:
        for name in ("rank", "suit_red", "suit_black"):
            if name not in heads:
                raise ValueError(f"missing head: {name}")
        self._rank_head = heads["rank"]
        self._suit_red = heads["suit_red"]
        self._suit_black = heads["suit_black"]
        self._rank_floor = rank_floor
        self._suit_floor = suit_floor

    def recognize_fused(
        self,
        fused_rank: np.ndarray,
        fused_suit: np.ndarray,
        is_red: bool | None,
    ) -> CardRecognition:
        if is_red is None:
            return CardRecognition(value=None, raw_score=0.0, slots=())
        rank_label, rank_margin = self._rank_head.predict(fused_rank)
        head = self._suit_red if is_red else self._suit_black
        suit_label, suit_margin = head.predict(fused_suit)
        family = RED_SUITS if is_red else BLACK_SUITS
        slot = CardSlotResult(
            rank_score=rank_margin,
            suit_score=suit_margin,
            rank=_RANK_MAP.get(rank_label),
            suit=_SUIT_MAP.get(suit_label) if suit_label in family else None,
        )
        raw = float(min(rank_margin, suit_margin))
        if (
            slot.rank is None
            or slot.suit is None
            or rank_margin <= self._rank_floor
            or suit_margin <= self._suit_floor
        ):
            return CardRecognition(value=None, raw_score=raw, slots=(slot,))
        return CardRecognition(
            value=(Card(rank=slot.rank, suit=slot.suit),),
            raw_score=raw,
            slots=(slot,),
        )

    def recognize(self, buffer: FusedSlotBuffer) -> CardRecognition:
        """Classify a slot buffer; UNKNOWN when fusion is under-sampled."""
        fused = buffer.fused()
        if fused is None:
            return CardRecognition(value=None, raw_score=0.0, slots=())
        return self.recognize_fused(*fused)
