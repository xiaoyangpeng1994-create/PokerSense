"""Stateful adapter: temporal-fusion card recognition behind the single-
frame ``CardRecognizer`` protocol.

The fused recognizer needs cross-frame accumulation (each card slot keeps a
``FusedSlotBuffer``), but ``VisionEngine`` is otherwise stateless. This
adapter keeps that state OUT of the engine: it multiplexes a per-slot buffer
by a caller-supplied ``slot_id``, ingests each frame's card crop, and returns
a ``CardRecognition`` only once the fusion has enough gated glyphs and both
margins clear the calibrated floors. Otherwise it returns UNKNOWN (fail
closed).

A card that changes (a slot's own pixels move past the slot gate — a new
hand, the board advancing a street, a slot emptying) resets that buffer, so
the fused result never blends glyphs from two different cards.
"""

from __future__ import annotations

from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer,
    FusedSlotBuffer,
)
from poker_engine.perceptual.vision.protocols import CardRecognition

__all__ = ["FusedCardRecognizerAdapter"]


class FusedCardRecognizerAdapter:
    """Per-slot temporal fusion behind the stateless CardRecognizer contract.

    The ``card_model`` argument doubles as the slot identity: callers pass an
    ``(group, slot_index)`` tuple (``group`` is ``"hero"`` or ``"board"``) so
    hero slots and board slots keep separate accumulation pools. The
    underlying per-slot signature gate automatically resets a buffer when the
    card in that slot changes (new hand / street advance / seat leave), so no
    explicit hand tracking is needed here.
    """

    def __init__(
        self,
        recognizer: FusedCardRecognizer,
        *,
        min_glyphs: int = 3,
        slot_gate: float = 10.0,
    ) -> None:
        self._recognizer = recognizer
        self._min = min_glyphs
        self._gate = slot_gate
        self._buffers: dict[tuple[str, int], FusedSlotBuffer] = {}

    @staticmethod
    def _key(card_model) -> tuple[str, int] | None:
        if isinstance(card_model, tuple) and len(card_model) == 2:
            group, idx = card_model
            if group in ("hero", "board") and isinstance(idx, int) and idx >= 0:
                return group, idx
        return None

    def recognize(self, roi_image, card_model=None) -> CardRecognition:
        """Ingest one slot's card crop and classify the fused glyphs.

        ``roi_image`` is the single-card crop; ``card_model`` is the
        ``(group, slot_index)`` identity. A missing colour router signal, an
        under-sampled fusion, or a below-floor margin all return UNKNOWN.
        """
        key = self._key(card_model)
        if key is None:
            # Unknown slot identity: nothing to accumulate — fail closed.
            return CardRecognition(value=None, raw_score=0.0, slots=())
        buf = self._buffers.get(key)
        if buf is None:
            # box is (x0, y0, x1, y1) in the CROP's own pixel space; the crop
            # is the whole single card, so its full extent is the box.
            h, w = roi_image.shape[:2]
            buf = self._buffers[key] = FusedSlotBuffer(
                (0, 0, w, h),
                min_glyphs=self._min,
                slot_gate=self._gate,
            )
        buf.ingest(roi_image)
        fused = buf.fused()
        if fused is None:
            return CardRecognition(value=None, raw_score=0.0, slots=())
        return self._recognizer.recognize_fused(*fused)

    def reset(self, key: tuple[str, int] | None = None) -> None:
        """Forget accumulation for one slot (or all slots when None)."""
        if key is None:
            self._buffers.clear()
        else:
            self._buffers.pop(key, None)
