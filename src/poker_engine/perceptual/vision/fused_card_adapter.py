"""V9 stateful adapter: temporal-fusion recognition behind the single-
frame ``CardRecognizer`` protocol.

The fused recognizer needs cross-frame accumulation (each card slot keeps a
``FusedSlotBuffer``), but ``VisionEngine`` is otherwise stateless. This
adapter keeps that state OUT of the engine: it multiplexes a per-slot buffer
by a caller-supplied ``slot_id``, ingests each frame's card crop, and returns
a ``CardRecognition`` only once the fusion has enough gated glyphs and both
margins clear the calibrated floors. Otherwise it returns UNKNOWN (fail
closed).

Source/frame/time/ROI discontinuities explicitly invalidate history. A coarse
pixel signature also resets visibly changed slots. Accepted current and fused
identities must not conflict. These checks are not a proof of card identity
when the classifier is uncertain or both reads make the same mistake.
"""

from __future__ import annotations

from datetime import datetime
import math

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
    frame lifecycle and per-slot signature gate control buffering. Canonical
    hand tracking remains a separate layer; similar uninterrupted card changes
    still require broader validation. Direct crop-only diagnostic callers may
    omit begin_frame, but then cannot claim frame/time continuity checks.
    """

    def __init__(
        self,
        recognizer: FusedCardRecognizer,
        *,
        min_glyphs: int = 3,
        slot_gate: float = 10.0,
        accept_candidates: bool = True,
        max_frame_gap_seconds: float = 1.0,
    ) -> None:
        if not isinstance(accept_candidates, bool):
            raise TypeError("accept_candidates must be a bool")
        if (isinstance(max_frame_gap_seconds, bool)
                or not isinstance(max_frame_gap_seconds, (int, float))
                or not math.isfinite(max_frame_gap_seconds)
                or max_frame_gap_seconds <= 0):
            raise ValueError("max_frame_gap_seconds must be finite and positive")
        self._recognizer = recognizer
        self._min = min_glyphs
        self._gate = slot_gate
        self._accept_candidates = accept_candidates
        self._buffers: dict[tuple[str, int], FusedSlotBuffer] = {}
        self._max_frame_gap = float(max_frame_gap_seconds)
        self._frame_seq: int | None = None
        self._timestamp: datetime | None = None
        self._source_context = None
        self._regions: dict = {}
        self._last_seen: dict[tuple[str, int], int] = {}
        self._attempted: set[tuple[str, int]] = set()
        self.identity_conflict_count = 0

    def begin_frame(self, frame_seq: int, timestamp: datetime,
                    source_context, regions: dict) -> None:
        """Optional lifecycle hook used by VisionEngine and offline wrappers.

        The time limit is an engineering freshness budget, not a measured
        poker rule. Replay must supply source timestamps, not processing time.
        Region absence/change resets only that group; stream gaps reset all.
        """
        if (isinstance(frame_seq, bool) or not isinstance(frame_seq, int)
                or frame_seq < 0 or not isinstance(timestamp, datetime)
                or timestamp.tzinfo is None or timestamp.utcoffset() is None):
            self.reset()
            raise ValueError("invalid frame identity/time")
        if (self._frame_seq is None or frame_seq != self._frame_seq + 1
                or source_context != self._source_context
                or timestamp < self._timestamp
                or (timestamp - self._timestamp).total_seconds() > self._max_frame_gap):
            self.reset()
        for group in ("hero", "board"):
            if (regions.get(group) is None
                    or regions.get(group) != self._regions.get(group)):
                for key in tuple(self._buffers):
                    if key[0] == group:
                        self.reset(key)
        self._frame_seq = frame_seq
        self._timestamp = timestamp
        self._source_context = source_context
        self._regions = dict(regions)
        self._attempted.clear()

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
        if self._frame_seq is not None:
            if self._regions.get(key[0]) is None:
                self.reset(key)
                return CardRecognition(value=None, raw_score=0.0, slots=())
            if key in self._attempted:
                # Never turn repeated calls on one frame into fresh samples.
                return CardRecognition(value=None, raw_score=0.0, slots=())
            self._attempted.add(key)
            if self._last_seen.get(key) != self._frame_seq - 1:
                self.reset(key)
            self._last_seen[key] = self._frame_seq
        if (roi_image is None or roi_image.size == 0 or roi_image.ndim != 3
                or roi_image.shape[2] != 3):
            self.reset(key)
            return CardRecognition(value=None, raw_score=0.0, slots=())
        buf = self._buffers.get(key)
        if buf is None or buf.region_shape != roi_image.shape[:2]:
            # box is (x0, y0, x1, y1) in the CROP's own pixel space; the crop
            # is the whole single card, so its full extent is the box.
            h, w = roi_image.shape[:2]
            buf = self._buffers[key] = FusedSlotBuffer(
                (0, 0, w, h),
                min_glyphs=self._min,
                slot_gate=self._gate,
            )
        if not buf.ingest(roi_image):
            return CardRecognition(value=None, raw_score=0.0, slots=())
        fused = buf.fused()
        if fused is None:
            return CardRecognition(value=None, raw_score=0.0, slots=())
        result = self._recognizer.recognize_fused(*fused)
        latest = buf.latest_glyphs()
        current = (self._recognizer.recognize_fused(*latest)
                   if latest is not None else None)
        if (current is not None and current.value is not None
                and result.value is not None
                and current.value != result.value):
            # A small rank/suit change can pass the whole-card pixel gate.
            # Do not let many old samples outvote an accepted current identity.
            # Both reads use the same calibrated head floors; no new guessed
            # card is promoted. This frame becomes sample1 of a new window.
            self.identity_conflict_count += 1
            buf.reset()
            buf.ingest(roi_image)
            return CardRecognition(value=None, raw_score=0.0, slots=())
        if not self._accept_candidates:
            # A changed feature pipeline needs fresh calibration. Preserve
            # diagnostic rank/suit components, but expose no accepted card.
            return CardRecognition(value=None, raw_score=result.raw_score,
                                   slots=result.slots)
        return result

    def reset(self, key: tuple[str, int] | None = None) -> None:
        """Forget accumulation for one slot (or all slots when None)."""
        if key is None:
            self._buffers.clear()
            self._last_seen.clear()
            self._attempted.clear()
            self._frame_seq = None
            self._timestamp = None
            self._source_context = None
            self._regions.clear()
        else:
            self._buffers.pop(key, None)
            self._last_seen.pop(key, None)
