"""Street detector: derive Street from a BoardSlotsRecognition (pure function).

Exact deterministic rules (plan §5):
- any slot UNKNOWN -> UNKNOWN
- all 5 slots CARD/EMPTY, pattern one of EEEEE/CCCEE/CCCCE/CCCCC -> VALID
  (PREFLOP/FLOP/TURN/RIVER respectively)
- otherwise (confident slots but non-standard pattern) -> CONFLICT
- legal pattern conflicting with independently-recognized board_cards -> CONFLICT
  (the conflict cross-check is applied by the caller with board_cards;
   this module provides `derive` plus a raw_score feature).

No State/history monotonic logic here.
"""

from __future__ import annotations

from poker_engine.core.enums import Street
from poker_engine.core.observation import ValidationStatus

from .protocols import BoardSlotOccupancy, BoardSlotsRecognition, StreetRecognition


def derive(board_slots: BoardSlotsRecognition) -> StreetRecognition:
    slots = board_slots.slots
    if len(slots) != 5:
        raise ValueError("Street derivation requires exactly 5 board slots")

    # raw feature: min relevant evidence (fallback to 0.0 for empty/unknown)
    raw_scores = [s.raw_score for s in slots]
    raw_score = float(min(raw_scores)) if raw_scores else 0.0

    occ = [s.occupancy for s in slots]

    # Rule 1: any UNKNOWN -> UNKNOWN
    if any(o is BoardSlotOccupancy.UNKNOWN for o in occ):
        return StreetRecognition(
            street=None, status=ValidationStatus.UNKNOWN,
            raw_score=raw_score, evidence=slots,
        )

    def as_card_mask(o):
        return o is BoardSlotOccupancy.CARD

    mask = tuple(as_card_mask(o) for o in occ)

    # Rule 2: strict positional patterns
    if mask == (False, False, False, False, False):
        street = Street.PREFLOP
    elif mask == (True, True, True, False, False):
        street = Street.FLOP
    elif mask == (True, True, True, True, False):
        street = Street.TURN
    elif mask == (True, True, True, True, True):
        street = Street.RIVER
    else:
        return StreetRecognition(
            street=None, status=ValidationStatus.CONFLICT,
            raw_score=raw_score, evidence=slots,
        )

    return StreetRecognition(
        street=street, status=ValidationStatus.VALID,
        raw_score=raw_score, evidence=slots,
    )


def board_card_count(board_slots: BoardSlotsRecognition) -> int:
    """Number of confidently-CARD slots (for cross-check with board_cards)."""
    return sum(
        1 for s in board_slots.slots if s.occupancy is BoardSlotOccupancy.CARD
    )


class TemplateStreetDetector:
    """StreetDetector protocol implementation wrapping the pure ``derive``."""

    def derive(self, board_slots: BoardSlotsRecognition) -> StreetRecognition:
        return derive(board_slots)


__all__ = ["derive", "board_card_count", "TemplateStreetDetector"]
