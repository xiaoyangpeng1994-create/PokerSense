"""Tests for StreetDetector (exact UNKNOWN/CONFLICT/VALID rules, plan §5)."""

from __future__ import annotations

import pytest

from poker_engine.core.enums import Rank, Street, Suit
from poker_engine.core.observation import ValidationStatus
from poker_engine.core.value_objects import Card
from poker_engine.perceptual.vision.protocols import (
    BoardSlotOccupancy,
    BoardSlotResult,
    BoardSlotsRecognition,
)
from poker_engine.perceptual.vision.street_detector import (
    board_card_count,
    derive,
)

_RANKS = (Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN)


def _slot(i, occupancy, raw=0.9, card=None):
    if occupancy is BoardSlotOccupancy.CARD and card is None:
        card = Card(_RANKS[i], Suit.SPADES)
    return BoardSlotResult(slot_index=i, occupancy=occupancy, card=card, raw_score=raw)


def _rec(occupancies):
    slots = tuple(_slot(i, o) for i, o in enumerate(occupancies))
    return BoardSlotsRecognition(slots=slots)


C = BoardSlotOccupancy.CARD
E = BoardSlotOccupancy.EMPTY
U = BoardSlotOccupancy.UNKNOWN


def test_preflop_all_empty():
    r = derive(_rec([E, E, E, E, E]))
    assert r.street is Street.PREFLOP
    assert r.status is ValidationStatus.VALID


def test_flop():
    r = derive(_rec([C, C, C, E, E]))
    assert r.street is Street.FLOP
    assert r.status is ValidationStatus.VALID


def test_turn():
    r = derive(_rec([C, C, C, C, E]))
    assert r.street is Street.TURN
    assert r.status is ValidationStatus.VALID


def test_river():
    r = derive(_rec([C, C, C, C, C]))
    assert r.street is Street.RIVER
    assert r.status is ValidationStatus.VALID


def test_any_unknown_is_unknown():
    r = derive(_rec([C, C, U, E, E]))
    assert r.status is ValidationStatus.UNKNOWN
    assert r.street is None


def test_nonstandard_confident_pattern_is_conflict():
    # CARD EMPTY CARD EMPTY EMPTY is not a legal pattern
    r = derive(_rec([C, E, C, E, E]))
    assert r.status is ValidationStatus.CONFLICT
    assert r.street is None


def test_board_card_count():
    assert board_card_count(_rec([C, C, C, E, E])) == 3
    assert board_card_count(_rec([C, C, C, C, C])) == 5


def test_raw_score_is_min():
    slots = tuple(
        _slot(i, C, raw=s) for i, s in enumerate([0.9, 0.8, 0.7, 0.6, 0.95])
    )
    r = derive(BoardSlotsRecognition(slots=slots))
    assert r.raw_score == 0.6


def test_requires_exactly_5():
    from poker_engine.perceptual.vision.protocols import BoardSlotsRecognition as B

    with pytest.raises(ValueError):
        derive(B(slots=tuple(_slot(i, E) for i in range(4))))
