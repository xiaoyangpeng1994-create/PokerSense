"""Deterministic fake scenario fixtures for Orchestrator tests.

All fixtures use real Frozen Core constructors. Fixed aware timestamps; no
datetime.now / random / uuid.
"""

from __future__ import annotations

from datetime import datetime, timezone

from poker_engine.core.enums import (
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount

UTC = timezone.utc

Ac = Card(Rank.ACE, Suit.CLUBS)
Kh = Card(Rank.KING, Suit.HEARTS)
Qh = Card(Rank.QUEEN, Suit.HEARTS)
Jh = Card(Rank.JACK, Suit.HEARTS)
Th = Card(Rank.TEN, Suit.HEARTS)
NineH = Card(Rank.NINE, Suit.HEARTS)


def ts(hour=1, minute=0):
    return datetime(2026, 8, 19, hour, minute, 0, tzinfo=UTC)


def player(seat=0, pid=None, hero=False):
    return PlayerState(
        player_id=pid or f"p{seat}",
        seat=seat,
        position=Position.BTN if seat == 0 else Position.SB,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=(seat == 0),
    )


def initial_state(hand_id="h1", street=Street.PREFLOP, hero=(), board=(), pot="1.5"):
    return PokerState(
        state_version=0,
        hand_id=hand_id,
        street=street,
        hero_cards=hero,
        board_cards=board,
        players=(player(0, hero=True), player(1)),
        pot=ChipAmount(pot),
        current_bet=ChipAmount("1"),
        to_call=ChipAmount("1"),
        actor=None,
    )


def _field(value, status=ValidationStatus.VALID, confidence=1.0):
    return ObservationField(
        value=value,
        confidence=confidence,
        source="test",
        evidence={},
        timestamp=ts(),
        validation_status=status,
    )


def _unknown():
    return _field(None, ValidationStatus.UNKNOWN)


def observation(
    frame_seq=1,
    hero=(),
    hero_status=ValidationStatus.VALID,
    board=(),
    board_status=ValidationStatus.VALID,
    pot=None,
    pot_status=ValidationStatus.VALID,
    street=None,
    street_status=ValidationStatus.VALID,
    actor=None,
    actor_status=ValidationStatus.UNKNOWN,
):
    return RawObservation(
        frame_seq=frame_seq,
        timestamp=ts(),
        hero_cards=_field(hero, hero_status),
        board_cards=_field(board, board_status),
        pot=(
            _field(ChipAmount(pot), pot_status)
            if pot is not None else _unknown()
        ),
        stacks=_unknown(),
        bet_size=_unknown(),
        action=_unknown(),
        street=(
            _field(street, street_status)
            if street is not None else _unknown()
        ),
        dealer_pos=_unknown(),
        actor=_field(actor, actor_status),
    )
