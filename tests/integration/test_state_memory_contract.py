"""Integration contract test: StateEngine output feeds HandMemory correctly.

Verifies Task 3 output is naturally compatible with Task 2 rules:
- record_state first, then record_event
- event.state_version references the recorded state
"""

from datetime import datetime, timezone

from poker_engine.core.enums import PlayerStatus, Position, Rank, Street, Suit
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState, StateContext
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.memory import InMemoryHandMemory
from poker_engine.state_engine import StateEngine

UTC = timezone.utc

QH = Card(Rank.QUEEN, Suit.HEARTS)
JH = Card(Rank.JACK, Suit.HEARTS)
TH = Card(Rank.TEN, Suit.HEARTS)
NH = Card(Rank.NINE, Suit.HEARTS)


def _aw():
    return datetime(2026, 8, 19, 1, 0, 0, tzinfo=UTC)


def _player(seat=0, pid=None, hero=False):
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


def _state(version=0, hand_id="h1", street=Street.PREFLOP, board=()):
    return PokerState(
        state_version=version,
        hand_id=hand_id,
        street=street,
        hero_cards=(),
        board_cards=board,
        players=(_player(0, hero=True), _player(1)),
        pot=ChipAmount("1.5"),
        current_bet=ChipAmount("1"),
        to_call=ChipAmount("1"),
    )


def _field(value, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=0.95, source="test",
        timestamp=_aw(), validation_status=status,
    )


def _obs(street=None, board=()):
    return RawObservation(
        frame_seq=1,
        timestamp=_aw(),
        hero_cards=_field(()),
        board_cards=_field(board),
        pot=_field(ChipAmount("1.5")),
        stacks=_field(()),
        bet_size=_field(ChipAmount("0")),
        action=_field(None, ValidationStatus.UNKNOWN),
        street=(
            _field(street)
            if street is not None
            else _field(None, ValidationStatus.UNKNOWN)
        ),
        dealer_pos=_field(None, ValidationStatus.UNKNOWN),
        actor=_field(None, ValidationStatus.UNKNOWN),
    )


def test_state_then_event_feed_contract():
    engine = StateEngine()
    memory = InMemoryHandMemory()
    ctx = StateContext()

    s0 = _state(version=0)
    memory.start_hand("h1", s0, started_at=_aw())

    r1 = engine.transition(
        s0, _obs(street=Street.FLOP, board=(QH, JH, TH)), ctx
    )
    assert r1.changed is True
    memory.record_state(r1.state)
    for e in r1.events:
        memory.record_event(e)

    assert memory.latest_state("h1").state_version == 1
    assert len(memory.events("h1")) == len(r1.events)


def test_invalid_transition_produces_no_state_or_event():
    engine = StateEngine()
    memory = InMemoryHandMemory()
    ctx = StateContext()

    s0 = _state(version=0, street=Street.TURN, board=(QH, JH, TH, NH))
    memory.start_hand("h1", s0, started_at=_aw())

    # board regression (4 -> 3) -> invalid
    r = engine.transition(
        s0, _obs(street=Street.TURN, board=(QH, JH, TH)), ctx
    )
    assert r.validation.is_valid is False
    assert r.changed is False
    assert memory.latest_state("h1").state_version == 0
    assert memory.events("h1") == ()
