"""Tests for State Engine (pure state reconciliation)."""

from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import (
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.events import EventType
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState, StateContext
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.state_engine import StateEngine, StateEngineError

UTC = timezone.utc

Ac = Card(Rank.ACE, Suit.CLUBS)
Kh = Card(Rank.KING, Suit.HEARTS)
Qh = Card(Rank.QUEEN, Suit.HEARTS)
Jh = Card(Rank.JACK, Suit.HEARTS)
Th = Card(Rank.TEN, Suit.HEARTS)
NineH = Card(Rank.NINE, Suit.HEARTS)


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


def _state(version=0, hand_id="h1", street=Street.PREFLOP,
           hero=(), board=(), pot="1.5", actor=None):
    return PokerState(
        state_version=version,
        hand_id=hand_id,
        street=street,
        hero_cards=hero,
        board_cards=board,
        players=(_player(0, hero=True), _player(1)),
        pot=ChipAmount(pot),
        current_bet=ChipAmount("1"),
        to_call=ChipAmount("1"),
        actor=actor,
    )


def _field(value, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=0.95, source="test",
        evidence={}, timestamp=_aw(), validation_status=status,
    )


def _unknown_field():
    return _field(None, ValidationStatus.UNKNOWN)


def _obs(hero=(), hero_status=ValidationStatus.VALID,
         board=(), board_status=ValidationStatus.VALID,
         pot=None, pot_status=ValidationStatus.VALID,
         street=None, street_status=ValidationStatus.VALID,
         action=None, action_status=ValidationStatus.UNKNOWN,
         actor=None, actor_status=ValidationStatus.UNKNOWN):
    return RawObservation(
        frame_seq=1,
        timestamp=_aw(),
        hero_cards=_field(hero, hero_status),
        board_cards=_field(board, board_status),
        pot=(
            _field(ChipAmount(pot), pot_status)
            if pot is not None else _unknown_field()
        ),
        stacks=_unknown_field(),
        bet_size=_unknown_field(),
        action=_field(action, action_status),
        street=(
            _field(street, street_status)
            if street is not None else _unknown_field()
        ),
        dealer_pos=_unknown_field(),
        actor=_field(actor, actor_status),
    )


@pytest.fixture
def engine():
    return StateEngine()


@pytest.fixture
def ctx():
    return StateContext()


# ---------- A. Purity ----------

def test_purity_same_result(engine, ctx):
    prev = _state()
    obs = _obs(pot="2.0")
    r1 = engine.transition(prev, obs, ctx)
    r2 = engine.transition(prev, obs, ctx)
    assert r1 == r2
    assert r1.state == r2.state
    assert r1.events == r2.events


def test_purity_does_not_mutate_inputs(engine, ctx):
    prev = _state()
    obs = _obs(pot="2.0")
    prev_copy = prev
    obs_copy = obs
    engine.transition(prev, obs, ctx)
    assert prev == prev_copy
    assert obs.hero_cards.value == obs_copy.hero_cards.value


# ---------- B. No-op ----------

def test_noop_unchanged_valid(engine, ctx):
    prev = _state(pot="1.5")
    # pot VALID but same value, everything else UNKNOWN
    obs = _obs(pot="1.5")
    r = engine.transition(prev, obs, ctx)
    assert r.changed is False
    assert r.state is prev
    assert r.events == ()


def test_noop_all_unknown(engine, ctx):
    prev = _state()
    obs = _obs(hero_status=ValidationStatus.UNKNOWN,
               board_status=ValidationStatus.UNKNOWN,
               street_status=ValidationStatus.UNKNOWN,
               pot_status=ValidationStatus.UNKNOWN)
    r = engine.transition(prev, obs, ctx)
    assert r.changed is False
    assert r.state is prev
    assert r.state.state_version == prev.state_version


# ---------- C. Single field update ----------

def test_single_field_pot_update(engine, ctx):
    prev = _state(pot="1.5")
    obs = _obs(pot="3.0")
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    assert r.state.pot == ChipAmount("3.0")
    # other fields unchanged
    assert r.state.street is prev.street
    assert r.state.hero_cards == prev.hero_cards
    assert r.state.board_cards == prev.board_cards
    assert r.state.players == prev.players


def test_single_field_street_update(engine, ctx):
    prev = _state(street=Street.PREFLOP)
    obs = _obs(street=Street.FLOP)
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    assert r.state.street is Street.FLOP
    # one STREET_CHANGE event
    assert len(r.events) == 1
    assert r.events[0].event_type is EventType.STREET_CHANGE


def test_single_field_hero_update(engine, ctx):
    prev = _state(hero=())
    obs = _obs(hero=(Ac, Kh))
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    assert r.state.hero_cards == (Ac, Kh)


def test_single_field_board_update(engine, ctx):
    prev = _state(board=())
    obs = _obs(board=(Qh, Jh, Th))
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    assert r.state.board_cards == (Qh, Jh, Th)
    # DEAL event
    assert any(e.event_type is EventType.DEAL for e in r.events)


# ---------- D. Observation status ----------

def test_unknown_does_not_overwrite(engine, ctx):
    prev = _state(pot="1.5")
    obs = _obs(pot=None, pot_status=ValidationStatus.UNKNOWN)
    r = engine.transition(prev, obs, ctx)
    assert r.state.pot == prev.pot
    assert r.changed is False


def test_low_confidence_does_not_overwrite(engine, ctx):
    prev = _state(street=Street.PREFLOP)
    obs = _obs(street=Street.TURN, street_status=ValidationStatus.LOW_CONFIDENCE)
    r = engine.transition(prev, obs, ctx)
    assert r.state.street is Street.PREFLOP
    assert r.changed is False


# ---------- E. Version ----------

def test_version_increments(engine, ctx):
    prev = _state(version=0)
    r1 = engine.transition(prev, _obs(pot="2.0"), ctx)
    assert r1.state.state_version == 1
    r2 = engine.transition(r1.state, _obs(pot="3.0"), ctx)
    assert r2.state.state_version == 2


def test_version_noop_unchanged(engine, ctx):
    prev = _state(version=5)
    r = engine.transition(prev, _obs(pot="1.5"), ctx)
    assert r.changed is False
    assert r.state.state_version == 5


def test_version_large_number(engine, ctx):
    prev = _state(version=999)
    r = engine.transition(prev, _obs(pot="999"), ctx)
    assert r.state.state_version == 1000


# ---------- F. Cards ----------

def test_board_progression_3_to_4(engine, ctx):
    prev = _state(street=Street.FLOP, board=(Qh, Jh, Th))
    obs = _obs(board=(Qh, Jh, Th, NineH), street=Street.TURN)
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    assert r.state.board_cards == (Qh, Jh, Th, NineH)


def test_board_regression_invalid(engine, ctx):
    prev = _state(street=Street.TURN, board=(Qh, Jh, Th, NineH))
    obs = _obs(board=(Qh, Jh, Th))  # 4 -> 3 regression
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.changed is False
    assert r.state is prev
    assert r.events == ()


def test_hero_regression_2_to_0_invalid(engine, ctx):
    prev = _state(hero=(Ac, Kh))
    obs = _obs(hero=())  # 2 -> 0 regression
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev


def test_duplicate_cards_invalid(engine, ctx):
    # duplicate board card via PokerState constructor -> invalid
    prev = _state(board=())
    obs = _obs(board=(Qh, Qh, Th))  # duplicate Qh
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False


# ---------- G. Street ----------

def test_street_progression_ok(engine, ctx):
    prev = _state(street=Street.FLOP)
    obs = _obs(street=Street.TURN)
    r = engine.transition(prev, obs, ctx)
    assert r.state.street is Street.TURN
    assert r.events[0].event_type is EventType.STREET_CHANGE


def test_street_same_noop(engine, ctx):
    prev = _state(street=Street.FLOP)
    obs = _obs(street=Street.FLOP)
    r = engine.transition(prev, obs, ctx)
    assert r.changed is False


def test_street_regression_invalid(engine, ctx):
    prev = _state(street=Street.TURN)
    obs = _obs(street=Street.FLOP)
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev


# ---------- H. Events ----------

def test_event_correct_metadata(engine, ctx):
    prev = _state(street=Street.PREFLOP)
    obs = _obs(street=Street.FLOP, board=(Qh, Jh, Th))
    r = engine.transition(prev, obs, ctx)
    assert r.changed is True
    for e in r.events:
        assert e.hand_id == "h1"
        assert e.state_version == r.state.state_version
        assert e.source == "state_engine"
        assert e.timestamp == obs.timestamp


def test_event_ordering_street_change_then_deal(engine, ctx):
    prev = _state(street=Street.PREFLOP, board=())
    obs = _obs(street=Street.FLOP, board=(Qh, Jh, Th))
    r = engine.transition(prev, obs, ctx)
    types = [e.event_type for e in r.events]
    assert types == [EventType.STREET_CHANGE, EventType.DEAL]


# ---------- K. Validation failure ----------

def test_invalid_keeps_previous(engine, ctx):
    prev = _state(street=Street.RIVER, board=(Qh, Jh, Th, NineH, Ac))
    obs = _obs(board=(Qh, Jh, Th, NineH))  # 5 -> 4 regression
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev
    assert r.events == ()
    assert r.changed is False


# ---------- context.previous_state mismatch ----------

def test_context_previous_state_mismatch_raises(engine):
    prev = _state(hand_id="h1")
    other = _state(hand_id="h2")
    ctx = StateContext(previous_state=other)
    with pytest.raises(StateEngineError):
        engine.transition(prev, _obs(pot="2.0"), ctx)


def test_context_previous_state_none_ok(engine):
    prev = _state()
    ctx = StateContext(previous_state=None)
    r = engine.transition(prev, _obs(pot="2.0"), ctx)
    assert r.changed is True


# ---------- type errors (programmer error) ----------

def test_non_poker_state_raises_typeerror(engine, ctx):
    with pytest.raises(TypeError):
        engine.transition("not-a-state", _obs(pot="2.0"), ctx)


def test_non_observation_raises_typeerror(engine, ctx):
    with pytest.raises(TypeError):
        engine.transition(_state(), "not-an-obs", ctx)


# ---------- Blocker 1: LOW_CONFIDENCE / CONFLICT warnings ----------

@pytest.mark.parametrize(
    "status", [ValidationStatus.LOW_CONFIDENCE, ValidationStatus.CONFLICT]
)
def test_low_conflict_warns_and_retains(engine, ctx, status):
    prev = _state(street=Street.PREFLOP)
    obs = _obs(street=Street.TURN, street_status=status)
    r = engine.transition(prev, obs, ctx)
    # retain previous
    assert r.state.street is Street.PREFLOP
    assert r.changed is False
    assert r.state.state_version == prev.state_version
    # deterministic warning present
    assert "street ignored" in r.validation.warnings[0]


@pytest.mark.parametrize(
    "status", [ValidationStatus.LOW_CONFIDENCE, ValidationStatus.CONFLICT]
)
def test_low_conflict_pot_warns_and_retains(engine, ctx, status):
    prev = _state(pot="1.5")
    obs = _obs(pot="9.9", pot_status=status)
    r = engine.transition(prev, obs, ctx)
    assert r.state.pot == ChipAmount("1.5")
    assert r.changed is False
    assert any("pot ignored" in w for w in r.validation.warnings)


def test_low_confidence_warning_deterministic(engine, ctx):
    prev = _state(street=Street.PREFLOP)
    obs = _obs(street=Street.TURN, street_status=ValidationStatus.LOW_CONFIDENCE)
    r1 = engine.transition(prev, obs, ctx)
    r2 = engine.transition(prev, obs, ctx)
    assert r1.validation.warnings == r2.validation.warnings


# ---------- Blocker 2: Card identity monotonicity ----------

def test_hero_2_to_different_2_invalid(engine, ctx):
    prev = _state(hero=(Ac, Kh))
    # same count but a different card identity -> invalid
    obs = _obs(hero=(Ac, Qh))
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev
    assert r.events == ()
    assert r.changed is False


def test_board_3_to_different_3_invalid(engine, ctx):
    prev = _state(street=Street.FLOP, board=(Qh, Jh, Th))
    # same count but one card replaced -> invalid
    obs = _obs(board=(Qh, Jh, NineH))
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev


def test_board_3_to_4_prefix_preserved_ok(engine, ctx):
    prev = _state(street=Street.TURN, board=(Qh, Jh, Th))
    obs = _obs(board=(Qh, Jh, Th, NineH))
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is True
    assert r.state.board_cards == (Qh, Jh, Th, NineH)


def test_board_3_to_4_replace_existing_invalid(engine, ctx):
    prev = _state(street=Street.TURN, board=(Qh, Jh, Th))
    # grows to 4 but one of the existing 3 was replaced -> invalid
    obs = _obs(board=(Qh, Jh, NineH, Ac))
    r = engine.transition(prev, obs, ctx)
    assert r.validation.is_valid is False
    assert r.state is prev
