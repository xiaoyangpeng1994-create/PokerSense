"""Tests for PokerState, StateContext, ValidationResult."""

from dataclasses import FrozenInstanceError

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Rank, Street, Suit
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState, StateContext, ValidationResult
from poker_engine.core.value_objects import Card, ChipAmount

Ac = Card(Rank.ACE, Suit.CLUBS)
Ad = Card(Rank.ACE, Suit.DIAMONDS)
Kc = Card(Rank.KING, Suit.CLUBS)
Kd = Card(Rank.KING, Suit.DIAMONDS)
Qh = Card(Rank.QUEEN, Suit.HEARTS)
Jh = Card(Rank.JACK, Suit.HEARTS)
Th = Card(Rank.TEN, Suit.HEARTS)
NineH = Card(Rank.NINE, Suit.HEARTS)


def _player(seat: int, pid: str = None, hero: bool = False) -> PlayerState:
    return PlayerState(
        player_id=pid or f"p{seat}",
        seat=seat,
        position=Position.BTN,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=(seat == 0),
    )


def _state(**overrides) -> PokerState:
    args = dict(
        state_version=1,
        hand_id="h1",
        street=Street.PREFLOP,
        hero_cards=(Ac, Ad),
        board_cards=(),
        players=(_player(0, hero=True), _player(1)),
        pot=ChipAmount("30"),
        current_bet=ChipAmount("10"),
        to_call=ChipAmount("10"),
    )
    args.update(overrides)
    return PokerState(**args)


# ---------- hero cards ----------

def test_hero_cards_zero_ok():
    s = _state(hero_cards=())
    assert len(s.hero_cards) == 0


def test_hero_cards_two_ok():
    s = _state(hero_cards=(Ac, Ad))
    assert len(s.hero_cards) == 2


def test_hero_cards_one_invalid():
    with pytest.raises(ValueError):
        _state(hero_cards=(Ac,))


# ---------- board cards ----------

@pytest.mark.parametrize("n", [0, 3, 4, 5])
def test_board_card_counts_ok(n):
    cards = (Qh, Jh, Th, NineH, Kc)[:n]
    s = _state(board_cards=cards)
    assert len(s.board_cards) == n


@pytest.mark.parametrize("n", [1, 2])
def test_board_card_counts_invalid(n):
    cards = (Qh, Jh, Th, NineH, Kc)[:n]
    with pytest.raises(ValueError):
        _state(board_cards=cards)


# ---------- duplicate cards ----------

def test_duplicate_board_cards_invalid():
    with pytest.raises(ValueError):
        _state(board_cards=(Qh, Qh, Th))


def test_hero_board_overlap_invalid():
    with pytest.raises(ValueError):
        _state(hero_cards=(Ac, Ad), board_cards=(Ac, Kc, Qh))


# ---------- players ----------

def test_duplicate_seat_invalid():
    with pytest.raises(ValueError):
        _state(players=(_player(0, "a"), _player(0, "b")))


def test_duplicate_player_id_invalid():
    with pytest.raises(ValueError):
        _state(players=(_player(0, "same"), _player(1, "same")))


def test_multiple_hero_invalid():
    with pytest.raises(ValueError):
        _state(players=(_player(0, hero=True), _player(1, hero=True)))


def test_actor_must_exist():
    with pytest.raises(ValueError):
        _state(actor=99)


def test_actor_none_ok():
    s = _state(actor=None)
    assert s.actor is None


def test_actor_existing_ok():
    s = _state(actor=1)
    assert s.actor == 1


def test_negative_state_version_invalid():
    with pytest.raises(ValueError):
        _state(state_version=-1)


def test_empty_hand_id_invalid():
    with pytest.raises(ValueError):
        _state(hand_id="")


# ---------- immutability ----------

def test_state_frozen():
    s = _state()
    with pytest.raises(FrozenInstanceError):
        s.pot = ChipAmount("99")  # type: ignore[misc]


def test_state_players_deep_immutable():
    s = _state()
    with pytest.raises(FrozenInstanceError):
        s.players += (_player(2),)  # type: ignore[operator]


def test_state_money_uses_chipamount():
    s = _state()
    assert isinstance(s.pot, ChipAmount)
    assert isinstance(s.current_bet, ChipAmount)
    assert isinstance(s.to_call, ChipAmount)


# ---------- StateContext ----------

def test_state_context_defaults():
    ctx = StateContext()
    assert ctx.previous_state is None
    assert ctx.platform_rules == {}
    assert ctx.recent_events == ()


def test_state_context_deep_freezes_mappings():
    rules = {"blind": {"sb": 1, "bb": 2}}
    ctx = StateContext(platform_rules=rules)
    rules["blind"]["sb"] = 999
    assert ctx.platform_rules["blind"]["sb"] == 1
    with pytest.raises(TypeError):
        ctx.platform_rules["blind"] = {}  # type: ignore[index]


# ---------- ValidationResult ----------

def test_validation_result_basic():
    vr = ValidationResult(is_valid=True, errors=(), warnings=())
    assert vr.is_valid is True
    assert vr.errors == ()


def test_validation_result_deep_immutable():
    vr = ValidationResult(is_valid=False, errors=["a", "b"])
    assert vr.errors == ("a", "b")
    with pytest.raises(FrozenInstanceError):
        vr.errors += ("c",)  # type: ignore[operator]


# ---------- 返修 v2 新增测试 ----------

def test_hero_cards_non_card_rejected():
    # hero_cards 元素必须是 Card，字符串 → TypeError
    with pytest.raises(TypeError):
        _state(hero_cards=("As", "Kh"))  # type: ignore[arg-type]


def test_board_cards_non_card_rejected():
    with pytest.raises(TypeError):
        _state(board_cards=(Qh, "Th", Kc))  # type: ignore[list-item]


def test_recent_events_must_be_state_events():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone

    ev = StateEvent(
        event_type=EventType.BET, hand_id="h1", state_version=0,
        timestamp=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    ctx = StateContext(recent_events=(ev,))
    assert len(ctx.recent_events) == 1
    assert isinstance(ctx.recent_events[0], StateEvent)


def test_recent_events_non_state_event_rejected():
    with pytest.raises(TypeError):
        StateContext(recent_events=("not_an_event",))  # type: ignore[arg-type]


def test_validation_result_errors_must_be_str():
    with pytest.raises(TypeError):
        ValidationResult(is_valid=False, errors=[1, 2])  # type: ignore[list-item]


def test_validation_result_warnings_must_be_str():
    with pytest.raises(TypeError):
        ValidationResult(is_valid=True, warnings=[None])  # type: ignore[list-item]
