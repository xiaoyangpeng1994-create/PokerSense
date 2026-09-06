"""Production ActionLineResolver tests.

The classification table mirrors the derivation GTOpen validates against, so
these tests pin both the happy paths and the fail-closed outcomes: any missing
or contradictory evidence must resolve to ``None`` instead of a guessed line.
"""

from __future__ import annotations

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Street
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.action_line import (
    ACTION_LINES,
    build_action_line_resolver,
    resolve_action_line,
)

from .helpers import NOW, card

_POSITIONS = (
    Position.UTG,
    Position.HJ,
    Position.CO,
    Position.BTN,
    Position.SB,
    Position.BB,
)


def _state(street: Street = Street.PREFLOP, hand_id: str = "hand-1"):
    players = tuple(
        PlayerState(
            player_id=f"seat-{seat}",
            seat=seat,
            position=_POSITIONS[seat],
            stack=ChipAmount("100"),
            committed_this_street=ChipAmount("0"),
            committed_this_hand=ChipAmount("0"),
            status=PlayerStatus.ACTIVE,
            has_cards=True,
            is_hero=seat == 0,
            is_dealer=seat == 3,
        )
        for seat in range(6)
    )
    return PokerState(
        state_version=1,
        hand_id=hand_id,
        street=street,
        hero_cards=(card("As"), card("Ks")),
        board_cards=(),
        players=players,
        pot=ChipAmount("1.5"),
        current_bet=ChipAmount("2"),
        to_call=ChipAmount("2"),
        actor=0,
    )


def _event(event_type, seat: int = 1, hand_id: str = "hand-1", **payload):
    body = {"seat_id": seat}
    body.update(payload)
    return StateEvent(
        event_type=event_type,
        hand_id=hand_id,
        state_version=1,
        payload=body,
        timestamp=NOW,
    )


@pytest.mark.parametrize(
    ("label", "history", "expected"),
    [
        ("empty history", (), "unopened"),
        ("folds only", (_event(EventType.FOLD, 1), _event(EventType.FOLD, 2)),
         "unopened"),
        ("checks only", (_event(EventType.CHECK, 1),), "unopened"),
        ("single limp", (_event(EventType.CALL, 1),), "limp"),
        ("two limps", (_event(EventType.CALL, 1), _event(EventType.CALL, 2)),
         "multi_limp"),
        ("open raise", (_event(EventType.RAISE, 1),), "raise"),
        ("iso raise over one limper",
         (_event(EventType.CALL, 1), _event(EventType.RAISE, 2)), "iso_raise"),
        ("three bet", (_event(EventType.RAISE, 1), _event(EventType.RAISE, 2)),
         "three_bet"),
        ("squeeze over a cold caller",
         (_event(EventType.RAISE, 1), _event(EventType.CALL, 2),
          _event(EventType.RAISE, 3)), "squeeze"),
        ("four bet",
         (_event(EventType.RAISE, 1), _event(EventType.RAISE, 2),
          _event(EventType.RAISE, 3)), "four_bet"),
        ("all in", (_event(EventType.ALL_IN, 1),), "all_in"),
    ],
)
def test_resolve_action_line_classifies_observed_history(label, history, expected):
    assert resolve_action_line(_state(), history) == expected


@pytest.mark.parametrize(
    ("label", "state", "history"),
    [
        ("postflop has no vocabulary", _state(Street.FLOP), ()),
        ("preflop bet event is impossible", _state(),
         (_event(EventType.BET, 1),)),
        ("street change contradicts preflop state", _state(),
         (StateEvent(EventType.STREET_CHANGE, "hand-1", 1),)),
        ("event without a seat", _state(),
         (StateEvent(EventType.RAISE, "hand-1", 1),)),
        ("unknown seat", _state(), (_event(EventType.RAISE, 99),)),
        ("non-event entry", _state(), ("not-an-event",)),
    ],
)
def test_resolve_action_line_fails_closed(label, state, history):
    assert resolve_action_line(state, history) is None


def test_resolve_action_line_ignores_events_from_an_earlier_hand():
    state = _state(hand_id="hand-2")
    history = (
        _event(EventType.RAISE, 1, hand_id="hand-1"),
        _event(EventType.RAISE, 2, hand_id="hand-1"),
    )

    assert resolve_action_line(state, history) == "unopened"


def test_resolve_action_line_rejects_non_state_input():
    assert resolve_action_line(object(), ()) is None


def test_resolver_tolerates_none_history():
    assert resolve_action_line(_state(), None) is None


def test_build_action_line_resolver_returns_the_production_resolver():
    assert build_action_line_resolver() is resolve_action_line


def test_vocabulary_matches_the_label_set_providers_advertise():
    assert ACTION_LINES == frozenset({
        "unopened", "limp", "multi_limp", "raise", "three_bet",
        "four_bet", "squeeze", "iso_raise", "all_in",
    })
