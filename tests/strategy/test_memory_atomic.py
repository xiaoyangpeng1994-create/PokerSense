"""Executable atomic HandMemory fixture and rollback tests."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Street
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandSummary
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory import HandConflictError, InMemoryHandMemory

from .helpers import NOW


FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _player(seat: int, *, hero: bool) -> PlayerState:
    return PlayerState(
        player_id="hero" if hero else f"p{seat}",
        seat=seat,
        position=Position.BB if hero else Position.BTN,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=not hero,
    )


def _state(hand_id: str, version: int) -> PokerState:
    return PokerState(
        state_version=version,
        hand_id=hand_id,
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=(_player(0, hero=False), _player(1, hero=True)),
        pot=ChipAmount("0"),
        current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"),
    )


def _event(hand_id: str, version: int, event_type: EventType) -> StateEvent:
    return StateEvent(
        event_type=event_type,
        hand_id=hand_id,
        state_version=version,
        timestamp=NOW,
    )


def _fixtures() -> list[dict]:
    return [
        item
        for item in (
            json.loads(line) for line in FIXTURES.read_text().splitlines()
        )
        if item["fixture_id"].startswith("MOCK-MEMORY-")
    ]


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda item: item["fixture_id"])
def test_atomic_memory_mock_cases_execute_production_store(fixture):
    operation = fixture["input"]["memory_operation"]
    expected = fixture["expected"]["memory"]
    memory = InMemoryHandMemory()
    if operation["fault"] == "existing_successor":
        memory.start_hand("h2", _state("h2", 0), started_at=NOW - timedelta(3))
        memory.complete_hand(
            "h2",
            HandSummary(final_pot=ChipAmount("0"), winners=()),
            ended_at=NOW - timedelta(2),
        )
        memory.start_hand("h1", _state("h1", 0), started_at=NOW - timedelta(1))
    else:
        memory.start_hand("h1", _state("h1", 0), started_at=NOW - timedelta(1))

    def execute() -> None:
        if operation["operation"] == "record_transition":
            event_hand = (
                "wrong" if operation["fault"] == "wrong_event_hand" else "h1"
            )
            memory.record_transition(
                _state("h1", 1),
                (
                    _event(event_hand, 1, EventType.DEAL),
                    _event(event_hand, 1, EventType.RAISE),
                ),
            )
        else:
            memory.replace_active_hand(
                _state("h2", 0),
                HandSummary(final_pot=ChipAmount("0"), winners=()),
                _event("h1", 0, EventType.HAND_END),
                ended_at=NOW,
                started_at=NOW,
            )

    if expected["committed"]:
        execute()
    else:
        with pytest.raises(HandConflictError):
            execute()

    assert memory.active_hand_id == expected["active_hand_id"]
    assert len(memory.states("h1")) == expected["previous_state_count"]
    assert len(memory.events("h1")) == expected["previous_event_count"]
    assert (
        memory.get_hand_history("h1") is not None
    ) is expected["previous_completed"]
    if operation["fault"] == "existing_successor":
        assert memory.hand_exists("h2")
        assert not memory.is_active("h2")
    else:
        assert memory.hand_exists("h2") is expected["successor_exists"]
