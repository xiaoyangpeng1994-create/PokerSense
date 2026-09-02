"""Tests for StateEvent and EventType."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from poker_engine.core.events import EventType, StateEvent

UTC = timezone.utc


def _aware() -> datetime:
    return datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)


def _event(payload=None) -> StateEvent:
    return StateEvent(
        event_type=EventType.BET,
        hand_id="h1",
        state_version=1,
        payload=payload if payload is not None else {},
        timestamp=_aware(),
    )


def test_valid_event():
    e = _event()
    assert e.event_type is EventType.BET
    assert e.hand_id == "h1"


def test_event_type_has_lifecycle_values():
    assert EventType.HAND_START.value == "hand_start"
    assert EventType.DEAL.value == "deal"
    assert EventType.STREET_CHANGE.value == "street_change"
    assert EventType.HAND_END.value == "hand_end"


def test_payload_external_mutation_no_effect():
    payload = {"amount": "10", "extra": []}
    e = _event(payload)
    payload["amount"] = "999"
    payload["injected"] = True
    assert e.payload["amount"] == "10"
    assert "injected" not in e.payload


def test_payload_nested_mutation_no_effect():
    payload = {
        "cards": ["Ah", "Kd"],
        "meta": {"players": ["a", "b"]},
    }
    e = _event(payload)
    payload["cards"].append("Qs")
    payload["meta"]["players"].append("c")
    assert list(e.payload["cards"]) == ["Ah", "Kd"]
    assert list(e.payload["meta"]["players"]) == ["a", "b"]


def test_payload_nested_frozen():
    payload = {"cards": ["Ah", "Kd"]}
    e = _event(payload)
    # nested list was deep-frozen into a tuple, so .append no longer exists
    with pytest.raises(AttributeError):
        e.payload["cards"].append("Qs")  # type: ignore[index]


def test_event_frozen():
    e = _event()
    with pytest.raises(FrozenInstanceError):
        e.hand_id = "h2"  # type: ignore[misc]


def test_event_negative_state_version_invalid():
    with pytest.raises(ValueError):
        StateEvent(
            event_type=EventType.BET, hand_id="h1",
            state_version=-1, timestamp=_aware(),
        )


def test_event_empty_hand_id_invalid():
    with pytest.raises(ValueError):
        StateEvent(
            event_type=EventType.BET, hand_id="",
            state_version=0, timestamp=_aware(),
        )


def test_event_naive_timestamp_rejected():
    with pytest.raises(TypeError):
        StateEvent(
            event_type=EventType.BET, hand_id="h1",
            state_version=0, timestamp=datetime(2026, 8, 18),
        )


# ---------- 返修 v2 新增测试 ----------

def test_event_default_timestamp_is_aware():
    e = StateEvent(event_type=EventType.BET, hand_id="h1", state_version=0)
    assert e.timestamp.tzinfo is not None
    assert e.timestamp.utcoffset() is not None


def test_event_source_empty_rejected():
    with pytest.raises(ValueError):
        StateEvent(
            event_type=EventType.BET, hand_id="h1",
            state_version=0, source="", timestamp=_aware(),
        )
