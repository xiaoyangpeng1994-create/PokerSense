"""Task 1E contract tests: invalid data + immutability after deserialize."""

import json

import pytest

from poker_engine.core.errors import SerializationError
from poker_engine.core.serialization import deserialize, serialize
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from . import helpers as H


# --- Invalid data: fail fast ---

def test_invalid_money_string():
    s = {"__type__": "ChipAmount", "value": "not-a-number", "schema_version": 1}
    with pytest.raises(Exception):
        deserialize(ChipAmount, s)


def test_invalid_money_negative_chipamount():
    s = {"__type__": "ChipAmount", "value": "-5", "schema_version": 1}
    with pytest.raises(Exception):
        deserialize(ChipAmount, s)


def test_money_value_must_be_string():
    s = {"__type__": "ChipDelta", "value": 12.5, "schema_version": 1}
    with pytest.raises(SerializationError):
        deserialize(ChipDelta, s)


def test_unknown_enum_value():
    # Card with unknown rank
    s = {"__type__": "Card", "rank": "Z", "suit": "s", "schema_version": 1}
    with pytest.raises(Exception):
        deserialize(H.Card, s)


def test_naive_datetime_rejected_on_serialize():
    from datetime import datetime
    from poker_engine.core.events import StateEvent, EventType
    with pytest.raises((SerializationError, TypeError)):
        StateEvent(
            event_type=EventType.FOLD, hand_id="h", state_version=0,
            timestamp=datetime(2026, 8, 19),  # naive
        )


def test_missing_required_field():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    del s["pot"]
    with pytest.raises(KeyError):
        deserialize(H.PokerState, s)


def test_wrong_field_type():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    s["state_version"] = "3"  # should be int
    with pytest.raises(TypeError):
        deserialize(H.PokerState, s)


def test_free_mapping_reserved_key_collision():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    ev = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={"__type__": "sneaky"},  # reserved key collision
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    # construction succeeds; serialization must reject the reserved key
    with pytest.raises(SerializationError):
        serialize(ev)


def test_malformed_nested_payload_raises():
    # payload with an object that can't be serialized
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    ev = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={"bad": object()},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    with pytest.raises(SerializationError):
        serialize(ev)


# --- Immutability after deserialize ---

def test_raw_observation_evidence_immutable_after_deser():
    obj = H.raw_observation()
    rt = H.roundtrip(obj)
    with pytest.raises(TypeError):
        rt.hero_cards.evidence["x"] = 1
    assert rt.hero_cards.evidence["box"] == (1, 2, 3)


def test_state_event_payload_immutable_after_deser():
    obj = H.state_event()
    rt = H.roundtrip(obj)
    with pytest.raises(TypeError):
        rt.payload["amount"] = ChipAmount("999")


def test_hand_summary_winnings_immutable_after_deser():
    obj = H.hand_summary()
    rt = H.roundtrip(obj)
    with pytest.raises(TypeError):
        rt.winnings["p0"] = ChipAmount("999")


def test_strategy_action_frequencies_immutable_after_deser():
    obj = H.strategy_report()
    rt = H.roundtrip(obj)
    with pytest.raises(TypeError):
        rt.action_frequencies[H.ActionType.CHECK] = 0.5


def test_state_context_mapping_immutable_after_deser():
    obj = H.state_context()
    rt = H.roundtrip(obj)
    with pytest.raises(TypeError):
        rt.platform_rules["blind"] = {}


# --- request_id preservation (for stale-result protection) ---

def test_request_context_request_id_preserved():
    obj = H.request_context()
    rt = H.roundtrip(obj)
    assert rt.request_id == "req-abc"
    assert rt.hand_id == "h1"
    assert rt.state_version == 3


# --- v2 返修：nested unknown type tag fail fast ---

def test_nested_unknown_type_tag_raises():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    obj = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s = serialize(obj)
    s = json.loads(json.dumps(s))
    # inject an unknown tag into the payload mapping
    s["payload"] = {"bad": {"__type__": "Nope", "x": 1}}
    with pytest.raises(SerializationError):
        deserialize(StateEvent, s)


def test_free_mapping_reserved_key_rejected_on_deser():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    obj = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s = serialize(obj)
    s = json.loads(json.dumps(s))
    s["payload"] = {"__type__": "sneaky"}
    with pytest.raises(SerializationError):
        deserialize(StateEvent, s)


# --- v2 返修：schema_version 类型 + non-finite float ---

def test_schema_version_bool_rejected():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    s["schema_version"] = True  # bool is int subclass; must be rejected
    with pytest.raises(SerializationError):
        deserialize(H.PokerState, s)


def test_generic_non_finite_float_rejected_on_serialize():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    obj = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={"bad": float("nan")},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    with pytest.raises(SerializationError):
        serialize(obj)
