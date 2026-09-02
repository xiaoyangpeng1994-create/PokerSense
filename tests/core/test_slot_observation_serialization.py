"""Serialization tests for ADR-002 SlotObservation additive contract (Task 7A)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from poker_engine.core.enums import ActionType, Street
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.serialization import deserialize, serialize
from poker_engine.core.value_objects import ChipAmount

UTC = timezone.utc


def _aware() -> datetime:
    return datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)


def _field(value=None, confidence=1.0, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=confidence, source="test",
        evidence={}, timestamp=_aware(), validation_status=status,
    )


def _raw(slot_stacks=(), slot_actions=(), slot_occupancies=()):
    return RawObservation(
        frame_seq=1,
        timestamp=_aware(),
        hero_cards=_field(()),
        board_cards=_field(()),
        pot=_field(ChipAmount("10")),
        stacks=_field(()),
        bet_size=_field(ChipAmount("0")),
        action=_field(None, status=ValidationStatus.UNKNOWN),
        street=_field(Street.PREFLOP),
        dealer_pos=_field(0),
        actor=_field(0),
        slot_stacks=slot_stacks,
        slot_actions=slot_actions,
        slot_occupancies=slot_occupancies,
    )


def _slot(slot_id, value, status=ValidationStatus.VALID):
    return SlotObservation(slot_id=slot_id, field=_field(value, 1.0, status))


def _rt(obj):
    return deserialize(type(obj), json.loads(json.dumps(serialize(obj))))


# --- SlotObservation round-trip ---

def test_slot_observation_roundtrip():
    s = SlotObservation(slot_id=2, field=_field(ChipAmount("100")))
    rt = _rt(s)
    assert rt == s
    assert rt.slot_id == 2
    assert rt.field.value == ChipAmount("100")


def test_slot_observation_type_tag():
    raw = _raw(slot_stacks=(_slot(0, ChipAmount("100")),))
    s = serialize(raw)
    assert s["slot_stacks"][0]["__type__"] == "SlotObservation"


# --- migration #1: historical v1 payload without slot_* ---

def test_historical_v1_payload_without_slots_deserializes():
    # A historical RawObservation payload has no slot_stacks/slot_actions keys.
    raw = _raw()
    data = serialize(raw)
    del data["slot_stacks"]
    del data["slot_actions"]
    del data["slot_occupancies"]
    assert "slot_stacks" not in data
    assert "slot_actions" not in data

    rt = deserialize(RawObservation, json.loads(json.dumps(data)))
    assert rt.slot_stacks == ()
    assert rt.slot_actions == ()
    assert rt.slot_occupancies == ()


# --- migration #2: new payload with slot_* exact round-trip ---

def test_new_payload_with_slots_roundtrips():
    raw = _raw(
        slot_stacks=(_slot(0, ChipAmount("100")), _slot(2, ChipAmount("250"))),
        slot_actions=(_slot(1, ActionType.CALL), _slot(4, ActionType.FOLD)),
        slot_occupancies=(_slot(0, True), _slot(4, False)),
    )
    rt = _rt(raw)
    assert rt == raw
    assert [s.slot_id for s in rt.slot_stacks] == [0, 2]
    assert [s.slot_id for s in rt.slot_actions] == [1, 4]
    assert rt.slot_stacks[1].field.value == ChipAmount("250")
    assert rt.slot_actions[1].field.value is ActionType.FOLD
    assert [s.field.value for s in rt.slot_occupancies] == [True, False]


# --- migration #3: new payload with empty slot_* round-trip ---

def test_new_payload_with_empty_slots_roundtrips():
    raw = _raw()
    rt = _rt(raw)
    assert rt == raw
    assert rt.slot_stacks == ()
    assert rt.slot_actions == ()
    assert rt.slot_occupancies == ()


def test_schema_version_remains_1():
    raw = _raw(slot_stacks=(_slot(0, ChipAmount("100")),))
    data = serialize(raw)
    assert data["schema_version"] == 1
