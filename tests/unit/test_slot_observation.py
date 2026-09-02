"""Tests for ADR-002 SlotObservation additive contract (Task 7A)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import ActionType, Street
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.value_objects import ChipAmount

UTC = timezone.utc


def _aware() -> datetime:
    return datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)


def _field(value=None, confidence=1.0, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=confidence, source="test",
        evidence={}, timestamp=_aware(), validation_status=status,
    )


def _slot(slot_id, value=None, confidence=1.0, status=ValidationStatus.VALID):
    return SlotObservation(
        slot_id=slot_id,
        field=_field(value, confidence, status),
    )


def _raw(**kw):
    base = dict(
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
    )
    base.update(kw)
    return RawObservation(**base)


# ---------- SlotObservation ----------

def test_slot_observation_valid():
    s = SlotObservation(slot_id=2, field=_field(ChipAmount("100")))
    assert s.slot_id == 2
    assert s.field.value == ChipAmount("100")


def test_slot_id_bool_rejected():
    with pytest.raises(TypeError):
        SlotObservation(slot_id=True, field=_field())


def test_slot_id_negative_rejected():
    with pytest.raises(ValueError):
        SlotObservation(slot_id=-1, field=_field())


def test_slot_observation_requires_field():
    with pytest.raises(TypeError):
        SlotObservation(slot_id=0, field="not-a-field")


def test_slot_observation_frozen():
    s = SlotObservation(slot_id=0, field=_field())
    with pytest.raises(FrozenInstanceError):
        s.slot_id = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        s.field = _field()  # type: ignore[misc]


# ---------- RawObservation additive fields ----------

def test_default_empty_slot_tuples():
    r = _raw()
    assert r.slot_stacks == ()
    assert r.slot_actions == ()


def test_slot_stacks_populated():
    r = _raw(slot_stacks=(_slot(0, ChipAmount("100")), _slot(1, ChipAmount("200"))))
    assert len(r.slot_stacks) == 2
    assert r.slot_stacks[0].slot_id == 0
    assert r.slot_stacks[0].field.value == ChipAmount("100")


def test_slot_actions_populated():
    r = _raw(slot_actions=(_slot(0, ActionType.CALL), _slot(2, ActionType.FOLD)))
    assert r.slot_actions[0].field.value is ActionType.CALL
    assert r.slot_actions[1].field.value is ActionType.FOLD


def test_duplicate_slot_id_rejected():
    with pytest.raises(ValueError):
        _raw(slot_stacks=(_slot(0), _slot(0)))


def test_non_ascending_slot_id_rejected():
    with pytest.raises(ValueError):
        _raw(slot_stacks=(_slot(1), _slot(0)))


def test_slot_tuple_immutable():
    r = _raw(slot_stacks=(_slot(0, ChipAmount("100")),))
    with pytest.raises((TypeError, AttributeError)):
        r.slot_stacks = ()  # type: ignore[misc]


def test_slot_field_deep_immutable():
    r = _raw(slot_stacks=(_slot(0, ChipAmount("100")),))
    # field itself is frozen
    with pytest.raises(FrozenInstanceError):
        r.slot_stacks[0].field.value = ChipAmount("1")  # type: ignore[misc]


def test_existing_stacks_action_unchanged():
    # existing fields still work exactly as before
    r = _raw(stacks=_field((ChipAmount("1"), ChipAmount("2"))))
    assert r.stacks.value == (ChipAmount("1"), ChipAmount("2"))
    assert r.action.value is None


# ---------- positional constructor compatibility (Blocker 1) ----------

def test_historical_positional_overall_confidence_preserved():
    # A historical positional call binds the trailing arg to overall_confidence.
    r = RawObservation(
        1,  # frame_seq
        _aware(),  # timestamp
        _field(()),  # hero_cards
        _field(()),  # board_cards
        _field(ChipAmount("10")),  # pot
        _field(()),  # stacks
        _field(ChipAmount("0")),  # bet_size
        _field(None, status=ValidationStatus.UNKNOWN),  # action
        _field(Street.PREFLOP),  # street
        _field(0),  # dealer_pos
        _field(0),  # actor
        0.95,  # overall_confidence (historically the trailing positional)
    )
    assert r.overall_confidence == 0.95
    assert r.slot_stacks == ()
    assert r.slot_actions == ()


# ---------- runtime slot payload type invariants (Blocker 2) ----------

def test_slot_stacks_with_wrong_type_rejected():
    with pytest.raises(TypeError):
        _raw(slot_stacks=(_slot(0, ActionType.CALL),))


def test_slot_actions_with_wrong_type_rejected():
    with pytest.raises(TypeError):
        _raw(slot_actions=(_slot(0, ChipAmount("100")),))


def test_slot_unknown_none_value_accepted():
    # None value (UNKNOWN) slot remains valid for both slot kinds.
    r = _raw(
        slot_stacks=(_slot(0, None, status=ValidationStatus.UNKNOWN),),
        slot_actions=(_slot(1, None, status=ValidationStatus.UNKNOWN),),
    )
    assert r.slot_stacks[0].field.value is None
    assert r.slot_actions[0].field.value is None
