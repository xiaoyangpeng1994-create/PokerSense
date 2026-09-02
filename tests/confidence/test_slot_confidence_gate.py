"""Confidence Gate per-slot tests for ADR-002 (Task 7A)."""

from __future__ import annotations

from datetime import datetime, timezone

from poker_engine.confidence import ConfidenceGate
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
        slot_id=slot_id, field=_field(value, confidence, status),
    )


def _raw(slot_stacks=(), slot_actions=(), slot_occupancies=(), **extra):
    params = dict(
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
    params.update(extra)
    return RawObservation(**params)


def test_slot_stack_below_threshold_gated():
    gate = ConfidenceGate()
    # stacks threshold = 0.99; slot with confidence 0.5 -> demoted
    raw = _raw(slot_stacks=(_slot(2, ChipAmount("100"), confidence=0.5),))
    r = gate.apply(raw)
    assert "slot_stacks[slot_id=2]" in r.blocked_fields
    assert (
        r.observation.slot_stacks[0].field.validation_status
        is ValidationStatus.UNKNOWN
    )
    assert r.observation.slot_stacks[0].field.value is None


def test_slot_action_below_threshold_gated():
    gate = ConfidenceGate()
    # action threshold = 0.99
    raw = _raw(slot_actions=(_slot(5, ActionType.RAISE, confidence=0.5),))
    r = gate.apply(raw)
    assert "slot_actions[slot_id=5]" in r.blocked_fields
    assert (
        r.observation.slot_actions[0].field.validation_status
        is ValidationStatus.UNKNOWN
    )


def test_slot_occupancy_uses_independent_slot_threshold():
    gate = ConfidenceGate(slot_thresholds={"occupancy": 0.8})
    raw = _raw(slot_occupancies=(_slot(4, False, confidence=0.79),))

    result = gate.apply(raw)

    assert "slot_occupancies[slot_id=4]" in result.blocked_fields
    assert result.observation.slot_occupancies[0].field.value is None


def test_slot_threshold_exact_passes():
    gate = ConfidenceGate()
    raw = _raw(slot_stacks=(_slot(0, ChipAmount("100"), confidence=0.99),))
    r = gate.apply(raw)
    assert "slot_stacks[slot_id=0]" not in r.blocked_fields
    assert (
        r.observation.slot_stacks[0].field.validation_status
        is ValidationStatus.VALID
    )


def test_blocked_path_encodes_slot_id_not_index():
    gate = ConfidenceGate()
    # slot_ids are 2 and 7 (not contiguous 0,1) — blocked path must encode 7, not 1
    raw = _raw(
        slot_stacks=(
            _slot(2, ChipAmount("100"), confidence=0.5),
            _slot(7, ChipAmount("200"), confidence=0.5),
        )
    )
    r = gate.apply(raw)
    assert "slot_stacks[slot_id=2]" in r.blocked_fields
    assert "slot_stacks[slot_id=7]" in r.blocked_fields
    assert "slot_stacks[0]" not in r.blocked_fields  # never tuple index
    assert "slot_stacks[1]" not in r.blocked_fields


def test_blocked_fields_ordering():
    gate = ConfidenceGate()
    raw = _raw(
        hero_cards=ObservationField(
            value=(), confidence=0.1, source="t", evidence={},
            timestamp=_aware(), validation_status=ValidationStatus.VALID,
        ),
        slot_stacks=(
            _slot(1, ChipAmount("200"), confidence=0.5),
            _slot(3, ChipAmount("100"), confidence=0.5),
        ),
        slot_actions=(_slot(2, ActionType.CALL, confidence=0.5),),
    )
    r = gate.apply(raw)
    # ordering: original fields first, then slot_stacks ascending, then slot_actions
    idx_hero = r.blocked_fields.index("hero_cards")
    idx_ss1 = r.blocked_fields.index("slot_stacks[slot_id=1]")
    idx_ss3 = r.blocked_fields.index("slot_stacks[slot_id=3]")
    idx_sa2 = r.blocked_fields.index("slot_actions[slot_id=2]")
    assert idx_hero < idx_ss1 < idx_ss3 < idx_sa2


def test_unknown_slot_not_reported_as_blocked():
    gate = ConfidenceGate()
    # UNKNOWN slot passes through unchanged -> NOT gate-blocked
    raw = _raw(
        slot_stacks=(
            _slot(0, None, confidence=0.0, status=ValidationStatus.UNKNOWN),
        )
    )
    r = gate.apply(raw)
    assert "slot_stacks[slot_id=0]" not in r.blocked_fields
    assert (
        r.observation.slot_stacks[0].field.validation_status
        is ValidationStatus.UNKNOWN
    )


def test_conflict_slot_not_reported_as_blocked():
    gate = ConfidenceGate()
    raw = _raw(
        slot_actions=(
            _slot(0, None, confidence=0.0, status=ValidationStatus.CONFLICT),
        )
    )
    r = gate.apply(raw)
    assert "slot_actions[slot_id=0]" not in r.blocked_fields
    assert (
        r.observation.slot_actions[0].field.validation_status
        is ValidationStatus.CONFLICT
    )


def test_original_observation_unchanged_after_gate():
    gate = ConfidenceGate()
    raw = _raw(slot_stacks=(_slot(0, ChipAmount("100"), confidence=0.5),))
    gate.apply(raw)
    assert raw.slot_stacks[0].field.validation_status is ValidationStatus.VALID
    assert raw.slot_stacks[0].field.value == ChipAmount("100")


def test_no_slots_no_extra_blocked():
    gate = ConfidenceGate()
    raw = _raw()
    r = gate.apply(raw)
    assert r.blocked_fields == ()  # no fields demoted
