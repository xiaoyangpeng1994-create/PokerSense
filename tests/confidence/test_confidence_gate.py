"""Unit tests for Confidence Gate (field-level sanitization)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from poker_engine.confidence import ConfidenceGate, ConfidenceGateError
from poker_engine.core.enums import ActionType, Street
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.value_objects import ChipAmount

UTC = timezone.utc
TS = datetime(2026, 8, 19, 1, 0, tzinfo=UTC)


def _field(value, confidence=1.0, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=confidence, source="test",
        evidence={"roi": [1, 2]}, timestamp=TS, validation_status=status,
    )


def _obs(hero_conf=1.0, board_conf=1.0, street_conf=1.0, pot_conf=1.0,
         stacks_conf=1.0, bet_size_conf=1.0, action_conf=1.0,
         hero_status=ValidationStatus.VALID):
    return RawObservation(
        frame_seq=1,
        timestamp=TS,
        hero_cards=_field((), hero_conf, hero_status),
        board_cards=_field((), board_conf),
        pot=_field(ChipAmount("10"), pot_conf),
        stacks=_field((), stacks_conf),
        bet_size=_field(ChipAmount("0"), bet_size_conf),
        action=_field(ActionType.CALL, action_conf),
        street=_field(Street.PREFLOP, street_conf),
        dealer_pos=_field(0, 1.0),
        actor=_field(0, 1.0),
        overall_confidence=0.5,
    )


# --- default thresholds ---

def test_default_thresholds():
    g = ConfidenceGate()
    t = dict(g.thresholds)
    assert t == {
        "hero_cards": 0.995,
        "board_cards": 0.995,
        "street": 0.999,
        "pot": 0.99,
        "stacks": 0.99,
        "bet_size": 0.99,
        "action": 0.99,
    }


def test_thresholds_immutable_exposure():
    g = ConfidenceGate()
    t = g.thresholds
    with pytest.raises(TypeError):
        t["pot"] = 0.5  # type: ignore[index]


# --- field gating ---

_FIELD_NAMES = [
    "hero_cards", "board_cards", "street", "pot", "stacks", "bet_size", "action",
]

_THRESHOLDS = {
    "hero_cards": 0.995,
    "board_cards": 0.995,
    "street": 0.999,
    "pot": 0.99,
    "stacks": 0.99,
    "bet_size": 0.99,
    "action": 0.99,
}


def _obs_field_conf(field: str, conf: float) -> RawObservation:
    """Build an observation with a single field's confidence overridden."""

    def mk(name, default_conf):
        if name == field:
            return _field(_value_for(name), conf)
        return _field(_value_for(name), default_conf)

    return RawObservation(
        frame_seq=1,
        timestamp=TS,
        hero_cards=mk("hero_cards", 0.995),
        board_cards=mk("board_cards", 0.995),
        pot=mk("pot", 0.99),
        stacks=mk("stacks", 0.99),
        bet_size=mk("bet_size", 0.99),
        action=mk("action", 0.99),
        street=mk("street", 0.999),
        dealer_pos=_field(0, 1.0),
        actor=_field(0, 1.0),
        overall_confidence=0.5,
    )


def _value_for(name):
    if name in ("hero_cards", "board_cards", "stacks"):
        return ()
    if name == "pot":
        return ChipAmount("10")
    if name == "bet_size":
        return ChipAmount("0")
    if name == "action":
        return ActionType.CALL
    if name == "street":
        return Street.PREFLOP
    return ()


@pytest.mark.parametrize("field", _FIELD_NAMES)
def test_below_threshold_blocked(field):
    g = ConfidenceGate()
    threshold = _THRESHOLDS[field]
    obs = _obs_field_conf(field, threshold - 0.0001)
    r = g.apply(obs)
    gated = getattr(r.observation, field)
    assert gated.validation_status is ValidationStatus.UNKNOWN
    assert gated.value is None
    assert field in r.blocked_fields


@pytest.mark.parametrize("field", _FIELD_NAMES)
def test_exact_threshold_passes(field):
    g = ConfidenceGate()
    threshold = _THRESHOLDS[field]
    obs = _obs_field_conf(field, threshold)
    r = g.apply(obs)
    gated = getattr(r.observation, field)
    assert gated.validation_status is ValidationStatus.VALID
    assert field not in r.blocked_fields


def test_above_threshold_passes():
    g = ConfidenceGate()
    obs = _obs(pot_conf=0.99 + 0.001)
    r = g.apply(obs)
    assert r.observation.pot.validation_status is ValidationStatus.VALID


# --- status preservation ---

def test_low_confidence_demoted_even_if_high_numeric():
    g = ConfidenceGate()
    obs = _obs(hero_conf=0.9999, hero_status=ValidationStatus.LOW_CONFIDENCE)
    r = g.apply(obs)
    assert r.observation.hero_cards.validation_status is ValidationStatus.UNKNOWN
    assert r.observation.hero_cards.value is None


def test_unknown_stays_unknown():
    g = ConfidenceGate()
    obs = _obs(pot_conf=0.0)
    # force UNKNOWN status on pot
    obs = RawObservation(
        frame_seq=1, timestamp=TS,
        hero_cards=_field(()), board_cards=_field(()),
        pot=_field(None, 1.0, ValidationStatus.UNKNOWN),
        stacks=_field(()), bet_size=_field(ChipAmount("0")),
        action=_field(ActionType.CALL), street=_field(Street.PREFLOP),
        dealer_pos=_field(0), actor=_field(0), overall_confidence=0.5,
    )
    r = g.apply(obs)
    assert r.observation.pot.validation_status is ValidationStatus.UNKNOWN


def test_conflict_stays_conflict():
    g = ConfidenceGate()
    obs = RawObservation(
        frame_seq=1, timestamp=TS,
        hero_cards=_field(()), board_cards=_field(()),
        pot=_field(ChipAmount("10"), 0.5, ValidationStatus.CONFLICT),
        stacks=_field(()), bet_size=_field(ChipAmount("0")),
        action=_field(ActionType.CALL), street=_field(Street.PREFLOP),
        dealer_pos=_field(0), actor=_field(0), overall_confidence=0.5,
    )
    r = g.apply(obs)
    assert r.observation.pot.validation_status is ValidationStatus.CONFLICT


# --- preserved evidence / no mutation ---

def test_sanitized_field_preserves_evidence():
    g = ConfidenceGate()
    obs = _obs(pot_conf=0.5)
    r = g.apply(obs)
    gated = r.observation.pot
    assert gated.confidence == 0.5
    assert gated.source == "test"
    # evidence deep-frozen: list became tuple
    assert gated.evidence == {"roi": (1, 2)}
    assert gated.timestamp == TS


def test_original_observation_unchanged():
    g = ConfidenceGate()
    obs = _obs(pot_conf=0.0)  # blocked
    _ = g.apply(obs)
    assert obs.pot.validation_status is ValidationStatus.VALID
    assert obs.pot.value == ChipAmount("10")


def test_actor_dealer_overall_unchanged():
    g = ConfidenceGate()
    obs = _obs(hero_conf=0.1)  # hero blocked only
    r = g.apply(obs)
    # actor/dealer_pos/overall_confidence untouched
    assert r.observation.actor.validation_status is ValidationStatus.VALID
    assert r.observation.dealer_pos.validation_status is ValidationStatus.VALID
    assert r.observation.overall_confidence == 0.5


def test_blocked_fields_deterministic_order():
    g = ConfidenceGate()
    obs = _obs(hero_conf=0.1, pot_conf=0.1, action_conf=0.1)
    r = g.apply(obs)
    # fields demoted in fixed order
    assert r.blocked_fields == ("hero_cards", "pot", "action")
    # determinism
    assert g.apply(obs).blocked_fields == r.blocked_fields


# --- threshold validation ---

def test_missing_threshold_key_raises():
    partial = {"hero_cards": 0.995}  # missing 6 others
    with pytest.raises(ConfidenceGateError):
        ConfidenceGate(thresholds=partial)


def test_unknown_threshold_key_raises():
    full = {k: v for k, v in ConfidenceGate().thresholds.items()}
    full["bogus"] = 0.5
    with pytest.raises(ConfidenceGateError):
        ConfidenceGate(thresholds=full)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_invalid_threshold_value_raises(bad):
    full = {k: v for k, v in ConfidenceGate().thresholds.items()}
    full["pot"] = bad
    with pytest.raises(ConfidenceGateError):
        ConfidenceGate(thresholds=full)


def test_bool_threshold_value_raises_typeerror():
    full = {k: v for k, v in ConfidenceGate().thresholds.items()}
    full["pot"] = True
    with pytest.raises(TypeError):
        ConfidenceGate(thresholds=full)


def test_negative_inf_rejected():
    full = {k: v for k, v in ConfidenceGate().thresholds.items()}
    full["pot"] = -math.inf
    with pytest.raises(ConfidenceGateError):
        ConfidenceGate(thresholds=full)
