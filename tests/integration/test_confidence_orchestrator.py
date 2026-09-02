"""Integration tests: Confidence Gate wired into the Application Orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone

from poker_engine.confidence import ConfidenceGate
from poker_engine.core.enums import Street
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.state_engine import StateEngine

from ..orchestrator import fixtures as F

UTC = timezone.utc


def _field(value, confidence=1.0, status=ValidationStatus.VALID):
    return ObservationField(
        value=value, confidence=confidence, source="test",
        evidence={}, timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        validation_status=status,
    )


def _obs(pot_conf=1.0, pot="1.5", street=Street.PREFLOP, street_conf=1.0):
    return RawObservation(
        frame_seq=1,
        timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        hero_cards=_field(()),
        board_cards=_field(()),
        pot=_field(ChipAmount(pot), pot_conf),
        stacks=_field(()),
        bet_size=_field(ChipAmount("0")),
        action=_field(None, 1.0, ValidationStatus.UNKNOWN),
        street=_field(street, street_conf),
        dealer_pos=_field(0),
        actor=_field(None, 1.0, ValidationStatus.UNKNOWN),
    )


def _build():
    return ApplicationOrchestrator(StateEngine(), InMemoryHandMemory())


def _start(orch, pot="100"):
    orch.start_hand(F.initial_state(pot=pot), started_at=F.ts())


def test_case_a_low_pot_confidence_blocked():
    orch = _build()
    _start(orch, pot="100")
    r = orch.process_observation(_obs(pot_conf=0.50, pot="200"))
    assert "pot" in r.confidence_gate.blocked_fields
    assert r.transition.changed is False
    assert r.persisted is False
    assert orch._hand_memory.latest_state("h1").pot == ChipAmount("100")  # noqa: SLF001


def test_case_b_pot_threshold_equality_passes():
    orch = _build()
    _start(orch, pot="100")
    r = orch.process_observation(_obs(pot_conf=0.99, pot="200"))
    assert "pot" not in r.confidence_gate.blocked_fields
    assert r.transition.changed is True
    assert r.persisted is True
    assert orch._hand_memory.latest_state("h1").pot == ChipAmount("200")  # noqa: SLF001


def test_case_c_street_below_threshold_blocked():
    orch = _build()
    _start(orch)
    r = orch.process_observation(
        _obs(street=Street.FLOP, street_conf=0.998)
    )
    assert "street" in r.confidence_gate.blocked_fields
    assert all(e.event_type.name != "STREET_CHANGE" for e in r.transition.events)
    assert r.persisted is False


def test_case_d_street_threshold_passes():
    orch = _build()
    _start(orch)
    r = orch.process_observation(
        _obs(street=Street.FLOP, street_conf=0.999)
    )
    assert "street" not in r.confidence_gate.blocked_fields
    assert any(e.event_type.name == "STREET_CHANGE" for e in r.transition.events)


def test_case_e_context_matches_gate_thresholds():
    from poker_engine.core.state import StateContext

    class CapturingStateEngine(StateEngine):
        def transition(self, previous_state, observation, context):
            self.captured_context = context  # type: ignore[attr-defined]
            return super().transition(previous_state, observation, context)

    spy = CapturingStateEngine()
    gate = ConfidenceGate()
    orch = ApplicationOrchestrator(spy, InMemoryHandMemory(), confidence_gate=gate)
    _start(orch)
    orch.process_observation(_obs(pot="2.0"))
    captured: StateContext = spy.captured_context  # type: ignore[attr-defined]
    assert dict(captured.confidence_thresholds) == dict(gate.thresholds)
