"""Unit tests for ApplicationOrchestrator."""

from __future__ import annotations

import pytest

from poker_engine.core.enums import Street
from poker_engine.core.hand import HandSummary
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import (
    ApplicationOrchestrator,
    OrchestratorError,
)
from poker_engine.state_engine import StateEngine

from . import fixtures as F


@pytest.fixture
def orch():
    return ApplicationOrchestrator(StateEngine(), InMemoryHandMemory())


def _started(orch):
    orch.start_hand(F.initial_state(), started_at=F.ts())
    return orch


# ---------- lifecycle ----------

def test_start_hand_delegates(orch):
    _started(orch)
    assert orch._hand_memory.active_hand_id == "h1"  # noqa: SLF001


def test_process_no_active_hand_raises(orch):
    with pytest.raises(OrchestratorError):
        orch.process_observation(F.observation(pot="2.0"))


# ---------- material change / persistence ----------

def test_material_change_persists(orch):
    _started(orch)
    r = orch.process_observation(F.observation(pot="2.0"))
    assert r.transition.changed is True
    assert r.persisted is True
    # memory now has version 1 + latest pot updated
    mem = orch._hand_memory  # noqa: SLF001
    latest = mem.latest_state("h1")
    assert latest.state_version == 1
    assert latest.pot == ChipAmount("2.0")


def test_noop_does_not_persist(orch):
    _started(orch)
    r = orch.process_observation(F.observation(pot="1.5"))  # unchanged
    assert r.transition.changed is False
    assert r.persisted is False
    mem = orch._hand_memory  # noqa: SLF001
    # still only version 0
    assert mem.latest_state("h1").state_version == 0


def test_invalid_does_not_persist(orch):
    _started(orch)
    # street regression: previous PREFLOP, observe nothing forward... use a
    # board regression instead: previous board () -> can't regress. Use pot
    # down is warning not invalid. Build a street regression via a turn state.
    orch2 = ApplicationOrchestrator(StateEngine(), InMemoryHandMemory())
    orch2.start_hand(
        F.initial_state(street=Street.TURN, pot="5.0"), started_at=F.ts()
    )
    r = orch2.process_observation(F.observation(street=Street.FLOP))
    assert r.transition.validation.is_valid is False
    assert r.persisted is False
    assert orch2._hand_memory.latest_state("h1").state_version == 0  # noqa: SLF001


def test_transition_composition(orch):
    _started(orch)
    r = orch.process_observation(F.observation(pot="2.0"))
    # OrchestrationResult composes StateTransitionResult, no duplicated fields
    assert hasattr(r, "transition")
    assert hasattr(r, "persisted")
    assert not hasattr(r, "state")  # fields come via .transition, not top-level


# ---------- event persistence order (state-first) ----------

def test_state_first_then_event(orch):
    _started(orch)
    r = orch.process_observation(
        F.observation(street=Street.FLOP, board=(F.Qh, F.Jh, F.Th))
    )
    assert r.persisted is True
    mem = orch._hand_memory  # noqa: SLF001
    # state snapshot exists for the event's state_version
    for e in r.transition.events:
        assert mem.get_state("h1", e.state_version) is not None


# ---------- deterministic persistence ----------

def test_repeated_scenario_deterministic(orch):
    _started(orch)
    orch.process_observation(F.observation(pot="2.0"))
    orch.process_observation(
        F.observation(street=Street.FLOP, board=(F.Qh, F.Jh, F.Th))
    )
    mem = orch._hand_memory  # noqa: SLF001
    snapshots = mem.states("h1")
    assert snapshots[1].pot == ChipAmount("2.0")
    assert snapshots[2].board_cards == (F.Qh, F.Jh, F.Th)


def test_completed_hand_write_rejected(orch):
    _started(orch)
    orch.process_observation(F.observation(pot="2.0"))
    orch.complete_hand(
        "h1",
        HandSummary(final_pot=ChipAmount("2.0"), winners=("p0",)),
        ended_at=F.ts(minute=5),
    )
    # after complete, process_observation should fail (no active hand)
    with pytest.raises(OrchestratorError):
        orch.process_observation(F.observation(pot="3.0"))


# ---------- type errors ----------

def test_bad_engine_type():
    with pytest.raises(TypeError):
        ApplicationOrchestrator("not-engine", InMemoryHandMemory())


def test_bad_memory_type():
    with pytest.raises(TypeError):
        ApplicationOrchestrator(StateEngine(), "not-memory")


def test_process_non_observation(orch):
    _started(orch)
    with pytest.raises(TypeError):
        orch.process_observation("not-an-obs")
