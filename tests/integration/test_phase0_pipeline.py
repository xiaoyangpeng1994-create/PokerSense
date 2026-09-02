"""End-to-end Phase 0 pipeline test: fake observation -> orchestrator -> memory."""

from __future__ import annotations

from poker_engine.core.enums import Street
from poker_engine.core.hand import HandSummary
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.state_engine import StateEngine

from ..orchestrator import fixtures as F


def _build():
    return ApplicationOrchestrator(StateEngine(), InMemoryHandMemory())


def test_full_preflop_to_river_pipeline():
    orch = _build()
    orch.start_hand(F.initial_state(), started_at=F.ts())

    # preflop pot update
    r1 = orch.process_observation(F.observation(frame_seq=1, pot="2.0"))
    assert r1.persisted

    # flop: street + board
    r2 = orch.process_observation(
        F.observation(frame_seq=2, street=Street.FLOP, board=(F.Qh, F.Jh, F.Th))
    )
    assert r2.persisted

    # turn
    r3 = orch.process_observation(
        F.observation(
            frame_seq=3, street=Street.TURN, board=(F.Qh, F.Jh, F.Th, F.NineH)
        )
    )
    assert r3.persisted

    # river
    r4 = orch.process_observation(
        F.observation(
            frame_seq=4,
            street=Street.RIVER,
            board=(F.Qh, F.Jh, F.Th, F.NineH, F.Ac),
        )
    )
    assert r4.persisted

    # complete
    history = orch.complete_hand(
        "h1",
        HandSummary(final_pot=ChipAmount("2.0"), winners=("p0",)),
        ended_at=F.ts(minute=10),
    )

    # reconstruction via memory readback
    mem = orch._hand_memory  # noqa: SLF001
    assert mem.latest_state("h1").street is Street.RIVER
    assert len(mem.states("h1")) == 5  # version 0..4
    assert history.hand_id == "h1"
    # events preserved in order (STREET_CHANGE before DEAL per transition)
    assert len(mem.events("h1")) >= 4
    # determinism: repeat the same input sequence on a fresh instance
    orch2 = _build()
    orch2.start_hand(F.initial_state(), started_at=F.ts())
    for obs in [
        F.observation(frame_seq=1, pot="2.0"),
        F.observation(
            frame_seq=2, street=Street.FLOP, board=(F.Qh, F.Jh, F.Th)
        ),
        F.observation(
            frame_seq=3,
            street=Street.TURN,
            board=(F.Qh, F.Jh, F.Th, F.NineH),
        ),
        F.observation(
            frame_seq=4,
            street=Street.RIVER,
            board=(F.Qh, F.Jh, F.Th, F.NineH, F.Ac),
        ),
    ]:
        orch2.process_observation(obs)
    mem2 = orch2._hand_memory  # noqa: SLF001
    assert mem2.states("h1") == mem.states("h1")
    assert mem2.events("h1") == mem.events("h1")


def test_duplicate_observation_no_extra_state():
    orch = _build()
    orch.start_hand(F.initial_state(), started_at=F.ts())
    orch.process_observation(F.observation(frame_seq=1, pot="2.0"))
    orch.process_observation(F.observation(frame_seq=2, pot="2.0"))  # dup
    mem = orch._hand_memory  # noqa: SLF001
    assert len(mem.states("h1")) == 2  # version 0 + version 1 only


def test_pot_regression_warns_not_invalid():
    orch = _build()
    orch.start_hand(F.initial_state(pot="5.0"), started_at=F.ts())
    r = orch.process_observation(F.observation(pot="3.0"))  # decrease
    assert r.transition.validation.is_valid is True
    assert r.persisted is False  # no material change (pot retained)
    assert any(
        "pot regression" in w for w in r.transition.validation.warnings
    )
    mem = orch._hand_memory  # noqa: SLF001
    assert mem.latest_state("h1").pot == ChipAmount("5.0")


def test_card_identity_conflict_invalid():
    orch = _build()
    orch.start_hand(
        F.initial_state(street=Street.FLOP, board=(F.Qh, F.Jh, F.Th)),
        started_at=F.ts(),
    )
    # board 3 -> 3 with a replaced card -> invalid, no persist
    r = orch.process_observation(
        F.observation(board=(F.Qh, F.Jh, F.NineH))
    )
    assert r.transition.validation.is_valid is False
    assert r.persisted is False
