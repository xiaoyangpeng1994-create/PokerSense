"""End-to-end tests for the realtime pipeline (Task 8).

Drives the full chain with a synthetic frame sequence:
    FrameSource -> Vision -> ChangeDetector -> Orchestrator(StateEngine)
    -> RealtimeAnalysis (state + equity + confidence)

No real platform is touched.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest

sys.path.insert(0, "tools")
sys.path.insert(0, "tests")

import run_benchmark  # noqa: E402
from tools.gen_wepoker_dataset import render_table  # noqa: E402

from poker_engine.memory.hand_memory import InMemoryHandMemory  # noqa: E402
from poker_engine.orchestrator import ApplicationOrchestrator  # noqa: E402
from poker_engine.realtime import (  # noqa: E402
    RealtimePipeline,
    SyntheticFrameSource,
)
from poker_engine.state_engine import StateEngine  # noqa: E402
from poker_engine.core.observation import ValidationStatus  # noqa: E402

from orchestrator.fixtures import initial_state  # noqa: E402
from profiles import relaxed_confidence_gate  # noqa: E402


def _build_pipeline(
    frames, *, hero_confirmation_frames=1, new_hand_state_factory=None
):
    """Assemble a realtime pipeline over a synthetic frame sequence."""
    # TEST-ONLY relaxed confidence gate (profiles.py) so synthetic template
    # confidence (max 0.9) can pass. Frozen production thresholds (0.995) are
    # NOT changed — see profiles.py audit note.
    orch = ApplicationOrchestrator(
        StateEngine(),
        InMemoryHandMemory(),
        confidence_gate=relaxed_confidence_gate(),
    )
    orch.start_hand(
        initial_state(hand_id="h1"), started_at=frames[0].timestamp
    )
    source = SyntheticFrameSource(tuple(frames))
    return RealtimePipeline(
        frame_source=source,
        vision=run_benchmark.build_engine(),
        table_map=run_benchmark.table_map(),
        orchestrator=orch,
        equity_trials=1000,
        equity_seed=7,
        hero_confirmation_frames=hero_confirmation_frames,
        new_hand_state_factory=new_hand_state_factory,
    )


def _frame(board_cards, hero_cards, street, pot, bet, stacks, actions, seq):
    img = render_table(
        tuple(board_cards), tuple(hero_cards), pot, bet, stacks, actions
    )
    return run_benchmark._frame(img, seq)


def test_realtime_pipeline_state_accumulates_across_streets():
    # A full preflop -> river progression as a frame sequence.
    frames = (
        _frame([], ["AS", "KD"], "PREFLOP", "10", "5",
               ("100", "200", "300"), ("CHECK", "CALL"), 0),
        _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
               ("100", "200", "300"), ("CHECK", "BET"), 1),
        _frame(["QH", "JD", "TC", "2S"], ["AS", "KD"], "TURN", "50", "20",
               ("150", "250", "350"), ("BET", "CALL"), 2),
        _frame(["QH", "JD", "TC", "2S", "7H"], ["AS", "KD"], "RIVER", "100",
               "40", ("200", "300", "400"), ("BET", "FOLD"), 3),
    )
    pipe = _build_pipeline(frames)

    board_sizes = []
    streets = []
    step = None
    while True:
        step = pipe.step()
        if step is None:
            break
        board_sizes.append(len(step.analysis.state.board_cards))
        streets.append(step.analysis.state.street)

    # board grows 0 -> 3 -> 4 -> 5, street advances preflop -> river.
    assert board_sizes == [0, 3, 4, 5]
    assert [s.value for s in streets] == ["preflop", "flop", "turn", "river"]


def test_latest_analysis_cannot_return_stale_card_confidence_after_loss(monkeypatch):
    frames = tuple(_frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
                          ("100", "200", "300"), ("CHECK", "BET"), seq)
                   for seq in range(2))
    pipe = _build_pipeline(frames)
    first = pipe.step()
    original = pipe._vision.process

    def unreadable(frame, table):
        obs = original(frame, table)
        return replace(obs, hero_cards=replace(
            obs.hero_cards, value=None, confidence=0.0,
            validation_status=ValidationStatus.UNKNOWN))

    monkeypatch.setattr(pipe._vision, "process", unreadable)
    lost = pipe.step()
    assert dict(lost.analysis.confidence.field_status)["hero_cards"] == "unknown"
    assert lost.analysis.state.hero_cards == first.analysis.state.hero_cards
    assert pipe.latest_analysis() == lost.analysis


def test_latest_analysis_matches_guarded_snapshot_without_poisoning_equity_cache():
    frames = tuple(_frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
                          ("100", "200", "300"), ("CHECK", "BET"), seq)
                   for seq in range(3))
    pipe = _build_pipeline(frames)
    first = pipe.step()
    pipe._equity_input_guard = lambda state, obs: "temporary_test_guard"
    blocked = pipe.step()
    assert blocked.analysis.equity.unavailable_reason == "temporary_test_guard"
    assert pipe.latest_analysis() == blocked.analysis
    pipe._equity_input_guard = None
    recovered = pipe.step()
    assert recovered.analysis.equity == first.analysis.equity
    assert pipe.latest_analysis() == recovered.analysis


def test_capture_failure_invalidates_snapshot_and_restarts_consensus(monkeypatch):
    frames = tuple(_frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
                          ("100", "200", "300"), ("CHECK", "BET"), seq)
                   for seq in range(3))
    pipe = _build_pipeline(frames, hero_confirmation_frames=2)
    pipe.step()
    before = pipe.step()
    assert dict(before.analysis.confidence.field_status)["hero_cards"] == "valid"
    next_frame = pipe._frame_source.next_frame

    def failed():
        raise RuntimeError("injected capture interruption")

    resets = []
    monkeypatch.setattr(pipe._vision, "reset_temporal", lambda: resets.append(True))
    monkeypatch.setattr(pipe._frame_source, "next_frame", failed)
    with pytest.raises(RuntimeError, match="injected capture"):
        pipe.step()
    unavailable = pipe.latest_analysis()
    assert unavailable.state == before.analysis.state
    assert unavailable.equity.unavailable_reason == "capture_unavailable"
    assert all(status == "unknown" for _, status in unavailable.confidence.field_status)
    assert resets == [True]
    monkeypatch.setattr(pipe._frame_source, "next_frame", next_frame)
    resumed = pipe.step()
    assert dict(resumed.analysis.confidence.field_status)["hero_cards"] == "unknown"


def test_change_detector_skips_duplicate_frames():
    # Identical consecutive frames must NOT trigger a fresh analysis.
    frames = (
        _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
               ("100", "200", "300"), ("CHECK", "BET"), 0),
        _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
               ("100", "200", "300"), ("CHECK", "BET"), 1),
    )
    pipe = _build_pipeline(frames)

    first = pipe.step()
    second = pipe.step()

    assert first.analysis_changed is True       # first frame always records
    assert second.analysis_changed is False     # duplicate frame: no change
    # a material change WAS detected on the first, none on the second
    assert second.change.changed is False


def test_hero_cards_require_two_matching_frames_when_configured():
    """A transient first read cannot seed the immutable hero hand."""
    frame = _frame([], ["AS", "KD"], "PREFLOP", "10", "5",
                   ("100", "200", "300"), ("CHECK", "CALL"), 0)
    next_frame = _frame([], ["AS", "KD"], "PREFLOP", "10", "5",
                        ("100", "200", "300"), ("CHECK", "CALL"), 1)
    pipe = _build_pipeline((frame, next_frame), hero_confirmation_frames=2)

    first = pipe.step()
    second = pipe.step()

    assert first is not None and second is not None
    assert first.analysis.state.hero_cards == ()
    assert dict(first.analysis.confidence.field_status)["hero_cards"] == "unknown"
    assert [str(card) for card in second.analysis.state.hero_cards] == ["As", "Kd"]
    assert dict(second.analysis.confidence.field_status)["hero_cards"] == "valid"


def test_confirmed_different_hero_cards_start_a_new_hand():
    """A newly dealt pair must not be rejected as a mutation of the old hand."""
    frames = (
        _frame([], ["AS", "KD"], "PREFLOP", "10", "5",
               ("100", "200", "300"), ("CHECK", "CALL"), 0),
        _frame([], ["AS", "KD"], "PREFLOP", "10", "5",
               ("100", "200", "300"), ("CHECK", "CALL"), 1),
        _frame([], ["QH", "8S"], "PREFLOP", "10", "5",
               ("100", "200", "300"), ("CHECK", "CALL"), 2),
        _frame([], ["QH", "8S"], "PREFLOP", "10", "5",
               ("100", "200", "300"), ("CHECK", "CALL"), 3),
    )
    pipe = _build_pipeline(
        frames,
        hero_confirmation_frames=2,
        new_hand_state_factory=lambda: initial_state(hand_id="h2"),
    )

    _first, second, pending, new_hand = (pipe.step() for _ in range(4))

    assert [str(card) for card in second.analysis.state.hero_cards] == ["As", "Kd"]
    assert dict(pending.analysis.confidence.field_status)["hero_cards"] == (
        ValidationStatus.UNKNOWN.value
    )
    assert pipe._orchestrator._hand_memory.active_hand_id == "h2"
    assert [str(card) for card in new_hand.analysis.state.hero_cards] == ["Qh", "8s"]
    assert new_hand.analysis_changed is True


def test_realtime_equity_is_a_valid_probability():
    frames = (
        _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
               ("100", "200", "300"), ("CHECK", "BET"), 0),
    )
    pipe = _build_pipeline(frames)
    step = pipe.step()
    assert step is not None
    eq = step.analysis.equity
    assert 0.0 <= eq.win_rate <= 1.0
    assert 0.0 <= eq.tie_rate <= 1.0
    assert eq.win_rate + eq.tie_rate <= 1.0 + 1e-9


def test_realtime_confidence_reflects_recognition():
    frames = (
        _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
               ("100", "200", "300"), ("CHECK", "BET"), 0),
    )
    pipe = _build_pipeline(frames)
    step = pipe.step()
    assert step is not None
    conf = step.analysis.confidence
    assert 0.0 <= conf.overall_confidence <= 1.0
    # field statuses are present for the tracked fields.
    names = {name for name, _st in conf.field_status}
    assert {"hero_cards", "board_cards", "street", "pot"} <= names


def test_realtime_pipeline_exhausts_cleanly():
    frames = (_frame([], ["AS", "KD"], "PREFLOP", "10", "5",
                     ("100", "200", "300"), ("CHECK", "CALL"), 0),)
    pipe = _build_pipeline(frames)
    first = pipe.step()
    assert first is not None
    assert pipe.step() is None  # exhausted


# ---------------------------------------------------------------------------
# Regression: production vs test-only confidence profile (audit requirement)
# ---------------------------------------------------------------------------

def test_production_threshold_gates_synthetic_to_unknown():
    # The SAME synthetic frame, gated with the PRODUCTION (default frozen)
    # ConfidenceGate, must demote low-confidence fields to UNKNOWN — proving
    # production thresholds are untouched and synthetic data cannot pass them.
    from poker_engine.confidence.gate import ConfidenceGate
    from poker_engine.core.observation import ValidationStatus

    img = _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
                 ("100", "200", "300"), ("CHECK", "BET"), 0)
    obs = run_benchmark.build_engine().process(img, run_benchmark.table_map())

    prod_gate = ConfidenceGate()  # default = frozen thresholds
    gated = prod_gate.apply(obs).observation
    assert gated.hero_cards.validation_status is ValidationStatus.UNKNOWN
    assert gated.board_cards.validation_status is ValidationStatus.UNKNOWN


def test_relaxed_profile_passes_same_frame_valid():
    from poker_engine.core.observation import ValidationStatus

    img = _frame(["QH", "JD", "TC"], ["AS", "KD"], "FLOP", "25", "10",
                 ("100", "200", "300"), ("CHECK", "BET"), 0)
    obs = run_benchmark.build_engine().process(img, run_benchmark.table_map())

    gated = relaxed_confidence_gate().apply(obs).observation
    assert gated.hero_cards.validation_status is ValidationStatus.VALID
    assert gated.board_cards.validation_status is ValidationStatus.VALID


def test_frozen_production_thresholds_unchanged():
    from poker_engine.confidence.gate import ConfidenceGate

    prod = ConfidenceGate().thresholds
    # Frozen production values remain exactly as defined.
    assert prod["hero_cards"] >= 0.995
    assert prod["board_cards"] >= 0.995
    assert prod["street"] >= 0.999


# ---------------------------------------------------------------------------
# equity strategy injectability + exact-vs-MC convergence (Task 9 anchor)
# ---------------------------------------------------------------------------

def test_exact_and_montecarlo_equity_strategies_converge_on_river():
    from poker_engine.core.enums import Rank, Street, Suit
    from poker_engine.core.value_objects import Card
    from poker_engine.realtime import (
        ExactRandomRangeEquity,
        MonteCarloRandomRangeEquity,
    )

    def C(s):
        return Card(Rank(s[0]), Suit(s[1].lower()))

    hero = (C("As"), C("Kd"))
    board = (C("Qh"), C("Jd"), C("Tc"), C("2s"), C("7h"))
    state = initial_state(hand_id="h", street=Street.RIVER, hero=hero, board=board)

    exact = ExactRandomRangeEquity().compute(state)
    mc = MonteCarloRandomRangeEquity(trials=10000, seed=3).compute(state)
    assert exact.win_rate == pytest.approx(mc.win_rate, abs=0.01)
    assert exact.tie_rate == pytest.approx(mc.tie_rate, abs=0.01)


def test_pipeline_accepts_injected_exact_equity_strategy():
    from poker_engine.realtime import ExactRandomRangeEquity
    from poker_engine.core.enums import Street

    # RIVER (board full) so exact enumeration is cheap (no board completions).
    frames = (
        _frame(["QH", "JD", "TC", "2S", "7H"], ["AS", "KD"], "RIVER", "100",
               "40", ("200", "300", "400"), ("BET", "FOLD"), 0),
    )
    orch = ApplicationOrchestrator(
        StateEngine(),
        InMemoryHandMemory(),
        confidence_gate=relaxed_confidence_gate(),
    )
    orch.start_hand(initial_state(hand_id="h1", street=Street.RIVER))
    pipe = RealtimePipeline(
        frame_source=SyntheticFrameSource(frames),
        vision=run_benchmark.build_engine(),
        table_map=run_benchmark.table_map(),
        orchestrator=orch,
        equity_strategy=ExactRandomRangeEquity(),
    )
    step = pipe.step()
    assert step is not None
    eq = step.analysis.equity
    assert 0.0 <= eq.win_rate <= 1.0
    assert 0.0 <= eq.tie_rate <= 1.0
