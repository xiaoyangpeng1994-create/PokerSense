"""Per-stage latency instrumentation on :class:`RealtimePipeline`.

Observation only: timings must never change what a step decides. These tests
pin both halves of that contract — the stages are recorded, and injecting a
fake clock does not alter the analysis.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "tools")
sys.path.insert(0, "tests")
sys.path.insert(0, "tests/realtime")

import run_benchmark  # noqa: E402
from tools.gen_wepoker_dataset import render_table  # noqa: E402

from orchestrator.fixtures import initial_state  # noqa: E402
from profiles import relaxed_confidence_gate  # noqa: E402

from poker_engine.memory.hand_memory import InMemoryHandMemory  # noqa: E402
from poker_engine.orchestrator import ApplicationOrchestrator  # noqa: E402
from poker_engine.realtime import RealtimePipeline, SyntheticFrameSource  # noqa: E402
from poker_engine.realtime.pipeline import (  # noqa: E402
    STAGE_ADVANCE,
    STAGE_BOUNDARY,
    STAGE_CAPTURE,
    STAGE_CHANGE,
    STAGE_CONSENSUS,
    STAGE_EQUITY,
    STAGE_TOTAL,
    STAGE_VISION,
)
from poker_engine.state_engine import StateEngine  # noqa: E402


def _frame(board_cards, hero_cards, pot, bet, stacks, actions, seq):
    img = render_table(
        tuple(board_cards), tuple(hero_cards), pot, bet, stacks, actions
    )
    return run_benchmark._frame(img, seq)


def _build_pipeline(frames, **kwargs):
    orch = ApplicationOrchestrator(
        StateEngine(),
        InMemoryHandMemory(),
        confidence_gate=relaxed_confidence_gate(),
    )
    orch.start_hand(initial_state(hand_id="h1"), started_at=frames[0].timestamp)
    return RealtimePipeline(
        frame_source=SyntheticFrameSource(tuple(frames)),
        vision=run_benchmark.build_engine(),
        table_map=run_benchmark.table_map(),
        orchestrator=orch,
        equity_trials=200,
        equity_seed=7,
        **kwargs,
    )


_SCENARIOS = (
    ([], ["AS", "KD"], "10", "5", ("100", "200", "300"), ("CHECK", "CALL")),
    (["QH", "JD", "TC"], ["AS", "KD"], "25", "10", ("100", "200", "300"),
     ("CHECK", "BET")),
    (["QH", "JD", "TC", "2S"], ["AS", "KD"], "50", "20",
     ("150", "250", "350"), ("BET", "CALL")),
)


def _frames():
    return tuple(
        _frame(bc, hc, pot, bet, stacks, actions, i)
        for i, (bc, hc, pot, bet, stacks, actions) in enumerate(_SCENARIOS)
    )


def test_step_records_all_expected_stages():
    pipeline = _build_pipeline(_frames())
    step = pipeline.step()
    assert step is not None
    recorded = {name for name, _ in step.timings}
    assert {
        STAGE_CAPTURE,
        STAGE_VISION,
        STAGE_CONSENSUS,
        STAGE_BOUNDARY,
        STAGE_CHANGE,
        STAGE_ADVANCE,
        STAGE_TOTAL,
    } <= recorded


def test_every_timing_is_a_finite_non_negative_number():
    pipeline = _build_pipeline(_frames())
    step = pipeline.step()
    assert step is not None
    for name, value in step.timings:
        assert isinstance(name, str) and name
        assert isinstance(value, float), name
        assert value >= 0.0, name
        assert value == value and value not in (float("inf"), float("-inf"))


def test_total_covers_the_sum_of_the_parts():
    """``total`` spans the whole step, so it cannot be less than any stage."""
    pipeline = _build_pipeline(_frames())
    step = pipeline.step()
    assert step is not None
    total = step.timing(STAGE_TOTAL)
    assert total is not None
    for name, value in step.timings:
        if name == STAGE_TOTAL:
            continue
        assert value <= total + 1e-9, name


def test_equity_is_timed_only_when_it_actually_ran():
    """Equity is change-gated; an unchanged frame must not report it."""
    pipeline = _build_pipeline(_frames())
    first = pipeline.step()
    assert first is not None
    assert first.analysis_changed
    assert first.timing(STAGE_EQUITY) is not None

    # Same picture twice (frame_seq still has to increase), so the second
    # step detects no material change and must not re-run equity.
    board, hero, pot, bet, stacks, actions = _SCENARIOS[0]
    duplicate = (
        _frame(board, hero, pot, bet, stacks, actions, 0),
        _frame(board, hero, pot, bet, stacks, actions, 1),
    )
    pipeline = _build_pipeline(duplicate)
    pipeline.step()
    second = pipeline.step()
    assert second is not None
    assert not second.analysis_changed
    assert second.timing(STAGE_EQUITY) is None
    assert second.timing(STAGE_ADVANCE) == 0.0


def test_timing_accessor_returns_none_for_unknown_stage():
    pipeline = _build_pipeline(_frames())
    step = pipeline.step()
    assert step is not None
    assert step.timing("no_such_stage") is None


def test_injected_clock_is_used_and_is_monotonic():
    """A fake clock makes the recorded costs deterministic and testable."""
    state = {"t": 0.0}

    def fake_clock():
        state["t"] += 0.010
        return state["t"]

    pipeline = _build_pipeline(_frames(), monotonic_clock=fake_clock)
    step = pipeline.step()
    assert step is not None
    # Two reads per stage (start + end) at a fixed 10ms tick => 10ms each.
    assert step.timing(STAGE_CAPTURE) == pytest.approx(10.0)
    assert step.timing(STAGE_VISION) == pytest.approx(10.0)
    assert step.timing(STAGE_CONSENSUS) == pytest.approx(10.0)
    # total spans every stage that ran, so it is the largest cost.
    assert step.timing(STAGE_TOTAL) >= step.timing(STAGE_CAPTURE)


def test_injected_clock_does_not_change_the_analysis():
    """Instrumentation is observation only — same frames, same answer."""
    frames = _frames()

    def fixed_clock():
        return 0.0

    plain = _build_pipeline(frames).step()
    clocked = _build_pipeline(frames, monotonic_clock=fixed_clock).step()
    assert plain is not None and clocked is not None
    assert plain.analysis.state == clocked.analysis.state
    assert plain.analysis.equity == clocked.analysis.equity
    assert plain.analysis.confidence == clocked.analysis.confidence
    assert plain.analysis_changed == clocked.analysis_changed
    assert plain.frame_seq == clocked.frame_seq


def test_pipeline_rejects_non_callable_clock():
    try:
        _build_pipeline(_frames(), monotonic_clock=1.0)
    except TypeError as exc:
        assert "monotonic_clock" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for non-callable clock")
