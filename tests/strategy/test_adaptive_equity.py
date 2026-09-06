"""Deadline-aware exact/Monte Carlo selection and cache integration tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from poker_engine.core.enums import Street
from poker_engine.core.errors import InvalidStateError
from poker_engine.strategy.adaptive_equity import (
    AdaptiveEquityPolicy,
    EquityComputationStatus,
    calculate_adaptive_equity,
)
from poker_engine.strategy.contracts import RangeDistribution
from poker_engine.strategy.equity_cache import (
    EquityCache,
    EquityCacheState,
    EquityMethod,
)

from .helpers import NOW, context


def _decision_context(*, street: Street = Street.RIVER, deadline_ms: int = 300):
    ctx = context(2, street=street)
    villain = RangeDistribution(
        seat_id=0,
        combo_weights={"QsQd": Decimal("0.25"), "8c6c": Decimal("0.75")},
        source="test-concrete-range",
        source_version="v1",
        confidence=0.8,
    )
    return replace(
        ctx,
        villain_ranges=(villain,),
        request=replace(ctx.request, deadline_ms=deadline_ms),
    )


def test_small_river_range_uses_exact_and_then_cache_hit():
    ctx = _decision_context()
    cache = EquityCache()

    first = calculate_adaptive_equity(ctx, now=NOW, cache=cache)
    second = calculate_adaptive_equity(ctx, now=NOW, cache=cache)

    assert first.method is EquityMethod.EXACT
    assert first.status is EquityComputationStatus.COMPLETE
    assert first.result.pot_equity == Decimal("0.75")
    assert first.confidence_low == first.confidence_high == Decimal("0.75")
    assert first.numerical_confidence == Decimal("1")
    assert first.cache_state is EquityCacheState.NOT_FOUND
    assert second.cache_state is EquityCacheState.HIT
    assert second.result == first.result
    assert any(item.startswith("cache://equity/") for item in second.evidence)


def test_deadline_budget_forces_seeded_monte_carlo_and_partial_status():
    ctx = _decision_context(street=Street.FLOP, deadline_ms=2)
    policy = AdaptiveEquityPolicy(
        exact_outcome_limit=0,
        minimum_mc_trials=1000,
        maximum_mc_trials=5000,
        mc_trials_per_ms=100,
        seed=17,
    )

    first = calculate_adaptive_equity(
        ctx, now=NOW, policy=policy, monotonic_clock=lambda: 0.0
    )
    second = calculate_adaptive_equity(
        ctx, now=NOW, policy=policy, monotonic_clock=lambda: 0.0
    )

    assert first == second
    assert first.method is EquityMethod.MONTE_CARLO
    assert first.status is EquityComputationStatus.PARTIAL
    assert first.trials == 200
    assert first.result.samples == 200
    assert first.confidence_low <= first.result.pot_equity <= first.confidence_high
    assert first.numerical_confidence <= Decimal("0.2")


def test_sufficient_deterministic_mc_can_be_complete():
    ctx = _decision_context(deadline_ms=10)
    always_loses = RangeDistribution(
        seat_id=0,
        combo_weights={"QsQd": Decimal("1")},
        source="test-concrete-range",
        source_version="v1",
        confidence=0.8,
    )
    ctx = replace(ctx, villain_ranges=(always_loses,))
    policy = AdaptiveEquityPolicy(
        exact_outcome_limit=0,
        minimum_mc_trials=500,
        maximum_mc_trials=1000,
        mc_trials_per_ms=100,
        seed=5,
    )

    report = calculate_adaptive_equity(
        ctx, now=NOW, policy=policy, monotonic_clock=lambda: 0.0
    )

    assert report.method is EquityMethod.MONTE_CARLO
    assert report.status is EquityComputationStatus.COMPLETE
    assert report.trials == 1000
    assert report.confidence_low == report.confidence_high == Decimal("0")
    assert report.numerical_confidence == Decimal("1")


def test_monte_carlo_confidence_metadata_survives_cache_hit():
    ctx = _decision_context(street=Street.FLOP, deadline_ms=5)
    policy = AdaptiveEquityPolicy(
        exact_outcome_limit=0,
        minimum_mc_trials=1000,
        maximum_mc_trials=1000,
        mc_trials_per_ms=100,
        seed=11,
    )
    cache = EquityCache()

    first = calculate_adaptive_equity(
        ctx,
        now=NOW,
        policy=policy,
        cache=cache,
        monotonic_clock=lambda: 0.0,
    )
    second = calculate_adaptive_equity(
        ctx,
        now=NOW,
        policy=policy,
        cache=cache,
        monotonic_clock=lambda: 0.0,
    )

    assert second.cache_state is EquityCacheState.HIT
    assert second.result == first.result
    assert second.confidence_low == first.confidence_low
    assert second.confidence_high == first.confidence_high
    assert second.numerical_confidence == first.numerical_confidence
    assert second.status == first.status


def test_expired_request_is_rejected_before_cache_or_computation():
    ctx = _decision_context()

    with pytest.raises(InvalidStateError, match="deadline_expired"):
        calculate_adaptive_equity(
            ctx,
            now=ctx.request.expires_at,
            cache=EquityCache(),
        )


def test_large_joint_range_skips_cartesian_materialization_and_uses_mc():
    ctx = _decision_context(deadline_ms=1)
    policy = AdaptiveEquityPolicy(
        joint_assignment_limit=1,
        minimum_mc_trials=10,
        maximum_mc_trials=10,
        mc_trials_per_ms=10,
        seed=9,
    )

    report = calculate_adaptive_equity(
        ctx, now=NOW, policy=policy, monotonic_clock=lambda: 0.0
    )

    assert report.method is EquityMethod.MONTE_CARLO
    assert report.trials == 10
    assert report.estimated_outcomes == 2


def test_adaptive_mc_returns_completed_samples_when_wall_deadline_fires():
    ctx = _decision_context(street=Street.FLOP, deadline_ms=1)
    policy = AdaptiveEquityPolicy(
        exact_outcome_limit=0,
        minimum_mc_trials=100,
        maximum_mc_trials=100,
        mc_trials_per_ms=100,
    )
    clock_values = iter((0.0, 2.0))

    report = calculate_adaptive_equity(
        ctx,
        now=NOW,
        policy=policy,
        monotonic_clock=lambda: next(clock_values),
    )

    assert report.status is EquityComputationStatus.PARTIAL
    assert report.trials == 16
    assert "planned=100" in report.evidence[0]


def test_exact_hysteresis_band_extends_exact_just_past_budget():
    # Turn with two villain combos: 2 assignments x 44 runouts = 88 outcomes.
    # deadline 20ms -> budget 80, hysteresis band lifts the limit to 88.
    ctx = _decision_context(street=Street.TURN, deadline_ms=20)

    report = calculate_adaptive_equity(
        ctx, now=NOW, monotonic_clock=lambda: 0.0
    )

    assert report.estimated_outcomes == 88
    assert report.method is EquityMethod.EXACT
    assert report.status is EquityComputationStatus.COMPLETE


def test_beyond_hysteresis_band_still_uses_monte_carlo():
    # deadline 10ms -> budget 40, band limit 44 < 88 outcomes.
    ctx = _decision_context(street=Street.TURN, deadline_ms=10)

    report = calculate_adaptive_equity(
        ctx, now=NOW, monotonic_clock=lambda: 0.0
    )

    assert report.method is EquityMethod.MONTE_CARLO


def test_exact_result_is_reused_when_shrinking_deadline_would_force_mc():
    cache = EquityCache()
    generous = _decision_context(street=Street.TURN, deadline_ms=300)
    tight = _decision_context(street=Street.TURN, deadline_ms=1)

    first = calculate_adaptive_equity(generous, now=NOW, cache=cache)
    second = calculate_adaptive_equity(tight, now=NOW, cache=cache)

    assert first.method is EquityMethod.EXACT
    assert second.method is EquityMethod.EXACT
    assert second.cache_state is EquityCacheState.HIT
    assert second.result == first.result


def test_mc_run_cut_short_by_wall_deadline_is_not_cached():
    ctx = _decision_context(street=Street.FLOP, deadline_ms=1)
    policy = AdaptiveEquityPolicy(
        exact_outcome_limit=0,
        minimum_mc_trials=100,
        maximum_mc_trials=100,
        mc_trials_per_ms=100,
    )
    cache = EquityCache()

    first_clock = iter((0.0, 2.0))
    first = calculate_adaptive_equity(
        ctx,
        now=NOW,
        policy=policy,
        cache=cache,
        monotonic_clock=lambda: next(first_clock),
    )

    assert first.trials == 16  # planned 100, wall deadline fired early
    assert len(cache) == 0  # truncated run must not poison the cache

    second_clock = iter((0.0, 2.0))
    second = calculate_adaptive_equity(
        ctx,
        now=NOW,
        policy=policy,
        cache=cache,
        monotonic_clock=lambda: next(second_clock),
    )

    assert second.cache_state is EquityCacheState.NOT_FOUND
    assert second.trials == 16
