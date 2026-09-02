"""Decision Fusion integration from routed baseline to Advice."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipDelta
from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.exploit_fusion import ExploitAdjustmentStatus
from poker_engine.strategy.fusion import DecisionFusion, OpponentAdjustmentInput
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.provider import (
    FakeProvider,
    LookupState,
    MatchKind,
    ProviderResult,
)
from poker_engine.strategy.router import RouteResult, StrategyRouter

from .helpers import NOW, candidate, capability, context


def _route(ctx):
    value = candidate(ctx)
    return RouteResult(LookupState.HIT_EXACT, value, ()), value


def _opponent(*, quality="0.8", samples=500):
    return OpponentAdjustmentInput(
        {
            ActionType.CHECK: ChipDelta("0"),
            ActionType.RAISE: ChipDelta("2"),
        },
        Decimal(quality),
        samples,
    )


def test_without_opponent_input_preserves_the_single_routed_baseline():
    ctx = context()
    route, baseline = _route(ctx)
    outcome = DecisionFusion().fuse(ctx, route, now=NOW)

    assert outcome.baseline is baseline
    assert outcome.selected is baseline
    assert outcome.adjustment is None
    assert outcome.advice.status is AdviceStatus.READY
    assert outcome.advice.match_kind is MatchKind.EXACT


def test_trusted_opponent_input_adjusts_baseline_and_builds_heuristic_advice():
    ctx = context()
    route, baseline = _route(ctx)
    outcome = DecisionFusion().fuse(
        ctx, route, opponent=_opponent(), now=NOW
    )

    assert outcome.baseline is baseline
    assert outcome.adjustment is not None
    assert outcome.adjustment.status is ExploitAdjustmentStatus.APPLIED
    assert outcome.selected is outcome.adjustment.candidate
    assert outcome.advice.status is AdviceStatus.READY
    assert outcome.advice.match_kind is MatchKind.HEURISTIC
    assert outcome.advice.action_probabilities[ActionType.RAISE] > Decimal("0.6")
    assert outcome.advice.action_ev[ActionType.RAISE] == ChipDelta("2")
    assert "opponent_adjustment_kl_bounded" in outcome.advice.assumptions


def test_weak_profile_records_baseline_result_without_downgrading_exact_advice():
    ctx = context()
    route, baseline = _route(ctx)
    outcome = DecisionFusion().fuse(
        ctx, route, opponent=_opponent(samples=2), now=NOW
    )

    assert outcome.selected is baseline
    assert outcome.adjustment.status is ExploitAdjustmentStatus.BASELINE
    assert outcome.adjustment.reasons == ("insufficient_profile_sample",)
    assert outcome.advice.match_kind is MatchKind.EXACT


def test_no_strategy_route_remains_equity_only_and_is_never_adjusted():
    ctx = context()
    route = RouteResult(LookupState.NO_STRATEGY, None, (), ("no_strategy",))
    outcome = DecisionFusion().fuse(
        ctx,
        route,
        opponent=_opponent(),
        math_report={"equity": Decimal("0.51")},
        now=NOW,
    )

    assert outcome.baseline is None
    assert outcome.selected is None
    assert outcome.adjustment is None
    assert outcome.advice.status is AdviceStatus.PARTIAL
    assert outcome.advice.match_kind is MatchKind.EQUITY_ONLY


def test_fusion_rejects_a_baseline_bound_to_another_request():
    ctx = context()
    route, baseline = _route(ctx)
    route = replace(route, selected=replace(baseline, request_id="other"))

    with pytest.raises(ValueError, match="baseline_context_mismatch"):
        DecisionFusion().fuse(ctx, route, now=NOW)


def test_orchestrator_uses_fusion_for_immediate_fast_advice():
    ctx = context()
    baseline = candidate(ctx, provider_id="fast", provider_version="v1")
    provider = FakeProvider(
        "fast",
        "v1",
        capability(),
        ProviderResult(LookupState.HIT_EXACT, "fast", baseline),
    )
    cycle = StrategyOrchestrator(StrategyRouter((provider,))).request(
        ctx,
        opponent_adjustment=_opponent(),
        now=NOW,
    )

    assert cycle.fast_advice.status is AdviceStatus.READY
    assert cycle.fast_advice.match_kind is MatchKind.HEURISTIC
    assert cycle.fast_advice.strategy_source == "fast"
    assert cycle.slow_handle is None


def test_opponent_input_freezes_q_values():
    values = {
        ActionType.CHECK: ChipDelta("0"),
        ActionType.RAISE: ChipDelta("1"),
    }
    input_value = OpponentAdjustmentInput(values, Decimal("0.8"), 500)
    values[ActionType.RAISE] = ChipDelta("99")

    assert input_value.q_values[ActionType.RAISE] == ChipDelta("1")
    with pytest.raises(TypeError):
        input_value.q_values[ActionType.RAISE] = ChipDelta("2")


@pytest.mark.parametrize(
    "args",
    (
        ({}, Decimal("0.8"), 500),
        ({ActionType.RAISE: Decimal("1")}, Decimal("0.8"), 500),
        ({ActionType.RAISE: ChipDelta("1")}, Decimal("1.1"), 500),
        ({ActionType.RAISE: ChipDelta("1")}, Decimal("0.8"), -1),
    ),
)
def test_invalid_opponent_adjustment_input_is_rejected(args):
    with pytest.raises((TypeError, ValueError)):
        OpponentAdjustmentInput(*args)
