"""Hard-gate contract, integration, and fail-closed regression tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from poker_engine.strategy.advice import AdviceStatus, build_advice, mark_stale
from poker_engine.strategy.fusion import DecisionFusion
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.provider import FakeProvider, LookupState, ProviderResult
from poker_engine.strategy.router import RouteResult, StrategyRouter
from poker_engine.strategy.safety import GateResult, GateStatus
from poker_engine.strategy.serialization import strategy_deserialize, strategy_serialize

from .helpers import NOW, candidate, capability, context


FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _route(ctx):
    value = candidate(ctx)
    return RouteResult(LookupState.HIT_EXACT, value, ()), value


def test_ready_advice_contains_all_builtin_pass_gates():
    ctx = context()
    route, _ = _route(ctx)
    advice = build_advice(ctx, route, now=NOW)

    assert advice.status is AdviceStatus.READY
    assert {item.name: item.status for item in advice.gate_results} == {
        "request_freshness": GateStatus.PASS,
        "confidence_components": GateStatus.PASS,
        "decision_context": GateStatus.PASS,
        "strategy_source": GateStatus.PASS,
        "legal_strategy_actions": GateStatus.PASS,
    }


def test_external_failed_gate_prevents_ready_and_hides_actions():
    ctx = context()
    route, _ = _route(ctx)
    gate = GateResult(
        "range_integrity", GateStatus.FAIL, ("range_card_collision",)
    )
    advice = build_advice(ctx, route, hard_gates=(gate,), now=NOW)

    assert advice.status is AdviceStatus.ABSTAIN
    assert not advice.action_probabilities
    assert advice.rejection_reasons == ("range_card_collision",)
    assert advice.gate_results[3] is gate


def test_external_pass_gate_allows_ready_and_survives_round_trip():
    ctx = context()
    route, _ = _route(ctx)
    gate = GateResult("range_integrity", GateStatus.PASS)
    advice = build_advice(ctx, route, hard_gates=(gate,), now=NOW)
    restored = strategy_deserialize(type(advice), strategy_serialize(advice))

    assert advice.status is AdviceStatus.READY
    assert restored == advice
    assert gate in restored.gate_results


def test_equity_only_partial_marks_strategy_and_legality_skipped():
    ctx = context()
    route = RouteResult(LookupState.NO_STRATEGY, None, (), ("no_strategy",))
    advice = build_advice(
        ctx, route, math_report={"equity": "0.51"}, now=NOW
    )

    statuses = {item.name: item.status for item in advice.gate_results}
    assert advice.status is AdviceStatus.PARTIAL
    assert statuses["strategy_source"] is GateStatus.SKIPPED
    assert statuses["legal_strategy_actions"] is GateStatus.SKIPPED


def test_no_strategy_is_a_named_failed_gate():
    ctx = context()
    route = RouteResult(LookupState.NO_STRATEGY, None, (), ("no_strategy",))
    advice = build_advice(ctx, route, now=NOW)

    gate = next(
        item for item in advice.gate_results if item.name == "strategy_source"
    )
    assert advice.status is AdviceStatus.ABSTAIN
    assert gate.status is GateStatus.FAIL
    assert gate.reasons == ("no_strategy",)


def test_stale_conversion_preserves_gate_audit():
    ctx = context()
    route, _ = _route(ctx)
    ready = build_advice(ctx, route, now=NOW)
    stale = mark_stale(ready, now=NOW)

    assert stale.status is AdviceStatus.STALE
    assert stale.gate_results == ready.gate_results


def test_fusion_and_orchestrator_forward_external_gate():
    ctx = context()
    route, value = _route(ctx)
    gate = GateResult("numerical_integrity", GateStatus.FAIL, ("ci_missing",))
    fused = DecisionFusion().fuse(ctx, route, hard_gates=(gate,), now=NOW)
    provider = FakeProvider(
        "mock-2p",
        "v1",
        capability(),
        ProviderResult(LookupState.HIT_EXACT, "mock-2p", value),
    )
    cycle = StrategyOrchestrator(StrategyRouter((provider,))).request(
        ctx, hard_gates=(gate,), now=NOW
    )

    assert fused.advice.status is AdviceStatus.ABSTAIN
    assert cycle.fast_advice.status is AdviceStatus.ABSTAIN
    assert gate in cycle.fast_advice.gate_results


@pytest.mark.parametrize(
    "gate",
    (
        lambda: GateResult("", GateStatus.PASS),
        lambda: GateResult("range", "PASS"),
        lambda: GateResult("range", GateStatus.FAIL),
        lambda: GateResult("range", GateStatus.PASS, ("unexpected",)),
    ),
)
def test_invalid_gate_contract_is_rejected(gate):
    with pytest.raises((TypeError, ValueError)):
        gate()


def test_duplicate_or_reserved_external_gate_is_rejected():
    ctx = context()
    route, _ = _route(ctx)
    gate = GateResult("range_integrity", GateStatus.PASS)
    with pytest.raises(ValueError, match="unique"):
        build_advice(ctx, route, hard_gates=(gate, gate), now=NOW)
    with pytest.raises(ValueError, match="reserved"):
        build_advice(
            ctx,
            route,
            hard_gates=(GateResult("decision_context", GateStatus.PASS),),
            now=NOW,
        )


def test_ready_advice_object_cannot_be_replaced_with_failed_gate():
    ctx = context()
    route, _ = _route(ctx)
    ready = build_advice(ctx, route, now=NOW)

    with pytest.raises(ValueError, match="failed gate"):
        replace(
            ready,
            gate_results=(
                GateResult("range", GateStatus.FAIL, ("range_invalid",)),
            ),
        )


@pytest.mark.parametrize(
    "fixture",
    [
        item for item in (
            json.loads(line) for line in FIXTURES.read_text().splitlines()
        )
        if item["fixture_id"].startswith("MOCK-HARD-GATE-")
    ],
)
def test_generated_hard_gate_fixtures_execute_through_advice(fixture):
    ctx = context()
    route, _ = _route(ctx)
    gates = tuple(
        GateResult(
            item["name"], GateStatus(item["status"]), tuple(item["reasons"])
        )
        for item in fixture["input"]["hard_gates"]
    )

    advice = build_advice(ctx, route, hard_gates=gates, now=NOW)

    expected = fixture["expected"]
    assert advice.status.value == expected["advice"]["status"]
    assert list(advice.rejection_reasons) == expected["advice"]["reason_codes"]
    assert [
        {
            "name": item.name,
            "status": item.status.value,
            "reasons": list(item.reasons),
        }
        for item in advice.gate_results if item.name in {
            gate["name"] for gate in fixture["input"]["hard_gates"]
        }
    ] == expected["gate_results"]
