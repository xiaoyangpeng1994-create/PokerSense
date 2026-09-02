import math
from decimal import Decimal

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.strategy.advice import AdviceStatus, build_advice
from poker_engine.strategy.confidence import aggregate_confidence
from poker_engine.strategy.provider import LookupState, MatchKind
from poker_engine.strategy.router import RouteResult

from .helpers import NOW, candidate, context


def route(value):
    state = (
        LookupState.HIT_EXACT
        if value.match_kind is MatchKind.EXACT
        else LookupState.HIT_APPROXIMATE
    )
    return RouteResult(state, value, ())


def test_confidence_uses_weakest_component_not_average():
    result = aggregate_confidence({
        "perception": 0.99,
        "state": 0.95,
        "range": 0.41,
        "numerical": 0.90,
    })
    assert result.complete
    assert result.overall == 0.41
    assert result.limiting_components == ("range",)


def test_equal_minimum_components_are_all_reported_deterministically():
    result = aggregate_confidence({"state": 0.5, "range": 0.5, "match": 0.9})
    assert result.limiting_components == ("range", "state")


def test_missing_required_component_yields_zero_without_fake_value():
    result = aggregate_confidence({
        "state": 0.9, "range": None, "optional": None
    }, required_components=("state", "range"))
    assert not result.complete
    assert result.overall == 0.0
    assert result.missing_components == ("range",)
    assert "range" not in result.components


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_confidence_component_is_rejected(value):
    with pytest.raises(ValueError, match="finite and in"):
        aggregate_confidence({"bad": value})


def test_advice_confidence_includes_match_provider_input_and_numerical():
    ctx = context()
    value = candidate(
        ctx,
        match_kind=MatchKind.INTERPOLATED,
        score=0.6,
    )
    advice = build_advice(
        ctx,
        route(value),
        confidence_components={"range": 0.7, "numerical": 0.55},
        now=NOW,
    )
    assert advice.status is AdviceStatus.READY
    assert advice.confidence == 0.55
    assert advice.confidence_factors == {
        "input_quality": 0.9,
        "evidence_chain": 1.0,
        "range": 0.7,
        "numerical": 0.55,
        "provider": 0.8,
        "state_match": 0.6,
    }


def test_missing_declared_confidence_component_forces_abstain():
    ctx = context()
    advice = build_advice(
        ctx,
        route(candidate(ctx)),
        confidence_components={"range": None},
        now=NOW,
    )
    assert advice.status is AdviceStatus.ABSTAIN
    assert not advice.action_probabilities
    assert advice.confidence == 0.0
    assert advice.missing_confidence_factors == ("range",)
    assert advice.rejection_reasons == (
        "missing_confidence_component:range",
    )


def test_match_score_caps_approximate_advice_confidence():
    ctx = context()
    value = candidate(
        ctx,
        match_kind=MatchKind.INTERPOLATED,
        score=0.2,
        probabilities={
            ActionType.CHECK: Decimal("0.4"),
            ActionType.RAISE: Decimal("0.6"),
        },
    )
    advice = build_advice(ctx, route(value), now=NOW)
    assert advice.confidence == 0.2
    assert advice.confidence_factors["state_match"] == 0.2
