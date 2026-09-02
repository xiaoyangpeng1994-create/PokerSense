from __future__ import annotations

from dataclasses import replace

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta
from poker_engine.strategy.advice import AdviceStatus, build_advice
from poker_engine.strategy.explanation import explain_advice
from poker_engine.strategy.provider import LookupState
from poker_engine.strategy.router import RouteResult
from poker_engine.strategy.training import (
    ActualActionRecord,
    build_hand_debrief,
)

from .helpers import NOW, candidate, context


def ready_advice():
    ctx = context()
    value = candidate(ctx)
    route = RouteResult(LookupState.HIT_EXACT, value, ())
    return build_advice(ctx, route, now=NOW)


def actual(advice, action=ActionType.RAISE, amount=ChipAmount("2.5")):
    return ActualActionRecord(
        hand_id=advice.hand_id,
        state_version=advice.state_version,
        request_id=advice.request_id,
        action=action,
        amount=amount,
        observed_at=NOW,
        evidence_ref="replay://actual/1",
    )


def test_actual_action_matching_advice_has_zero_ev_loss():
    advice = ready_advice()
    debrief = build_hand_debrief(advice, actual(advice))
    assert debrief.action_deviation is False
    assert debrief.size_deviation is False
    assert debrief.ev_loss == ChipDelta("0")
    assert debrief.training_tags == ("matched_advice",)


def test_action_deviation_reports_ev_loss_only_from_available_action_evs():
    advice = ready_advice()
    debrief = build_hand_debrief(
        advice, actual(advice, ActionType.CHECK, None)
    )
    assert debrief.action_deviation is True
    assert debrief.size_deviation is None
    assert debrief.ev_loss == ChipDelta("1.25")
    assert debrief.training_tags == (
        "action_deviation", "positive_ev_loss"
    )


def test_missing_counterfactual_ev_never_invents_ev_loss():
    advice = replace(ready_advice(), action_ev={})
    debrief = build_hand_debrief(
        advice, actual(advice, ActionType.CHECK, None)
    )
    assert debrief.ev_loss is None
    assert "ev_loss_unavailable" in debrief.training_tags


def test_same_action_wrong_size_is_a_size_deviation():
    advice = ready_advice()
    debrief = build_hand_debrief(
        advice, actual(advice, amount=ChipAmount("3"))
    )
    assert debrief.action_deviation is False
    assert debrief.size_deviation is True
    assert "size_deviation" in debrief.training_tags


def test_non_ready_advice_records_action_without_strategy_judgment():
    ctx = context()
    route = RouteResult(
        LookupState.NO_STRATEGY, None, (), ("strategy_unavailable",)
    )
    advice = build_advice(ctx, route, math_report={"equity": "0.5"}, now=NOW)
    debrief = build_hand_debrief(
        advice, actual(advice, ActionType.CHECK, None)
    )
    assert advice.status is AdviceStatus.PARTIAL
    assert debrief.action_deviation is None
    assert debrief.ev_loss is None
    assert debrief.training_tags == ("strategy_unavailable",)


def test_actual_action_must_bind_exact_advice_identity():
    advice = ready_advice()
    with pytest.raises(ValueError, match="does not match"):
        build_hand_debrief(
            advice, replace(actual(advice), state_version=99)
        )


def test_ready_explanation_snapshot_preserves_exact_values():
    advice = ready_advice()
    zh = explain_advice(advice, language="zh")
    en = explain_advice(advice, language="en")
    assert zh.summary == "首选动作：raise"
    assert zh.key_factors == (
        "频率=0.6",
        "尺度=2.5",
        "策略来源=mock-2p@v1",
        "匹配=exact:1.0",
        "置信度=0.8",
    )
    assert en.summary == "Preferred action: raise"
    assert en.key_factors[0] == "frequency=0.6"


def test_abstain_explanation_contains_reason_without_actions():
    ctx = context(missing_fields=("pot",))
    route = RouteResult(LookupState.NO_STRATEGY, None, (), ("pot",))
    advice = build_advice(ctx, route, now=NOW)
    explanation = explain_advice(advice, language="zh")
    assert explanation.summary == "当前建议状态：ABSTAIN"
    assert explanation.key_factors[0] == "原因=pot"
