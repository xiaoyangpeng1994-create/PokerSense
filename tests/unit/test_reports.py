"""Tests for report contracts: EquityReport, StrategyReport, ReasoningReport,
Decision.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.reports import (
    Decision,
    DecisionPath,
    EquityMethod,
    EquityReport,
    ReasoningReport,
    StrategyReport,
    StrategySource,
)
from poker_engine.core.value_objects import ChipAmount

UTC = timezone.utc


def _aware():
    return datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)


# ---------- EquityReport ----------

def test_equity_report_valid():
    r = EquityReport(
        win_rate=0.5, tie_rate=0.1, pot_odds=2.0, implied_odds=1.5,
        estimated_ev=ChipAmount("10"), method=EquityMethod.MONTECARLO,
        timestamp=_aware(),
    )
    assert r.method is EquityMethod.MONTECARLO


def test_equity_win_rate_out_of_range():
    with pytest.raises(ValueError):
        EquityReport(
            win_rate=1.5, tie_rate=0.0, pot_odds=2.0, implied_odds=1.0,
            estimated_ev=ChipAmount("10"), method=EquityMethod.ENUMERATION,
            timestamp=_aware(),
        )


def test_equity_estimated_ev_must_be_chipamount():
    with pytest.raises(TypeError):
        EquityReport(
            win_rate=0.5, tie_rate=0.0, pot_odds=2.0, implied_odds=1.0,
            estimated_ev="10", method=EquityMethod.ENUMERATION,
            timestamp=_aware(),
        )


def test_equity_method_enum_required():
    with pytest.raises(TypeError):
        EquityReport(
            win_rate=0.5, tie_rate=0.0, pot_odds=2.0, implied_odds=1.0,
            estimated_ev=ChipAmount("10"), method="enumeration",
            timestamp=_aware(),
        )


# ---------- StrategyReport ----------

def test_strategy_report_valid():
    r = StrategyReport(
        action_frequencies={ActionType.FOLD: 0.2, ActionType.CALL: 0.8},
        bet_sizes=(ChipAmount("10"),),
        ev=ChipAmount("5"),
        strategy_source=StrategySource.CACHE,
        confidence=0.9,
        cache_hit=True,
    )
    assert r.strategy_source is StrategySource.CACHE
    assert r.cache_hit is True


def test_strategy_action_freq_keys_must_be_actiontype():
    with pytest.raises(TypeError):
        StrategyReport(action_frequencies={"fold": 0.5})


def test_strategy_bet_sizes_must_be_chipamount():
    with pytest.raises(TypeError):
        StrategyReport(bet_sizes=("10",))


def test_strategy_source_enum_required():
    with pytest.raises(TypeError):
        StrategyReport(strategy_source="cache")


def test_strategy_cache_hit_must_be_bool():
    with pytest.raises(TypeError):
        StrategyReport(cache_hit="yes")  # type: ignore[arg-type]


# ---------- ReasoningReport ----------

def test_reasoning_report_valid():
    r = ReasoningReport(
        analysis_summary="hero has nut flush draw",
        key_factors=("SPR = 2.8", "villain high turn aggression"),
        suggested_action=ActionType.RAISE,
        suggested_size=ChipAmount("20"),
        confidence=0.85,
        source="poker_skill",
        hand_id="h1",
        state_version=3,
        request_id="r1",
    )
    assert r.suggested_action is ActionType.RAISE
    assert r.suggested_size == ChipAmount("20")
    # structurally no reasoning_chain attribute
    assert not hasattr(r, "reasoning_chain")


def test_reasoning_key_factors_must_be_str():
    with pytest.raises(TypeError):
        ReasoningReport(
            analysis_summary="x", key_factors=(1, 2),
            suggested_action=ActionType.CALL, suggested_size=ChipAmount("0"),
            confidence=0.5, source="mock",
            hand_id="h1", request_id="r1",
        )


def test_reasoning_no_chain_of_thought_field():
    # ReasoningReport must NOT store a full chain of thought.
    fields = ReasoningReport.__dataclass_fields__  # type: ignore[attr-defined]
    assert "reasoning_chain" not in fields
    assert "chain_of_thought" not in fields


# ---------- Decision ----------

def test_decision_valid():
    d = Decision(
        action=ActionType.RAISE,
        confidence=0.9,
        evidence_chain=("EquityReport", "StrategyReport"),
        raise_size=ChipAmount("15"),
        fast_or_slow=DecisionPath.FAST,
    )
    assert d.action is ActionType.RAISE
    assert d.raise_size == ChipAmount("15")
    assert d.fast_or_slow is DecisionPath.FAST


def test_decision_raise_size_must_be_chipamount():
    with pytest.raises(TypeError):
        Decision(action=ActionType.RAISE, confidence=0.9, raise_size="15")


def test_decision_raise_size_none_ok():
    d = Decision(action=ActionType.CALL, confidence=0.9)
    assert d.raise_size is None


def test_decision_evidence_chain_must_be_str():
    with pytest.raises(TypeError):
        Decision(action=ActionType.CALL, confidence=0.9, evidence_chain=(1, 2))


def test_decision_path_enum_required():
    with pytest.raises(TypeError):
        Decision(action=ActionType.CALL, confidence=0.9, fast_or_slow="fast")


def test_decision_frozen():
    d = Decision(action=ActionType.CALL, confidence=0.9)
    with pytest.raises(FrozenInstanceError):
        d.action = ActionType.FOLD  # type: ignore[misc]


# ---------- 返修 v2 新增测试 ----------

def _equity(**overrides):
    args = dict(
        win_rate=0.5, tie_rate=0.1, pot_odds=2.0, implied_odds=1.5,
        estimated_ev=ChipAmount("10"), method=EquityMethod.MONTECARLO,
        timestamp=_aware(),
    )
    args.update(overrides)
    return EquityReport(**args)


def test_equity_pot_odds_bool_rejected():
    with pytest.raises(TypeError):
        _equity(pot_odds=True)


def test_equity_pot_odds_nan_rejected():
    with pytest.raises(ValueError):
        _equity(pot_odds=float("nan"))


def test_equity_implied_odds_infinity_rejected():
    with pytest.raises(ValueError):
        _equity(implied_odds=float("inf"))


def test_equity_implied_odds_non_numeric_rejected():
    with pytest.raises(TypeError):
        _equity(implied_odds="1.5")  # type: ignore[arg-type]


def test_equity_pot_odds_negative_rejected():
    with pytest.raises(ValueError):
        _equity(pot_odds=-1.0)


def _reasoning(**overrides):
    args = dict(
        analysis_summary="x",
        key_factors=(),
        suggested_action=ActionType.CALL,
        suggested_size=ChipAmount("0"),
        confidence=0.5,
        source="mock",
        hand_id="h1",
        request_id="r1",
    )
    args.update(overrides)
    return ReasoningReport(**args)


def test_reasoning_hand_id_empty_rejected():
    with pytest.raises(ValueError):
        _reasoning(hand_id="")


def test_reasoning_request_id_empty_rejected():
    with pytest.raises(ValueError):
        _reasoning(request_id="")
