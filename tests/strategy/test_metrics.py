"""Exact derived strategy metric tests."""

from decimal import Decimal

import pytest

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import EffectiveStack, PotState
from poker_engine.strategy.metrics import (
    MetricStatus,
    calculate_pairwise_spr,
    calculate_pot_odds,
    normalize_action_size,
)


def test_pairwise_spr_uses_each_opponent_effective_stack():
    values = calculate_pairwise_spr(
        (
            EffectiveStack(0, ChipAmount("50")),
            EffectiveStack(1, ChipAmount("25")),
        ),
        (
            PotState("main", ChipAmount("15"), (0, 1, 2)),
            PotState("side-1", ChipAmount("5"), (1, 2)),
        ),
    )

    assert [(item.opponent_seat, item.value, item.status) for item in values] == [
        (0, Decimal("2.5"), MetricStatus.KNOWN),
        (1, Decimal("1.25"), MetricStatus.KNOWN),
    ]


def test_pairwise_spr_marks_zero_pot_unknown():
    values = calculate_pairwise_spr(
        (EffectiveStack(0, ChipAmount("50")),),
        (PotState("main", ChipAmount("0"), (0, 1)),),
    )

    assert values[0].status is MetricStatus.UNKNOWN
    assert values[0].value is None


def test_decimal_pot_odds_matches_exact_hand_calculation():
    odds = calculate_pot_odds(ChipAmount("100"), ChipAmount("25"))

    assert odds.required_equity == Decimal("0.2")
    assert not odds.no_call_cost


def test_zero_call_cost_is_explicit_not_unknown():
    odds = calculate_pot_odds(ChipAmount("0"), ChipAmount("0"))

    assert odds.required_equity == Decimal("0")
    assert odds.no_call_cost


def test_action_size_normalizes_all_requested_bases_exactly():
    value = normalize_action_size(
        additional_amount=ChipAmount("10"),
        total_street_amount=ChipAmount("30"),
        pot_before_action=ChipAmount("40"),
        big_blind=ChipAmount("2"),
        current_bet=ChipAmount("20"),
    )

    assert value.size_bb == Decimal("5")
    assert value.pot_fraction == Decimal("0.25")
    assert value.raise_multiplier == Decimal("1.5")


def test_action_size_preserves_unknown_zero_bases_as_none():
    value = normalize_action_size(
        additional_amount=ChipAmount("1"),
        total_street_amount=ChipAmount("1"),
        pot_before_action=ChipAmount("0"),
        big_blind=ChipAmount("2"),
        current_bet=ChipAmount("0"),
    )

    assert value.size_bb == Decimal("0.5")
    assert value.pot_fraction is None
    assert value.raise_multiplier is None


def test_action_size_rejects_invalid_amount_semantics_or_blind():
    with pytest.raises(ValueError, match="below"):
        normalize_action_size(
            additional_amount=ChipAmount("20"),
            total_street_amount=ChipAmount("10"),
            pot_before_action=ChipAmount("40"),
            big_blind=ChipAmount("2"),
            current_bet=ChipAmount("5"),
        )
    with pytest.raises(ValueError, match="> 0"):
        normalize_action_size(
            additional_amount=ChipAmount("1"),
            total_street_amount=ChipAmount("1"),
            pot_before_action=ChipAmount("40"),
            big_blind=ChipAmount("0"),
            current_bet=ChipAmount("0"),
        )
