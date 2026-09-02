from decimal import Decimal

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta
from poker_engine.strategy.ev import (
    EvStatus,
    calculate_aggressive_ev,
    calculate_call_ev,
    calculate_ev_gap,
)


@pytest.mark.parametrize(
    ("share", "pot", "call", "expected"),
    [
        ("0", "10", "5", "-5"),
        ("1", "10", "5", "10"),
        ("0.333333333333333333", "10", "5", "-5E-18"),
        ("0.5", "0.5", "0.5", "0.0"),
        ("0.25", "100", "25", "6.25"),
    ],
)
def test_call_ev_exact_decimal_formula(share, pot, call, expected):
    result = calculate_call_ev(
        expected_pot_share=Decimal(share),
        pot_before_call=ChipAmount(pot),
        to_call=ChipAmount(call),
    )
    assert result.status is EvStatus.COMPLETE
    assert result.ev == ChipDelta(expected)
    assert result.assumptions == ("immediate_call_ev_no_future_actions",)


def test_call_ev_rejects_float_and_out_of_range_share():
    with pytest.raises(TypeError, match="Decimal"):
        calculate_call_ev(
            expected_pot_share=0.5,
            pot_before_call=ChipAmount("10"),
            to_call=ChipAmount("5"),
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        calculate_call_ev(
            expected_pot_share=Decimal("1.01"),
            pot_before_call=ChipAmount("10"),
            to_call=ChipAmount("5"),
        )


def test_aggressive_ev_aggregates_explicit_net_branches():
    result = calculate_aggressive_ev(
        pot_before_action=ChipAmount("10"),
        amount=ChipAmount("7.5"),
        fold_probability=Decimal("0.40"),
        call_probability=Decimal("0.50"),
        raise_probability=Decimal("0.10"),
        call_continuation_ev=ChipDelta("3"),
        raise_continuation_ev=ChipDelta("-7.5"),
    )
    assert result.status is EvStatus.COMPLETE
    assert result.ev == ChipDelta("4.750")
    assert result.components["amount"] == Decimal("7.5")


@pytest.mark.parametrize(
    ("call_probability", "raise_probability", "call_ev", "raise_ev", "missing"),
    [
        ("0.5", "0.1", None, ChipDelta("-5"), ("call_continuation_ev",)),
        ("0.4", "0.2", ChipDelta("1"), None, ("raise_continuation_ev",)),
        ("0.4", "0.2", None, None,
         ("call_continuation_ev", "raise_continuation_ev")),
    ],
)
def test_aggressive_ev_is_unknown_when_positive_branch_value_missing(
    call_probability, raise_probability, call_ev, raise_ev, missing
):
    fold_probability = Decimal("1") - Decimal(call_probability) - Decimal(
        raise_probability
    )
    result = calculate_aggressive_ev(
        pot_before_action=ChipAmount("10"),
        amount=ChipAmount("5"),
        fold_probability=fold_probability,
        call_probability=Decimal(call_probability),
        raise_probability=Decimal(raise_probability),
        call_continuation_ev=call_ev,
        raise_continuation_ev=raise_ev,
    )
    assert result.status is EvStatus.UNKNOWN
    assert result.ev is None
    assert result.missing_inputs == missing


def test_zero_probability_branch_does_not_require_continuation_value():
    result = calculate_aggressive_ev(
        pot_before_action=ChipAmount("10"),
        amount=ChipAmount("5"),
        fold_probability=Decimal("1"),
        call_probability=Decimal("0"),
        raise_probability=Decimal("0"),
        call_continuation_ev=None,
        raise_continuation_ev=None,
    )
    assert result.status is EvStatus.COMPLETE
    assert result.ev == ChipDelta("10")


def test_aggressive_branch_probabilities_must_sum_exactly():
    with pytest.raises(ValueError, match="sum exactly"):
        calculate_aggressive_ev(
            pot_before_action=ChipAmount("10"),
            amount=ChipAmount("5"),
            fold_probability=Decimal("0.3"),
            call_probability=Decimal("0.3"),
            raise_probability=Decimal("0.3"),
            call_continuation_ev=ChipDelta("0"),
            raise_continuation_ev=ChipDelta("0"),
        )


def test_ev_gap_requires_every_legal_action():
    result = calculate_ev_gap(
        (ActionType.FOLD, ActionType.CALL, ActionType.RAISE),
        {
            ActionType.FOLD: ChipDelta("0"),
            ActionType.CALL: ChipDelta("1"),
            ActionType.RAISE: None,
        },
    )
    assert result.complete is False
    assert result.gap is None
    assert result.missing_actions == (ActionType.RAISE,)


def test_ev_gap_returns_best_second_and_exact_gap():
    result = calculate_ev_gap(
        (ActionType.FOLD, ActionType.CALL, ActionType.RAISE),
        {
            ActionType.FOLD: ChipDelta("0"),
            ActionType.CALL: ChipDelta("2.25"),
            ActionType.RAISE: ChipDelta("3.00"),
        },
    )
    assert result.complete is True
    assert result.best_action is ActionType.RAISE
    assert result.second_action is ActionType.CALL
    assert result.gap == ChipDelta("0.75")


def test_ev_gap_tie_is_deterministic_and_zero():
    result = calculate_ev_gap(
        (ActionType.CALL, ActionType.RAISE),
        {
            ActionType.CALL: ChipDelta("2"),
            ActionType.RAISE: ChipDelta("2"),
        },
    )
    assert result.best_action is ActionType.RAISE
    assert result.second_action is ActionType.CALL
    assert result.gap == ChipDelta("0")
