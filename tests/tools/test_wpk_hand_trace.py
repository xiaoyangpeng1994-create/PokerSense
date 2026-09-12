"""Redacted real-hand arithmetic and anti-lookahead regression anchors."""

import json
import warnings

import pytest

from tools.verify_wpk_hand_trace import (
    DEFAULT_TRACE, audit_trace, verify_project_math, visible_board, visible_hole_cards,
)


@pytest.fixture
def trace():
    return json.loads(DEFAULT_TRACE.read_text(encoding="utf-8"))


def test_project_pots_and_exact_equity_match_observed_hand(trace):
    result = verify_project_math(trace)
    assert result["passed"] is True
    assert result["call_threshold_before_fees"] == "32/173"
    assert result["wrong_threshold_if_raw_total_used"] == "8/73"


def test_later_hole_cards_and_river_are_hidden_from_turn_decision(trace):
    assert visible_hole_cards(trace, 8660) == {0: ("Ac", "Qh")}
    assert visible_board(trace, 8660) == ("Jh", "5s", "Qs", "9s")
    assert 6 in visible_hole_cards(trace, 8700)
    assert visible_board(trace, 8760)[-1] == "8c"


def test_independent_rules_reconcile_bets_but_do_not_claim_fees_verified(trace):
    pytest.importorskip("pokerkit")
    result = audit_trace(trace)
    assert result["betting"]["passed"] is True
    assert result["betting"]["actions"][-1]["paid"] == 64
    assert result["settlement_status"] == "PARTIAL"
    assert result["fee_model_verified"] is False
    assert result["strategy_golden_eligible"] is False


def test_wrong_contribution_cannot_silently_pass_reference_checks(trace):
    trace["source_seats"][0]["hand_contributed"] += 1
    with pytest.raises(AssertionError):
        verify_project_math(trace)


def test_unobserved_burns_do_not_randomly_conflict_with_known_board(trace):
    pytest.importorskip("pokerkit")
    from tools.verify_wpk_hand_trace import verify_betting

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        runs = [verify_betting(trace) for _ in range(10)]
    assert all(run == runs[0] for run in runs)
