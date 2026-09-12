import json
from pathlib import Path

import pytest

from tools.verify_aa8_settlement import amount, reconcile


def review():
    path = Path(__file__).resolve().parents[1] / (
        "fixtures/aa_reference_hands/eight_session_56dd_settlement_v1.json")
    return json.loads(path.read_text())


def test_reviewed_cash_differences_stay_separate():
    result = reconcile(review())
    assert result["outside_displayed_pot_before_settlement"] == "6"
    assert result["payout_difference"] == "13"
    assert result["closing_cash_difference"] == "19"
    assert result["hero_net_visible_change"] == "316"
    assert result["hero_win_label_matches"] and result["hero_beats_revealed_opponent"]
    assert result["decomposition_is_algebraic_not_independent_validation"]
    assert not result["rake_policy_verified"]
    assert not result["full_action_truth_verified"]
    assert result["checkpoint_cash_and_pot_consistent_under_hypothesis"]


def test_waiting_newcomer_cannot_enter_old_hand():
    data = review()
    data["opening_slots"].append(6)
    with pytest.raises(ValueError, match="roster"):
        reconcile(data)


def test_changed_credit_is_not_corrected_to_match_win_label():
    data = review()
    data["after_settlement"]["4"] = "515"
    result = reconcile(data)
    assert result["payout_difference"] == "14"
    assert not result["hero_win_label_matches"]


def test_checkpoint_disagreement_is_exposed():
    data = review()
    data["monetary_checkpoints"][1]["pot"] = "82"
    result = reconcile(data)
    assert not result["checkpoint_cash_and_pot_consistent_under_hypothesis"]
    assert result["monetary_checkpoints"][1][
        "discrepancy_under_displayed_post_hypothesis"] == "1"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", None, 3])
def test_invalid_money_rejected(value):
    with pytest.raises(ValueError):
        amount(value)


def test_duplicate_card_and_excess_credit_rejected():
    data = review()
    data["opponent"][0] = data["hero"][0]
    with pytest.raises(ValueError, match="distinct"):
        reconcile(data)
    data = review()
    data["after_settlement"]["4"] = "9999"
    with pytest.raises(ValueError, match="inflow"):
        reconcile(data)
