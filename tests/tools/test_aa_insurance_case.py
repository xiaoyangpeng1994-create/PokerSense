from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.verify_aa_insurance_case import reconcile, river_outs


def review():
    path = Path(__file__).resolve().parents[1] / (
        "fixtures/aa_reference_hands/insurance_32880_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_source_reviewed_insurance_window_reconciles_but_not_general_rake():
    result = reconcile(review())
    assert result["pot_less_credits"] == "174"
    assert result["unexplained_after_premium"] == "0"
    assert result["hero_insured_pot_less_premium_matches_credit"]
    assert result["displayed_outs_match"]
    assert result["remaining_cards_at_purchase_notice"] == 40
    assert result["quote_rounding_residual"] == "0.2"
    assert not result["general_rake_policy_verified"]
    assert not result["automated_visual_extraction"]


def test_unexplained_money_not_silently_assigned_to_insurance():
    case = review()
    case["after_balances"]["5"] = "450"
    assert reconcile(case)["unexplained_after_premium"] == "2"


def test_actual_later_river_does_not_change_prior_outs():
    case = review()
    changed = deepcopy(case)
    changed["river"] = "3c"
    assert reconcile(case)["calculated_outs"] == reconcile(changed)["calculated_outs"]


def test_duplicate_known_card_rejected():
    case = review()
    with pytest.raises(ValueError):
        river_outs(case["hero"], case["insured_opponent"], case["turn_board"], ["Kc"])


def test_quote_without_consistent_purchase_notice_not_accepted():
    case = review()
    case["purchase_notice_text"] = "购买保险中"
    with pytest.raises(ValueError, match="notice"):
        reconcile(case)
