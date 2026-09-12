"""Frozen-batch scoring separates wrong accepts, abstentions and absence."""

import pytest

from tools.validate_wpk_card_batch import score_slots, summarize


def test_wrong_identity_is_not_hidden_by_other_abstentions():
    result = score_slots(["Ac", "Qh"], ["Qc", None], 2)
    assert result["counts"]["wrong_accepted"] == 1
    assert result["counts"]["abstain_on_positive"] == 1
    assert not result["complete_match"]


def test_a_card_on_an_empty_slot_is_a_false_positive():
    result = score_slots([], [None, "Ac"], 2)
    assert result["counts"]["false_positive_on_absent"] == 1


def test_all_abstain_cannot_pass_required_coverage():
    rows = [{"hero": score_slots(["Ac", "Qh"], [None, None], 2),
             "board": score_slots(["2c", "3s", "4h"], [None] * 5, 5),
             "require_hero_complete": True, "require_board_complete": True}]
    result = summarize(rows, {"max_wrong_accepted": 0,
                              "min_required_complete_fraction": 0.95})
    assert not result["passed"]
    assert result["fields"]["hero"]["accepted_precision"] is None


def test_optional_folded_cards_still_count_if_wrongly_accepted():
    base = {"hero": score_slots(["Ac", "Qh"], ["Ac", "Qh"], 2),
            "board": score_slots([], [None] * 5, 5),
            "require_hero_complete": True, "require_board_complete": True}
    folded = base | {"hero": score_slots(["4s", "Jd"], ["4c", None], 2),
                     "require_hero_complete": False}
    criteria = {"max_wrong_accepted": 0, "min_required_complete_fraction": 0.95}
    result = summarize([base, folded], criteria)
    assert result["fields"]["hero"]["coverage"] == 1
    assert not result["passed"]


def test_empty_batch_and_invalid_dimensions_are_rejected():
    assert not summarize([], {"max_wrong_accepted": 0,
                              "min_required_complete_fraction": 0.95})["passed"]
    with pytest.raises(ValueError):
        score_slots(["Ac"], [], 2)
