from copy import deepcopy

import pytest

from tools.aa8_unmarked_money import infer_call


def inputs():
    before = {"stacks": {"0": "104", "3": "15"}, "pot": "474"}
    after = {"stacks": {"0": "0", "3": "15"}, "pot": "578"}
    context = {"actor": 0, "street_price": "221", "actor_street_wager": "63",
               "reviewed_single_action_interval": True}
    return before, after, context


def test_short_call_only_conditional_on_context():
    result = infer_call(*inputs())
    assert result["action"] == "call" and result["all_in"]
    assert result["debit"] == "104"
    assert not result["context_automated"]
    assert not result["strategy_eligible"]
    assert not result["visual_action_glyph_inferred"]


@pytest.mark.parametrize("value", [None, "NaN", "Infinity", "-1", 578, "577"])
def test_unknown_invalid_or_inconsistent_pot_abstains(value):
    before, after, context = inputs()
    after["pot"] = value
    assert infer_call(before, after, context)["status"] == "UNKNOWN"


def test_multiple_cash_changes_or_missing_context_abstain():
    before, after, context = inputs()
    after["stacks"]["3"] = "14"
    assert infer_call(before, after, context)["action"] is None
    before, after, context = inputs()
    context["reviewed_single_action_interval"] = False
    assert infer_call(before, after, context)["action"] is None


def test_cannot_explain_raise_as_call_or_add_missing_roster():
    before, after, context = inputs()
    context["street_price"] = "100"
    assert infer_call(before, after, context)["action"] is None
    before, after, context = inputs()
    changed = deepcopy(after)
    changed["stacks"]["6"] = "100"
    assert infer_call(before, changed, context)["action"] is None
