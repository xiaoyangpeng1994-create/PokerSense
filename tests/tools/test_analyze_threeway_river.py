from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import pytest

from tools.analyze_threeway_river import (
    analyze_file, complete_report, scenario_from_dict, summary,
)


EXAMPLE = Path(__file__).resolve().parents[2] / (
    "configs/strategy/examples/threeway-river-response-manual.json")


def test_manual_example_has_real_followup_actions_and_costs():
    report = analyze_file(EXAMPLE)
    result = report["result"]
    assert result["status"] == "COMPLETE_CONDITIONAL_ABSTRACTION"
    assert result["joint_assignments"] == 4
    actions = result["root_actions"]
    assert [(r["action"]["kind"], r["additional_cost"]) for r in actions] == [
        ("fold", "0"), ("call", "20"), ("raise", "40"), ("raise", "80")]
    assert result["best_action"] in [r["action"] for r in actions]
    assert result["hero_policy"]  # Later decisions are retained for audit.
    assert result["advice_emitted"] is False and result["strategy_eligible"] is False
    displayed = summary(report)
    assert displayed["scope"].endswith("NOT_GTO")
    assert displayed["root_players"] == displayed["river_start_players"] == 3
    assert displayed["table_players"] == 6
    assert displayed["strategy_eligible"] is displayed["advice_emitted"] is False
    assert displayed["no_model_robust_policy_claim"] is True
    assert displayed["assumptions"]


def test_nine_named_model_pairs_report_sensitivity_not_guaranteed_robustness():
    report = analyze_file(EXAMPLE, sensitivity=True)
    s = report["sensitivity"]
    assert s["tested_model_pairs"] == len(s["cases"]) == 9
    assert s["status"] == "MODEL_SENSITIVE"
    assert s["no_model_robust_policy_claim"] is True
    choices = {(c["result"]["best_action"]["kind"],
                c["result"]["best_action"]["target"]) for c in s["cases"]}
    assert len(choices) > 1
    assert len({json.dumps(c["result"]["root_posterior"], sort_keys=True)
                for c in s["cases"]}) > 1
    local = report["local_sensitivity"]
    assert len(local["cases"]) == 16
    assert local["relative_weight_changes"] == ["-10%", "+10%"]
    assert all(c["result"]["best_action"] is not None for c in local["cases"])
    displayed = summary(report)
    assert displayed["sensitivity"]["cases"][0]["model_pair"] == [
        {"seat_id": 1, "style": "tight"}, {"seat_id": 2, "style": "tight"}]
    assert len(report["range_sensitivity"]["cases"]) == 8
    assert report["range_sensitivity"]["status"] != "INCOMPLETE"


def test_budget_failure_never_selects_a_partial_best_action():
    report = analyze_file(EXAMPLE, sensitivity=True, max_nodes=1)
    assert report["result"]["best_action"] is None
    assert report["result"]["root_actions"] == []
    assert report["sensitivity"]["status"] == "INCOMPLETE"
    assert complete_report(report) is False


@pytest.mark.parametrize("field", ["models", "ranges"])
@pytest.mark.parametrize("size", [0, 1])
def test_incomplete_baseline_cannot_crash_or_be_repaired_by_sweep(
    tmp_path, field, size,
):
    data = json.loads(EXAMPLE.read_text())
    data[field] = data[field][:size]
    path = tmp_path / "missing-opponent.json"
    path.write_text(json.dumps(data))
    report = analyze_file(path, sensitivity=True)
    assert report["result"]["status"] == "BLOCKED"
    for key in ("sensitivity", "local_sensitivity", "range_sensitivity"):
        assert report[key]["cases"] == []
        assert report[key]["reason"] == "baseline_analysis_blocked"
    assert summary(report)["best_action"] is None
    assert complete_report(report) is False


def test_partial_model_sweep_is_not_a_successful_sensitivity_run(tmp_path):
    data = json.loads(EXAMPLE.read_text())
    data["action_order"] = [0, 1, 2]
    data["history"] = []
    for model in data["models"]:
        model["weights"] = {"fold": "1", "check": "1"}
        model["category_weights"] = {}
        model["price_multipliers"] = []
    p = tmp_path / "fast-baseline.json"
    p.write_text(json.dumps(data))
    report = analyze_file(p, sensitivity=True, max_nodes=20)
    assert report["result"]["best_action"] is not None
    assert report["sensitivity"]["status"] == "INCOMPLETE"
    assert complete_report(report) is False


@pytest.mark.parametrize("bad_band", [
    {"upper_ratio": 0.2, "weights": {}},
    {"upper_ratio": "1", "weights": {"call": 1.0}},
    {"upper_ratio": "1", "callback": "other_private_cards"},
])
def test_price_band_parser_rejects_coercion_and_extra_fields(bad_band):
    data = json.loads(EXAMPLE.read_text())
    data["models"][0]["price_multipliers"] = [bad_band]
    with pytest.raises(ValueError):
        scenario_from_dict(data)


@pytest.mark.parametrize("mutation", [
    "wrong_range_anchor", "bool_order", "float_target", "float_weight",
    "unlabelled_model", "callback", "missing_history", "missing_fee",
    "unlabelled_range", "noncanonical_category", "caller_live_flag",
])
def test_manual_input_is_not_silently_coerced_or_relabelled(mutation):
    data = deepcopy(json.loads(EXAMPLE.read_text()))
    if mutation == "wrong_range_anchor":
        data["range_start"] = "current_node"
    elif mutation == "bool_order":
        data["action_order"][0] = True
    elif mutation == "float_target":
        data["aggression_targets"][0] = 10.0
    elif mutation == "float_weight":
        data["models"][0]["weights"]["call"] = 3.0
    elif mutation == "unlabelled_model":
        data["models"][0]["source"] = "empirical"
    elif mutation == "callback":
        data["models"][0]["callback"] = "read_other_players_cards"
    elif mutation == "missing_history":
        del data["history"]
    elif mutation == "missing_fee":
        del data["other_fees"]
    elif mutation == "unlabelled_range":
        data["range_assumptions"] = ""
    elif mutation == "noncanonical_category":
        data["models"][0]["category_weights"]["00"] = {"call": "1"}
    else:
        data["live_approved"] = True
    with pytest.raises((ValueError, TypeError)):
        scenario_from_dict(data)


def test_zero_reachable_response_mass_is_blocked_not_uniform_fallback(tmp_path):
    data = json.loads(EXAMPLE.read_text())
    data["models"][0]["weights"] = {
        "fold": "0", "call": "0", "check": "0", "bet": "0", "raise": "0"}
    data["models"][0]["category_weights"] = {}
    p = tmp_path / "zero.json"
    p.write_text(json.dumps(data))
    assert analyze_file(p)["result"]["best_action"] is None


def test_large_range_sweep_is_explicitly_incomplete_not_silently_truncated(tmp_path):
    data = json.loads(EXAMPLE.read_text())
    combos = list(combinations(
        ("Ac", "Ad", "Ah", "As", "Kc", "Kd", "Kh", "Ks", "Tc"), 2))
    data["ranges"][0]["combos"] = {a + b: "1" for a, b in combos[:33]}
    p = tmp_path / "many-combos.json"
    p.write_text(json.dumps(data))
    report = analyze_file(p, sensitivity=True)
    assert report["result"]["best_action"] is not None
    assert report["range_sensitivity"]["status"] == "INCOMPLETE"
    assert report["range_sensitivity"]["cases"] == []
    assert report["range_sensitivity"]["reason"] == "range_sweep_budget_exceeded"
    assert complete_report(report) is False
