from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.analyze_terminal_multiway import (
    analyze_file, scenario_from_dict, unique_object,
)


EXAMPLE = Path(__file__).resolve().parents[2] / (
    "configs/strategy/examples/terminal-multiway-river-manual.json")


def test_example_runs_true_four_way_sidepots_and_conditional_action():
    report = analyze_file(EXAMPLE)
    assert report["table_players"] == 6 and report["all_in_opponents"] == 3
    result = report["result"]
    assert result["recommendation"] == "CALL"
    assert result["call_cost"] == "60"
    assert result["call_net_ev"]["exact"] == "256"
    assert [p["amount"] for p in result["call_pots"]] == ["110", "130", "80"]
    assert result["advice_emitted"] is False and result["strategy_eligible"] is False


@pytest.mark.parametrize("change", [
    "float_money", "bool_seat", "missing_fee", "missing_assumptions",
    "hidden_permission", "range_float_weight", "bad_card", "unknown_mode",
])
def test_cli_does_not_coerce_unknown_or_undisclosed_input(change):
    data = deepcopy(json.loads(EXAMPLE.read_text()))
    if change == "float_money":
        data["current_bet"] = 100.0
    elif change == "bool_seat":
        data["seats"][0]["seat_id"] = False
    elif change == "missing_fee":
        del data["other_fees"]
    elif change == "missing_assumptions":
        data["range_assumptions"] = ""
    elif change == "hidden_permission":
        data["live_approved"] = True
    elif change == "range_float_weight":
        data["ranges"][0]["combos"]["KhKd"] = 1.0
    elif change == "bad_card":
        data["hero_cards"][0] = "Ax"
    else:
        data["mode"] = "live"
    with pytest.raises((TypeError, ValueError)):
        scenario_from_dict(data)


def test_duplicate_input_keys_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="duplicate"):
        json.loads('{"mode":"manual_hypothesis","mode":"live"}',
                   object_pairs_hook=unique_object)


def test_unsupported_nonterminal_state_reports_blocked(tmp_path):
    data = json.loads(EXAMPLE.read_text())
    data["seats"][1].update(status="ACTIVE", stack="100")
    path = tmp_path / "nonterminal.json"
    path.write_text(json.dumps(data))
    report = analyze_file(path)
    assert report["result"]["status"] == "BLOCKED"
    assert report["result"]["recommendation"] is None
