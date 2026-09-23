"""Pairing rejects changed worlds and incomplete candidates."""

from copy import deepcopy

import pytest

from tools.compare_strategy_evaluation_v1 import compare_reports
from tools.strategy_evaluation_v1 import (
    DEFAULT_BASELINE, DEFAULT_RESULTS, _case, book_from_data, digest,
    planning_scenarios, read_json, stable_json,
)


def reseal(report):
    body = {key: value for key, value in report.items()
            if key not in ("runtime", "deterministic_sha256")}
    report["deterministic_sha256"] = digest(stable_json(body).encode())
    return report


def test_frozen_report_self_control_has_zero_paired_deltas():
    baseline = read_json(DEFAULT_RESULTS)
    result = compare_reports(baseline, baseline)
    assert result["status"] == "SELF_CONTROL"
    assert result["paired_count"] == 30
    assert result["candidate_blocked_count"] == 0
    assert result["positive_delta_count"] == 0
    assert result["negative_delta_count"] == 0
    assert result["equal_delta_count"] == 30
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False


def test_changed_world_rejected_even_with_valid_resealed_digest():
    baseline = read_json(DEFAULT_RESULTS)
    changed = reseal(deepcopy(baseline))
    changed["cases"][0]["world_scenario_sha256"] = "0" * 64
    reseal(changed)
    with pytest.raises(ValueError, match="changed_evaluation_world"):
        compare_reports(baseline, changed)

    with pytest.raises(ValueError, match="frozen_baseline_report_required"):
        compare_reports(changed, baseline)


def test_blocked_candidate_is_retained_in_denominator():
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    plans, _, _, _ = planning_scenarios(
        "configs/strategy/examples/threeway-river-response-manual.json",
        "configs/strategy/examples/threeway-validation-protocol-v1.json")
    book = book_from_data(read_json(DEFAULT_BASELINE)["books"]["n6-facing_bet"])
    blocked = _case("n6-facing_bet", "reference_control", book,
                    plans["n6-facing_bet"], (), 1)
    blocked.pop("elapsed_ms")
    assert blocked["status"] == "BLOCKED"
    assert blocked["legal_joint_assignments"] is None
    candidate["cases"][0] = blocked
    candidate["complete_count"] = 29
    candidate["blocked_count"] = 1
    reseal(candidate)
    result = compare_reports(baseline, candidate)
    assert result["status"] == "INCOMPLETE"
    assert result["case_count"] == 30
    assert result["paired_count"] == 29
    assert result["candidate_blocked_count"] == 1
    assert result["cases"][0]["reasons"] == [
        "global_evaluation_node_budget_exceeded"]
