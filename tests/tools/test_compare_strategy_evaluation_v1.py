"""Pairing rejects changed worlds and incomplete candidates."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from tools.compare_strategy_evaluation_v1 import compare_reports
from tools.strategy_evaluation_v1 import (
    DEFAULT_BASELINE, DEFAULT_RESULTS, _case, book_data, book_from_data, digest,
    planning_scenarios, public_candidates, read_json, stable_json, world_scenario,
)
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    compile_policy_book, policy_book_hash,
)
from poker_engine.strategy.threeway_river_v1 import RiverAction


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


def test_actual_world_specific_reoptimization_is_rejected():
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    plans, _, _, _ = planning_scenarios(
        "configs/strategy/examples/threeway-river-response-manual.json",
        "configs/strategy/examples/threeway-validation-protocol-v1.json")
    calling = next(item for item in public_candidates((1, 2))
                   if item.candidate_id == "calling_public_v1")
    world, overrides = world_scenario(
        plans["n6-facing_bet"], "value_heavy", calling)
    # Genuine evaluator output can still leak knowledge of one test world.
    book = compile_policy_book(world, policy_id="leaked-value-heavy")
    row = _case("n6-facing_bet", "value_heavy", book, world, overrides, 20000)
    row.pop("elapsed_ms")
    candidate["cases"][1] = row
    with pytest.raises(ValueError, match="candidate_policy_changed_between_worlds"):
        compare_reports(baseline, reseal(candidate))


@pytest.mark.parametrize("field,value", [
    ("history_likelihood", "1/999"),
    ("delta_vs_check_fold_chips", "999"),
    ("nodes", 999),
    ("reasons", ["unreported_fallback"]),
])
def test_same_book_world_diagnostics_cannot_drift(field, value):
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    candidate["cases"][0][field] = value
    with pytest.raises(ValueError, match="candidate_case_replay_mismatch"):
        compare_reports(baseline, reseal(candidate))


@pytest.mark.parametrize("metric,field,value", [
    ("check_fold", "net_ev_chips", "999"),
    ("check_call", "conditional_net_ev_bb", "-999"),
    ("frozen_policy", "fallback_probability", "-1"),
    ("frozen_policy", "fallback_probability", "2"),
    ("frozen_policy", "expected_fallback_count", "10000"),
    ("frozen_policy", "conditional_net_ev_bb", "999"),
])
def test_metric_controls_units_and_fallback_cannot_drift(metric, field, value):
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    candidate["cases"][0]["metrics"][metric][field] = value
    with pytest.raises(ValueError, match="candidate_case_replay_mismatch"):
        compare_reports(baseline, reseal(candidate))


@pytest.mark.parametrize("field,value,error", [
    ("complete_count", 29, "evaluation_case_counts_mismatch"),
    ("blocked_count", True, "evaluation_case_counts_mismatch"),
    ("base_commit", "0" * 40, "candidate_evaluation_identity_mismatch"),
    ("baseline_file_sha256", "0" * 64, "candidate_evaluation_identity_mismatch"),
])
def test_candidate_report_identity_and_counts_are_checked(field, value, error):
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    candidate[field] = value
    with pytest.raises(ValueError, match=error):
        compare_reports(baseline, reseal(candidate))


@pytest.mark.parametrize("field,value", [
    ("status", "TIMED_OUT_BUT_COMPLETE"),
    ("policy_book_sha256", "not-a-book-hash"),
    ("declared_world_sha256", "0"),
])
def test_unknown_status_or_malformed_identity_is_rejected(field, value):
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    candidate["cases"][0][field] = value
    with pytest.raises(ValueError, match="invalid_case_status_or_identity"):
        compare_reports(baseline, reseal(candidate))


def test_blocked_case_cannot_hide_partial_metrics_or_a_world_specific_book():
    baseline = read_json(DEFAULT_RESULTS)
    candidate = deepcopy(baseline)
    row = candidate["cases"][0]
    row.update(status="BLOCKED", reasons=["timeout"],
               legal_joint_assignments=None, world_scenario_sha256=None)
    candidate.update(complete_count=29, blocked_count=1)
    with pytest.raises(ValueError, match="blocked_case_claims_partial"):
        compare_reports(baseline, reseal(candidate))
    row["policy_book_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="candidate_policy_changed_between_worlds"):
        compare_reports(baseline, reseal(candidate))


@pytest.fixture
def replayed_candidate():
    """An explicit test policy edit, frozen before all five world evaluations."""
    candidate = read_json(DEFAULT_RESULTS)
    books = read_json(DEFAULT_BASELINE)["books"]
    plans, protocol, _, _ = planning_scenarios(
        "configs/strategy/examples/threeway-river-response-manual.json",
        "configs/strategy/examples/threeway-validation-protocol-v1.json")
    group = "n6-unopened"
    book = book_from_data(books[group])
    decisions = tuple(replace(item, action=RiverAction(0, "check", Decimal(0)))
                      if item.history == plans[group].history else item
                      for item in book.decisions)
    changed = replace(book, policy_id="test-fixed-root-check", decisions=decisions)
    changed = replace(changed, book_sha256=policy_book_hash(changed))
    books[group] = book_data(changed)
    calling = next(item for item in public_candidates((1, 2))
                   if item.candidate_id == "calling_public_v1")
    for world_name in protocol["worlds"]:
        world, overrides = world_scenario(plans[group], world_name, calling)
        row = _case(group, world_name, changed, world, overrides, 20000)
        row.pop("elapsed_ms")
        index = next(index for index, old in enumerate(candidate["cases"])
                     if old["case_id"] == row["case_id"])
        candidate["cases"][index] = row
    return reseal(candidate), books


def test_changed_policy_requires_complete_serialized_books_and_exact_replay(
        replayed_candidate):
    candidate, books = replayed_candidate
    baseline = read_json(DEFAULT_RESULTS)
    with pytest.raises(ValueError, match="candidate_book_payload_required"):
        compare_reports(baseline, candidate)
    with pytest.raises(ValueError, match="complete_candidate_books_required"):
        compare_reports(baseline, candidate, {})
    result = compare_reports(baseline, candidate, books)
    assert result["status"] == "COMPARABLE_SYNTHETIC"
    assert result["paired_count"] == 30
    assert result["candidate_blocked_count"] == 0
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False


def test_resealed_invented_ev_with_valid_new_book_is_rejected(replayed_candidate):
    candidate, books = replayed_candidate
    row = next(row for row in candidate["cases"]
               if row["case_id"] == "n6-unopened/value_heavy")
    row["metrics"]["frozen_policy"]["net_ev_chips"] = "1000000"
    row["metrics"]["frozen_policy"]["conditional_net_ev_bb"] = "500000"
    row["delta_vs_check_fold_chips"] = "1000000"
    row["delta_vs_check_call_chips"] = "1000000"
    with pytest.raises(ValueError, match="candidate_case_replay_mismatch"):
        compare_reports(read_json(DEFAULT_RESULTS), reseal(candidate), books)


def test_candidate_book_payload_tampering_is_rejected(replayed_candidate):
    candidate, books = replayed_candidate
    books["n6-unopened"]["decisions"][0]["action"]["kind"] = "fold"
    with pytest.raises(ValueError, match="frozen_book_hash_mismatch"):
        compare_reports(read_json(DEFAULT_RESULTS), candidate, books)


def test_even_self_control_requires_frozen_source_guard(monkeypatch):
    def reject():
        raise ValueError("baseline_source_drift")
    monkeypatch.setattr(
        "tools.compare_strategy_evaluation_v1.load_frozen_baseline", reject)
    baseline = read_json(DEFAULT_RESULTS)
    with pytest.raises(ValueError, match="baseline_source_drift"):
        compare_reports(baseline, baseline)
