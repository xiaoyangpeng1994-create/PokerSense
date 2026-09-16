#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for the C single-factor opponent-range experiment.

The committed result sample must be the current output of the real driver, the
declared baseline factor must still reproduce the original 30-world reading, the
two policy books must be frozen across all three worlds, and every world must
reconcile exactly. The original 30-world study and the uncertainty study are
re-run here, so a range perturbation can never silently drift the baseline.
"""

import importlib.util
import json
import os
import subprocess
import sys
from fractions import Fraction

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLES = os.path.join(REPO_ROOT, "configs", "strategy", "examples")
INPUT = os.path.join(EXAMPLES, "threeway-river-response-manual.json")
STUDY_PROTOCOL = os.path.join(EXAMPLES, "threeway-validation-protocol-v1.json")
PROTOCOL = os.path.join(EXAMPLES, "strategy-diag-c-range-protocol-v1.json")
SAMPLE = os.path.join(EXAMPLES, "strategy-diag-c-range-experiment-v1.json")
DRIVER = os.path.join(REPO_ROOT, "tools", "threeway_range_experiment.py")
STUDY_TOOL = os.path.join(REPO_ROOT, "tools", "validate_threeway_models.py")
UNCERTAINTY_TOOL = os.path.join(REPO_ROOT, "tools", "study_opponent_uncertainty.py")
FACTORS = ("0.5", "1", "2")
BOOKS = ("manual_reference", "training_selected")
TARGET_FACTORS = ["0.5", "1", "2"]


def env():
    values = dict(os.environ)
    values["PYTHONPATH"] = "src" + os.pathsep + "."
    values["PYTHONUTF8"] = "1"
    return values


def run_tool(tool, *args):
    return subprocess.run(
        [sys.executable, tool, *[str(arg) for arg in args]], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env())


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def driver():
    spec = importlib.util.spec_from_file_location("range_experiment", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sample():
    return load(SAMPLE)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Run the original 30-world production study once."""
    out = tmp_path_factory.mktemp("diag-c-study") / "study"
    proc = run_tool(STUDY_TOOL, "--input", INPUT, "--protocol", STUDY_PROTOCOL,
                    "--output", out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return load(os.path.join(str(out), "report.json"))


def world_of(sample, factor):
    return next(item for item in sample["worlds"] if item["factor"] == factor)


def test_check_mode_accepts_the_committed_sample():
    proc = run_tool(DRIVER, "--check")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "STRATEGY-DIAG-C-ONE-RANGE-PARAMETER-V1" in proc.stdout
    assert '"matches_original_reading": true' in proc.stdout


def test_fresh_run_reproduces_the_committed_sample(tmp_path, sample):
    out = tmp_path / "experiment.json"
    proc = run_tool(DRIVER, "--output", out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert driver().stable_view(load(out)) == driver().stable_view(sample)


def test_baseline_factor_reproduces_the_original_study_reading(sample, study):
    record = next(item for item in study["evaluations"]
                  if item["group"] == sample["study_group"]
                  and item["world"] == sample["study_world"])
    reading = sample["readings_by_factor"]["1"]
    assert sample["baseline_verification"]["matches_original_reading"] is True
    assert sample["baseline_verification"]["mismatches"] == []
    assert reading["selected_minus_manual_reference"] == record[
        "selected_minus_manual_reference_chips"]["exact"]
    readings = (("training_selected", "training_selected_frozen_policy"),
                ("manual_reference", "manual_reference_frozen_policy"))
    for name, key in readings:
        assert reading[key] == record["evaluations"][name]["metrics"][0][
            "net_ev_chips"]["exact"]
    assert world_of(sample, "1")["world_scenario_sha256"] == record[
        "evaluations"]["training_selected"]["world_scenario_sha256"]
    loss = next(item for item in study["negative_comparisons"]
                if item["group"] == sample["study_group"]
                and item["world"] == sample["study_world"])
    assert loss["delta_vs_reference"]["exact"] == reading[
        "selected_minus_manual_reference"]
    assert Fraction(reading["selected_minus_manual_reference"]) < 0
    assert study["selected_id"] == sample["selected_id"]


def test_only_the_declared_opponent_combo_weight_changes():
    module = driver()
    protocol, _ = module.load_protocol(PROTOCOL)
    varied = protocol["single_varied_object"]
    plan, _, _ = module.planning_context(INPUT, STUDY_PROTOCOL, protocol)
    seat, combos = varied["target_opponent_seat"], tuple(varied["target_combos"])
    worlds = {text: module.perturb_target_range(
        plan, seat, combos, module.Decimal(text))
        for text in varied["relative_factors"]}
    assert worlds["1"].ranges == plan.ranges
    for text in ("0.5", "2"):
        other = worlds[text]
        for attribute in ("rules", "models", "seats", "history", "action_order",
                          "aggression_targets", "max_aggressions", "hero_cards",
                          "board_cards", "hero_seat"):
            assert getattr(other, attribute) == getattr(plan, attribute)
        for seat_id in (1, 2):
            before = next(r for r in plan.ranges if r.seat_id == seat_id)
            after = next(r for r in other.ranges if r.seat_id == seat_id)
            if seat_id != seat:
                assert after == before
                continue
            assert set(after.combo_weights) == set(before.combo_weights)
            factor = module.Decimal(text)
            for combo, weight in before.combo_weights.items():
                expected = weight * (factor if combo in combos
                                     else module.Decimal(1))
                assert after.combo_weights[combo] == expected
    target = next(r for r in plan.ranges if r.seat_id == seat)
    assert 0 < len(varied["target_combos"]) < len(target.combo_weights)


def test_books_are_frozen_once_and_reused_in_every_world(sample):
    declared = {item["name"]: item["policy_book_sha256_before"]
                for item in sample["frozen_books"]}
    assert sorted(declared) == sorted(BOOKS)
    for item in sample["frozen_books"]:
        assert item["unchanged_across_worlds"] is True
        assert item["policy_book_sha256_before"] == item[
            "policy_book_sha256_after"]
    for factor in FACTORS:
        world = world_of(sample, factor)
        for name in BOOKS:
            book = world["books"][name]
            assert book["policy_unchanged_during_world"] is True
            assert book["policy_hash_before"] == book["policy_hash_after"] == (
                declared[name])
            assert book["status"] == "COMPLETE_CONDITIONAL_FIXED_POLICY"


def test_world_evaluation_never_calls_an_optimizing_entry_point(monkeypatch):
    module = driver()
    protocol, _ = module.load_protocol(PROTOCOL)
    varied = protocol["single_varied_object"]
    plan, selected, _ = module.planning_context(INPUT, STUDY_PROTOCOL, protocol)
    books = module.freeze_books(plan, selected, protocol,
                                protocol["compute"]["max_nodes"])
    base, overrides = module.world_scenario(plan, protocol[
        "original_failing_scenario"]["study_world"], selected)

    def forbidden(*args, **kwargs):
        raise AssertionError("world evaluation attempted to optimize Hero")

    engine = sys.modules["poker_engine.strategy.threeway_policy_evaluation_v1"]
    monkeypatch.setattr(engine, "analyze_threeway_river", forbidden)
    monkeypatch.setattr(
        sys.modules["poker_engine.strategy.threeway_river_v1"]._Tree, "value",
        forbidden)
    for text in TARGET_FACTORS:
        world = module.perturb_target_range(
            base, varied["target_opponent_seat"], tuple(varied["target_combos"]),
            module.Decimal(text))
        for name in BOOKS:
            evaluation, elapsed = module.evaluate_world(
                books[name], world, overrides, protocol["compute"])
            assert evaluation.status == "COMPLETE_CONDITIONAL_FIXED_POLICY"
            assert evaluation.policy_hash_before == evaluation.policy_hash_after
            assert evaluation.policy_hash_before == books[name].book_sha256
            assert elapsed > 0 and evaluation.path_ledgers


def test_every_world_reconciles_and_keeps_the_shared_baselines(sample):
    for factor in FACTORS:
        world = world_of(sample, factor)
        assert world["status"] == "COMPLETE_CONDITIONAL_FIXED_POLICY"
        assert world["joint_assignments"] == 4
        assert world["nonzero_posterior_entries"] == 4
        assert sum(Fraction(value) for value in world["root_posterior"]) == 1
        for name in BOOKS:
            book = world["books"][name]
            metrics = {item["name"]: item for item in book["metrics"]}
            assert Fraction(book["trace"]["reach_probability_sum"]) == 1
            assert Fraction(book["trace"]["contribution_sum_chips"]) == Fraction(
                metrics["frozen_policy"]["net_ev_chips"])
            for reconciliation in book["reconciliations"]:
                assert reconciliation["distinct_books"] is False
                assert reconciliation["right_policy_book_sha256"] is None
                assert Fraction(reconciliation["total_ev_difference_chips"]) == (
                    Fraction(reconciliation["contribution_difference_sum_chips"]))
                assert sum(Fraction(row["contribution_chips_difference"])
                           for row in reconciliation["rows"]) == Fraction(
                    reconciliation["contribution_difference_sum_chips"])
        reference = world["books"]["manual_reference"]["metrics"]
        selected = world["books"]["training_selected"]["metrics"]
        for index in (1, 2):
            assert reference[index]["net_ev_chips"] == selected[index][
                "net_ev_chips"]
        cross = world["cross_book_branch_difference"]
        assert cross["distinct_books"] is True
        assert cross["left_policy_book_sha256"] != cross[
            "right_policy_book_sha256"]
        assert Fraction(cross["total_ev_difference_chips"]) == Fraction(
            world["selected_minus_manual_reference_chips"])
        assert Fraction(cross["contribution_difference_sum_chips"]) == Fraction(
            cross["total_ev_difference_chips"])
        assert cross["rows_total"] == 20
        assert cross["rows_with_nonzero_difference"] + cross[
            "zero_difference_rows"] == cross["rows_total"]


def test_perturbation_outcomes_are_exact_and_descriptively_reported(sample):
    readings = sample["readings_by_factor"]
    assert sorted(readings) == list(FACTORS)
    assert Fraction(readings["1"]["selected_minus_manual_reference"]) < 0
    assert Fraction(readings["0.5"]["selected_minus_manual_reference"]) > 0
    assert Fraction(readings["2"]["selected_minus_manual_reference"]) < 0
    assert world_of(sample, "0.5")["failure_gate_fired"] == []
    assert world_of(sample, "2")["failure_gate_fired"] == [
        "selected_minus_manual_reference", "selected_delta_vs_check_fold",
        "selected_delta_vs_check_call"]
    direction = sample["descriptive_direction_over_0.5_1_2"]
    assert direction["factor_order"] == TARGET_FACTORS
    assert direction["direction"]["history_likelihood"] == "increasing"
    assert direction["qualification"] == "descriptive_only_no_improvement_claim"
    assert sample["strategy_eligible"] is False
    assert sample["advice_emitted"] is False
    assert sample["fixed_single_varied_object"]["target_combos"] == ["JhJd"]
    for factor in FACTORS:
        weights = dict(world_of(sample, factor)["books"]["training_selected"][
            "perturbed_range"]["combo_weights"])
        assert weights["JhJd"] == factor
        assert weights["TcTd"] == "3"


@pytest.mark.parametrize("mutation", ["whole_range", "empty", "unknown_combo",
                                      "zero_factor", "negative_factor",
                                      "unknown_seat"])
def test_invalid_range_perturbations_are_rejected(mutation):
    module = driver()
    protocol, _ = module.load_protocol(PROTOCOL)
    plan, _, _ = module.planning_context(INPUT, STUDY_PROTOCOL, protocol)
    seat, combos, factor = 1, ("JhJd",), module.Decimal("1")
    if mutation == "whole_range":
        combos = ("JhJd", "TcTd")
    elif mutation == "empty":
        combos = ()
    elif mutation == "unknown_combo":
        combos = ("AhAd",)
    elif mutation == "zero_factor":
        factor = module.Decimal("0")
    elif mutation == "negative_factor":
        factor = module.Decimal("-1")
    else:
        seat = 0
    with pytest.raises(ValueError):
        module.perturb_target_range(plan, seat, combos, factor)


@pytest.mark.parametrize("broken", [
    {"scope": "SOMETHING_ELSE"},
    {"status_of_this_file": "NOT_FROZEN"},
    {"relative_factors": ["1", "2", "4"]},
    {"relative_factors": ["0.5", "1"]},
    {"target_combos": ["JhJd", "TcTd"]},
    {"trace": False},
])
def test_protocol_mutations_are_rejected(tmp_path, broken):
    module = driver()
    original = load(PROTOCOL)
    mutated = dict(original)
    for key, value in broken.items():
        if key in ("relative_factors", "target_combos"):
            mutated["single_varied_object"] = dict(
                original["single_varied_object"], **{key: value})
            if key == "target_combos":
                applicability = dict(
                    original["original_failing_scenario"]["applicability_check"],
                    target_opponent_combos=value,
                )
                mutated["original_failing_scenario"] = dict(
                    original["original_failing_scenario"],
                    applicability_check=applicability,
                    )
                mutated["single_varied_object"] = dict(
                    mutated["single_varied_object"],
                    untouched_combos_of_that_opponent=[],
                )
        elif key == "trace":
            mutated["compute"] = dict(original["compute"], trace=value)
        else:
            mutated[key] = value
    path = tmp_path / "mutated-protocol.json"
    path.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_protocol(path)


def test_budget_exhaustion_blocks_every_world_without_partial_readings(tmp_path):
    module = driver()
    protocol = load(PROTOCOL)
    protocol["compute"] = dict(protocol["compute"], trace_max_nodes=1)
    path = tmp_path / "budget-protocol.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    report = module.run_experiment(INPUT, path, STUDY_PROTOCOL)
    assert [item["status"] for item in report["worlds"]] == ["BLOCKED"] * 3
    assert all(item["cross_book_branch_difference"] is None
               for item in report["worlds"])
    assert all(item["books"]["training_selected"]["trace"] is None
               for item in report["worlds"])
    assert all(value is None for reading in report["readings_by_factor"].values()
               for value in reading.values())
    assert "trace_node_budget_exceeded" in report["worlds"][0]["reasons"]
    assert report["baseline_verification"]["matches_original_reading"] is False


def test_original_study_reading_and_uncertainty_study_still_hold(tmp_path, study):
    assert study["screening_status"] == "FAIL_STRESS_SCREEN"
    assert study["promotion_decision"] == "NO_EMPIRICAL_PROMOTION"
    assert study["calibration_status"] == "NOT_REAL_CALIBRATION"
    assert study["world_cases"] == 30 and study["all_complete"] is True
    assert len(study["evaluations"]) == 30
    assert len(study["negative_comparisons"]) == 18
    assert study["fallback_cases"] == []
    out = tmp_path / "uncertainty"
    proc = run_tool(UNCERTAINTY_TOOL, "--input", INPUT, "--output", out)
    # exit 2 is the declared INSUFFICIENT_EVIDENCE signal, not a crash
    assert proc.returncode == 2
    report = load(os.path.join(str(out), "report.json"))
    assert len(report["groups"]) == 6
    assert {item["validation"]["status"] for item in report["groups"]} == {
        "FAIL_DECLARED_WORLD_SCREEN", "INSUFFICIENT_EVIDENCE"}
    assert report["strategy_eligible"] is False
