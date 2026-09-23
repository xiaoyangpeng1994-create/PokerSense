"""Regression gate for the exact pre-optimization AA river baseline."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import json

import pytest

from poker_engine.strategy.threeway_policy_evaluation_v1 import policy_book_hash
from poker_engine.strategy.threeway_river_v1 import RiverAction
from tools import strategy_evaluation_v1 as evaluation
from tools.strategy_evaluation_v1 import (
    DEFAULT_BASELINE, DEFAULT_RESULTS, _case, book_data, book_from_data,
    load_frozen_baseline, planning_scenarios, read_json, run_benchmark, source_hashes,
    verify_published_results, write_json,
)


def test_baseline_replay_matches_frozen_results_and_preserves_safety():
    frozen = read_json(DEFAULT_BASELINE)
    expected = read_json(DEFAULT_RESULTS)
    assert frozen["source_sha256"] == source_hashes()
    assert frozen["groups"] == 6
    assert len({item["book_sha256"] for item in frozen["books"].values()}) == 6

    actual = run_benchmark()
    verify_published_results(actual, expected)
    assert (actual["case_count"], actual["complete_count"],
            actual["blocked_count"]) == (30, 30, 0)
    assert len(actual["negative_vs_simple_policy_case_ids"]) == 12
    assert all(row["metrics"]["frozen_policy"]["fallback_probability"] == "0"
               for row in actual["cases"])
    assert all(row["nodes"] <= frozen["max_nodes"] for row in actual["cases"])
    assert actual["real_hand_acceptance_pending"] is True
    assert actual["strategy_eligible"] is False
    assert actual["advice_emitted"] is False
    assert len(actual["capacity_probes"]) == 6
    assert all(row["status"] == "BLOCKED" and row["best_action"] is None
               and row["strategy_eligible"] is False
               and row["advice_emitted"] is False
               for row in actual["capacity_probes"])


def test_frozen_baseline_exposes_model_knowledge_regret_not_gto_regret():
    cases = {row["case_id"]: row for row in run_benchmark()["cases"]}
    value_heavy = cases["n6-unopened/value_heavy"]
    assert value_heavy["frozen_root_action"]["kind"] == "bet"
    assert value_heavy["world_model_best_action"]["kind"] == "check"
    assert Fraction(value_heavy["delta_vs_check_fold_chips"]) == -80
    assert Fraction(value_heavy["model_knowledge_regret_chips"]) == 80
    assert value_heavy["model_knowledge_regret_scope"] == (
        "SAME_KERNEL_WITH_WORLD_MODEL_KNOWN_IN_ADVANCE")
    out_of_family = cases["n6-unopened/check_trap"]
    assert out_of_family["model_knowledge_regret_chips"] is None
    assert out_of_family["model_knowledge_regret_scope"] == (
        "NOT_COMPUTED_FOR_OUT_OF_FAMILY_OVERRIDE")


def test_book_and_protocol_tampering_rejected(tmp_path):
    frozen = read_json(DEFAULT_BASELINE)
    changed = deepcopy(frozen["books"]["n6-facing_bet"])
    changed["decisions"][0]["action"]["target"] = "999"
    with pytest.raises(ValueError, match="frozen_book_hash_mismatch"):
        book_from_data(changed)

    protocol = read_json(frozen["protocol_path"])
    protocol["worlds"].pop()
    altered_path = tmp_path / "protocol.json"
    altered_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected_baseline_protocol"):
        run_benchmark(protocol_path=altered_path)


def test_published_results_and_safety_claim_tampering_rejected(tmp_path):
    actual = run_benchmark()
    published = read_json(DEFAULT_RESULTS)
    published["cases"][0]["metrics"]["frozen_policy"]["net_ev_chips"] = (
        "999999")
    with pytest.raises(ValueError, match="published_results_digest_mismatch"):
        verify_published_results(actual, published)

    baseline = read_json(DEFAULT_BASELINE)
    baseline["advice_emitted"] = True
    altered_path = tmp_path / "baseline.json"
    altered_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical_baseline_artifact_required"):
        run_benchmark(baseline_path=altered_path)


def test_self_consistent_rehashed_policy_cannot_be_relabeled_baseline_v1(tmp_path):
    frozen = read_json(DEFAULT_BASELINE)
    plans, _, _, _ = planning_scenarios(
        evaluation.DEFAULT_INPUT, evaluation.DEFAULT_PROTOCOL)
    group = "n6-facing_bet"
    book = book_from_data(frozen["books"][group])
    altered = replace(book, decisions=tuple(
        replace(item, action=RiverAction(0, "fold"))
        if item.history == plans[group].history else item
        for item in book.decisions))
    altered = replace(altered, book_sha256=policy_book_hash(altered))
    frozen["books"][group] = book_data(altered)
    # This is a valid, self-consistent policy, but is not the frozen V1 policy.
    assert book_from_data(frozen["books"][group]) == altered
    changed = tmp_path / "relabeled-baseline.json"
    write_json(changed, frozen)
    with pytest.raises(ValueError, match="canonical_baseline_artifact_required"):
        run_benchmark(baseline_path=changed)


def test_canonical_baseline_guard_rejects_source_drift(monkeypatch):
    changed = source_hashes()
    changed["src/poker_engine/strategy/threeway_river_v1.py"] = "0" * 64
    monkeypatch.setattr(evaluation, "source_hashes", lambda: changed)
    with pytest.raises(ValueError, match="baseline_source_drift"):
        load_frozen_baseline()


def test_freeze_cannot_assign_v1_identity_to_changed_input(tmp_path, monkeypatch):
    original_run = evaluation.subprocess.run

    def exact_base_head(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return evaluation.subprocess.CompletedProcess(
                command, 0, stdout=evaluation.BASE_COMMIT + "\n")
        return original_run(command, **kwargs)

    # Isolate the input-identity gate from git history (CI may use depth 1).
    # Policy compilation still runs; the source gate has its own drift test.
    monkeypatch.setattr(evaluation.subprocess, "run", exact_base_head)
    monkeypatch.setattr(evaluation, "require_base_sources", lambda: (
        read_json(DEFAULT_BASELINE)["source_sha256"]))
    changed_input = tmp_path / "changed-input.json"
    changed_input.write_text(
        evaluation.DEFAULT_INPUT.read_text(encoding="utf-8") + "\n",
        encoding="utf-8")
    output = tmp_path / "not-v1.json"
    with pytest.raises(ValueError,
                       match="freeze_does_not_reproduce_canonical_baseline"):
        evaluation.freeze_baseline(input_path=changed_input, output=output)
    assert not output.exists()


def test_frozen_artifact_cannot_be_overwritten(tmp_path):
    artifact = tmp_path / "freeze.json"
    write_json(artifact, {"baseline_id": "BASELINE_V1"})
    with pytest.raises(FileExistsError):
        write_json(artifact, {"baseline_id": "RELABELED"})
    assert read_json(artifact)["baseline_id"] == "BASELINE_V1"


def test_over_budget_case_is_reported_as_blocked():
    plans, _, _, _ = planning_scenarios(
        "configs/strategy/examples/threeway-river-response-manual.json",
        "configs/strategy/examples/threeway-validation-protocol-v1.json")
    plan = plans["n6-facing_bet"]
    combos = ("2c2d", "2h2s", "3c3d", "3h3s", "4c4d", "4h4s",
              "5c5d", "5h5s", "6c6d", "6h6s", "7c7d", "7h7s")
    world = replace(plan, ranges=tuple(replace(
        item, combo_weights={combo: Decimal(1) for combo in combos})
        for item in plan.ranges))
    book = book_from_data(read_json(DEFAULT_BASELINE)["books"]["n6-facing_bet"])
    row = _case("n6-facing_bet", "over_budget", book, world, (), 20000)
    assert row["status"] == "BLOCKED"
    assert row["legal_joint_assignments"] is None
    assert row["joint_range_product"] == 144
    assert row["reasons"] == ["joint_assignment_budget_exceeded"]
