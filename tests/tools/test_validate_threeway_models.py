from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from poker_engine.strategy.threeway_policy_evaluation_v1 import PolicyEvaluation
from tools import validate_threeway_models as driver


EXAMPLES = Path(__file__).resolve().parents[2] / "configs/strategy/examples"
INPUT = EXAMPLES / "threeway-river-response-manual.json"
PROTOCOL = EXAMPLES / "threeway-validation-protocol-v1.json"


def test_mapping_proxy_nested_in_world_contract_serializes(tmp_path):
    plan = driver.public_scenario(json.loads(INPUT.read_text()), 6, "facing_bet")
    assert isinstance(plan.ranges[0].combo_weights, MappingProxyType)
    driver.dump(tmp_path / "world.json", {"scenario": plan})
    result = json.loads((tmp_path / "world.json").read_text())
    assert isinstance(result["scenario"]["ranges"][0]["combo_weights"], dict)


def test_freeze_selection_and_policy_books_precede_validation_and_worlds(
        tmp_path, monkeypatch):
    output = tmp_path / "study"
    observations = driver.synthetic_observations
    evaluate = driver.evaluate_policy_book
    observed = []

    def generate(*args, **kwargs):
        assert (output / "candidates-frozen.json").exists()
        if kwargs["partition"] == "validation":
            assert (output / "selection-before-validation.json").exists()
        return observations(*args, **kwargs)

    def fixed_evaluate(book, *args, **kwargs):
        assert (output / "policy-books-before-worlds.json").exists()
        saved = json.loads((output / "policy-books-before-worlds.json").read_text())
        assert len(saved) == 6
        assert all(len(pair) == 2 for pair in saved.values())
        observed.append(book.book_sha256)
        return evaluate(book, *args, **kwargs)

    monkeypatch.setattr(driver, "synthetic_observations", generate)
    monkeypatch.setattr(driver, "evaluate_policy_book", fixed_evaluate)
    result = driver.run_study(INPUT, PROTOCOL, output)
    assert len(observed) == 60 and result["world_cases"] == 30
    assert result["all_complete"]
    assert result["screening_status"] == "FAIL_STRESS_SCREEN"
    assert len(result["negative_comparisons"]) > 0
    assert len(result["evaluations"]) == 30
    assert result["promotion_decision"] == "NO_EMPIRICAL_PROMOTION"


def test_planning_budget_failures_preserve_all_declared_cases(tmp_path):
    result = driver.run_study(INPUT, PROTOCOL, tmp_path / "blocked", max_nodes=1)
    assert result["screening_status"] == "INCOMPLETE"
    assert not result["all_complete"] and len(result["planning_errors"]) == 12
    assert len(result["evaluations"]) == 30
    assert all(not case["complete"] for case in result["evaluations"])
    assert all(e["status"] == "BLOCKED" for case in result["evaluations"]
               for e in case["evaluations"].values())


def test_evaluation_blocked_case_not_dropped(tmp_path, monkeypatch):
    original = driver.evaluate_policy_book
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return PolicyEvaluation("BLOCKED", reasons=("synthetic_budget_probe",))
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "evaluate_policy_book", evaluate)
    report = driver.run_study(INPUT, PROTOCOL, tmp_path / "one-blocked")
    assert len(calls) == 60 and len(report["evaluations"]) == 30
    assert report["screening_status"] == "INCOMPLETE"
    assert report["evaluations"][0]["complete"] is False
    assert report["evaluations"][0]["evaluations"]["manual_reference"]["reasons"] == [
        "synthetic_budget_probe"]


def test_negative_check_call_delta_alone_fails_screen(tmp_path, monkeypatch):
    original = driver.evaluate_policy_book

    def evaluate(book, *args, **kwargs):
        value = original(book, *args, **kwargs)
        if book.policy_id.endswith("training_selected"):
            # Force the two other screen inputs nonnegative without removing
            # the independent check-call comparison that the protocol promises.
            metrics = (replace(value.metrics[0], net_ev_chips=Fraction(10 ** 6)),
                       *value.metrics[1:])
            return replace(value, metrics=metrics,
                           delta_vs_check_fold_chips=Fraction(1),
                           delta_vs_check_call_chips=Fraction(-1))
        return value

    monkeypatch.setattr(driver, "evaluate_policy_book", evaluate)
    report = driver.run_study(INPUT, PROTOCOL, tmp_path / "check-call-loss")
    assert report["screening_status"] == "FAIL_STRESS_SCREEN"
    assert len(report["negative_comparisons"]) == 30
    assert all(row["delta_vs_check_call"]["exact"] == "-1"
               for row in report["negative_comparisons"])


def test_facing_bet_label_requires_replayed_positive_debt():
    data = json.loads(INPUT.read_text())
    data["history"] = []
    data["action_order"] = [0, 1, 2]
    with pytest.raises(ValueError, match="positive_actual_to_call"):
        driver.public_scenario(data, 6, "facing_bet")
    assert driver.public_scenario(data, 6, "unopened").history == ()


@pytest.mark.parametrize("field,value", [
    ("source_kind", "reviewed_all_decisions"), ("training_seed", True),
    ("decisions_per_session", 1), ("table_sizes", [6]),
    ("worlds", ["reference_control"]), ("generating_candidate_id", []),
    ("candidate_ids", ["chosen_after_validation"]), ("qualification", ""),
])
def test_protocol_changes_cannot_silently_reduce_or_relabel_study(field, value):
    data = json.loads(PROTOCOL.read_text())
    data[field] = value
    with pytest.raises(ValueError):
        driver.validate_protocol(data)


def test_same_random_stream_and_extra_protocol_fields_rejected():
    data = json.loads(PROTOCOL.read_text())
    data["validation_seed"] = data["training_seed"]
    with pytest.raises(ValueError, match="separate"):
        driver.validate_protocol(data)
    data = json.loads(PROTOCOL.read_text())
    data["drop_failed_cases"] = True
    with pytest.raises(ValueError, match="exact"):
        driver.validate_protocol(data)


def test_existing_evidence_directory_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    evidence = output / "report.json"
    evidence.write_text("original evidence")
    with pytest.raises(FileExistsError):
        driver.run_study(INPUT, PROTOCOL, output)
    assert evidence.read_text() == "original evidence"


def test_duplicate_protocol_json_field_rejected_before_output(tmp_path):
    protocol = tmp_path / "duplicate.json"
    protocol.write_text(PROTOCOL.read_text().replace(
        '"schema_version":', '"schema_version":1,"schema_version":', 1))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="duplicate"):
        driver.run_study(INPUT, protocol, output)
    assert not output.exists()


def test_only_own_public_models_and_complete_sampling_stream_generated():
    candidates = driver.public_candidates((1, 2))
    rows = driver.synthetic_observations(candidates[0], partition="train", seed=4,
                                         sessions=2, count=12)
    repeated = driver.synthetic_observations(candidates[0], partition="train", seed=4,
                                             sessions=2, count=12)
    assert rows == repeated and len(rows) == 24
    assert len({r.row_id for r in rows}) == 24
    assert all(r.source_kind == "synthetic" and r.own_category is None
               and r.sampling_frame == "all_decisions" for r in rows)


def test_protocol_input_not_mutated_by_world_or_root_builders():
    data = json.loads(INPUT.read_text())
    before = deepcopy(data)
    plan = driver.public_scenario(data, 8, "unopened")
    for world in driver.WORLDS:
        driver.world_scenario(plan, world, driver.public_candidates((1, 2))[0])
    assert data == before
