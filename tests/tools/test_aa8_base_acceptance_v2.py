import copy
import json
from pathlib import Path

import pytest

from tools.aa8_base_acceptance_v2 import (
    BASE_FIELDS, declaration_template, evaluate_evidence,
)


def fixture():
    repo = Path(__file__).resolve().parents[2]
    scope = json.loads((repo / "configs/reproduction/aa8_candidate_v2/"
                       "base_visual_scope.json").read_text(encoding="utf-8"))
    declaration = declaration_template()
    declaration.update(declared_at_utc="2020-01-01T00:00:00Z", hardware_min_seconds=10)
    registry = {"role": "holdout", "audit_sha256": "audit",
                "candidate_range_seconds": ["300", "600"], "hands": [{
                    "id": "h", "first_frame": 1, "last_frame": 3,
                    "preroll_first_frame": 0, "boundaries_verified": True,
                    "opportunities_reviewed": True,
                    "action_opportunity_frames": [1, 2, 3]}]}
    pool = {"audit_sha256": "audit", "samples": [{
        "global_frame": f, "role": "holdout", "pts_seconds": str(301 + f / 30),
        "sha256": "a" * 64} for f in range(4)]}
    values = {"actor": 0, "street_wagers": dict.fromkeys(map(str, range(8)), "0"),
              "actions": [], "pot": "0",
              "stacks": dict.fromkeys(map(str, range(8)), "100"),
              "street": "preflop", "hand": "h", "hero_cards": ["Ah", "Kd"],
              "board_cards": [],
              "participation": dict.fromkeys(map(str, range(8)), "active")}
    labels, predictions = [], []
    for frame in range(1, 4):
        fields = {key: {"status": "KNOWN", "value": copy.deepcopy(values[key]),
                        "origin": "automatic"} for key in BASE_FIELDS}
        labels.append({"frame": frame, "source_sha256": "a" * 64, "fields": fields,
                       "scene": "supported", "scene_evidence": {
                           "source_reviewed": True, "source_sha256": "a" * 64},
                       "decision_opportunity": True})
        predictions.append({
            "frame": frame, "source_sha256": "a" * 64,
            "evaluation_fields": copy.deepcopy(fields), "visual_scope": {
                "scope_id": "aa8_base_visual_v1", "status": "BASE_CANDIDATE_UNVERIFIED",
                "unknown_cash_policy": "UNALLOCATED",
                "automatically_exempt_from_acceptance": False},
            "actions_complete_and_canonical_verified": True,
            "causal_street_wagers_v2": {"canonical_verified": True},
            "observed_state_v2": {"authoritative_hand_boundary": True}})
    gold = {"labels": labels, "prediction_used_for_labels": False,
            "source_review_started_at_utc": "2020-01-01T00:00:01Z",
            "prediction_start_utc": "2020-01-01T00:00:02Z"}
    hardware = {key: True for key in (
        "user_authorized", "target_device", "end_to_end", "freshness_checked",
        "reconnect_tested", "recovery_verified", "latency_budget_met")}
    hardware.update(duration_seconds=10, unexplained_gaps=0,
                    undetected_stale_frames=0, crashes=0)
    return scope, declaration, registry, pool, gold, predictions, hardware


def special_middle(args):
    _, _, registry, _, gold, predictions, _ = args
    registry["hands"][0]["action_opportunity_frames"] = [1, 3]
    label, pred = gold["labels"][1], predictions[1]
    label.update(scene="special", decision_opportunity=False)
    label["scene_evidence"]["mode"] = "insurance"
    for field in BASE_FIELDS:
        label["fields"][field] = {
            "status": "UNKNOWN", "value": None, "required": False,
            "exemption": {"reason": "special_ui", "source_reviewed": True,
                          "source_sha256": "a" * 64}}
        pred["evaluation_fields"][field] = {"status": "UNKNOWN", "value": None}
    pred["visual_scope"].update(status="DEFERRED_MODE_OBSERVED",
                                observed_deferred_modes=["insurance"])
    predictions[2]["visual_scope"]["resume_evidence"] = {
        "fresh_context_verified": True, "source_frames": [3]}


def test_synthetic_base_pass_never_full_functional_or_strategy_pass():
    result = evaluate_evidence(*fixture())
    assert result["status"] == "BASE_VISUAL_PASS"
    assert not result["full_visual_acceptance"]
    assert not result["strategy_eligible"]
    assert len(result["coverage"]) == 10


def test_special_frame_retained_and_guarded_not_dropped():
    args = fixture()
    special_middle(args)
    result = evaluate_evidence(*args)
    assert result["status"] == "BASE_VISUAL_PASS"
    assert result["owned_frames"] == 3
    assert result["coverage"]["actor"] == {
        "total": 3, "required": 2, "matched_required": 2,
        "known_predictions": 2, "guarded": 1}


def test_missing_special_guard_fails_even_with_unknown_outputs():
    args = fixture()
    special_middle(args)
    args[5][1]["visual_scope"]["status"] = "BASE_CANDIDATE_UNVERIFIED"
    assert "special_guard_missed" in evaluate_evidence(*args)["failures"]


def test_unknown_mode_not_assumed_supported():
    args = fixture()
    args[4]["labels"][1]["scene"] = "unknown"
    assert "scene_unknown_not_normal" in evaluate_evidence(*args)["failures"]


def test_post_prediction_scope_declaration_rejected():
    args = fixture()
    args[1]["declared_at_utc"] = "2020-01-01T00:00:03Z"
    assert evaluate_evidence(*args)["status"] == "BASE_VISUAL_PARTIAL"


def test_sparse_gold_stays_partial():
    args = fixture()
    args[4]["labels"].pop()
    assert "whole_frame_coverage:gold" in evaluate_evidence(*args)["failures"]


@pytest.mark.parametrize("key,child", [
    ("actions_complete_and_canonical_verified", None),
    ("causal_street_wagers_v2", "canonical_verified"),
    ("observed_state_v2", "authoritative_hand_boundary")])
def test_canonical_false_never_promoted(key, child):
    args = fixture()
    if child:
        args[5][0][key][child] = False
    else:
        args[5][0][key] = False
    assert evaluate_evidence(*args)["status"] == "BASE_VISUAL_PARTIAL"


def test_required_action_unknown_fails():
    args = fixture()
    args[5][0]["evaluation_fields"]["actions"] = {"status": "UNKNOWN", "value": None}
    assert "base_field_incomplete:actions" in evaluate_evidence(*args)["failures"]


def test_resume_requires_new_source_context():
    args = fixture()
    special_middle(args)
    args[5][2]["visual_scope"]["resume_evidence"]["source_frames"] = [1]
    assert "resume_without_fresh_context" in evaluate_evidence(*args)["failures"]


def test_receipt_cannot_auto_exempt():
    args = fixture()
    args[5][0]["visual_scope"]["automatically_exempt_from_acceptance"] = True
    assert evaluate_evidence(*args)["status"] == "BASE_VISUAL_PARTIAL"


def test_hardware_still_required():
    args = fixture()
    args[6]["freshness_checked"] = False
    assert "hardware:freshness_checked" in evaluate_evidence(*args)["failures"]


def test_cannot_replace_cash_policy_with_rake_guess():
    args = fixture()
    args[5][0]["visual_scope"]["unknown_cash_policy"] = "ASSUME_RAKE"
    assert "unallocated_cash_policy_lost" in evaluate_evidence(*args)["failures"]


def test_original_thirteen_field_gate_unchanged():
    from tools.aa8_acceptance import FIELDS
    assert len(FIELDS) == 13
    assert tuple(BASE_FIELDS) == FIELDS[:10]
    assert set(FIELDS) - set(BASE_FIELDS) == {"insurance", "mushroom", "bomb"}


def test_no_reading_predictions_to_fill_gold():
    args = fixture()
    args[4]["prediction_used_for_labels"] = True
    assert "gold_prediction_leakage" in evaluate_evidence(*args)["failures"]


def test_every_known_action_moment_remains_required():
    args = fixture()
    args[4]["labels"][0]["decision_opportunity"] = False
    args[2]["hands"][0]["action_opportunity_frames"] = [2, 3]
    assert "known_action_opportunity_excluded" in evaluate_evidence(*args)["failures"]
