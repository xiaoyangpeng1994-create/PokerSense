from copy import deepcopy
import hashlib
import json

import pytest

from poker_engine.strategy.opponent_dataset_v1 import (
    audit_opponent_dataset, calibrate_opponent_dataset,
)


def candidate_document():
    return {"schema_version": 1, "opponent_id": "example-opponent-not-real",
            "context": {"platform_id": "synthetic-example-platform",
                        "rule_fingerprint": "c" * 64},
            "candidate_scope": "public_legal_menu_and_price_only", "candidates": [
                {"candidate_id": name, "weights": {"fold": fold, "call": call},
                 "price_multipliers": []}
                for name, fold, call in (
                    ("folding", "9", "1"), ("calling", "1", "9"))]}


def documents():
    candidates = candidate_document()
    raw = json.dumps(candidates, indent=2).encode()
    candidate_hash = hashlib.sha256(raw).hexdigest()
    opportunities, decisions = [], []
    for session, seat, action in (("train-example", 1, "fold"),
                                  ("validation-example", 4, "call")):
        identity = {"opportunity_id": session + "-decision1", "session_id": session,
                    "hand_id": session + "-hand1",
                    "opponent_id": candidates["opponent_id"]}
        opportunities.append(identity)
        decisions.append({**identity, "source_kind": "synthetic",
                          "context": deepcopy(candidates["context"]),
                          "source_hash": hashlib.sha256(session.encode()).hexdigest(),
                          "seat_id": seat, "table_size": 6, "active_count": 3,
                          "position": "BTN", "street": "river", "pot": "60",
                          "to_call": "20", "legal_actions": [
                              {"kind": "fold", "target": "0"},
                              {"kind": "call", "target": "0"}],
                          "observed_action": {"kind": action, "target": "0"},
                          "audit_status": "reviewed", "unknown_reasons": []})
    data = {"schema_version": 1, "source_kind": "synthetic",
            "context": deepcopy(candidates["context"]),
            "opponent_id": candidates["opponent_id"], "coverage": {
                "status": "synthetic_complete", "reviewer": "synthetic-fixture",
                "evidence_hash": "a" * 64}, "protocol": {
                    "training_sessions": ["train-example"],
                    "validation_sessions": ["validation-example"],
                    "candidates_sha256": candidate_hash},
            "opportunities": opportunities, "decisions": decisions}
    return data, candidates, candidate_hash


def calibrate(data, candidates, sha):
    return calibrate_opponent_dataset(data, candidates, candidates_sha256=sha)


def test_stable_opponent_across_changed_session_seat_and_validation_not_selection():
    data, cs, sha = documents()
    report = calibrate(data, cs, sha)
    assert report["selection"].selected_id == "folding"
    assert report["calibration"].status == "NOT_REAL_CALIBRATION"
    assert report["calibration"].validation_score.mean_log_loss > 2
    assert report["audit"]["records"][1]["input"]["seat_id"] == 4
    assert report["selection"].training[0].actor_seat == 0
    assert report["selection"].training[0].own_category is None
    assert report["range_model"] is None and not report["strategy_eligible"]


@pytest.mark.parametrize("field,value", [
    ("pot", None), ("to_call", None), ("position", None),
    ("legal_actions", None), ("observed_action", None),
    ("source_hash", None), ("active_count", None), ("seat_id", None),
    ("audit_status", "unaudited"), ("unknown_reasons", ["animation_ambiguous"]),
])
def test_unknown_rows_retained_and_no_complete_case_calibration(field, value):
    data, cs, sha = documents()
    data["decisions"][1][field] = value
    report = calibrate(data, cs, sha)
    assert report["calibration"] is None and report["selection"] is None
    assert report["audit"]["opportunity_count"] == 2
    assert len(report["audit"]["records"]) == 2
    assert report["audit"]["records"][1]["input"][field] == value
    assert report["audit"]["records"][1]["reasons"]


def test_missing_row_kept_as_missing_opportunity_not_dropped():
    data, cs, sha = documents()
    data["decisions"].pop()
    report = calibrate(data, cs, sha)
    assert report["audit"]["missing_ids"] == ["validation-example-decision1"]
    assert len(report["audit"]["records"]) == 2 and report["calibration"] is None


def test_showdown_only_coverage_claim_not_accepted():
    data, cs, sha = documents()
    data["coverage"]["status"] = "showdown_only"
    assert calibrate(data, cs, sha)["calibration"] is None


def test_reviewed_source_needs_full_coverage_declaration_and_remains_unverified():
    data, cs, sha = documents()
    data["source_kind"] = "reviewed_public_decisions"
    for row in data["decisions"]:
        row["source_kind"] = "reviewed_public_decisions"
    data["coverage"]["status"] = "unreviewed"
    assert calibrate(data, cs, sha)["calibration"] is None
    data["coverage"]["status"] = "reviewed_complete"
    report = calibrate(data, cs, sha)
    assert report["calibration"].status == "REVIEW_REQUIRED_NOT_EMPIRICALLY_APPROVED"
    assert report["audit"]["provenance_unverified"]


@pytest.mark.parametrize("fault", [
    "different_opponent", "source_mixing", "session_overlap", "hand_overlap",
    "duplicate_opportunity", "unledgered_row", "changed_candidates",
])
def test_identity_provenance_and_protocol_conflation_rejected(fault):
    data, cs, sha = documents()
    if fault == "different_opponent":
        data["decisions"][1]["opponent_id"] = "other-person-same-seat"
    elif fault == "source_mixing":
        data["decisions"][1]["source_kind"] = "reviewed_public_decisions"
    elif fault == "session_overlap":
        data["protocol"]["validation_sessions"] = ["train-example"]
    elif fault == "hand_overlap":
        for collection in ("decisions", "opportunities"):
            data[collection][1]["hand_id"] = data[collection][0]["hand_id"]
    elif fault == "duplicate_opportunity":
        data["opportunities"].append(deepcopy(data["opportunities"][0]))
    elif fault == "unledgered_row":
        data["decisions"][1]["opportunity_id"] = "not-in-ledger"
    else:
        data["protocol"]["candidates_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        calibrate(data, cs, sha)


def test_candidate_hidden_category_or_callback_fields_rejected():
    data, cs, sha = documents()
    cs["candidates"][0]["category_weights"] = {"0": {"fold": "1"}}
    with pytest.raises(ValueError, match="exact_candidate"):
        calibrate(data, cs, sha)


def test_incomplete_legal_menu_is_retained_as_blocked_row():
    data, cs, sha = documents()
    data["decisions"][0]["legal_actions"] = [{"kind": "fold", "target": "0"}]
    report = calibrate(data, cs, sha)
    assert report["calibration"] is None
    assert "incomplete_or_inconsistent_declared_legal_menu" in (
        report["audit"]["records"][0]["reasons"])


def test_input_audit_does_not_mutate_or_remove_raw_records():
    data, cs, sha = documents()
    before = deepcopy(data)
    audit_opponent_dataset(data, cs, candidates_sha256=sha)
    assert data == before


@pytest.mark.parametrize("size", [7, 8])
def test_legitimate_lj_position_supported(size):
    data, cs, sha = documents()
    data["decisions"][0].update(table_size=size, position="LJ")
    assert calibrate(data, cs, sha)["calibration"] is not None


@pytest.mark.parametrize("field,value", [
    ("platform_id", "different-platform"), ("rule_fingerprint", "b" * 64),
])
def test_per_row_population_mismatch_is_retained_and_blocks_all_fit(field, value):
    data, cs, sha = documents()
    data["decisions"][1]["context"][field] = value
    report = calibrate(data, cs, sha)
    assert report["calibration"] is None
    assert "row_platform_or_rule_context_missing_or_mismatch" in (
        report["audit"]["records"][1]["reasons"])


def test_candidate_context_mismatch_rejected():
    data, cs, sha = documents()
    cs["context"]["platform_id"] = "other-platform"
    with pytest.raises(ValueError, match="population_context"):
        calibrate(data, cs, sha)


def test_price_ratio_uses_exact_fraction_sum_beyond_decimal_precision():
    from fractions import Fraction
    data, cs, sha = documents()
    pot = "100000000000000000000000000000000000000000000000001"
    data["decisions"][0].update(pot=pot, to_call="2")
    _, _, rows = audit_opponent_dataset(data, cs, candidates_sha256=sha)
    assert rows[0].price_ratio == Fraction(2, int(pot) + 2)
