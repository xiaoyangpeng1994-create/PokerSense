import hashlib
import json

import pytest

from tools.build_aa8_decision_readiness import CONSTRAINTS, build


def write(path, value, *, lines=False):
    payload = ("\n".join(json.dumps(row) for row in value) + "\n"
               if lines else json.dumps(value))
    path.write_text(payload, encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    paths = {name: tmp_path / ("observations.jsonl" if name == "observations"
                               else name + ".json") for name in (
        "protocol", "registry", "review", "review_result", "report",
        "observations")}
    labels = [{
        "review_id": "A01", "hand_id": "hand1", "action_frame": 10,
        "actor_slot": 1, "candidate_glyph": "check",
        "review_status": "MATCH_VISIBLE_COMPLETED_ACTION",
        "actual_action": "check", "amount": "0", "amount_status": "ZERO",
        "duplicate_of": None, "street": "flop", "legal_actions": None,
        "evidence_frames": [10]}, {
        "review_id": "A02", "hand_id": "hand1", "action_frame": 11,
        "actor_slot": 4, "candidate_glyph": "fold",
        "review_status": "MATCH_VISIBLE_COMPLETED_ACTION",
        "actual_action": "fold", "amount": "0", "amount_status": "ZERO",
        "duplicate_of": None, "street": "flop", "legal_actions": None,
        "evidence_frames": [11]}]
    sample_hash = "a" * 64
    write(paths["review"], {"labels": labels})
    write(paths["review_result"], {
        "labels": labels, "source_samples_sha256": sample_hash})
    registry = {
        "source_session_id": "session1", "source_audit_sha256": "b" * 64,
        "samples_sha256": sample_hash,
        "window": {"first_frame": 10, "last_frame": 12,
                   "start_pts": "10", "end_pts_exclusive": "13"},
        "hands": [{
            "hand_id": "hand1", "start_frame": 10, "last_gameplay_frame": 11,
            "end_frame": 11, "next_start_frame": 12, "temporal_complete": True,
            "opening_state_status": "ORDINARY_UNKNOWN"}, {
            "hand_id": "tail", "start_frame": 12, "last_gameplay_frame": 12,
            "end_frame": 12, "next_start_frame": None,
            "temporal_complete": False,
            "opening_state_status": "INCOMPLETE_RECORDING_TAIL"}]}
    write(paths["registry"], registry)
    observations = [{
        "frame": frame, "pts_seconds": str(frame),
        "source_sha256": str(frame % 10) * 64, "strategy_eligible": False}
        for frame in range(10, 13)]
    write(paths["observations"], observations, lines=True)
    write(paths["report"], {
        "frames": 3, "observations_sha256": sha(paths["observations"]),
        "events": [{"frame": 10, "slot": 1, "glyph": "check"},
                   {"frame": 11, "slot": 4, "glyph": "fold"}]})
    protocol = {
        "schema_version": 1,
        "status": "CURRENT_ACTION_CANDIDATES_NOT_ALL_OPPORTUNITIES_OR_DATA",
        "registry_sha256": sha(paths["registry"]),
        "review_config_sha256": sha(paths["review"]),
        "review_result_sha256": sha(paths["review_result"]),
        "v2_report_sha256": sha(paths["report"]),
        "observations_sha256": sha(paths["observations"]),
        "expected_reviewed_candidates": 2, "expected_matched_actions": 2,
        "expected_false_candidates": 0, "expected_opponent_matches": 1,
        "expected_hero_matches": 1, "constraints": CONSTRAINTS}
    write(paths["protocol"], protocol)
    return paths, protocol


def run(paths):
    return build(paths["protocol"], paths["registry"], paths["review"],
                 paths["review_result"], paths["report"], paths["observations"])


def refresh(paths, protocol, name):
    key = {"registry": "registry_sha256", "review": "review_config_sha256",
           "review_result": "review_result_sha256", "report": "v2_report_sha256",
           "observations": "observations_sha256"}[name]
    protocol[key] = sha(paths[name])
    write(paths["protocol"], protocol)


def test_current_action_candidates_become_blocked_review_queue(tmp_path):
    paths, _ = fixture(tmp_path)
    report = run(paths)
    assert report["candidate_inventory"] == {
        "reviewed": 2, "matched": 2, "false": 0,
        "opponent_matched": 1, "hero_matched": 1,
        "all_actual_opportunities_reviewed": False}
    assert report["audit"]["data_readiness"] == "BLOCKED"
    assert report["audit"]["eligible_target_count"] == 0
    assert len(report["dataset"]["opportunity_ledger"]) == 2
    assert report["dataset"]["opportunities"][0]["evidence"][
        "action_confirmation_sha256"] == "0" * 64
    assert not report["model_fit_executed"] and not report["advice_emitted"]


def test_input_hash_change_is_rejected(tmp_path):
    paths, _ = fixture(tmp_path)
    paths["observations"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_hash"):
        run(paths)


def test_v2_events_must_equal_every_reviewed_match(tmp_path):
    paths, protocol = fixture(tmp_path)
    report = json.loads(paths["report"].read_text())
    report["events"].pop()
    write(paths["report"], report)
    refresh(paths, protocol, "report")
    with pytest.raises(ValueError, match="v2_events"):
        run(paths)


def test_review_result_must_equal_frozen_review_config(tmp_path):
    paths, protocol = fixture(tmp_path)
    result = json.loads(paths["review_result"].read_text())
    result["labels"][0]["actual_action"] = "fold"
    write(paths["review_result"], result)
    refresh(paths, protocol, "review_result")
    with pytest.raises(ValueError, match="identity_mismatch"):
        run(paths)
