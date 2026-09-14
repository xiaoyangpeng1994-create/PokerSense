import hashlib
import json
import subprocess
import sys

import pytest

from tools.audit_aa8_actor_episodes import CONSTRAINTS, build


def write(path, value, *, lines=False):
    payload = ("\n".join(json.dumps(row) for row in value) + "\n"
               if lines else json.dumps(value))
    path.write_bytes(payload.encode())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {name: tmp_path / ("observations.jsonl" if name == "observations"
                               else name + ".json") for name in (
        "protocol", "registry", "review", "result", "report", "observations")}
    labels = [{
        "review_id": "A1", "hand_id": "hand1", "action_frame": 3,
        "actor_slot": 1, "candidate_glyph": "check",
        "review_status": "MATCH_VISIBLE_COMPLETED_ACTION",
        "actual_action": "check", "amount": "0", "amount_status": "ZERO",
        "street": "flop"}, {
        "review_id": "A2", "hand_id": "hand1", "action_frame": 7,
        "actor_slot": 3, "candidate_glyph": "fold",
        "review_status": "MATCH_VISIBLE_COMPLETED_ACTION",
        "actual_action": "fold", "amount": "0", "amount_status": "ZERO",
        "street": "flop"}]
    write(paths["review"], {"labels": labels})
    write(paths["result"], {"labels": labels, "source_samples_sha256": "a" * 64})
    write(paths["registry"], {
        "source_session_id": "session1", "samples_sha256": "a" * 64,
        "window": {"first_frame": 1, "last_frame": 8},
        "hands": [{"hand_id": "hand1", "start_frame": 1,
                   "last_gameplay_frame": 7, "end_frame": 7,
                   "temporal_complete": True},
                  {"hand_id": "tail", "start_frame": 8, "end_frame": 8,
                   "last_gameplay_frame": 8,
                   "temporal_complete": False}]})
    actor = {1: 1, 2: 1, 3: None, 4: 2, 5: None, 6: 3, 7: 3, 8: None}
    observations = [{
        "frame": frame, "pts_seconds": str(frame),
        "source_sha256": format(frame, "x") * 64,
        "strategy_eligible": False, "scene_supported": True,
        "current_actor": actor[frame], "actor_evidence": {"actor": actor[frame]},
        "board_count": 3}
        for frame in range(1, 9)]
    write(paths["observations"], observations, lines=True)
    write(paths["report"], {
        "frames": 8, "observations_sha256": sha(paths["observations"]),
        "events": [{"frame": 3, "slot": 1, "glyph": "check"},
                   {"frame": 7, "slot": 3, "glyph": "fold"}]})
    protocol = {
        "schema_version": 1,
        "status": "ACTOR_EPISODES_ARE_CANDIDATES_NOT_ALL_OPPORTUNITIES",
        "registry_sha256": sha(paths["registry"]),
        "review_config_sha256": sha(paths["review"]),
        "review_result_sha256": sha(paths["result"]),
        "v2_report_sha256": sha(paths["report"]),
        "observations_sha256": sha(paths["observations"]),
        "expected_frames": 8, "expected_complete_hands": 1,
        "expected_reviewed_matches": 2,
        "standard_confirmation_lag_frames": 2,
        "maximum_candidate_confirmation_lag_frames": 4,
        "constraints": CONSTRAINTS}
    write(paths["protocol"], protocol)
    return paths, protocol


def run(paths):
    return build(paths["protocol"], paths["registry"], paths["review"],
                 paths["result"], paths["report"], paths["observations"])


def refresh(paths, protocol, *names):
    mapping = {"registry": "registry_sha256", "review": "review_config_sha256",
               "result": "review_result_sha256", "report": "v2_report_sha256",
               "observations": "observations_sha256"}
    for name in names:
        protocol[mapping[name]] = sha(paths[name])
    write(paths["protocol"], protocol)


def test_contiguous_actor_runs_and_unknown_gaps_are_all_retained(tmp_path):
    paths, _ = fixture(tmp_path)
    report = run(paths)
    assert report["actor_episode_count"] == 3
    assert report["reviewed_action_candidate_count"] == 2
    assert report["bound_reviewed_action_count"] == 2
    assert report["standard_lag_match_count"] == 2
    assert report["unmatched_actor_episode_count"] == 1
    assert report["unbound_reviewed_action_count"] == 0
    assert report["unknown_actor_frame_count"] == 2
    assert report["unknown_span_count"] == 2
    assert report["supported_actor_unknown_frame_count"] == 2
    assert report["complete_legal_menus"] == 0
    assert not report["all_actual_decision_opportunities_reviewed"]


def test_delayed_confirmation_is_disclosed_not_predecision_truth(tmp_path):
    paths, protocol = fixture(tmp_path)
    review = json.loads(paths["review"].read_text())
    result = json.loads(paths["result"].read_text())
    review["labels"][0]["action_frame"] = 5
    result["labels"][0]["action_frame"] = 5
    pipeline = json.loads(paths["report"].read_text())
    pipeline["events"][0]["frame"] = 5
    write(paths["review"], review)
    write(paths["result"], result)
    write(paths["report"], pipeline)
    refresh(paths, protocol, "review", "result", "report")
    report = run(paths)
    assert report["delayed_lag_match_count"] == 1
    episode = next(row for row in report["episodes"]
                   if row["reviewed_action"]["review_id"] == "A1")
    assert episode["confirmation_lag_frames"] == 3
    assert episode["predecision_truth"] is None


def test_action_outside_maximum_lag_is_retained_unbound(tmp_path):
    paths, protocol = fixture(tmp_path)
    review = json.loads(paths["review"].read_text())
    result = json.loads(paths["result"].read_text())
    review["labels"][0]["action_frame"] = 7
    result["labels"][0]["action_frame"] = 7
    pipeline = json.loads(paths["report"].read_text())
    pipeline["events"][0]["frame"] = 7
    write(paths["review"], review)
    write(paths["result"], result)
    write(paths["report"], pipeline)
    refresh(paths, protocol, "review", "result", "report")
    report = run(paths)
    assert report["unbound_reviewed_action_count"] == 1
    assert report["unmatched_actor_episode_count"] == 2


def test_unknown_frame_splits_same_actor_into_two_proposals(tmp_path):
    paths, protocol = fixture(tmp_path)
    rows = [json.loads(line) for line in paths["observations"].read_text().splitlines()]
    rows[3]["current_actor"] = 1
    rows[3]["actor_evidence"] = {"actor": 1}
    write(paths["observations"], rows, lines=True)
    pipeline = json.loads(paths["report"].read_text())
    pipeline["observations_sha256"] = sha(paths["observations"])
    write(paths["report"], pipeline)
    refresh(paths, protocol, "observations", "report")
    report = run(paths)
    assert sum(row["actor_slot"] == 1 for row in report["episodes"]) == 2


def test_hash_or_observation_contract_drift_is_rejected(tmp_path):
    paths, _ = fixture(tmp_path)
    paths["observations"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_hash"):
        run(paths)


def test_actor_in_unsupported_scene_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    rows = [json.loads(line) for line in paths["observations"].read_text().splitlines()]
    rows[0]["scene_supported"] = False
    write(paths["observations"], rows, lines=True)
    pipeline = json.loads(paths["report"].read_text())
    pipeline["observations_sha256"] = sha(paths["observations"])
    write(paths["report"], pipeline)
    refresh(paths, protocol, "observations", "report")
    with pytest.raises(ValueError, match="unsupported_scene"):
        run(paths)


def test_pts_must_increase_and_actor_type_cannot_silently_become_unknown(tmp_path):
    paths, protocol = fixture(tmp_path)
    rows = [json.loads(line) for line in paths["observations"].read_text().splitlines()]
    rows[1]["pts_seconds"] = "0"
    write(paths["observations"], rows, lines=True)
    pipeline = json.loads(paths["report"].read_text())
    pipeline["observations_sha256"] = sha(paths["observations"])
    write(paths["report"], pipeline)
    refresh(paths, protocol, "observations", "report")
    with pytest.raises(ValueError, match="strictly_increasing"):
        run(paths)
    paths, protocol = fixture(tmp_path / "actor")
    rows = [json.loads(line) for line in paths["observations"].read_text().splitlines()]
    rows[0]["current_actor"] = "1"
    rows[0]["actor_evidence"] = {"actor": "1"}
    write(paths["observations"], rows, lines=True)
    pipeline = json.loads(paths["report"].read_text())
    pipeline["observations_sha256"] = sha(paths["observations"])
    write(paths["report"], pipeline)
    refresh(paths, protocol, "observations", "report")
    with pytest.raises(ValueError, match="exact_slot"):
        run(paths)


def test_cli_creates_once_without_promoting_episode_candidates(tmp_path):
    paths, _ = fixture(tmp_path)
    output = tmp_path / "episodes.json"
    command = [sys.executable, "-m", "tools.audit_aa8_actor_episodes"]
    for name in ("protocol", "registry", "review", "result", "report",
                 "observations"):
        option = {"result": "review-result", "report": "v2-report"}.get(
            name, name)
        command.extend(["--" + option, str(paths[name])])
    command.extend(["--output", str(output)])
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    assert json.loads(raw)["ready_for_offline_calibration"] is False
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert output.read_bytes() == raw
