import hashlib
import json
from pathlib import Path

import pytest

from tools import aa8_offline_session_evidence as module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root):
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixture(tmp_path, monkeypatch):
    audit, samples, pipeline = (tmp_path / name for name in (
        "audit", "samples", "pipeline"))
    for path in (audit, samples, pipeline):
        path.mkdir()
    split_path, registry_path = tmp_path / "split.json", tmp_path / "registry.json"
    audit_path, samples_path = audit / "report.json", samples / "samples.json"
    report_path, observations_path = pipeline / "report.json", pipeline / (
        "observations.jsonl")
    source = {
        "state": "verified",
        "segments": [
            {"file": "cross.mkv", "sha256": "a" * 64, "decoded_frames": 4,
             "first_pts": "8", "last_pts": "11"},
            {"file": "dev.mkv", "sha256": "b" * 64, "decoded_frames": 6,
             "first_pts": "10", "last_pts": "19.9"},
        ],
    }
    write_json(audit_path, source)
    write_manifest(audit)
    audit_hash = digest(audit_path)
    plan = {"source_audit_sha256": audit_hash, "ranges": [
        {"start_inclusive": "0", "end_exclusive": "10",
         "role": "holdout_candidate"},
        {"start_inclusive": "10", "end_exclusive": "20",
         "role": "development"},
    ]}
    write_json(split_path, plan)
    plan_hash = digest(split_path)
    (samples / "frames").mkdir()
    sample_rows = []
    for index in range(6):
        path = samples / "frames" / f"frame_{100 + index:06d}.png"
        path.write_bytes(f"frame-{index}".encode())
        sample_rows.append({
            "segment": "dev.mkv", "local_frame": index,
            "global_frame": 100 + index, "pts_seconds": str(10 + index),
            "file": path.relative_to(samples).as_posix(),
            "sha256": digest(path), "role": "development"})
    sample_doc = {"audit_sha256": audit_hash, "plan_sha256": plan_hash,
                  "range_seconds": ["10", "20"], "samples": sample_rows}
    write_json(samples_path, sample_doc)
    write_manifest(samples)
    observations = []
    events = []
    for index, sample in enumerate(sample_rows):
        transitions = ([{"frame": 101, "slot": 2, "glyph": "call"}]
                       if index == 1 else ([{
                           "frame": 104, "slot": 3, "glyph": "fold"}]
                           if index == 4 else []))
        events.extend(transitions)
        observations.append({
            "frame": sample["global_frame"], "source_sha256": sample["sha256"],
            "strategy_eligible": False, "glyph_transitions": transitions,
            "current_actor": 2 if index == 0 else None,
            "pot": {"value": "20"}, "board_count": 5,
        })
    observations_path.write_text("".join(
        json.dumps(row) + "\n" for row in observations), encoding="utf-8")
    observations_hash = digest(observations_path)
    pipeline_doc = {"observations_sha256": observations_hash, "frames": 6,
                    "independent_holdout": False, "strategy_eligible": False,
                    "events": events}
    write_json(report_path, pipeline_doc)
    registry = {
        "schema_version": 1,
        "status": "DEVELOPMENT_REVIEWED_BOUNDARIES_NOT_CALIBRATION",
        "source_session_id": "session-a", "capture_path": (
            "physical_phone_capture_card"), "emulator_used": False,
        "source_audit_sha256": audit_hash, "split_plan_sha256": plan_hash,
        "samples_sha256": digest(samples_path),
        "pipeline_report_sha256": digest(report_path),
        "observations_sha256": observations_hash,
        "window": {
            "start_pts": "10", "end_pts_exclusive": "20",
            "first_frame": 100, "last_frame": 105,
            "segments": [{"file": "dev.mkv", "sha256": "b" * 64,
                          "decoded_frames": 6, "first_global_frame": 100,
                          "last_global_frame": 105}],
            "excluded_cross_boundary_segment": {
                "file": "cross.mkv",
                "reason": "crosses_protected_to_development_boundary"}},
        "hands": [
            {"hand_id": "complete", "start_frame": 100,
             "last_gameplay_frame": 101, "end_frame": 102,
             "next_start_frame": 103, "temporal_complete": True,
             "boundary_status": "DEVELOPMENT_MANUAL_REVIEW",
             "opening_state_status": "PARTIAL",
             "evidence": [{"frame": 100, "sha256": sample_rows[0]["sha256"],
                           "meaning": "deal"},
                          {"frame": 101, "sha256": sample_rows[1]["sha256"],
                           "meaning": "gameplay"},
                          {"frame": 103, "sha256": sample_rows[3]["sha256"],
                           "meaning": "next"}]},
            {"hand_id": "tail", "start_frame": 103,
             "last_gameplay_frame": 104, "end_frame": 105,
             "next_start_frame": None, "temporal_complete": False,
             "boundary_status": "INCOMPLETE_RECORDING_TAIL",
             "opening_state_status": "PARTIAL",
             "evidence": [{"frame": 103, "sha256": sample_rows[3]["sha256"],
                           "meaning": "start"},
                          {"frame": 104, "sha256": sample_rows[4]["sha256"],
                           "meaning": "gameplay"},
                          {"frame": 105, "sha256": sample_rows[5]["sha256"],
                           "meaning": "end"}]},
        ],
        "review_constraints": ["development_only"],
    }
    write_json(registry_path, registry)
    return {"registry": registry, "registry_path": registry_path, "audit": audit,
            "split": split_path, "samples": samples, "pipeline": pipeline,
            "samples_path": samples_path, "report_path": report_path,
            "observations_path": observations_path}


def run(item):
    return module.analyze(item["registry_path"], item["audit"], item["split"],
                          item["samples"], item["pipeline"])


def save_registry(item):
    write_json(item["registry_path"], item["registry"])


def refresh_samples(item):
    write_manifest(item["samples"])
    item["registry"]["samples_sha256"] = digest(item["samples_path"])
    save_registry(item)


def test_complete_hand_candidates_are_retained_without_false_calibration(
        tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    report = run(item)
    assert report["frame_count"] == 6
    assert report["temporally_complete_hand_candidates"] == 1
    assert report["incomplete_tail_hands"] == 1
    assert report["all_pipeline_glyph_candidates"] == 2
    assert report["complete_hand_action_candidates"] == 1
    assert report["opponent_action_candidates"] == 1
    assert report["actor_match_within_previous_12_frames"] == 1
    assert report["opportunities"][0]["candidate_predecision_pot"] == "20"
    assert report["complete_legal_action_menus"] == 0
    assert not report["ready_for_opponent_calibration"]
    assert not report["ldplayer_or_emulator_supported"]


def test_missing_actor_is_retained_as_a_candidate_gap(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in item["observations_path"].read_text(
        encoding="utf-8").splitlines()]
    rows[0]["current_actor"] = None
    item["observations_path"].write_text("".join(
        json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    observation_hash = digest(item["observations_path"])
    pipeline = json.loads(item["report_path"].read_text())
    pipeline["observations_sha256"] = observation_hash
    write_json(item["report_path"], pipeline)
    item["registry"]["observations_sha256"] = observation_hash
    item["registry"]["pipeline_report_sha256"] = digest(item["report_path"])
    save_registry(item)
    report = run(item)
    assert report["complete_hand_action_candidates"] == 1
    assert report["actor_match_within_previous_12_frames"] == 0
    assert report["opportunities"][0]["actor_match_frame"] is None


def test_emulator_source_is_rejected(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["emulator_used"] = True
    save_registry(item)
    with pytest.raises(ValueError, match="physical_offline"):
        run(item)


def test_protected_window_overlap_is_rejected(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["window"]["start_pts"] = "9"
    save_registry(item)
    with pytest.raises(ValueError, match="not_wholly"):
        run(item)


def test_registry_hash_change_is_rejected(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["samples_sha256"] = "0" * 64
    save_registry(item)
    with pytest.raises(ValueError, match="input_hash"):
        run(item)


def test_noncontiguous_sample_inventory_is_rejected(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    samples = json.loads(item["samples_path"].read_text())
    del samples["samples"][2]
    write_json(item["samples_path"], samples)
    refresh_samples(item)
    with pytest.raises(ValueError, match="contiguous"):
        run(item)


def test_observation_source_mismatch_is_rejected(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    rows = item["observations_path"].read_text().splitlines()
    first = json.loads(rows[0])
    first["source_sha256"] = "f" * 64
    rows[0] = json.dumps(first)
    item["observations_path"].write_text("\n".join(rows) + "\n", encoding="utf-8")
    observation_hash = digest(item["observations_path"])
    pipeline = json.loads(item["report_path"].read_text())
    pipeline["observations_sha256"] = observation_hash
    write_json(item["report_path"], pipeline)
    item["registry"]["observations_sha256"] = observation_hash
    item["registry"]["pipeline_report_sha256"] = digest(item["report_path"])
    save_registry(item)
    with pytest.raises(ValueError, match="source_or_order"):
        run(item)


def test_pipeline_cannot_drop_a_failed_event(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    pipeline = json.loads(item["report_path"].read_text())
    pipeline["events"] = pipeline["events"][:-1]
    write_json(item["report_path"], pipeline)
    item["registry"]["pipeline_report_sha256"] = digest(item["report_path"])
    save_registry(item)
    with pytest.raises(ValueError, match="event_inventory"):
        run(item)


def test_complete_hand_requires_immediate_next_boundary(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["hands"][0]["next_start_frame"] = 104
    save_registry(item)
    with pytest.raises(ValueError, match="next_boundary"):
        run(item)


def test_next_boundary_must_be_the_next_registered_hand(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["hands"][1]["start_frame"] = 104
    save_registry(item)
    with pytest.raises(ValueError, match="next_boundary"):
        run(item)


def test_complete_hand_requires_start_gameplay_and_next_evidence(
        tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    evidence = item["registry"]["hands"][0]["evidence"]
    evidence[1] = dict(evidence[0])
    save_registry(item)
    with pytest.raises(ValueError, match="boundary_evidence_missing"):
        run(item)


def test_only_final_entry_can_be_incomplete_tail(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["hands"] = list(reversed(item["registry"]["hands"]))
    save_registry(item)
    with pytest.raises(ValueError, match="incomplete_tail|ordered_nonoverlapping"):
        run(item)


def test_duplicate_evidence_hash_cannot_replace_required_frame(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    item["registry"]["hands"][0]["evidence"][0]["sha256"] = (
        item["registry"]["hands"][0]["evidence"][1]["sha256"])
    save_registry(item)
    with pytest.raises(ValueError, match="evidence_not_bound"):
        run(item)


def test_shipped_registry_discloses_no_emulator_or_calibration_eligibility():
    registry = json.loads(Path(
        "configs/reproduction/aa8_late_development_hands_v1.json").read_text())
    assert registry["capture_path"] == "physical_phone_capture_card"
    assert registry["emulator_used"] is False
    assert len([hand for hand in registry["hands"]
                if hand["temporal_complete"]]) == 2
    assert registry["hands"][-1]["boundary_status"] == (
        "INCOMPLETE_RECORDING_TAIL")
    assert "complete_legal_action_menus_are_not_available" in (
        registry["review_constraints"])


def test_protected_pts_cannot_be_relabelled_development(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    samples = json.loads(item["samples_path"].read_text())
    samples["samples"][0]["pts_seconds"] = "9"
    write_json(item["samples_path"], samples)
    refresh_samples(item)
    original, reads = module.sha256, []

    def guarded(path):
        reads.append(path.resolve())
        return original(path)

    monkeypatch.setattr(module, "sha256", guarded)
    with pytest.raises(ValueError, match="sample_pts_path"):
        run(item)
    assert not [path for path in reads if path.suffix == ".png"]


def test_duplicate_local_frame_cannot_relabel_a_whole_segment(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    samples = json.loads(item["samples_path"].read_text())
    samples["samples"][2]["local_frame"] = 1
    write_json(item["samples_path"], samples)
    refresh_samples(item)
    with pytest.raises(ValueError, match="safe_development_source"):
        run(item)


def test_manifest_traversal_rejected_before_outside_file_read(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    manifest = item["samples"] / "SHA256SUMS"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(f"{digest(outside)}  ../outside.txt\n")
    original, reads = module.sha256, []

    def guarded(path):
        reads.append(path.resolve())
        return original(path)

    monkeypatch.setattr(module, "sha256", guarded)
    with pytest.raises(ValueError, match="unsafe_or_malformed"):
        run(item)
    assert outside.resolve() not in reads


def test_undeclared_frame_is_rejected_without_reading_it(tmp_path, monkeypatch):
    item = fixture(tmp_path, monkeypatch)
    extra = item["samples"] / "frames" / "frame_000099.png"
    extra.write_text("not-declared", encoding="utf-8")
    write_manifest(item["samples"])
    original, reads = module.sha256, []

    def guarded(path):
        reads.append(path.resolve())
        return original(path)

    monkeypatch.setattr(module, "sha256", guarded)
    with pytest.raises(ValueError, match="undeclared_sample_frames"):
        run(item)
    assert extra.resolve() not in reads
