import hashlib
import json

import pytest

from tools.aa8_glyph_dedupe_regression import analyze


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    paths = {name: tmp_path / f"{name}.json"
             for name in ("protocol", "old", "new", "review", "registry")}
    old_events = [
        {"frame": 1, "slot": 7, "glyph": "fold"},
        {"frame": 11, "slot": 0, "glyph": "check"},
        {"frame": 12, "slot": 4, "glyph": "call"},
        {"frame": 13, "slot": 4, "glyph": "fold"},
    ]
    new_events = old_events[:3]
    policy = {"stable_frames": 2, "clear_frames": 5,
              "rapid_change_guard_frames": 5,
              "unsupported_scene_frames_suspended": True,
              "unconfirmed_streaks_cross_suspension": False,
              "stable_clear_required_after_suppression": True,
              "hand_epoch_reset": "not_applied_without_authoritative_boundary"}
    old_inputs = {
        r"G:\old-worktree\tools\aa8_visual_pipeline.py": "a" * 64,
        r"G:\private\samples.json": "c" * 64,
    }
    new_inputs = {
        r"G:\new-worktree\tools\aa8_visual_pipeline_v2.py": "b" * 64,
        r"G:\new-worktree\tools\aa8_glyph_transitions_v2.py": "d" * 64,
        r"G:\private\samples.json": "c" * 64,
    }
    write(paths["old"], {"events": old_events, "frames": 4,
                         "input_hashes": old_inputs})
    write(paths["new"], {"events": new_events,
                         "frames": 4, "input_hashes": new_inputs,
                         "glyph_transition_policy": policy})
    write(paths["review"], {"labels": [
        {"action_frame": 11, "actor_slot": 0, "candidate_glyph": "check",
         "review_status": "MATCH_VISIBLE_COMPLETED_ACTION"},
        {"action_frame": 12, "actor_slot": 4, "candidate_glyph": "call",
         "review_status": "MATCH_VISIBLE_COMPLETED_ACTION"},
        {"action_frame": 13, "actor_slot": 4, "candidate_glyph": "fold",
         "review_status": "FALSE_POSITIVE_POST_ACTION_DISPLAY"},
    ]})
    write(paths["registry"], {"hands": [
        {"start_frame": 10, "end_frame": 19, "temporal_complete": True}]})
    protocol = {
        "schema_version": 1,
        "status": "DEVELOPMENT_REGRESSION_NOT_ACCURACY_OR_ACCEPTANCE",
        "old_report_sha256": sha(paths["old"]),
        "new_report_sha256": sha(paths["new"]),
        "review_sha256": sha(paths["review"]),
        "registry_sha256": sha(paths["registry"]),
        "expected_old_all_events": 4, "expected_new_all_events": 3,
        "expected_complete_hand_candidates": 3,
        "expected_retained_matches": 2,
        "expected_removed_false_candidates": 1,
        "expected_unchanged_outside_complete_hands": 1,
        "expected_frames": 4,
        "expected_changed_input_hashes": {},
        "expected_removed_input_hashes": {
            "tools/aa8_visual_pipeline.py": "a" * 64},
        "expected_added_input_hashes": {
            "tools/aa8_visual_pipeline_v2.py": "b" * 64,
            "tools/aa8_glyph_transitions_v2.py": "d" * 64},
        "expected_unchanged_input_bindings": 1,
        "constraints": [
            "same_preregistered_development_frames_and_model_inputs",
            "remove_exactly_reviewed_false_candidates",
            "retain_every_reviewed_match",
            "no_new_events",
            "outside_complete_hands_unchanged",
            "not_independent_accuracy_or_strategy_evidence",
        ],
    }
    write(paths["protocol"], protocol)
    return paths, protocol


def refresh(paths, protocol):
    hash_keys = {
        "old": "old_report_sha256",
        "new": "new_report_sha256",
        "review": "review_sha256",
        "registry": "registry_sha256",
    }
    for name, key in hash_keys.items():
        protocol[key] = sha(paths[name])
    write(paths["protocol"], protocol)


def run(paths):
    return analyze(paths["protocol"], paths["old"], paths["new"],
                   paths["review"], paths["registry"])


def test_exact_false_candidate_removed_and_every_match_retained(tmp_path):
    paths, _ = fixture(tmp_path)
    report = run(paths)
    assert report["old_all_events"] == 4 and report["new_all_events"] == 3
    assert report["retained_reviewed_matches"] == 2
    assert report["removed_reviewed_false_candidates"] == 1
    assert report["unchanged_outside_complete_hands"] == 1
    assert report["removed_events"] == [[13, 4, "fold"]]
    assert not report["strategy_eligible"] and not report["independent_accuracy"]


def test_retained_false_candidate_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["events"].append({"frame": 13, "slot": 4, "glyph": "fold"})
    write(paths["new"], new)
    protocol["expected_new_all_events"] = 4
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="retains_false"):
        run(paths)


def test_missing_reviewed_match_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["events"].pop()
    write(paths["new"], new)
    protocol["expected_new_all_events"] = 2
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="misses_match"):
        run(paths)


def test_new_event_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["events"].append({"frame": 14, "slot": 3, "glyph": "bet"})
    write(paths["new"], new)
    protocol["expected_new_all_events"] = 4
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="misses_match|added_new"):
        run(paths)


def test_event_outside_complete_hands_must_not_change(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["events"][0]["slot"] = 6
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="outside_complete_hands"):
        run(paths)


def test_transition_policy_is_bound(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["glyph_transition_policy"]["rapid_change_guard_frames"] = 6
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="transition_policy"):
        run(paths)


def test_frame_count_drift_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["frames"] = 5
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="frame_count"):
        run(paths)


def test_model_or_input_hash_drift_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["input_hashes"][r"G:\private\samples.json"] = "d" * 64
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="changed_input_hashes"):
        run(paths)


def test_extra_or_missing_input_binding_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    del new["input_hashes"][r"G:\private\samples.json"]
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="removed_input_hashes"):
        run(paths)


def test_extra_input_binding_is_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["input_hashes"][r"G:\private\other-model.npz"] = "e" * 64
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="added_input_hashes"):
        run(paths)


def test_constraints_and_count_types_are_exact(tmp_path):
    paths, protocol = fixture(tmp_path)
    protocol["constraints"].pop()
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="exact_development"):
        run(paths)
    paths, protocol = fixture(tmp_path)
    protocol["expected_retained_matches"] = True
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="exact_development"):
        run(paths)


def test_reordered_retained_events_are_rejected(tmp_path):
    paths, protocol = fixture(tmp_path)
    new = json.loads(paths["new"].read_text())
    new["events"][1], new["events"][2] = new["events"][2], new["events"][1]
    write(paths["new"], new)
    refresh(paths, protocol)
    with pytest.raises(ValueError, match="chronology|order"):
        run(paths)


def test_input_hash_change_is_rejected(tmp_path):
    paths, _ = fixture(tmp_path)
    paths["old"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="input_hash"):
        run(paths)
