import hashlib
import json
import os
import subprocess
import sys

import pytest

from poker_engine.strategy.decision_opportunities_v1 import (
    audit_decision_opportunities,
)
from tools.audit_aa8_actor_episodes import (
    CONSTRAINTS as EPISODE_CONSTRAINTS,
    build_documents as build_episode_documents,
)
from tools.build_aa8_artifact_bundle import build_bundle
from tools.build_aa8_decision_readiness import (
    CONSTRAINTS as READINESS_CONSTRAINTS,
    build_documents as build_readiness_documents,
)
from tools.verify_decision_opportunity_artifacts import (
    LIMITATIONS, ROLE_SPECS, verify_bundle,
)


def write(path, value, *, lines=False):
    payload = ("\n".join(json.dumps(row) for row in value) + "\n"
               if lines else json.dumps(value) + "\n")
    path.write_bytes(payload.encode())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    paths = {role: root / (role + (".jsonl" if spec[0] == "jsonl" else ".json"))
             for role, spec in ROLE_SPECS.items()}
    label = {
        "review_id": "A1", "hand_id": "hand1", "action_frame": 1,
        "actor_slot": 1, "candidate_glyph": "check",
        "review_status": "MATCH_VISIBLE_COMPLETED_ACTION",
        "actual_action": "check", "amount": "0", "amount_status": "ZERO",
        "street": "flop"}
    registry = {
        "source_session_id": "session1", "source_audit_sha256": "d" * 64,
        "samples_sha256": "a" * 64,
        "window": {"first_frame": 1, "last_frame": 2,
                   "start_pts": "1", "end_pts_exclusive": "3"},
        "hands": [{"hand_id": "hand1", "start_frame": 1,
                   "last_gameplay_frame": 1, "end_frame": 1,
                   "next_start_frame": 2, "temporal_complete": True,
                   "opening_state_status": "ORDINARY_UNKNOWN"},
                  {"hand_id": "tail", "start_frame": 2,
                   "last_gameplay_frame": 2, "end_frame": 2,
                   "next_start_frame": None, "temporal_complete": False,
                   "opening_state_status": "INCOMPLETE_RECORDING_TAIL"}]}
    write(paths["hand_registry"], registry)
    write(paths["action_review_config"], {"labels": [label]})
    write(paths["action_review_result"], {
        "labels": [label], "source_samples_sha256": "a" * 64})
    observations = [{"frame": 1, "pts_seconds": "1",
                     "source_sha256": "b" * 64,
                     "strategy_eligible": False, "scene_supported": True,
                     "current_actor": 1, "actor_evidence": {"actor": 1},
                     "board_count": 3},
                    {"frame": 2, "pts_seconds": "2",
                     "source_sha256": "c" * 64,
                     "strategy_eligible": False, "scene_supported": True,
                     "current_actor": None, "actor_evidence": {"actor": None},
                     "board_count": 3}]
    write(paths["v2_observations"], observations, lines=True)
    write(paths["v2_report"], {
        "frames": 2, "observations_sha256": sha(paths["v2_observations"]),
        "events": [{"frame": 1, "slot": 1, "glyph": "check"}]})
    input_hashes = {
        "registry_sha256": sha(paths["hand_registry"]),
        "review_config_sha256": sha(paths["action_review_config"]),
        "review_result_sha256": sha(paths["action_review_result"]),
        "v2_report_sha256": sha(paths["v2_report"]),
        "observations_sha256": sha(paths["v2_observations"]),
    }
    dataset_id = "aa8-current-action-candidates-NOT-DATA-v1"
    readiness_protocol = {
        "schema_version": 1,
        "status": "CURRENT_ACTION_CANDIDATES_NOT_ALL_OPPORTUNITIES_OR_DATA",
        **input_hashes, "expected_reviewed_candidates": 1,
        "expected_matched_actions": 1, "expected_false_candidates": 0,
        "expected_opponent_matches": 1, "expected_hero_matches": 0,
        "constraints": READINESS_CONSTRAINTS}
    episode_protocol = {
        "schema_version": 1,
        "status": "ACTOR_EPISODES_ARE_CANDIDATES_NOT_ALL_OPPORTUNITIES",
        **input_hashes, "expected_frames": 2, "expected_complete_hands": 1,
        "expected_reviewed_matches": 1, "standard_confirmation_lag_frames": 2,
        "maximum_candidate_confirmation_lag_frames": 4,
        "constraints": EPISODE_CONSTRAINTS}
    write(paths["readiness_protocol"], readiness_protocol)
    write(paths["episode_protocol"], episode_protocol)
    readiness = build_readiness_documents(
        readiness_protocol, registry, {"labels": [label]},
        {"labels": [label], "source_samples_sha256": "a" * 64},
        {"frames": 2, "observations_sha256": input_hashes[
            "observations_sha256"],
         "events": [{"frame": 1, "slot": 1, "glyph": "check"}]},
        observations, input_hashes)
    write(paths["decision_readiness"], readiness)
    episode_report = build_episode_documents(
        episode_protocol, registry, {"labels": [label]},
        {"labels": [label], "source_samples_sha256": "a" * 64},
        {"frames": 2, "observations_sha256": input_hashes[
            "observations_sha256"],
         "events": [{"frame": 1, "slot": 1, "glyph": "check"}]},
        observations, input_hashes)
    write(paths["actor_episode_census"], episode_report)
    files, bindings = [], []
    for index, (role, path) in enumerate(paths.items(), start=1):
        file_id = f"file-{index}"
        file_format, schema_id = ROLE_SPECS[role]
        files.append({"file_id": file_id, "relative_path": path.name,
                      "sha256": sha(path), "size_bytes": path.stat().st_size,
                      "format": file_format, "schema_id": schema_id})
        bindings.append({"owner_kind": "dataset", "owner_id": dataset_id,
                         "role": role, "file_id": file_id})
    manifest = {"schema_version": 1,
                "status": "CURRENT_AA8_BLOCKED_ARTIFACT_BUNDLE_V1",
                "dataset_id": dataset_id, "files": files,
                "bindings": bindings, "limitations": LIMITATIONS}
    manifest_path = root / "manifest.json"
    write(manifest_path, manifest)
    return root, manifest_path, manifest, paths


def rewrite_manifest(path, manifest):
    write(path, manifest)
    return sha(path)


def test_closed_snapshot_verifies_but_remains_blocked(tmp_path):
    root, manifest_path, _, _ = bundle(tmp_path)
    report = verify_bundle(root, manifest_path.name, sha(manifest_path))
    assert report["artifact_binding_status"] == (
        "VERIFIED_CURRENT_BLOCKED_METADATA_SNAPSHOTS")
    assert report["verified_file_count"] == 9
    assert report["semantic_checks"]["actor_episode_count"] == 1
    assert report["blockers"] and not report["ready_for_offline_calibration"]
    assert not report["model_fit_executed"] and not report["advice_emitted"]


@pytest.mark.parametrize("unsafe", [
    "../outside.json", "/absolute.json", "C:/drive.json", r"bad\name.json",
    "bad:name.json", "./dot.json", "trail. ",
])
def test_unsafe_manifest_paths_are_rejected(tmp_path, unsafe):
    root, manifest_path, manifest, _ = bundle(tmp_path)
    manifest["files"][0]["relative_path"] = unsafe
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="unsafe"):
        verify_bundle(root, manifest_path.name, digest)


def test_external_manifest_hash_and_artifact_hash_are_both_required(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    with pytest.raises(ValueError, match="external_sha256"):
        verify_bundle(root, manifest_path.name, "0" * 64)
    paths["v2_report"].write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="hash_or_size"):
        verify_bundle(root, manifest_path.name, sha(manifest_path))


def test_extra_file_and_missing_binding_are_rejected(tmp_path):
    root, manifest_path, manifest, _ = bundle(tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="extra_or_missing"):
        verify_bundle(root, manifest_path.name, sha(manifest_path))
    (root / "extra.json").unlink()
    manifest["bindings"].pop()
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="closure"):
        verify_bundle(root, manifest_path.name, digest)


def test_duplicate_json_keys_are_rejected_from_hashed_bytes(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    target = paths["v2_report"]
    target.write_bytes(b'{"frames":1,"frames":1}\n')
    item = next(row for row in manifest["files"]
                if row["relative_path"] == target.name)
    item.update(sha256=sha(target), size_bytes=target.stat().st_size)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        verify_bundle(root, manifest_path.name, digest)


def test_case_alias_and_hardlink_are_rejected(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    manifest["files"][1]["relative_path"] = manifest["files"][0][
        "relative_path"].upper()
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="unique_exact"):
        verify_bundle(root, manifest_path.name, digest)
    root, manifest_path, manifest, paths = bundle(tmp_path / "other")
    target = paths["episode_protocol"]
    target.unlink()
    try:
        os.link(paths["readiness_protocol"], target)
    except OSError:
        pytest.skip("hardlinks unavailable")
    item = next(row for row in manifest["files"] if row["relative_path"] == target.name)
    item.update(sha256=sha(target), size_bytes=target.stat().st_size)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="hardlinked"):
        verify_bundle(root, manifest_path.name, digest)


def test_symlink_and_output_inside_bundle_are_rejected(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    target = paths["episode_protocol"]
    target.unlink()
    try:
        target.symlink_to(paths["readiness_protocol"].name)
    except OSError:
        pytest.skip("symlinks unavailable")
    item = next(row for row in manifest["files"] if row["relative_path"] == target.name)
    item.update(sha256=sha(target), size_bytes=target.stat().st_size)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="symlink|reparse"):
        verify_bundle(root, manifest_path.name, digest)


def test_cli_output_must_be_new_and_outside_bundle(tmp_path):
    root, manifest_path, _, _ = bundle(tmp_path)
    output = root / "report.json"
    run = subprocess.run([
        sys.executable, "-m", "tools.verify_decision_opportunity_artifacts",
        "--bundle-root", str(root), "--manifest", manifest_path.name,
        "--manifest-sha256", sha(manifest_path), "--output", str(output)],
        capture_output=True)
    assert run.returncode == 2 and not output.exists()


def test_logical_source_swap_is_rejected_even_with_updated_hash(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    readiness = json.loads(paths["decision_readiness"].read_text())
    readiness["audit"]["dataset_id"] = "another-dataset"
    write(paths["decision_readiness"], readiness)
    item = next(row for row in manifest["files"]
                if row["relative_path"] == paths["decision_readiness"].name)
    item.update(sha256=sha(paths["decision_readiness"]),
                size_bytes=paths["decision_readiness"].stat().st_size)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="dataset_identity"):
        verify_bundle(root, manifest_path.name, digest)


def test_manifest_dataset_and_candidate_inventory_are_cross_bound(tmp_path):
    root, manifest_path, manifest, _ = bundle(tmp_path)
    manifest["dataset_id"] = "attacker-dataset"
    for binding in manifest["bindings"]:
        binding["owner_id"] = "attacker-dataset"
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="dataset_identity"):
        verify_bundle(root, manifest_path.name, digest)
    root, manifest_path, manifest, paths = bundle(tmp_path / "inventory")
    readiness = json.loads(paths["decision_readiness"].read_text())
    readiness["candidate_inventory"]["reviewed"] = 999
    write(paths["decision_readiness"], readiness)
    update_artifact(manifest, paths["decision_readiness"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="candidate_inventory"):
        verify_bundle(root, manifest_path.name, digest)


def test_bundle_builder_copies_exact_roles_and_does_not_overwrite(tmp_path):
    _, _, _, paths = bundle(tmp_path / "source")
    output = tmp_path / "closed-bundle"
    result = build_bundle(paths, output)
    report = verify_bundle(output, "manifest.json", result["manifest_sha256"])
    assert report["verified_file_count"] == len(ROLE_SPECS)
    with pytest.raises(FileExistsError):
        build_bundle(paths, output)


def update_artifact(manifest, path):
    item = next(row for row in manifest["files"]
                if row["relative_path"] == path.name)
    item.update(sha256=sha(path), size_bytes=path.stat().st_size)


def test_protocol_and_episode_content_cannot_be_spoofed_by_new_hash(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    write(paths["episode_protocol"], {"schema_version": 999,
                                      "status": "PROMOTION_ALLOWED"})
    update_artifact(manifest, paths["episode_protocol"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="protocol_or_wrapper"):
        verify_bundle(root, manifest_path.name, digest)
    root, manifest_path, manifest, paths = bundle(tmp_path / "episode")
    episode = json.loads(paths["actor_episode_census"].read_text())
    episode["episodes"][0]["hand_id"] = "wrong-hand"
    update = paths["actor_episode_census"]
    write(update, episode)
    update_artifact(manifest, update)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="census_differs"):
        verify_bundle(root, manifest_path.name, digest)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 999),
    ("schema_version", True),
    ("expected_reviewed_candidates", 999),
    ("expected_reviewed_candidates", True),
    ("expected_false_candidates", False),
])
def test_readiness_protocol_semantics_are_rebuilt(tmp_path, field, value):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    protocol = json.loads(paths["readiness_protocol"].read_text())
    protocol[field] = value
    write(paths["readiness_protocol"], protocol)
    update_artifact(manifest, paths["readiness_protocol"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="readiness_protocol|review_counts"):
        verify_bundle(root, manifest_path.name, digest)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True),
    ("expected_complete_hands", True),
])
def test_episode_protocol_rejects_boolean_integer_aliases(tmp_path, field, value):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    protocol = json.loads(paths["episode_protocol"].read_text())
    protocol[field] = value
    write(paths["episode_protocol"], protocol)
    update_artifact(manifest, paths["episode_protocol"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="actor_episode_protocol"):
        verify_bundle(root, manifest_path.name, digest)


def test_manifest_schema_version_rejects_boolean_integer_alias(tmp_path):
    root, manifest_path, manifest, _ = bundle(tmp_path)
    manifest["schema_version"] = True
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="blocked_bundle_manifest"):
        verify_bundle(root, manifest_path.name, digest)


@pytest.mark.parametrize("mutation", ["scope", "dataset"])
def test_whole_readiness_wrapper_is_rebuilt_from_sources(tmp_path, mutation):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    readiness = json.loads(paths["decision_readiness"].read_text())
    if mutation == "scope":
        readiness["scope"] = "ATTACKER_SCOPE"
    else:
        readiness["dataset"]["platform_binding"][
            "platform_id"] = "attacker-platform"
        readiness["audit"] = audit_decision_opportunities(
            readiness["dataset"])
    write(paths["decision_readiness"], readiness)
    update_artifact(manifest, paths["decision_readiness"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="readiness_differs"):
        verify_bundle(root, manifest_path.name, digest)


def test_missing_or_nonnull_promotion_fields_are_rejected(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    episodes = json.loads(paths["actor_episode_census"].read_text())
    del episodes["ready_for_offline_calibration"]
    write(paths["actor_episode_census"], episodes)
    update_artifact(manifest, paths["actor_episode_census"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="census_differs"):
        verify_bundle(root, manifest_path.name, digest)
    root, manifest_path, manifest, paths = bundle(tmp_path / "readiness")
    readiness = json.loads(paths["decision_readiness"].read_text())
    readiness["audit"]["selection"] = {"attacker": True}
    write(paths["decision_readiness"], readiness)
    update_artifact(manifest, paths["decision_readiness"])
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="recalculation"):
        verify_bundle(root, manifest_path.name, digest)


def test_non_utf8_json_is_rejected_even_when_hash_matches(tmp_path):
    root, manifest_path, manifest, paths = bundle(tmp_path)
    path = paths["episode_protocol"]
    path.write_bytes(json.dumps({"schema_version": 1}).encode("utf-16"))
    update_artifact(manifest, path)
    digest = rewrite_manifest(manifest_path, manifest)
    with pytest.raises((ValueError, UnicodeDecodeError), match="utf8|UTF-8"):
        verify_bundle(root, manifest_path.name, digest)
