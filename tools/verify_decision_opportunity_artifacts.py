"""Verify one closed AA8 metadata bundle without reading media or promoting it."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata

from tools.audit_decision_opportunities import unique_object
from tools.audit_aa8_actor_episodes import (
    CONSTRAINTS as EPISODE_CONSTRAINTS,
    KEYS as EPISODE_PROTOCOL_KEYS,
    build_documents as rebuild_actor_episodes,
)
from tools.build_aa8_decision_readiness import (
    CONSTRAINTS as READINESS_CONSTRAINTS,
    PROTOCOL_KEYS as READINESS_PROTOCOL_KEYS,
    build_documents as rebuild_decision_readiness,
)
from poker_engine.strategy.decision_opportunities_v1 import (
    audit_decision_opportunities,
)


SHA256 = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS = {
    "schema_version", "status", "dataset_id", "files", "bindings",
    "limitations",
}
FILE_KEYS = {
    "file_id", "relative_path", "sha256", "size_bytes", "format", "schema_id",
}
BINDING_KEYS = {"owner_kind", "owner_id", "role", "file_id"}
ROLE_SPECS = {
    "decision_readiness": ("json", "aa8-decision-readiness-wrapper-v1"),
    "actor_episode_census": ("json", "aa8-actor-episode-census-v1"),
    "readiness_protocol": ("json", "aa8-decision-readiness-protocol-v1"),
    "episode_protocol": ("json", "aa8-actor-episode-protocol-v1"),
    "hand_registry": ("json", "aa8-late-hand-registry-v1"),
    "action_review_config": ("json", "aa8-action-review-config-v1"),
    "action_review_result": ("json", "aa8-action-review-result-v1"),
    "v2_report": ("json", "aa8-glyph-v2-report-v1"),
    "v2_observations": ("jsonl", "aa8-glyph-v2-observations-v1"),
}
LIMITATIONS = [
    "metadata_json_jsonl_only_no_media_read",
    "raw_recording_not_rehashed_or_semantically_reviewed",
    "stable_identity_rule_legal_menu_and_episode_coverage_artifacts_missing",
    "verified_snapshot_cannot_grant_offline_calibration_or_live_use",
]
MAX_BYTES = 64 * 1024 * 1024
MAX_JSONL_LINES = 10000
MAX_JSONL_LINE_BYTES = 256 * 1024


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("exact_" + name + "_fields_required")


def _is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _relative(value):
    if (not isinstance(value, str) or not value or "\x00" in value
            or "\\" in value or ":" in value
            or unicodedata.normalize("NFC", value) != value):
        raise ValueError("unsafe_or_nonnormalized_bundle_path")
    raw_parts = value.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError("unsafe_relative_bundle_path")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts
            or any(part in ("", ".", "..") or part.endswith((" ", "."))
                   for part in path.parts)):
        raise ValueError("unsafe_relative_bundle_path")
    return path


def _root(path):
    raw = path.lstat()
    if path.is_symlink() or _is_reparse(raw) or not stat.S_ISDIR(raw.st_mode):
        raise ValueError("bundle_root_must_be_plain_directory")
    return path.resolve(strict=True)


def _plain_ancestors(root, target):
    current = target
    chain = []
    while current != root:
        chain.append(current)
        current = current.parent
        if len(chain) > 64:
            raise ValueError("bundle_path_depth_exceeded")
    for item in reversed(chain):
        info = item.lstat()
        if item.is_symlink() or _is_reparse(info):
            raise ValueError("bundle_symlink_or_reparse_forbidden")


def _snapshot(root, relative, *, maximum=MAX_BYTES):
    logical = _relative(relative)
    path = root.joinpath(*logical.parts)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("bundle_path_escapes_root")
    _plain_ancestors(root, path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("bundle_artifact_must_be_regular_file_and_not_hardlinked")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("opened_artifact_not_regular_file")
        chunks, total = [], 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ValueError("bundle_artifact_exceeds_size_limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()

    def identity(item):
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                item.st_mode, item.st_nlink,
                getattr(item, "st_file_attributes", 0))
    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise ValueError("bundle_artifact_changed_during_read")
    if (identity(after) != identity(final) or path.resolve(strict=True) != resolved
            or path.is_symlink() or _is_reparse(final)):
        raise ValueError("bundle_artifact_path_changed_during_read")
    raw = b"".join(chunks)
    return {
        "raw": raw, "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw), "file_identity": (opened.st_dev, opened.st_ino),
    }


def _json(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("utf8_bom_forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=unique_object)
    except UnicodeDecodeError as exc:
        raise ValueError("utf8_json_required") from exc


def _jsonl(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("utf8_bom_forbidden")
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("utf8_jsonl_required") from exc
    if not lines or len(lines) > MAX_JSONL_LINES:
        raise ValueError("bounded_nonempty_jsonl_required")
    result = []
    for line in lines:
        if not line or len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            raise ValueError("bounded_nonempty_jsonl_line_required")
        result.append(json.loads(line, object_pairs_hook=unique_object))
    return result


def _enumerate_bundle(root):
    paths = []
    for base, folders, files in os.walk(root, followlinks=False):
        if len(paths) + len(files) > 256:
            raise ValueError("bundle_file_count_limit_exceeded")
        base_path = Path(base)
        for folder in folders:
            info = (base_path / folder).lstat()
            if (base_path / folder).is_symlink() or _is_reparse(info):
                raise ValueError("bundle_directory_link_forbidden")
        for name in files:
            paths.append((base_path / name).relative_to(root).as_posix())
    return sorted(paths)


def _cross_bind(documents, manifest_dataset_id):
    readiness = documents["decision_readiness"]
    episodes = documents["actor_episode_census"]
    readiness_protocol = documents["readiness_protocol"]
    episode_protocol = documents["episode_protocol"]
    registry = documents["hand_registry"]
    review = documents["action_review_config"]
    result = documents["action_review_result"]
    report = documents["v2_report"]
    observations = documents["v2_observations"]
    objects = (readiness, episodes, readiness_protocol, episode_protocol,
               registry, review, result, report)
    if any(not isinstance(value, dict) for value in objects):
        raise ValueError("artifact_documents_must_be_objects")
    if (set(readiness) != {
            "scope", "input_hashes", "candidate_inventory", "dataset", "audit",
            "strategy_eligible", "advice_emitted", "model_fit_executed"}
            or set(readiness_protocol) != READINESS_PROTOCOL_KEYS
            or readiness_protocol["constraints"] != READINESS_CONSTRAINTS
            or readiness_protocol["status"] != (
                "CURRENT_ACTION_CANDIDATES_NOT_ALL_OPPORTUNITIES_OR_DATA")
            or set(episode_protocol) != EPISODE_PROTOCOL_KEYS
            or episode_protocol["constraints"] != EPISODE_CONSTRAINTS
            or episode_protocol["status"] != (
                "ACTOR_EPISODES_ARE_CANDIDATES_NOT_ALL_OPPORTUNITIES")):
        raise ValueError("artifact_protocol_or_wrapper_schema_invalid")
    dataset_id = readiness.get("dataset", {}).get("dataset_id")
    if (dataset_id != "aa8-current-action-candidates-NOT-DATA-v1"
            or readiness.get("audit", {}).get("dataset_id") != dataset_id
            or manifest_dataset_id != dataset_id):
        raise ValueError("manifest_and_readiness_dataset_identity_disagree")
    role_hashes = documents["_hashes"]
    source_hashes = {
        "registry_sha256": role_hashes["hand_registry"],
        "review_config_sha256": role_hashes["action_review_config"],
        "review_result_sha256": role_hashes["action_review_result"],
        "v2_report_sha256": role_hashes["v2_report"],
        "observations_sha256": role_hashes["v2_observations"],
    }
    if (readiness["input_hashes"] != source_hashes
            or episodes.get("input_hashes") != source_hashes):
        raise ValueError("saved_output_artifact_hash_bindings_disagree")
    for key, digest in source_hashes.items():
        if (readiness_protocol[key] != digest or episode_protocol[key] != digest):
            raise ValueError("protocol_artifact_hash_bindings_disagree")
    labels = result.get("labels")
    if (labels != review.get("labels")
            or result.get("source_samples_sha256") != registry.get("samples_sha256")
            or report.get("observations_sha256") != role_hashes["v2_observations"]
            or report.get("frames") != len(observations)):
        raise ValueError("artifact_source_chain_disagrees")
    matched = [row for row in labels
               if row["review_status"] == "MATCH_VISIBLE_COMPLETED_ACTION"]
    expected_inventory = {
        "reviewed": len(labels), "matched": len(matched),
        "false": len(labels) - len(matched),
        "opponent_matched": sum(row["actor_slot"] != 4 for row in matched),
        "hero_matched": sum(row["actor_slot"] == 4 for row in matched),
        "all_actual_opportunities_reviewed": False}
    if readiness["candidate_inventory"] != expected_inventory:
        raise ValueError("readiness_candidate_inventory_differs_from_review")
    rebuilt_readiness = rebuild_decision_readiness(
        readiness_protocol, registry, review, result, report, observations,
        source_hashes)
    if rebuilt_readiness != readiness:
        raise ValueError(
            "saved_decision_readiness_differs_from_source_recalculation")
    recalculated_audit = audit_decision_opportunities(readiness["dataset"])
    if recalculated_audit != readiness["audit"]:
        raise ValueError("saved_readiness_audit_differs_from_recalculation")
    if (recalculated_audit["data_readiness"] != "BLOCKED"
            or readiness["candidate_inventory"][
                "all_actual_opportunities_reviewed"] is not False):
        raise ValueError("current_readiness_blocked_contract_required")
    rebuilt_episodes = rebuild_actor_episodes(
        episode_protocol, registry, review, result, report, observations,
        source_hashes)
    if rebuilt_episodes != episodes:
        raise ValueError("saved_actor_episode_census_differs_from_recalculation")
    required_false = (
        "ready_for_offline_calibration", "model_fit_executed",
        "strategy_eligible", "advice_emitted", "live_use")
    if (readiness["strategy_eligible"] is not False
            or readiness["advice_emitted"] is not False
            or readiness["model_fit_executed"] is not False
            or any(recalculated_audit[key] is not False for key in required_false)
            or any(episodes[key] is not False for key in required_false)
            or any(recalculated_audit[key] is not None for key in (
                "selection", "calibration", "range_model"))):
        raise ValueError("artifact_chain_attempts_forbidden_promotion")
    return {
        "dataset_id": dataset_id,
        "actor_episode_count": episodes["actor_episode_count"],
        "bound_reviewed_action_count": episodes["bound_reviewed_action_count"],
        "unmatched_actor_episode_count": episodes[
            "unmatched_actor_episode_count"],
        "unbound_reviewed_action_count": episodes[
            "unbound_reviewed_action_count"],
        "unknown_span_count": episodes["unknown_span_count"],
        "observation_count": len(observations),
        "decision_readiness_recalculated": True,
        "readiness_audit_recalculated": True,
        "actor_episode_census_recalculated": True,
    }


def verify_bundle(bundle_root, manifest_relative, expected_manifest_sha256):
    root = _root(bundle_root)
    if not isinstance(expected_manifest_sha256, str) or SHA256.fullmatch(
            expected_manifest_sha256) is None:
        raise ValueError("external_manifest_sha256_required")
    manifest_path = _relative(manifest_relative).as_posix()
    manifest_snapshot = _snapshot(root, manifest_path, maximum=4 * 1024 * 1024)
    if manifest_snapshot["sha256"] != expected_manifest_sha256:
        raise ValueError("manifest_differs_from_external_sha256")
    manifest = _json(manifest_snapshot["raw"])
    _exact(manifest, MANIFEST_KEYS, "artifact_manifest")
    if (type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["status"] != "CURRENT_AA8_BLOCKED_ARTIFACT_BUNDLE_V1"
            or manifest["limitations"] != LIMITATIONS
            or not isinstance(manifest["files"], list)
            or not isinstance(manifest["bindings"], list)):
        raise ValueError("exact_current_blocked_bundle_manifest_required")
    files, paths, aliases = {}, {}, set()
    for item in manifest["files"]:
        _exact(item, FILE_KEYS, "artifact_file")
        file_id, logical = item["file_id"], _relative(item["relative_path"])
        alias = logical.as_posix().casefold()
        if (not isinstance(file_id, str) or not file_id
                or file_id in files or logical.as_posix() in paths or alias in aliases
                or type(item["size_bytes"]) is not int
                or item["size_bytes"] < 0 or not isinstance(item["sha256"], str)
                or SHA256.fullmatch(item["sha256"]) is None):
            raise ValueError("unique_exact_artifact_file_identity_required")
        files[file_id] = item
        paths[logical.as_posix()] = file_id
        aliases.add(alias)
    bindings, roles = {}, {}
    for binding in manifest["bindings"]:
        _exact(binding, BINDING_KEYS, "artifact_binding")
        identity = (binding["owner_kind"], binding["owner_id"], binding["role"])
        if (identity in bindings or binding["file_id"] not in files
                or binding["owner_kind"] != "dataset"
                or binding["owner_id"] != manifest["dataset_id"]
                or binding["role"] in roles):
            raise ValueError("unique_exact_artifact_binding_required")
        bindings[identity] = binding["file_id"]
        roles[binding["role"]] = binding["file_id"]
    if set(roles) != set(ROLE_SPECS) or set(roles.values()) != set(files):
        raise ValueError("artifact_binding_closure_must_be_exact")
    actual_paths = _enumerate_bundle(root)
    if actual_paths != sorted([manifest_path, *paths]):
        raise ValueError("bundle_contains_extra_or_missing_files")
    documents, verified, file_identities = {"_hashes": {}}, [], set()
    for role, file_id in roles.items():
        item = files[file_id]
        expected_format, expected_schema = ROLE_SPECS[role]
        if (item["format"] != expected_format
                or item["schema_id"] != expected_schema):
            raise ValueError("artifact_format_or_schema_role_mismatch")
        snapshot = _snapshot(root, item["relative_path"])
        if (snapshot["sha256"] != item["sha256"]
                or snapshot["size_bytes"] != item["size_bytes"]):
            raise ValueError("artifact_hash_or_size_mismatch:" + file_id)
        if snapshot["file_identity"] in file_identities:
            raise ValueError("hardlinked_or_aliased_artifact_forbidden")
        file_identities.add(snapshot["file_identity"])
        documents[role] = (_json(snapshot["raw"])
                           if item["format"] == "json" else _jsonl(snapshot["raw"]))
        documents["_hashes"][role] = snapshot["sha256"]
        verified.append({
            "file_id": file_id, "role": role, "sha256": snapshot["sha256"],
            "size_bytes": snapshot["size_bytes"], "schema_id": item["schema_id"]})
    checks = _cross_bind(documents, manifest["dataset_id"])
    if _enumerate_bundle(root) != actual_paths:
        raise ValueError("bundle_file_set_changed_during_verification")
    return {
        "engineering_status": "PASS",
        "artifact_binding_status": "VERIFIED_CURRENT_BLOCKED_METADATA_SNAPSHOTS",
        "manifest_sha256": manifest_snapshot["sha256"],
        "dataset_id": checks["dataset_id"],
        "verified_file_count": len(verified),
        "verified_binding_count": len(bindings),
        "verified_files": verified, "semantic_checks": checks,
        "blockers": [
            "source_dataset_is_blocked",
            "raw_recording_not_in_bundle_or_read",
            "stable_identity_artifacts_missing",
            "real_rule_artifact_missing",
            "complete_legal_menu_artifacts_missing",
            "independent_episode_coverage_missing"],
        "ready_for_offline_calibration": False,
        "model_fit_executed": False, "selection": None, "calibration": None,
        "range_model": None, "strategy_eligible": False,
        "advice_emitted": False, "live_use": False,
        "human_signoff_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        root = _root(args.bundle_root)
        output = args.output.resolve()
        if output.is_relative_to(root) or args.output.exists():
            raise ValueError("output_must_be_new_and_outside_bundle")
        report = verify_bundle(
            root, args.manifest, args.manifest_sha256)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"artifact bundle rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "engineering_status", "artifact_binding_status", "verified_file_count",
        "verified_binding_count", "blockers")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
