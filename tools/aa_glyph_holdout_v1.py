"""Metadata-only eligibility freeze and closed eight-seat confusion accounting.

Never imports a recognizer, opens media, or treats historical predictions as
holdout results. This version is a negative-only exclusion audit.
Positive eligibility is never
supported; known historical-v1 structures corroborate exclusions only.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import stat


CLASSES = ("fold", "muck", "all_in")
FLAGS = {"template", "tuning_or_debug", "manual_action_review",
         "template_episode_overlap"}


def digest(path):
    return hashlib.sha256(plain_bytes(path)).hexdigest()


def plain_path(path, *, directory=False):
    path = Path(path)
    for component in (path, *path.parents):
        info = component.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("symlink or reparse path forbidden")
    info = path.lstat()
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("directory required")
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("plain single-link JSON required")


def plain_bytes(path):
    plain_path(path)
    before = Path(path).stat()
    if before.st_size > 16 * 1024 * 1024:
        raise ValueError("metadata too large")
    raw = Path(path).read_bytes()
    after = Path(path).stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("metadata changed during read")
    return raw


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject)


def load(path):
    return strict_json(plain_bytes(path))


def write_new(path, value):
    plain_path(Path(path).parent, directory=True)
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def public_safe(value):
    if isinstance(value, dict):
        for key, item in value.items():
            public_safe(key)
            public_safe(item)
    elif isinstance(value, list):
        for item in value:
            public_safe(item)
    elif isinstance(value, str):
        if (any(c in value for c in (":", "/", "\\"))
                or ".." in value or value.lower().endswith(
                    (".png", ".jpg", ".mkv", ".mp4", ".env", ".pem"))
                or re.fullmatch(r"\+?\d{10,15}", value)
                or re.search(r"(?:ghp_|github_pat_|sk-)[A-Za-z0-9]", value)):
            raise ValueError("private path, media or sensitive identifier")


def interval(row):
    a, b = row["first"], row["last"]
    if type(a) is not int or type(b) is not int or not 0 <= a <= b:
        raise ValueError("invalid frame interval")
    return a, b


def overlap(a, b):
    return a["source"] == b["source"] and max(a["first"], b["first"]) <= min(
        a["last"], b["last"])


HISTORY_KEYS = {
    'action-review': (
        "candidate_count classification_correction complete_opportunity_census  "
        "duplicates_found_in_reviewed_candidates false_fold_events false_folds  "
        "full_recall legal_state_acceptance matched_glyph_transitions  "
        "missing_actions pot_conservation_acceptance reviewer rows status  "
        "strategy_eligible visible_amount_delta_checks"
    ).split(),
    'boundaries': (
        "advice_emitted boundaries boundary_definition counts  "
        "cross_session_canonical_dataset frame_pts_index_sha256 hands  "
        "ready_for_calibration schema_version source_receipt_sha256  "
        "source_segment_frame_counts source_session_id  "
        "source_specific_canonical_boundaries status strategy_eligible  "
        "total_frames"
    ).split(),
    'media-review': (
        "authorization_scope_sha256 automatic_promotion boundaries  "
        "data_readiness evidence media_uploaded next_step observer  "
        "other_recordings_read platform privacy recognition_model_executed  "
        "remaining_blockers reviewed_at_utc reviewer schema_version source  "
        "status strategy_executed technical"
    ).split(),
    'receipt': (
        "advice_emitted all_opportunities_reviewed authorization_sha256  "
        "authorization_use_receipt_sha256 blockers data_readiness ended_at_utc  "
        "exit_code ffmpeg_binary_sha256 ffmpeg_contract_sha256  "
        "forced_termination hardware_fingerprint_sha256 human_signoff_required  "
        "identity_status legal_menu_count live_control metadata_files  "
        "model_fit_executed observed_device observed_device_sha256  "
        "ordered_segment_manifest_sha256 phase plan_sha256 platform_verified  "
        "progress ready_for_calibration ready_for_offline_review  "
        "recording_root rule_fingerprint schema_version segments session_id  "
        "source_integrity_status source_partition source_recording_group_id  "
        "special_modes started_at_utc status stop_reason strategy_eligible"
    ).split(),
    'selection': (
        "advice_emitted calibration_set canonical_cross_session_dataset  "
        "exact_boundary_registry_sha256 excluded frozen_against_source_hashes  "
        "independent_holdout media_extracted recognition_executed  "
        "same_session_only schema_version selected_frame_count  "
        "selected_hand_count selected_hands selection_role  "
        "source_receipt_sha256 source_session_id status strategy_eligible  "
        "validation_set"
    ).split(),
    'v3-comparison': (
        "additional_events_in_reviewed_hands all_event_count  "
        "confirmed_original_matches false_muck_folds_remaining  "
        "independent_holdout missing_confirmed training_and_regression_overlap"
    ).split(),
    'v3-final': (
        "events frames predictions_sha256 reference_frames status  "
        "training_cases_included"
    ).split(),
    'v3-first': (
        "events frames predictions_sha256 reference_frames status  "
        "training_cases_included"
    ).split(),
}


def exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("unknown or missing fields: " + name)


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("exact nonnegative integer required")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid logical ID")
    return value


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")
    return value


def unique(values):
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError("duplicate list values")
    return values


def tiling(rows, first, last):
    cursor = first
    for row in rows:
        a, b = interval(row)
        if a != cursor or b > last:
            raise ValueError("interval gap overlap or out of bounds")
        cursor = b + 1
    if cursor != last + 1:
        raise ValueError("incomplete source coverage")


def declarations(config):
    public_safe(config)
    exact(config, {"version", "scope", "sources", "evidence", "episodes",
                   "candidates"}, "config")
    if (type(config["version"]) is not int or config["version"] != 2
            or config["scope"] != "template-disjoint intra-session holdout"):
        raise ValueError("negative-only schema version 2 required")
    if len(config["sources"]) != 1:
        raise ValueError("single session adapter required")
    source = config["sources"][0]
    exact(source, {"id", "receipt_sha256", "media", "first", "last"}, "source")
    identifier(source["id"])
    sha256(source["receipt_sha256"])
    interval(source)
    if source["first"] != 0:
        raise ValueError("full source must start at frame zero")
    for row in source["media"]:
        exact(row, {"id", "sha256", "first", "last"}, "media")
        identifier(row["id"])
        sha256(row["sha256"])
    unique([r["id"] for r in source["media"]])
    tiling(source["media"], source["first"], source["last"])
    ev = config["evidence"]
    for row in ev:
        exact(row, {"id", "file", "sha256"}, "evidence")
        identifier(row["id"])
        sha256(row["sha256"])
        if not re.fullmatch(r"[a-z0-9-]+\.json", row["file"]):
            raise ValueError("metadata JSON evidence only")
    unique([r["id"] for r in ev])
    unique([r["file"] for r in ev])
    if {r["id"] for r in ev} != set(HISTORY_KEYS):
        raise ValueError("missing or unknown evidence roles")
    rows = config["candidates"]
    unique([r["id"] for r in rows])
    for row in rows:
        exact(row, {"id", "source", "first", "last", "slots", "scene",
                    "media_ids", "exposure", "clean_proven", "episode_complete",
                    "evidence"}, "candidate")
        identifier(row["id"])
        identifier(row["scene"])
        if row["source"] != source["id"]:
            raise ValueError("unknown source")
        if (row["slots"] != list(range(8))
                or any(type(s) is not int for s in row["slots"])):
            raise ValueError("eight exact physical seats required")
        exact(row["exposure"], FLAGS, "exposure")
        if any(v not in ("YES", "NO", "UNKNOWN")
               for v in row["exposure"].values()):
            raise ValueError("invalid exposure")
        for key in ("clean_proven", "episode_complete"):
            if type(row[key]) is not bool:
                raise ValueError("exact boolean declaration required")
        unique(row["evidence"])
        if not row["evidence"] or not set(row["evidence"]) <= set(HISTORY_KEYS):
            raise ValueError("missing exposure evidence")
        unique(row["media_ids"])
        expected = [m["id"] for m in source["media"]
                    if max(m["first"], row["first"]) <= min(m["last"], row["last"])]
        if row["media_ids"] != expected:
            raise ValueError("wrong candidate media set")
        # No declarations can positively authorize this negative-only tool.
        if (all(v == "NO" for v in row["exposure"].values())
                and row["clean_proven"] and row["episode_complete"]):
            raise ValueError("POSITIVE_ELIGIBILITY_UNSUPPORTED")
    tiling(rows, source["first"], source["last"])
    return source


def verified_evidence(config, directory):
    declarations(config)
    root = Path(directory)
    plain_path(root, directory=True)
    root = root.resolve(strict=True)
    if sorted(p.name for p in root.iterdir()) != sorted(
            r["file"] for r in config["evidence"]):
        raise ValueError("missing duplicate or unexpected evidence file")
    values = {}
    for row in config["evidence"]:
        path = root / row["file"]
        raw = plain_bytes(path)
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("evidence hash mismatch")
        value = strict_json(raw)
        exact(value, HISTORY_KEYS[row["id"]], row["id"] + " historical-v1")
        values[row["id"]] = value
    return values


def semantic_exposures(config, ev):
    source = declarations(config)
    hashes = {r["id"]: r["sha256"] for r in config["evidence"]}
    receipt, bounds, selected = (ev[k] for k in ("receipt", "boundaries", "selection"))
    if (receipt["schema_version"] != 1 or type(receipt["schema_version"]) is not int
            or type(bounds["schema_version"]) is not int
            or bounds["schema_version"] != 1
            or type(selected["schema_version"]) is not int
            or selected["schema_version"] != 1):
        raise ValueError("unsupported historical schema")
    if (source["receipt_sha256"] != hashes["receipt"]
            or bounds["source_receipt_sha256"] != hashes["receipt"]
            or selected["source_receipt_sha256"] != hashes["receipt"]
            or selected["exact_boundary_registry_sha256"] != hashes["boundaries"]
            or receipt["session_id"] != bounds["source_session_id"]
            or receipt["session_id"] != selected["source_session_id"]):
        raise ValueError("historical source binding conflict")
    counts = bounds["source_segment_frame_counts"]
    exact(receipt["progress"], {"frame", "drop_frames", "dup_frames",
                                "out_time", "progress"}, "receipt progress")
    if integer(receipt["progress"]["frame"]) != sum(integer(n) for n in counts):
        raise ValueError("receipt frame total conflict")
    if (len(counts) != len(source["media"])
            or len(receipt["segments"]) != len(counts)):
        raise ValueError("media count conflict")
    parts = zip(counts, source["media"], receipt["segments"])
    for index, (count, media, seg) in enumerate(parts):
        exact(seg, {"segment_index", "relative_path", "csv_start_pts",
                    "csv_end_pts_exclusive", "csv_gap_seconds", "sha256",
                    "size_bytes", "mtime_ns"}, "receipt segment")
        if (integer(count) == 0 or integer(seg["segment_index"]) != index
                or seg["relative_path"] != f"segment_{index:04d}.mkv"
                or sha256(seg["sha256"]) != media["sha256"]
                or media["last"] - media["first"] + 1 != count):
            raise ValueError("receipt media SHA or interval conflict")
    if (integer(bounds["total_frames"]) != sum(counts)
            or sum(counts) != source["last"] + 1):
        raise ValueError("source total conflict")
    hands = bounds["hands"]
    for hand in hands:
        required = {"hand_id", "start_global_frame", "end_global_frame",
                    "temporal_complete", "status"}
        optional = {"opening_boundary", "closing_boundary", "start_pts_seconds",
                    "end_pts_seconds", "privacy_usable", "ordinary_mode",
                    "preflop_only_candidate"}
        if not required <= set(hand) or set(hand) - required - optional:
            raise ValueError("unknown or missing historical hand fields")
        identifier(hand["hand_id"])
    unique([h["hand_id"] for h in hands])
    actual_hands = [(h["hand_id"], integer(h["start_global_frame"]),
                     integer(h["end_global_frame"])) for h in hands]
    if actual_hands != [(c["id"], c["first"], c["last"])
                        for c in config["candidates"]]:
        raise ValueError("candidate and historical hand boundaries conflict")
    by_id = {c["id"]: c for c in config["candidates"]}
    training = selected["selected_hands"]
    unique([h["hand_id"] for h in training])
    total = 0
    for h in training:
        exact(h, {"hand_id", "role", "start_global_frame", "end_global_frame",
                  "start_pts_seconds", "end_pts_seconds", "frame_count",
                  "preflop_only_candidate", "temporal_complete", "ordinary_mode",
                  "privacy_usable"}, "selected hand")
        c = by_id.get(h["hand_id"])
        if (c is None or h["role"] != "development"
                or integer(h["start_global_frame"]) != c["first"]
                or integer(h["end_global_frame"]) != c["last"]
                or integer(h["frame_count"]) != c["last"] - c["first"] + 1):
            raise ValueError("development selection conflict")
        total += h["frame_count"]
    if (integer(selected["selected_frame_count"]) != total
            or integer(selected["selected_hand_count"]) != len(training)):
        raise ValueError("development counts conflict")
    trained_ids = {h["hand_id"] for h in training}
    template_sets = []
    for key in ("v3-first", "v3-final"):
        v = ev[key]
        if (v["status"] != "DEVELOPMENT_REGRESSION_NOT_HOLDOUT"
                or integer(v["frames"]) != total
                or v["training_cases_included"] is not True):
            raise ValueError("V3 historical regression conflict")
        frames = unique(v["reference_frames"])
        template_sets.append({integer(f) for f in frames})
        seen = set()
        event_hands = set()
        for e in v["events"]:
            exact(e, {"hand_id", "frame", "slot", "glyph"}, "V3 event")
            validate_event(e, by_id)
            identity = (e["hand_id"], e["frame"], e["slot"], e["glyph"])
            if identity in seen:
                raise ValueError("duplicate V3 event")
            seen.add(identity)
            event_hands.add(e["hand_id"])
        if event_hands != trained_ids:
            raise ValueError("regression hands conflict")
    if template_sets[0] != template_sets[1] or template_sets[0] != {4565, 8680, 11761}:
        raise ValueError("V3 template evidence conflict")
    episodes = config["episodes"]
    if len(episodes) != len(template_sets[0]):
        raise ValueError("missing or duplicate template enclosure")
    unique([e["template_frame"] for e in episodes])
    if {integer(e["template_frame"]) for e in episodes} != template_sets[0]:
        raise ValueError("template enclosure set conflict")
    templated_ids = set()
    for e in episodes:
        exact(e, {"source", "first", "last", "template_frame", "extent"}, "enclosure")
        interval(e)
        candidates = [c for c in by_id.values()
                      if c["first"] <= e["template_frame"] <= c["last"]]
        if len(candidates) != 1:
            raise ValueError("template must belong to one whole hand")
        c = candidates[0]
        if (e["source"] != source["id"] or interval(e) != interval(c)
                or e["extent"] != "CONSERVATIVE_WHOLE_HAND_EXCLUSION_NOT_EXACT_EPISODE"
                or c["exposure"]["template"] != "YES"
                or c["exposure"]["template_episode_overlap"] != "YES"):
            raise ValueError("template whole hand enclosure conflict")
        templated_ids.add(c["id"])
    action = ev["action-review"]
    reviewed_ids, seen = set(), set()
    for e in action["rows"]:
        exact(e, {"event_id", "hand_id", "frame", "slot", "glyph", "review_status",
                  "manual_action", "manual_street", "visible_stack_decrease",
                  "amount_semantics", "legal_amount", "evidence_page"}, "review event")
        validate_event(e, by_id)
        identifier(e["event_id"])
        if e["event_id"] in seen or e["review_status"] not in (
                "MATCH_VISIBLE_GLYPH_TRANSITION", "FALSE_FOLD_SHOWDOWN_MUCK"):
            raise ValueError("duplicate or unsupported action review")
        seen.add(e["event_id"])
        reviewed_ids.add(e["hand_id"])
    if integer(action["candidate_count"]) != len(seen):
        raise ValueError("review count conflict")
    matches = sum(e["review_status"] == "MATCH_VISIBLE_GLYPH_TRANSITION"
                  for e in action["rows"])
    if (integer(action["matched_glyph_transitions"]) != matches
            or integer(action["false_folds"]) != len(seen) - matches):
        raise ValueError("review label counts conflict")
    comparison = ev["v3-comparison"]
    if (comparison["training_and_regression_overlap"] is not True
            or comparison["independent_holdout"] is not False
            or integer(comparison["all_event_count"]) != len(ev["v3-final"]["events"])):
        raise ValueError("V3 comparison conflict")
    return {c["id"]: {
        "template": "YES" if c["id"] in templated_ids else "UNKNOWN",
        "template_episode_overlap": "YES" if c["id"] in templated_ids else "UNKNOWN",
        "tuning_or_debug": "YES" if c["id"] in trained_ids else "UNKNOWN",
        "manual_action_review": "YES" if c["id"] in reviewed_ids else "UNKNOWN"}
        for c in config["candidates"]}


def validate_event(e, candidates):
    c = candidates.get(e["hand_id"])
    if (c is None or not c["first"] <= integer(e["frame"]) <= c["last"]
            or integer(e["slot"]) > 7
            or e["glyph"] not in ("fold", "check", "call", "aggressive", "all_in")):
        raise ValueError("event hand frame or class conflict")


def eligibility(config, evidence):
    derived = semantic_exposures(config, evidence)
    decisions = []
    for c in config["candidates"]:
        flags = derived[c["id"]]
        reasons = [k + "_" + v for k, v in sorted(flags.items())]
        reasons.append("NEGATIVE_ONLY_NO_CLEAN_PROVENANCE_PROOF")
        decisions.append({"id": c["id"], "source": c["source"],
                          "first": c["first"], "last": c["last"],
                          "eligible": False, "reasons": reasons,
                          "declared_exposure": c["exposure"],
                          "derived_exposure": flags})
    return decisions


def schedule(decisions):
    """Exhaustive temporal/seat order, no scores or prediction-driven selection."""
    return [(d["id"], frame, slot)
            for d in sorted(decisions, key=lambda r: (r["source"], r["first"]))
            if d["eligible"] for frame in range(d["first"], d["last"] + 1)
            for slot in range(8)]


def confusion(queue, labels, outputs):
    """Exhaustive frame-slot multiclass counts; not episode/event recall.

    Caller supplies human labels in the frozen non-model-driven queue order.
    missing/UNKNOWN labels or missing machine rows fail, never shrink denominator.
    """
    if (len(queue) != len(set(queue)) or len(labels) != len(queue)
            or len(outputs) != len(queue)):
        raise ValueError("exact census required")
    allowed = {*CLASSES, "none"}
    totals = {str(slot): {c: dict(
        TP=0, FN=0, FP=0, TN=0, opportunities=0, negatives=0)
                          for c in CLASSES} for slot in range(8)}
    for expected, human, machine in zip(queue, labels, outputs):
        if (tuple(human["key"]) != expected
                or tuple(machine["key"]) != expected
                or human["label"] not in allowed
                or machine["label"] not in allowed
                or type(expected[2]) is not int or not 0 <= expected[2] < 8):
            raise ValueError("census identity or label mismatch")
        for cls in CLASSES:
            pos = human["label"] == cls
            pred = machine["label"] == cls
            row = totals[str(expected[2])][cls]
            row["opportunities" if pos else "negatives"] += 1
            row["TP" if pos and pred else "FN" if pos else
                "FP" if pred else "TN"] += 1
    return totals


def freeze(config_path, evidence_dir, output):
    config = load(config_path)
    evidence = verified_evidence(config, evidence_dir)
    decisions = eligibility(config, evidence)
    value = {"schema_version": 1, "scope": config["scope"],
             "frozen_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
             "config_sha256": digest(config_path),
             "tool_sha256": digest(__file__), "decisions": decisions,
             "model_executed": False,
             "status": "NO_ELIGIBLE_HOLDOUT"}
    write_new(output, value)
    return value


def conclude(config_path, evidence_dir, manifest, expected_sha, out):
    if digest(manifest) != expected_sha:
        raise ValueError("frozen manifest hash mismatch")
    frozen = load(manifest)
    if (frozen["config_sha256"] != digest(config_path)
            or frozen["tool_sha256"] != digest(__file__)):
        raise ValueError("frozen config or implementation changed")
    config = load(config_path)
    evidence = verified_evidence(config, evidence_dir)
    decisions = eligibility(config, evidence)
    if decisions != frozen["decisions"]:
        raise ValueError("frozen decisions mismatch")
    if any(d["eligible"] for d in decisions):
        raise ValueError("eligible material needs separate label-first validation")
    target = Path(out)
    target.mkdir(exist_ok=False)
    for name in ("human-labels", "machine-output"):
        write_new(target / (name + ".json"), {
            "status": "NOT_RUN_NO_ELIGIBLE_HOLDOUT", "rows": [],
            "manifest_sha256": expected_sha})
    metrics = {str(s): {c: {"denominator": 0, "TP": None, "FN": None,
                            "FP": None, "TN": None} for c in CLASSES}
               for s in range(8)}
    result = {"status": "NO_ELIGIBLE_HOLDOUT", "scope": config["scope"],
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "frozen_manifest_sha256": expected_sha,
              "eligible_frame_slots": 0, "metrics": metrics,
              "metrics_by_hand": {d["id"]: metrics for d in decisions},
              "labels_sha256": digest(target / "human-labels.json"),
              "output_sha256": digest(target / "machine-output.json"),
              "model_executed": False, "media_read": False,
              "decisions": decisions, "command": sys.argv,
              "python": sys.version, "code_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True,
                  cwd=Path(__file__).resolve().parents[1]).strip(),
              "tool_sha256": digest(__file__)}
    # Version metadata only; no image/model module imports are needed.
    from importlib.metadata import version
    result["numpy"] = version("numpy")
    result["opencv"] = version("opencv-python")
    write_new(target / "report.json", result)
    write_new(target / "SHA256.json", {
        p.name: digest(p) for p in sorted(target.glob("*.json"))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "conclude"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.config, args.evidence_dir, args.output)
    else:
        if args.manifest is None or args.manifest_sha256 is None:
            parser.error("conclude requires externally pinned manifest")
        result = conclude(args.config, args.evidence_dir, args.manifest,
                          args.manifest_sha256, args.output)
    print(json.dumps({"status": result["status"], "model_executed": False}))


if __name__ == "__main__":
    main()
