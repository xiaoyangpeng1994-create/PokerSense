"""Verify an offline AA8 development window and audit action opportunities."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

from tools.capture_card_calibration.hashing import verify_sha256sums


SHA256 = re.compile(r"[0-9a-f]{64}")
REGISTRY_KEYS = {
    "schema_version", "status", "source_session_id", "capture_path",
    "emulator_used", "source_audit_sha256", "split_plan_sha256",
    "samples_sha256", "pipeline_report_sha256", "observations_sha256",
    "window", "hands", "review_constraints",
}
HAND_KEYS = {
    "hand_id", "start_frame", "last_gameplay_frame", "end_frame",
    "next_start_frame", "temporal_complete", "boundary_status",
    "opening_state_status", "evidence",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"exact_{name}_fields_required")


def decimal(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name}_decimal_string_required")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name}_finite_decimal_required") from exc
    if not result.is_finite():
        raise ValueError(f"{name}_finite_decimal_required")
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def development_range(start, end, plan):
    return any(
        item["role"] == "development"
        and decimal(item["start_inclusive"], "range_start") <= start
        and end <= decimal(item["end_exclusive"], "range_end")
        for item in plan["ranges"]
    )


def _validate_registry(registry):
    exact(registry, REGISTRY_KEYS, "registry")
    if (registry["schema_version"] != 1
            or registry["status"] != (
                "DEVELOPMENT_REVIEWED_BOUNDARIES_NOT_CALIBRATION")
            or not isinstance(registry["source_session_id"], str)
            or not registry["source_session_id"]
            or registry["capture_path"] != "physical_phone_capture_card"
            or registry["emulator_used"] is not False):
        raise ValueError("physical_offline_development_registry_required")
    for key in ("source_audit_sha256", "split_plan_sha256", "samples_sha256",
                "pipeline_report_sha256", "observations_sha256"):
        if not isinstance(registry[key], str) or not SHA256.fullmatch(registry[key]):
            raise ValueError("lowercase_sha256_identity_required")
    if (not isinstance(registry["review_constraints"], list)
            or not registry["review_constraints"]
            or not all(isinstance(item, str) and item
                       for item in registry["review_constraints"])):
        raise ValueError("explicit_review_constraints_required")


def _validate_window(registry, audit, plan, samples):
    window = registry["window"]
    exact(window, {"start_pts", "end_pts_exclusive", "first_frame",
                   "last_frame", "segments", "excluded_cross_boundary_segment"},
          "window")
    start = decimal(window["start_pts"], "window_start")
    end = decimal(window["end_pts_exclusive"], "window_end")
    if start < 0 or end <= start or not development_range(start, end, plan):
        raise ValueError("window_not_wholly_in_development_range")
    if samples["range_seconds"] != [str(start), str(end)]:
        raise ValueError("sample_window_identity_mismatch")
    rows = samples["samples"]
    frames = [row["global_frame"] for row in rows]
    if (not frames or frames != list(range(frames[0], frames[-1] + 1))
            or frames[0] != window["first_frame"]
            or frames[-1] != window["last_frame"]
            or any(row.get("role") != "development" for row in rows)):
        raise ValueError("contiguous_development_frame_inventory_required")
    by_segment = {item["file"]: item for item in audit["segments"]}
    declared = window["segments"]
    if not isinstance(declared, list) or not declared:
        raise ValueError("nonempty_segment_inventory_required")
    selected_names = []
    for item in declared:
        exact(item, {"file", "sha256", "decoded_frames", "first_global_frame",
                     "last_global_frame"}, "segment")
        source = by_segment.get(item["file"])
        segment_rows = [row for row in rows if row["segment"] == item["file"]]
        if (source is None or source["sha256"] != item["sha256"]
                or source["decoded_frames"] != item["decoded_frames"]
                or not segment_rows
                or segment_rows[0]["global_frame"] != item["first_global_frame"]
                or segment_rows[-1]["global_frame"] != item["last_global_frame"]
                or not development_range(decimal(source["first_pts"], "segment_start"),
                                         decimal(source["last_pts"], "segment_end"),
                                         plan)):
            raise ValueError("segment_not_bound_to_safe_development_source")
        selected_names.append(item["file"])
    if set(selected_names) != {row["segment"] for row in rows}:
        raise ValueError("segment_inventory_does_not_cover_samples")
    excluded = window["excluded_cross_boundary_segment"]
    if (not isinstance(excluded, dict)
            or set(excluded) != {"file", "reason"}
            or excluded["file"] in selected_names
            or excluded["reason"] != "crosses_protected_to_development_boundary"):
        raise ValueError("explicit_cross_boundary_segment_exclusion_required")
    return rows


def _validate_hands(registry, samples_by_frame):
    hands = registry["hands"]
    if not isinstance(hands, list) or len(hands) < 2:
        raise ValueError("reviewed_hand_and_tail_registry_required")
    ids, previous_end = set(), None
    for index, hand in enumerate(hands):
        exact(hand, HAND_KEYS, "hand")
        if (not isinstance(hand["hand_id"], str) or not hand["hand_id"]
                or hand["hand_id"] in ids):
            raise ValueError("unique_hand_id_required")
        ids.add(hand["hand_id"])
        values = [hand[k] for k in ("start_frame", "last_gameplay_frame",
                                    "end_frame")]
        if (any(type(value) is not int for value in values)
                or not values[0] <= values[1] <= values[2]
                or values[0] not in samples_by_frame
                or values[1] not in samples_by_frame
                or values[2] not in samples_by_frame
                or previous_end is not None and values[0] <= previous_end):
            raise ValueError("ordered_nonoverlapping_hand_frames_required")
        previous_end = values[2]
        complete = hand["temporal_complete"]
        if complete is True:
            if (hand["boundary_status"] != "DEVELOPMENT_MANUAL_REVIEW"
                    or type(hand["next_start_frame"]) is not int
                    or hand["next_start_frame"] != hand["end_frame"] + 1):
                raise ValueError("complete_hand_requires_next_boundary")
        elif complete is False:
            if (index != len(hands) - 1 or hand["next_start_frame"] is not None
                    or hand["boundary_status"] != "INCOMPLETE_RECORDING_TAIL"):
                raise ValueError("only_final_hand_may_be_incomplete_tail")
        else:
            raise ValueError("explicit_temporal_complete_boolean_required")
        if (not isinstance(hand["opening_state_status"], str)
                or not hand["opening_state_status"]):
            raise ValueError("opening_state_status_required")
        evidence = hand["evidence"]
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise ValueError("multiple_boundary_evidence_frames_required")
        for item in evidence:
            exact(item, {"frame", "sha256", "meaning"}, "evidence")
            sample = samples_by_frame.get(item["frame"])
            if (sample is None or sample["sha256"] != item["sha256"]
                    or not isinstance(item["meaning"], str)
                    or not item["meaning"]):
                raise ValueError("boundary_evidence_not_bound_to_sample")
    return hands


def analyze(registry_path, audit_dir, split_plan_path, samples_dir, pipeline_dir):
    registry = load_json(registry_path)
    _validate_registry(registry)
    audit_path = audit_dir / "report.json"
    samples_path = samples_dir / "samples.json"
    pipeline_report_path = pipeline_dir / "report.json"
    observations_path = pipeline_dir / "observations.jsonl"
    if (sha256(audit_path) != registry["source_audit_sha256"]
            or sha256(split_plan_path) != registry["split_plan_sha256"]
            or sha256(samples_path) != registry["samples_sha256"]
            or sha256(pipeline_report_path) != registry["pipeline_report_sha256"]
            or sha256(observations_path) != registry["observations_sha256"]):
        raise ValueError("registry_input_hash_mismatch")
    if verify_sha256sums(audit_dir) or verify_sha256sums(samples_dir):
        raise ValueError("source_or_extracted_sample_integrity_failure")
    audit, plan, samples = (load_json(path) for path in (
        audit_path, split_plan_path, samples_path))
    if (audit.get("state") != "verified"
            or plan.get("source_audit_sha256") != registry["source_audit_sha256"]
            or samples.get("audit_sha256") != registry["source_audit_sha256"]
            or samples.get("plan_sha256") != registry["split_plan_sha256"]):
        raise ValueError("source_split_sample_identity_mismatch")
    sample_rows = _validate_window(registry, audit, plan, samples)
    samples_by_frame = {row["global_frame"]: row for row in sample_rows}
    hands = _validate_hands(registry, samples_by_frame)
    pipeline = load_json(pipeline_report_path)
    if (pipeline.get("observations_sha256") != registry["observations_sha256"]
            or pipeline.get("frames") != len(sample_rows)
            or pipeline.get("independent_holdout") is not False
            or pipeline.get("strategy_eligible") is not False):
        raise ValueError("development_pipeline_contract_mismatch")
    observations = [json.loads(line) for line in observations_path.read_text(
        encoding="utf-8").splitlines() if line]
    if len(observations) != len(sample_rows):
        raise ValueError("observation_frame_count_mismatch")
    for sample, row in zip(sample_rows, observations):
        if (row.get("frame") != sample["global_frame"]
                or row.get("source_sha256") != sample["sha256"]
                or row.get("strategy_eligible") is not False):
            raise ValueError("observation_source_or_order_mismatch")
    flattened = [event for row in observations for event in row["glyph_transitions"]]
    if flattened != pipeline.get("events"):
        raise ValueError("pipeline_event_inventory_mismatch")
    rows_by_frame = {row["frame"]: row for row in observations}
    opportunity_rows = []
    for event in flattened:
        hand = next((item for item in hands
                     if item["start_frame"] <= event["frame"] <= item["end_frame"]),
                    None)
        if hand is None or hand["temporal_complete"] is not True:
            continue
        actor_frame = next((frame for frame in range(
            event["frame"], max(hand["start_frame"], event["frame"] - 12) - 1, -1)
            if rows_by_frame[frame].get("current_actor") == event["slot"]), None)
        actor_row = rows_by_frame[actor_frame] if actor_frame is not None else None
        opportunity_rows.append({
            "hand_id": hand["hand_id"], "action_frame": event["frame"],
            "actor_slot": event["slot"], "glyph": event["glyph"],
            "source_sha256": samples_by_frame[event["frame"]]["sha256"],
            "actor_match_frame": actor_frame,
            "actor_match_within_previous_12_frames": actor_frame is not None,
            "candidate_predecision_pot": (
                actor_row["pot"].get("value") if actor_row else None),
            "candidate_board_count": (
                actor_row.get("board_count") if actor_row else None),
            "legal_actions": None, "amount": None,
            "audit_status": "CANDIDATE_NOT_REVIEWED_DECISION_TRUTH",
        })
    complete = [hand for hand in hands if hand["temporal_complete"]]
    report = {
        "schema_version": 1,
        "status": "DEVELOPMENT_EVIDENCE_ONLY_NOT_REAL_CALIBRATION",
        "source_session_id": registry["source_session_id"],
        "capture_path": registry["capture_path"], "emulator_used": False,
        "development_window": registry["window"],
        "frame_count": len(observations), "registered_hands": hands,
        "temporally_complete_hand_candidates": len(complete),
        "incomplete_tail_hands": len(hands) - len(complete),
        "all_pipeline_glyph_candidates": len(flattened),
        "complete_hand_action_candidates": len(opportunity_rows),
        "opponent_action_candidates": sum(
            row["actor_slot"] != 4 for row in opportunity_rows),
        "actor_match_within_previous_12_frames": sum(
            row["actor_match_within_previous_12_frames"]
            for row in opportunity_rows),
        "complete_legal_action_menus": 0, "reviewed_decision_truth_rows": 0,
        "opportunities": opportunity_rows,
        "blockers": [
            "one_recording_session_cannot_supply_session_disjoint_validation",
            "complete_legal_action_menus_not_reviewed",
            "glyph_confirmation_is_not_exact_action_time_or_semantic_truth",
            "action_amounts_not_bound_to_each_candidate",
            "first_complete_hand_opening_contains_unresolved_special_bomb_pot",
            "recording_tail_hand_is_incomplete",
        ],
        "all_decisions_coverage_certified": False,
        "ready_for_opponent_calibration": False,
        "strategy_eligible": False, "advice_emitted": False,
        "ldplayer_or_emulator_supported": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "audit", "split-plan", "samples", "pipeline",
                 "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.registry, args.audit, args.split_plan,
                         args.samples, args.pipeline)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"AA8 offline evidence rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "status", "frame_count", "temporally_complete_hand_candidates",
        "complete_hand_action_candidates", "opponent_action_candidates",
        "ready_for_opponent_calibration")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
