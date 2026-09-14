"""Verify a complete manual review of AA8 development action candidates."""

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from tools.aa8_offline_session_evidence import (
    parse_safe_manifest, sha256, verify_manifest_hashes,
)


TOP_KEYS = {
    "schema_version", "status", "source_report_sha256", "source_samples_sha256",
    "review_scope", "labels", "constraints",
}
LABEL_KEYS = {
    "review_id", "hand_id", "action_frame", "actor_slot", "candidate_glyph",
    "review_status", "actual_action", "amount", "amount_status", "duplicate_of",
    "street", "legal_actions", "evidence_frames",
}
MATCH = "MATCH_VISIBLE_COMPLETED_ACTION"
POST_ACTION = "FALSE_POSITIVE_POST_ACTION_DISPLAY"
STALE = "FALSE_POSITIVE_STALE_REAPPEARANCE"
FALSE_STATUSES = {POST_ACTION, STALE}
ACTUAL_ACTIONS = {"check", "fold", "call", "bet", "raise", "all_in"}
BOARD_STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"exact_{name}_fields_required")


def amount(value):
    if not isinstance(value, str):
        raise ValueError("exact_amount_decimal_string_required")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("finite_nonnegative_amount_required") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("finite_nonnegative_amount_required")
    return result


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(source_report_path, review_path, samples_dir):
    review, source = load(review_path), load(source_report_path)
    exact(review, TOP_KEYS, "review")
    samples_path = samples_dir / "samples.json"
    if (review["schema_version"] != 1
            or review["status"] != "DEVELOPMENT_AUTHOR_REVIEW_NOT_INDEPENDENT"
            or review["review_scope"] != "all_36_complete_hand_glyph_candidates"
            or sha256(source_report_path) != review["source_report_sha256"]
            or sha256(samples_path) != review["source_samples_sha256"]):
        raise ValueError("review_source_or_scope_mismatch")
    if (not isinstance(review["constraints"], list)
            or not review["constraints"]
            or not all(isinstance(item, str) and item
                       for item in review["constraints"])):
        raise ValueError("explicit_review_constraints_required")
    candidates = source.get("opportunities")
    labels = review["labels"]
    if (not isinstance(candidates, list) or not isinstance(labels, list)
            or len(labels) != len(candidates) or len(labels) != 36):
        raise ValueError("all_36_candidates_must_be_reviewed_once")
    samples = load(samples_path)
    sample_rows = samples.get("samples")
    if not isinstance(sample_rows, list):
        raise ValueError("sample_inventory_required")
    samples_by_frame = {row["global_frame"]: row for row in sample_rows}
    if len(samples_by_frame) != len(sample_rows):
        raise ValueError("unique_sample_frames_required")
    manifest_entries = parse_safe_manifest(samples_dir)
    manifest_hashes = {
        relative: expected for relative, (expected, _) in manifest_entries.items()}
    hand_ranges = {hand["hand_id"]: (hand["start_frame"], hand["end_frame"])
                   for hand in source["registered_hands"]}
    reviewed, evidence_files = {}, set()
    for index, (candidate, label) in enumerate(zip(candidates, labels), 1):
        exact(label, LABEL_KEYS, "label")
        expected_id = f"A{index:02d}"
        identity = (candidate["hand_id"], candidate["action_frame"],
                    candidate["actor_slot"], candidate["glyph"])
        claimed = (label["hand_id"], label["action_frame"], label["actor_slot"],
                   label["candidate_glyph"])
        if label["review_id"] != expected_id or claimed != identity:
            raise ValueError("review_label_order_or_candidate_identity_mismatch")
        action_sample = samples_by_frame.get(candidate["action_frame"])
        if (action_sample is None or candidate.get("source_sha256") != (
                action_sample.get("sha256"))):
            raise ValueError("candidate_source_hash_differs_from_action_frame")
        if (label["legal_actions"] is not None
                or label["street"] not in {"preflop", "flop", "turn", "river"}):
            raise ValueError("legal_menu_must_remain_unknown_and_street_explicit")
        board = candidate["candidate_board_count"]
        if board in BOARD_STREETS and label["street"] != BOARD_STREETS[board]:
            raise ValueError("street_differs_from_causal_candidate_board")
        frames = label["evidence_frames"]
        bounds = hand_ranges.get(label["hand_id"])
        if (not isinstance(frames, list) or len(frames) < 2
                or len(set(frames)) != len(frames)
                or label["action_frame"] not in frames or bounds is None
                or any(type(frame) is not int or not bounds[0] <= frame <= bounds[1]
                       for frame in frames)):
            raise ValueError("ordered_in_hand_action_evidence_required")
        for frame in frames:
            sample = samples_by_frame.get(frame)
            if sample is None:
                raise ValueError("review_evidence_frame_missing")
            expected_path = f"frames/frame_{frame:06d}.png"
            if (sample.get("file") != expected_path
                    or manifest_hashes.get(expected_path) != sample.get("sha256")):
                raise ValueError("review_evidence_path_or_hash_mismatch")
            evidence_files.add(expected_path)
        status = label["review_status"]
        if status == MATCH:
            actual = label["actual_action"]
            if actual not in ACTUAL_ACTIONS or label["duplicate_of"] is not None:
                raise ValueError("matched_action_and_no_duplicate_required")
            allowed = ({label["candidate_glyph"]}
                       if label["candidate_glyph"] != "aggressive"
                       else {"bet", "raise"})
            if actual not in allowed:
                raise ValueError("actual_action_differs_from_visible_glyph")
            chips = amount(label["amount"])
            if (actual in {"check", "fold"}
                    and (chips != 0 or label["amount_status"] != "ZERO_CHIP_ACTION")):
                raise ValueError("passive_zero_chip_action_required")
            if (actual not in {"check", "fold"}
                    and (chips <= 0 or label["amount_status"] != (
                        "VISUAL_STACK_DELTA_MATCH"))):
                raise ValueError("positive_visual_stack_delta_required")
        elif status in FALSE_STATUSES:
            if (label["actual_action"] is not None or label["amount"] is not None
                    or label["amount_status"] is not None):
                raise ValueError("false_positive_cannot_become_action_or_amount")
            duplicate = label["duplicate_of"]
            if status == STALE:
                original = reviewed.get(duplicate)
                if (original is None or original["review_status"] != MATCH
                        or original["hand_id"] != label["hand_id"]
                        or original["action_frame"] >= label["action_frame"]
                        or original["actor_slot"] != label["actor_slot"]
                        or original["candidate_glyph"] != label["candidate_glyph"]):
                    raise ValueError("stale_reappearance_requires_prior_same_action")
            elif duplicate is not None:
                raise ValueError("post_action_display_is_not_duplicate_action")
        else:
            raise ValueError("explicit_match_or_false_positive_status_required")
        reviewed[label["review_id"]] = label
    verify_manifest_hashes(manifest_entries, evidence_files)
    matches = [label for label in labels if label["review_status"] == MATCH]
    false_rows = [label for label in labels if label["review_status"] in FALSE_STATUSES]
    report = {
        "schema_version": 1,
        "status": "DEVELOPMENT_VISIBLE_ACTION_REVIEW_NOT_DECISION_CALIBRATION",
        "source_report_sha256": review["source_report_sha256"],
        "source_samples_sha256": review["source_samples_sha256"],
        "candidates_reviewed": len(labels), "matched_visible_actions": len(matches),
        "false_positive_candidates": len(false_rows),
        "opponent_matched_visible_actions": sum(
            label["actor_slot"] != 4 for label in matches),
        "hero_matched_visible_actions": sum(
            label["actor_slot"] == 4 for label in matches),
        "matched_action_types": dict(Counter(
            label["actual_action"] for label in matches)),
        "false_positive_types": dict(Counter(
            label["review_status"] for label in false_rows)),
        "matched_amount_candidates": len(matches),
        "complete_legal_action_menus": 0,
        "full_decision_truth_rows": 0,
        "labels": labels,
        "blockers": [
            "complete_legal_action_menus_unavailable",
            "review_is_author_development_review_not_independent_labeling",
            "one_recording_session_cannot_supply_session_disjoint_validation",
            "exact_click_time_not_observed_for_opponents",
            "amounts_are_visual_stack_delta_candidates_not_canonical_ledger",
        ],
        "ready_for_opponent_calibration": False,
        "strategy_eligible": False, "advice_emitted": False,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-report", "review", "samples", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.source_report, args.review, args.samples)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"AA8 action review rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "status", "candidates_reviewed", "matched_visible_actions",
        "false_positive_candidates", "opponent_matched_visible_actions",
        "complete_legal_action_menus", "ready_for_opponent_calibration")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
