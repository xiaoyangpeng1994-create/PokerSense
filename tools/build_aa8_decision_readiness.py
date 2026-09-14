"""Build the current AA8 blocked opportunity-review queue from JSON evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from poker_engine.strategy.decision_opportunities_v1 import (
    MODES, audit_decision_opportunities,
)
from tools.audit_decision_opportunities import unique_object


PROTOCOL_KEYS = {
    "schema_version", "status", "registry_sha256", "review_config_sha256",
    "review_result_sha256", "v2_report_sha256", "observations_sha256",
    "expected_reviewed_candidates", "expected_matched_actions",
    "expected_false_candidates", "expected_opponent_matches",
    "expected_hero_matches", "constraints",
}
CONSTRAINTS = [
    "json_and_jsonl_metadata_only_no_media_read",
    "all_reviewed_matches_seed_queue_but_do_not_prove_all_opportunities",
    "glyph_confirmation_is_not_predecision_or_action_onset",
    "seat_is_not_stable_player_identity",
    "legal_menus_rules_and_special_modes_remain_unknown",
    "one_recording_session_cannot_be_train_and_validation",
    "model_fit_strategy_advice_and_live_use_forbidden",
]
INPUT_HASH_KEYS = (
    "registry_sha256", "review_config_sha256", "review_result_sha256",
    "v2_report_sha256", "observations_sha256",
)
PROTOCOL_COUNT_KEYS = (
    "expected_reviewed_candidates", "expected_matched_actions",
    "expected_false_candidates", "expected_opponent_matches",
    "expected_hero_matches",
)
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _snapshot(path):
    before = path.lstat()
    if (path.is_symlink() or _is_reparse(before)
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
        raise ValueError("readiness_input_must_be_plain_single_link_file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise ValueError("readiness_input_exceeds_size_limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()

    def identity(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                value.st_mode, value.st_nlink,
                getattr(value, "st_file_attributes", 0))

    if (identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or identity(after) != identity(final)
            or path.is_symlink() or _is_reparse(final)):
        raise ValueError("readiness_input_changed_during_read")
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def _decode(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("utf8_bom_forbidden")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("utf8_readiness_input_required") from exc


def _load(raw):
    return json.loads(_decode(raw), object_pairs_hook=unique_object)


def _action(label):
    kind = label["actual_action"]
    semantics = "none" if kind in ("check", "fold") else (
        "additional" if kind in ("call", "all_in") else "additional_candidate")
    return {"action": kind, "amount": label["amount"],
            "amount_semantics": semantics}


def _empty_snapshot():
    return {
        "dealer_seat": None, "board": None, "actor_status": None,
        "actor_stack": None, "actor_street_committed": None,
        "actor_hand_committed": None, "current_bet": None, "pot_before": None,
        "to_call": None, "minimum_raise_increment": None,
        "betting_reopened": None,
        "public_history_through_sequence": None, "state_quality": "UNKNOWN",
        "decision_price_fraction": None,
    }


def _evidence(label, observation, review_hash):
    return {
        "actor_frame": None, "actor_pts": None, "actor_sha256": None,
        "menu_frame": None, "menu_pts": None, "menu_sha256": None,
        "state_frame": None, "state_pts": None, "state_sha256": None,
        "action_onset_frame": None, "action_onset_pts": None,
        "action_onset_sha256": None,
        "action_confirmation_frame": label["action_frame"],
        "action_confirmation_pts": str(observation["pts_seconds"]),
        "action_confirmation_sha256": observation["source_sha256"],
        "review_bundle_sha256": review_hash,
    }


def _special(hand, review_hash):
    values = {mode: "UNKNOWN" for mode in MODES}
    if hand["opening_state_status"] == "PARTIAL_SPECIAL_BOMB_POT":
        values["bomb"] = "PRESENT"
    return {**values, "block_state_updates": True,
            "evidence_sha256": review_hash}


def build_documents(protocol, registry, review, result, report, observations,
                    bindings):
    if (not isinstance(protocol, dict) or set(protocol) != PROTOCOL_KEYS
            or type(protocol["schema_version"]) is not int
            or protocol["schema_version"] != 1
            or protocol["status"] != (
                "CURRENT_ACTION_CANDIDATES_NOT_ALL_OPPORTUNITIES_OR_DATA")
            or protocol["constraints"] != CONSTRAINTS
            or any(type(protocol[key]) is not int or protocol[key] < 0
                   for key in PROTOCOL_COUNT_KEYS)):
        raise ValueError("exact_readiness_protocol_required")
    if (not isinstance(bindings, dict)
            or set(bindings) != set(INPUT_HASH_KEYS)):
        raise ValueError("exact_readiness_input_hash_bindings_required")
    for key in INPUT_HASH_KEYS:
        if protocol[key] != bindings[key]:
            raise ValueError("readiness_input_hash_mismatch:" + key)
    if (result["labels"] != review["labels"]
            or result["source_samples_sha256"] != registry["samples_sha256"]
            or report["observations_sha256"] != protocol["observations_sha256"]
            or report["frames"] != registry["window"]["last_frame"]
            - registry["window"]["first_frame"] + 1):
        raise ValueError("review_registry_v2_report_identity_mismatch")
    if not isinstance(observations, list) or len(observations) != report["frames"]:
        raise ValueError("observation_count_differs_from_v2_report")
    by_frame = {}
    for row in observations:
        frame = row.get("frame")
        if (type(frame) is not int or frame in by_frame
                or not isinstance(row.get("source_sha256"), str)
                or row.get("strategy_eligible") is not False):
            raise ValueError("ordered_source_bound_observations_required")
        by_frame[frame] = row
    if list(by_frame) != list(range(
            registry["window"]["first_frame"],
            registry["window"]["last_frame"] + 1)):
        raise ValueError("contiguous_observation_frames_required")
    complete = {hand["hand_id"]: hand for hand in registry["hands"]
                if hand["temporal_complete"]}
    matched = [label for label in result["labels"]
               if label["review_status"] == "MATCH_VISIBLE_COMPLETED_ACTION"]
    false_rows = [label for label in result["labels"] if label not in matched]
    expected = {
        "expected_reviewed_candidates": len(result["labels"]),
        "expected_matched_actions": len(matched),
        "expected_false_candidates": len(false_rows),
        "expected_opponent_matches": sum(row["actor_slot"] != 4 for row in matched),
        "expected_hero_matches": sum(row["actor_slot"] == 4 for row in matched),
    }
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("review_counts_differ_from_protocol")
    v2_complete = [(row["frame"], row["slot"], row["glyph"])
                   for row in report["events"]
                   if any(hand["start_frame"] <= row["frame"] <= hand["end_frame"]
                          for hand in complete.values())]
    reviewed = [(row["action_frame"], row["actor_slot"], row["candidate_glyph"])
                for row in matched]
    if v2_complete != reviewed:
        raise ValueError("v2_events_do_not_equal_reviewed_matches")

    session_id = registry["source_session_id"]
    rule_id = "aa8-real-room-rules-UNKNOWN"
    unknown_rule = {
        "rule_profile_id": rule_id, "rule_fingerprint": None, "table_size": 8,
        "small_blind": None, "big_blind": None, "ante": None,
        "ante_mode": "unknown", "straddle_mode": "unknown",
        "straddle_amount": None, "rake_percent": None, "rake_cap_bb": None,
        "rake_application": "unknown", "rake_rounding": "unknown",
        "rake_distribution": "unknown", "minimum_chip": None,
        "verification_status": "unknown", "source": "unknown",
        "source_sha256": None, "independent_review_status": "PENDING",
    }
    hand_counts = {hid: sum(row["hand_id"] == hid for row in matched)
                   for hid in complete}
    all_hands = []
    for hand in complete.values():
        hid = hand["hand_id"]
        all_hands.append({
            "hand_id": hid, "session_id": session_id,
            "start_frame": hand["start_frame"],
            "last_gameplay_frame": hand["last_gameplay_frame"],
            "last_gameplay_pts": str(by_frame[
                hand["last_gameplay_frame"]]["pts_seconds"]),
            "end_frame": hand["end_frame"],
            "next_start_frame": hand["next_start_frame"],
            "temporal_complete": hand["temporal_complete"],
            "dealer_seat": None, "occupied_seats": [],
            "seat_player_map": {str(i): None for i in range(8)},
            "table_size": 8, "rule_profile_id": rule_id,
            "rule_fingerprint": None, "observed_optional_straddle": None,
            "special_mode_summary": {
                mode: ("PRESENT" if mode == "bomb" and hand[
                    "opening_state_status"] == "PARTIAL_SPECIAL_BOMB_POT"
                       else "UNKNOWN") for mode in MODES},
            "public_action_timeline_sha256": protocol["observations_sha256"],
            "expected_opportunity_count": hand_counts.get(hid, 0),
            "coverage_status": "ACTION_CANDIDATES_ONLY",
            "timeline_binding_status": "ACTION_CANDIDATES_ONLY",
        })
    ordered = sorted(matched, key=lambda row: (
        registry["hands"].index(next(
            hand for hand in registry["hands"] if hand["hand_id"] == row["hand_id"])),
        row["action_frame"], row["review_id"]))
    sequence = {}
    ledger, details = [], []
    review_hash = protocol["review_result_sha256"]
    for label in ordered:
        hid = label["hand_id"]
        sequence[hid] = sequence.get(hid, 0) + 1
        identity = {
            "opportunity_id": "candidate-" + label["review_id"],
            "canonical_opportunity_id": (
                session_id + ":" + hid + ":candidate:" + label["review_id"]),
            "session_id": session_id, "hand_id": hid,
            "decision_sequence": sequence[hid], "actor_player_id": None,
            "actor_seat": label["actor_slot"], "is_hero": label["actor_slot"] == 4,
        }
        ledger.append(identity)
        details.append({
            **identity, "position": None, "street": label["street"],
            "table_size": 8, "active_count": None, "pot_eligible_seats": [],
            "pending_action_seats": [],
            "rule_fingerprint": None, "predecision_snapshot": _empty_snapshot(),
            "evidence": _evidence(label, by_frame[label["action_frame"]], review_hash),
            "legal_menu": None, "observed_action": _action(label),
            "special_mode_state": _special(complete[hid], review_hash),
            "rule_binding_status": "UNKNOWN", "row_status": "UNKNOWN",
            "audit_status": "reviewed_visible_action_only",
            "unknown_reasons": [
                "opportunity_census_not_reviewed",
                "stable_player_identity_unknown",
                "predecision_causal_anchor_unknown",
                "complete_legal_menu_unknown",
                "real_rule_fingerprint_unknown",
                "special_mode_state_unknown",
                "observed_amount_candidate_not_canonical",
            ],
            "author_review": "VISIBLE_ACTION_ONLY",
            "independent_review": "VISIBLE_ACTION_ONLY",
        })
    dataset = {
        "schema_version": 1,
        "dataset_id": "aa8-current-action-candidates-NOT-DATA-v1",
        "source_kind": "reviewed_physical_capture_card",
        "dataset_scope": {
            "calibration_unit": "per_stable_opponent", "target_player_ids": [],
            "hero_player_id": None,
            "included_streets": list(("preflop", "flop", "turn", "river")),
            "included_active_counts": list(range(2, 9)),
            "included_modes": ["ordinary"],
            "contiguous_session_policy": (
                "all_complete_hands_and_explicit_censored_edges"),
            "all_table_decisions_required": True,
        },
        "platform_binding": {"platform_id": "aa_poker",
                             "capture_path": "physical_phone_capture_card",
                             "emulator_used": False},
        "rule_profiles": [unknown_rule],
        "split_protocol": {
            "frozen_before_label_review": False,
            "training_session_ids": [session_id], "validation_session_ids": [],
            "split_evidence_sha256": None,
        },
        "coverage_review": {
            "status": "INCOMPLETE", "author_reviewer": "existing-action-review",
            "independent_reviewer": None, "evidence_sha256": review_hash,
            "expected_opportunity_count": len(ledger),
            "listed_opportunity_count": len(ledger),
        },
        "sessions": [{
            "session_id": session_id,
            "source_recording_group_id": session_id, "split": "training",
            "platform_id": "aa_poker", "rule_profile_id": rule_id,
            "rule_fingerprint": None,
            "source_recording_sha256": None,
            "source_audit_sha256": registry["source_audit_sha256"],
            "sample_manifest_sha256": registry["samples_sha256"],
            "first_frame": registry["window"]["first_frame"],
            "last_frame": registry["window"]["last_frame"],
            "start_pts": registry["window"]["start_pts"],
            "end_pts_exclusive": registry["window"]["end_pts_exclusive"],
            "complete_hand_ids": list(complete),
            "censored_edge_intervals": [
                {"first_frame": registry["window"]["first_frame"],
                 "last_frame": min(
                     hand["start_frame"] for hand in complete.values()) - 1,
                 "reason": "leading_censored_context"},
                {"first_frame": next(hand["start_frame"] for hand in registry["hands"]
                 if not hand["temporal_complete"]),
                 "last_frame": registry["window"]["last_frame"],
                 "reason": "trailing_incomplete_hand"}],
            "coverage_status": "ACTION_CANDIDATES_ONLY",
            "coverage_evidence_sha256": review_hash,
        }],
        "participants": [], "hands": all_hands,
        "opportunity_ledger": ledger, "opportunities": details,
    }
    audit = audit_decision_opportunities(dataset)
    return {
        "scope": "AA8_CURRENT_ACTION_CANDIDATE_READINESS_NOT_DATA",
        "input_hashes": {key: protocol[key] for key in INPUT_HASH_KEYS},
        "candidate_inventory": {
            "reviewed": len(result["labels"]), "matched": len(matched),
            "false": len(false_rows), "opponent_matched": expected[
                "expected_opponent_matches"],
            "hero_matched": expected["expected_hero_matches"],
            "all_actual_opportunities_reviewed": False,
        },
        "dataset": dataset, "audit": audit,
        "strategy_eligible": False, "advice_emitted": False,
        "model_fit_executed": False,
    }


def build(protocol_path, registry_path, review_path, review_result_path,
          report_path, observations_path):
    paths = {
        "protocol": protocol_path,
        "registry_sha256": registry_path,
        "review_config_sha256": review_path,
        "review_result_sha256": review_result_path,
        "v2_report_sha256": report_path,
        "observations_sha256": observations_path,
    }
    snapshots = {key: _snapshot(path) for key, path in paths.items()}
    protocol = _load(snapshots["protocol"][0])
    registry = _load(snapshots["registry_sha256"][0])
    review = _load(snapshots["review_config_sha256"][0])
    result = _load(snapshots["review_result_sha256"][0])
    report = _load(snapshots["v2_report_sha256"][0])
    lines = _decode(snapshots["observations_sha256"][0]).splitlines()
    if not lines or len(lines) > 10000:
        raise ValueError("bounded_nonempty_observations_required")
    observations = [json.loads(line, object_pairs_hook=unique_object)
                    for line in lines]
    bindings = {key: value[1] for key, value in snapshots.items()
                if key != "protocol"}
    return build_documents(
        protocol, registry, review, result, report, observations, bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "registry", "review", "review-result", "v2-report",
                 "observations", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        report = build(args.protocol, args.registry, args.review,
                       args.review_result, args.v2_report, args.observations)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"AA8 decision readiness rejected: {exc}\n")
    print(json.dumps({"candidate_inventory": report["candidate_inventory"],
                      "data_readiness": report["audit"]["data_readiness"],
                      "eligible_target_count": report["audit"][
                          "eligible_target_count"],
                      "blockers": report["audit"]["blockers"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
