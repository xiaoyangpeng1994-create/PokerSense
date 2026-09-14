"""Enumerate AA8 actor-cue episodes from bound development JSONL only."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from tools.audit_decision_opportunities import unique_object


SHA256 = re.compile(r"[0-9a-f]{64}")
KEYS = {
    "schema_version", "status", "registry_sha256", "review_config_sha256",
    "review_result_sha256", "v2_report_sha256", "observations_sha256",
    "expected_frames", "expected_complete_hands", "expected_reviewed_matches",
    "standard_confirmation_lag_frames",
    "maximum_candidate_confirmation_lag_frames", "constraints",
}
CONSTRAINTS = [
    "json_jsonl_only_no_media_read",
    "contiguous_same_actor_frames_form_one_raw_episode",
    "unknown_frames_split_episodes_and_are_not_silently_filled",
    "reviewed_action_binding_is_one_to_one_same_hand_and_seat",
    "glyph_confirmation_lag_is_not_action_onset_or_predecision_truth",
    "unmatched_episodes_and_actions_are_retained",
    "episode_census_does_not_prove_all_actual_opportunities",
    "model_fit_strategy_advice_and_live_use_forbidden",
]


def _is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _snapshot(path, maximum=64 * 1024 * 1024):
    before = path.lstat()
    if (path.is_symlink() or _is_reparse(before)
            or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
        raise ValueError("actor_episode_input_must_be_plain_file")
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
            if size > maximum:
                raise ValueError("actor_episode_input_exceeds_size_limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()

    def identity(value):
        return (value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_mode, value.st_nlink,
                getattr(value, "st_file_attributes", 0))

    if (identity(before) != identity(opened)
            or identity(opened) != identity(after)
            or identity(after) != identity(final)
            or path.is_symlink() or _is_reparse(final)):
        raise ValueError("actor_episode_input_changed_during_read")
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def _decode(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("utf8_bom_forbidden")
    return raw.decode("utf-8", errors="strict")


def _load(raw):
    return json.loads(_decode(raw), object_pairs_hook=unique_object)


def _pts(value):
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError("finite_decimal_pts_required") from None
    if not result.is_finite() or result < 0:
        raise ValueError("finite_decimal_pts_required")
    return result


def _episode(hand_id, sequence, actor, rows):
    first, last = rows[0], rows[-1]
    board_counts = {row.get("board_count") for row in rows}
    street = ({0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(
        next(iter(board_counts))) if len(board_counts) == 1 else None)
    return {
        "episode_id": f"{hand_id}:actor-episode:{sequence:03d}",
        "hand_id": hand_id, "episode_sequence": sequence,
        "actor_slot": actor, "is_hero": actor == 4,
        "first_frame": first["frame"], "last_frame": last["frame"],
        "first_pts": str(first["pts_seconds"]),
        "last_pts": str(last["pts_seconds"]),
        "first_source_sha256": first["source_sha256"],
        "last_source_sha256": last["source_sha256"],
        "frame_count": len(rows), "reviewed_action": None,
        "street_candidate": street,
        "street_consistent": street is not None,
        "street_binding_status": None,
        "confirmation_lag_frames": None,
        "status": "UNMATCHED_ACTOR_EPISODE_CANDIDATE",
        "predecision_truth": None, "legal_menu": None,
        "actual_opportunity_verified": False,
    }


def build_documents(protocol, registry, review, result, report, rows, bindings):
    if (not isinstance(protocol, dict) or set(protocol) != KEYS
            or type(protocol["schema_version"]) is not int
            or protocol["schema_version"] != 1
            or protocol["status"] != (
                "ACTOR_EPISODES_ARE_CANDIDATES_NOT_ALL_OPPORTUNITIES")
            or protocol["constraints"] != CONSTRAINTS
            or type(protocol["expected_frames"]) is not int
            or protocol["expected_frames"] <= 0
            or type(protocol["expected_complete_hands"]) is not int
            or protocol["expected_complete_hands"] < 0
            or type(protocol["expected_reviewed_matches"]) is not int
            or protocol["expected_reviewed_matches"] < 0
            or type(protocol["standard_confirmation_lag_frames"]) is not int
            or type(protocol[
                "maximum_candidate_confirmation_lag_frames"]) is not int
            or not 0 < protocol["standard_confirmation_lag_frames"] <= protocol[
                "maximum_candidate_confirmation_lag_frames"]):
        raise ValueError("exact_actor_episode_protocol_required")
    if set(bindings) != {
            "registry_sha256", "review_config_sha256", "review_result_sha256",
            "v2_report_sha256", "observations_sha256"}:
        raise ValueError("exact_actor_episode_input_bindings_required")
    for key, digest in bindings.items():
        if digest != protocol[key]:
            raise ValueError("actor_episode_input_hash_mismatch:" + key)
    if (result.get("labels") != review.get("labels")
            or result.get("source_samples_sha256") != registry.get("samples_sha256")
            or report.get("observations_sha256") != protocol["observations_sha256"]
            or report.get("frames") != protocol["expected_frames"]):
        raise ValueError("actor_episode_source_documents_disagree")
    window = registry["window"]
    frames = [row.get("frame") for row in rows]
    if (len(rows) != protocol["expected_frames"]
            or frames != list(range(window["first_frame"], window["last_frame"] + 1))):
        raise ValueError("contiguous_bound_observation_window_required")
    previous_pts = None
    for row in rows:
        if (not SHA256.fullmatch(str(row.get("source_sha256", "")))
                or row.get("strategy_eligible") is not False
                or type(row.get("scene_supported")) is not bool):
            raise ValueError("source_bound_nonstrategy_observation_required")
        point = _pts(row.get("pts_seconds"))
        if previous_pts is not None and point <= previous_pts:
            raise ValueError("observation_pts_must_be_strictly_increasing")
        previous_pts = point
        actor = row.get("current_actor")
        if actor is not None and (type(actor) is not int or not 0 <= actor < 8):
            raise ValueError("current_actor_must_be_none_or_exact_slot")
        evidence = row.get("actor_evidence")
        if (not isinstance(evidence, dict)
                and not (actor is None and row["scene_supported"] is False)):
            raise ValueError("actor_evidence_object_required")
        if isinstance(evidence, dict) and actor is None and (
                evidence.get("actor") is not None
                or evidence.get("hero_turn") is True):
            raise ValueError("unknown_actor_conflicts_with_actor_evidence")
    complete = [hand for hand in registry["hands"] if hand["temporal_complete"]]
    if len(complete) != protocol["expected_complete_hands"]:
        raise ValueError("complete_hand_count_differs_from_protocol")
    by_frame = {row["frame"]: row for row in rows}
    episodes, unknown_spans = [], []
    complete_frame_count = actor_frame_count = 0
    for hand in complete:
        hand_rows = [by_frame[frame] for frame in range(
            hand["start_frame"], hand["last_gameplay_frame"] + 1)]
        complete_frame_count += len(hand_rows)
        unknown_kind, unknown_rows = None, []

        def close_unknown():
            nonlocal unknown_rows
            if unknown_rows:
                unknown_spans.append({
                    "hand_id": hand["hand_id"], "classification": unknown_kind,
                    "first_frame": unknown_rows[0]["frame"],
                    "last_frame": unknown_rows[-1]["frame"],
                    "first_pts": str(unknown_rows[0]["pts_seconds"]),
                    "last_pts": str(unknown_rows[-1]["pts_seconds"]),
                    "first_source_sha256": unknown_rows[0]["source_sha256"],
                    "last_source_sha256": unknown_rows[-1]["source_sha256"],
                    "frame_count": len(unknown_rows),
                    "actual_opportunity_interpretation": None})
                unknown_rows = []

        for row in hand_rows:
            value = row.get("current_actor")
            kind = None if type(value) is int else (
                "SUPPORTED_ACTOR_UNKNOWN" if row["scene_supported"]
                else "UNSUPPORTED_SCENE")
            if kind != unknown_kind:
                close_unknown()
                unknown_kind = kind
            if kind is not None:
                unknown_rows.append(row)
        close_unknown()
        actor, episode_rows, sequence = None, [], 0

        def close():
            nonlocal episode_rows, sequence
            if episode_rows:
                sequence += 1
                episodes.append(_episode(
                    hand["hand_id"], sequence, actor, episode_rows))
                episode_rows = []

        for row in hand_rows:
            value = row.get("current_actor")
            value = value if type(value) is int and 0 <= value < 8 else None
            if value is not None and row["scene_supported"] is not True:
                raise ValueError("actor_cannot_exist_in_unsupported_scene")
            evidence = row.get("actor_evidence")
            if value is not None and not (
                    isinstance(evidence, dict)
                    and (evidence.get("actor") == value
                         or value == 4 and evidence.get("hero_turn") is True)):
                raise ValueError("actor_value_and_evidence_disagree")
            if value != actor:
                close()
                actor = value
            if actor is not None:
                episode_rows.append(row)
                actor_frame_count += 1
        close()
    matched = [label for label in result["labels"]
               if label["review_status"] == "MATCH_VISIBLE_COMPLETED_ACTION"]
    if len(matched) != protocol["expected_reviewed_matches"]:
        raise ValueError("reviewed_match_count_differs_from_protocol")
    complete_events = [(row["frame"], row["slot"], row["glyph"])
                       for row in report["events"]
                       if any(hand["start_frame"] <= row["frame"]
                              <= hand["last_gameplay_frame"]
                              for hand in complete)]
    review_events = [(row["action_frame"], row["actor_slot"],
                      row["candidate_glyph"]) for row in matched]
    if complete_events != review_events:
        raise ValueError("v2_complete_events_differ_from_reviewed_matches")
    used, unbound = set(), []
    standard = delayed = 0
    maximum = protocol["maximum_candidate_confirmation_lag_frames"]
    standard_limit = protocol["standard_confirmation_lag_frames"]
    for label in matched:
        candidates = [
            (index, episode) for index, episode in enumerate(episodes)
            if index not in used and episode["hand_id"] == label["hand_id"]
            and episode["actor_slot"] == label["actor_slot"]
            and episode["last_frame"] <= label["action_frame"]
            and label["action_frame"] - episode["last_frame"] <= maximum]
        if not candidates:
            unbound.append({
                "review_id": label["review_id"], "hand_id": label["hand_id"],
                "actor_slot": label["actor_slot"],
                "action_frame": label["action_frame"],
                "actual_action": label["actual_action"]})
            continue
        index, episode = max(candidates, key=lambda item: item[1]["last_frame"])
        lag = label["action_frame"] - episode["last_frame"]
        status = ("MATCHED_WITHIN_STANDARD_CONFIRMATION_LAG"
                  if lag <= standard_limit
                  else "MATCHED_DELAYED_CONFIRMATION_CANDIDATE")
        street_binding = (
            "MATCH" if episode["street_candidate"] == label["street"]
            else "EPISODE_STREET_AMBIGUOUS" if not episode["street_consistent"]
            else "MISMATCH")
        if street_binding == "MISMATCH":
            unbound.append({
                "review_id": label["review_id"], "hand_id": label["hand_id"],
                "actor_slot": label["actor_slot"],
                "action_frame": label["action_frame"],
                "actual_action": label["actual_action"],
                "reason": "episode_street_candidate_mismatch"})
            continue
        used.add(index)
        standard += int(lag <= standard_limit)
        delayed += int(lag > standard_limit)
        episode.update(
            reviewed_action={key: label[key] for key in (
                "review_id", "action_frame", "candidate_glyph", "actual_action",
                "amount", "amount_status", "street")},
            confirmation_lag_frames=lag, status=status)
        episode["street_binding_status"] = street_binding
    unmatched = [episode["episode_id"] for index, episode in enumerate(episodes)
                 if index not in used]
    actor_segments = [{
        "hand_id": row["hand_id"],
        "classification": "ACTOR_EPISODE_CANDIDATE",
        "episode_id": row["episode_id"], "actor_slot": row["actor_slot"],
        "first_frame": row["first_frame"], "last_frame": row["last_frame"],
        "first_pts": row["first_pts"], "last_pts": row["last_pts"],
        "first_source_sha256": row["first_source_sha256"],
        "last_source_sha256": row["last_source_sha256"],
        "frame_count": row["frame_count"],
        "actual_opportunity_interpretation": None}
        for row in episodes]
    actor_segments.extend({**row, "episode_id": None, "actor_slot": None}
                          for row in unknown_spans)
    hand_order = {hand["hand_id"]: index for index, hand in enumerate(complete)}
    actor_segments.sort(key=lambda row: (
        hand_order[row["hand_id"]], row["first_frame"]))
    for hand in complete:
        cursor = hand["start_frame"]
        selected = [row for row in actor_segments if row["hand_id"] == hand["hand_id"]]
        for segment in selected:
            if segment["first_frame"] != cursor:
                raise ValueError("actor_segments_do_not_partition_gameplay")
            cursor = segment["last_frame"] + 1
        if cursor != hand["last_gameplay_frame"] + 1:
            raise ValueError("actor_segments_do_not_partition_gameplay")
    return {
        "schema_version": 1,
        "status": "DEVELOPMENT_ACTOR_EPISODE_CANDIDATES_ONLY",
        "input_hashes": {key: protocol[key] for key in bindings},
        "source_session_id": registry["source_session_id"],
        "complete_hand_count": len(complete),
        "complete_hand_frame_count": complete_frame_count,
        "actor_candidate_frame_count": actor_frame_count,
        "unknown_actor_frame_count": complete_frame_count - actor_frame_count,
        "supported_actor_unknown_frame_count": sum(
            row["frame_count"] for row in unknown_spans
            if row["classification"] == "SUPPORTED_ACTOR_UNKNOWN"),
        "unsupported_scene_frame_count": sum(
            row["frame_count"] for row in unknown_spans
            if row["classification"] == "UNSUPPORTED_SCENE"),
        "unknown_span_count": len(unknown_spans), "unknown_spans": unknown_spans,
        "actor_segment_count": len(actor_segments),
        "actor_segments": actor_segments,
        "actor_episode_count": len(episodes),
        "reviewed_action_candidate_count": len(matched),
        "bound_reviewed_action_count": len(used),
        "standard_lag_match_count": standard,
        "delayed_lag_match_count": delayed,
        "unmatched_actor_episode_count": len(unmatched),
        "unmatched_actor_episode_ids": unmatched,
        "unbound_reviewed_action_count": len(unbound),
        "unbound_reviewed_actions": unbound,
        "episodes": episodes,
        "all_actual_decision_opportunities_reviewed": False,
        "episode_detector_accuracy_verified": False,
        "predecision_truth_complete": False,
        "complete_legal_menus": 0,
        "ready_for_offline_calibration": False,
        "model_fit_executed": False, "strategy_eligible": False,
        "advice_emitted": False, "live_use": False,
    }


def build(protocol_path, registry_path, review_path, review_result_path,
          report_path, observations_path):
    paths = {
        "protocol": protocol_path, "registry_sha256": registry_path,
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
    rows = [json.loads(line, object_pairs_hook=unique_object) for line in lines]
    bindings = {key: value[1] for key, value in snapshots.items()
                if key != "protocol"}
    return build_documents(
        protocol, registry, review, result, report, rows, bindings)


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
        parser.exit(2, f"AA8 actor episode audit rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "status", "actor_episode_count", "reviewed_action_candidate_count",
        "bound_reviewed_action_count",
        "standard_lag_match_count", "delayed_lag_match_count",
        "unmatched_actor_episode_count", "unbound_reviewed_action_count")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
