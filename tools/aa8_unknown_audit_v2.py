"""Offline diagnostic of saved development abstentions, with no image access."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.aa8_action_transfer import inventory, sha


def classify(row):
    modes = row.get("special_modes", {})
    reason = row.get("causal_street_wagers_v2", {}).get("reason", "UNSPECIFIED")
    if (modes.get("insurance") == "VISIBLE"
            or modes.get("block_state_updates") is True
            or modes.get("critical_hit_title", {}).get(
                "critical_hit_animation") is True):
        return "POSITIVE_SPECIAL_MODE_PAUSE"
    if row.get("scene_supported") is not True:
        return "UNSUPPORTED_SCENE_NOT_ORDINARY"
    if reason == "collection_or_center_change_invalidates_epoch":
        return "CENTER_CHANGE_OR_MISSING_NOT_DISTINGUISHABLE"
    if reason in ("two_stable_frames_required", "pending_action_not_yet_accounted"):
        return "TEMPORAL_CONFIRMATION_WAIT"
    hand = row.get("hand_transition", {})
    if hand.get("center_deal", {}).get("visible") is True or hand.get("candidate"):
        return "POSITIVE_DEAL_TRANSITION"
    if type(row.get("current_actor")) is int and 0 <= row["current_actor"] < 8:
        return "ACTION_OPPORTUNITY_MODE_UNVERIFIED"
    return "NO_ACTOR_AND_MODE_UNVERIFIED"


def summarize_interval(rows):
    first, last = rows[0], rows[-1]
    categories = Counter(classify(r) for r in rows)
    triggers = []
    for row in rows:
        state = row.get("causal_street_wagers_v2", {})
        if state.get("reason") == "collection_or_center_change_invalidates_epoch":
            logged = "observed_center_v2" in row
            center = (row.get("observed_center_v2") or {}).get("value")
            triggers.append({"frame": row["frame"], "followup_frames_in_interval":
                             last["frame"] - row["frame"],
                             "observed_center": center if logged else "NOT_LOGGED",
                             "null_vs_numeric_change": (
                                 "NULL_READING" if center is None else
                                 "CURRENT_VALUE_PRESENT_BASELINE_NOT_LOGGED") if logged
                             else "UNKNOWN_FROM_SAVED_DATA"})
    return {"first_frame": first["frame"], "last_frame": last["frame"],
            "frames": len(rows), "duration_seconds": float(last["pts_seconds"]) - float(
                first["pts_seconds"]), "categories": dict(categories),
            "reasons": dict(Counter(
                r["causal_street_wagers_v2"].get("reason") for r in rows)),
            "streets": dict(Counter(str(r.get("observed_state_v2", {}).get(
                "street_candidate")) for r in rows)),
            "actors": dict(Counter(str(r.get("current_actor")) for r in rows)),
            "center_invalidation_triggers": triggers,
            "initial_unknown_regions": first["causal_street_wagers_v2"].get(
                "unknown_regions", []), "ordinary_mode_confirmed": False}


def analyze(rows):
    intervals, current = [], []
    unknown, reason_counts, strata = [], Counter(), Counter()
    previous = None
    for row in rows:
        if previous is not None and row["frame"] <= previous:
            raise ValueError("strictly ordered observations required")
        is_unknown = row.get("causal_street_wagers_v2", {}).get(
            "status") == "WAGERS_UNKNOWN"
        if current and (not is_unknown or row["frame"] != previous + 1):
            intervals.append(summarize_interval(current))
            current = []
        if is_unknown:
            current.append(row)
            unknown.append(row)
            reason = row["causal_street_wagers_v2"].get("reason", "UNSPECIFIED")
            reason_counts[reason] += 1
            strata[(classify(row), reason, str(row.get("observed_state_v2", {}).get(
                "street_candidate")), str(row.get("current_actor")))] += 1
        previous = row["frame"]
    if current:
        intervals.append(summarize_interval(current))
    return {"total_frames": len(rows), "unknown_frames": len(unknown),
            "categories": dict(Counter(classify(r) for r in unknown)),
            "reasons": dict(reason_counts), "intervals": intervals,
            "strata": [{"category": k[0], "reason": k[1], "street": k[2],
                        "actor": k[3], "frames": value}
                       for k, value in strata.most_common()],
            "center_triggers": [v for interval in intervals
                                for v in interval["center_invalidation_triggers"]],
            "ordinary_mode_not_inferred_from_missing_special_detection": True,
            "center_values_not_present_in_saved_observations": not any(
                "observed_center_v2" in row for row in rows),
            "full_visual_acceptance": False}


def run(observations, pools, output):
    allowed, audit = {}, None
    for pool in pools:
        meta = json.loads((pool / "samples.json").read_text())
        if audit is not None and meta["audit_sha256"] != audit:
            raise ValueError("mixed recording sources")
        audit = meta["audit_sha256"]
        allowed.update(inventory(pool, audit))
    rows = [json.loads(line) for line in observations.read_text().splitlines()]
    for row in rows:
        if row["source_sha256"] != allowed[row["frame"]]["sha256"]:
            raise ValueError("development metadata source hash mismatch")
    result = analyze(rows)
    result.update(observations_sha256=sha(observations),
                  implementation_sha256=sha(Path(__file__)),
                  source_manifest_sha256={
                      str(p): sha(p / "samples.json") for p in pools},
                  media_read=False, modifications_to_candidate=False)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    keys = ("total_frames", "unknown_frames", "categories", "center_triggers")
    print(json.dumps({k: result[k] for k in keys}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.observations, args.pool, args.output)
