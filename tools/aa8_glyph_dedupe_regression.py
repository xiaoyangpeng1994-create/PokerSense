"""Compare AA8 glyph de-duplication against frozen development review labels."""

import argparse
import hashlib
import json
from pathlib import Path


KEYS = {
    "schema_version", "status", "old_report_sha256", "new_report_sha256",
    "review_sha256", "registry_sha256", "expected_old_all_events",
    "expected_new_all_events", "expected_complete_hand_candidates",
    "expected_retained_matches", "expected_removed_false_candidates",
    "expected_unchanged_outside_complete_hands", "expected_frames",
    "expected_changed_input_hashes", "expected_removed_input_hashes",
    "expected_added_input_hashes", "expected_unchanged_input_bindings", "constraints",
}
CONSTRAINTS = [
    "same_preregistered_development_frames_and_model_inputs",
    "remove_exactly_reviewed_false_candidates",
    "retain_every_reviewed_match",
    "no_new_events",
    "outside_complete_hands_unchanged",
    "not_independent_accuracy_or_strategy_evidence",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def event(value):
    if (not isinstance(value, dict)
            or set(value) != {"frame", "slot", "glyph"}
            or type(value["frame"]) is not int or type(value["slot"]) is not int
            or not isinstance(value["glyph"], str)):
        raise ValueError("exact_typed_glyph_event_required")
    return value["frame"], value["slot"], value["glyph"]


def input_hashes(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("pipeline_input_hashes_required")
    result = {}
    for raw_path, digest in value.items():
        if (not isinstance(raw_path, str) or not raw_path
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("exact_pipeline_input_hash_required")
        path = raw_path.replace("\\", "/")
        lowered = path.lower()
        markers = [marker for marker in ("/configs/", "/tools/")
                   if marker in lowered]
        key = path[lowered.index(markers[0]) + 1:] if markers else path
        if key in result:
            raise ValueError("duplicate_normalized_pipeline_input_path")
        result[key] = digest
    return result


def analyze(protocol_path, old_path, new_path, review_path, registry_path):
    protocol = load(protocol_path)
    count_keys = {
        "expected_old_all_events", "expected_new_all_events",
        "expected_complete_hand_candidates", "expected_retained_matches",
        "expected_removed_false_candidates",
        "expected_unchanged_outside_complete_hands", "expected_frames",
        "expected_unchanged_input_bindings",
    }
    if (not isinstance(protocol, dict) or set(protocol) != KEYS
            or protocol["schema_version"] != 1
            or protocol["status"] != (
                "DEVELOPMENT_REGRESSION_NOT_ACCURACY_OR_ACCEPTANCE")
            or protocol["constraints"] != CONSTRAINTS
            or any(type(protocol[key]) is not int or protocol[key] < 0
                   for key in count_keys)
            or protocol["expected_frames"] < 1
            or not isinstance(protocol["expected_changed_input_hashes"], dict)
            or not isinstance(protocol["expected_removed_input_hashes"], dict)
            or not protocol["expected_removed_input_hashes"]
            or not isinstance(protocol["expected_added_input_hashes"], dict)
            or not protocol["expected_added_input_hashes"]):
        raise ValueError("exact_development_regression_protocol_required")
    for key, path in (("old_report_sha256", old_path),
                      ("new_report_sha256", new_path),
                      ("review_sha256", review_path),
                      ("registry_sha256", registry_path)):
        if sha256(path) != protocol[key]:
            raise ValueError("regression_input_hash_mismatch:" + key)
    old, new, review, registry = (load(path) for path in (
        old_path, new_path, review_path, registry_path))
    if (old.get("frames") != protocol["expected_frames"]
            or new.get("frames") != protocol["expected_frames"]):
        raise ValueError("pipeline_frame_count_differs_from_protocol")
    old_inputs = input_hashes(old.get("input_hashes"))
    new_inputs = input_hashes(new.get("input_hashes"))
    old_keys, new_keys = set(old_inputs), set(new_inputs)
    removed_inputs = {key: old_inputs[key] for key in old_keys - new_keys}
    added_inputs = {key: new_inputs[key] for key in new_keys - old_keys}
    changed_inputs = {
        key: {"old": old_inputs[key], "new": new_inputs[key]}
        for key in old_keys & new_keys if old_inputs[key] != new_inputs[key]
    }
    if changed_inputs != protocol["expected_changed_input_hashes"]:
        raise ValueError("pipeline_changed_input_hashes_differ_from_protocol")
    if removed_inputs != protocol["expected_removed_input_hashes"]:
        raise ValueError("pipeline_removed_input_hashes_differ_from_protocol")
    if added_inputs != protocol["expected_added_input_hashes"]:
        raise ValueError("pipeline_added_input_hashes_differ_from_protocol")
    unchanged_inputs = len(old_keys & new_keys) - len(changed_inputs)
    if unchanged_inputs != protocol["expected_unchanged_input_bindings"]:
        raise ValueError("pipeline_unchanged_input_count_differs_from_protocol")
    old_events = [event(item) for item in old["events"]]
    new_events = [event(item) for item in new["events"]]
    if (len(set(old_events)) != len(old_events)
            or len(set(new_events)) != len(new_events)):
        raise ValueError("duplicate_event_identity_in_pipeline_report")
    if (any(first[0] > second[0]
            for first, second in zip(old_events, old_events[1:]))
            or any(first[0] > second[0]
                   for first, second in zip(new_events, new_events[1:]))):
        raise ValueError("pipeline_event_chronology_changed")

    def is_subsequence(full, subset):
        cursor = iter(full)
        return all(any(candidate == item for candidate in cursor)
                   for item in subset)

    complete = [hand for hand in registry["hands"] if hand["temporal_complete"]]

    def inside(item):
        return any(hand["start_frame"] <= item[0] <= hand["end_frame"]
                   for hand in complete)

    old_complete_order = [item for item in old_events if inside(item)]
    new_complete_order = [item for item in new_events if inside(item)]
    old_outside_order = [item for item in old_events if not inside(item)]
    new_outside_order = [item for item in new_events if not inside(item)]
    old_complete, new_complete = set(old_complete_order), set(new_complete_order)
    old_outside, new_outside = set(old_outside_order), set(new_outside_order)
    all_labels, matches, false_rows = set(), set(), set()
    all_label_order, match_order = [], []
    for label in review["labels"]:
        item = (label["action_frame"], label["actor_slot"],
                label["candidate_glyph"])
        if item in all_labels:
            raise ValueError("duplicate_review_candidate_identity")
        all_labels.add(item)
        all_label_order.append(item)
        if label["review_status"] == "MATCH_VISIBLE_COMPLETED_ACTION":
            matches.add(item)
            match_order.append(item)
        elif label["review_status"].startswith("FALSE_POSITIVE_"):
            false_rows.add(item)
        else:
            raise ValueError("review_contains_unresolved_candidate")
    expected = {
        "expected_old_all_events": len(old_events),
        "expected_new_all_events": len(new_events),
        "expected_complete_hand_candidates": len(all_labels),
        "expected_retained_matches": len(matches),
        "expected_removed_false_candidates": len(false_rows),
        "expected_unchanged_outside_complete_hands": len(old_outside),
    }
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("regression_count_differs_from_preregistered_protocol")
    if old_complete != all_labels or old_complete_order != all_label_order:
        raise ValueError("old_pipeline_does_not_equal_reviewed_candidate_set")
    if new_complete != matches or new_complete_order != match_order:
        raise ValueError("new_pipeline_misses_match_or_retains_false_candidate")
    if old_complete - new_complete != false_rows:
        raise ValueError("removed_events_are_not_exact_reviewed_false_candidates")
    if old_outside != new_outside or old_outside_order != new_outside_order:
        raise ValueError("events_outside_complete_hands_changed")
    if set(new_events) - set(old_events):
        raise ValueError("dedupe_pipeline_added_new_events")
    if not is_subsequence(old_events, new_events):
        raise ValueError("new_pipeline_event_order_is_not_old_subsequence")
    policy = new.get("glyph_transition_policy")
    if policy != {"stable_frames": 2, "clear_frames": 5,
                  "rapid_change_guard_frames": 5,
                  "unsupported_scene_frames_suspended": True,
                  "unconfirmed_streaks_cross_suspension": False,
                  "stable_clear_required_after_suppression": True,
                  "hand_epoch_reset": (
                      "not_applied_without_authoritative_boundary")}:
        raise ValueError("declared_transition_policy_missing_or_changed")
    return {
        "status": "PASS_DEVELOPMENT_GLYPH_DEDUPE_REGRESSION",
        "old_all_events": len(old_events), "new_all_events": len(new_events),
        "complete_hand_candidates": len(all_labels),
        "retained_reviewed_matches": len(matches),
        "removed_reviewed_false_candidates": len(false_rows),
        "unchanged_outside_complete_hands": len(old_outside),
        "frames": protocol["expected_frames"],
        "changed_input_hashes": changed_inputs,
        "removed_input_hashes": removed_inputs,
        "added_input_hashes": added_inputs,
        "unchanged_input_bindings": unchanged_inputs,
        "removed_events": [list(item) for item in sorted(false_rows)],
        "added_events": [], "missed_reviewed_matches": [],
        "development_only": True, "independent_accuracy": False,
        "strategy_eligible": False, "advice_emitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "old", "new", "review", "registry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.protocol, args.old, args.new,
                         args.review, args.registry)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"AA8 glyph regression rejected: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
