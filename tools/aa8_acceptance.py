"""Offline, fail-closed AA8 release evidence gate (never opens capture/media)."""
import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tools.aa8_holdout_plan import validate_freeze


FIELDS = ("actor", "street_wagers", "actions", "pot", "stacks", "street",
          "hand", "participation", "hero_cards", "board_cards",
          "insurance", "mushroom", "bomb")
SCENARIOS = ("street_transition", "hand_transition", "waiting_seat",
             "unmarked_action", "insurance", "mushroom", "bomb")
ARTIFACTS = ("implementation", "model", "parameters", "dataset", "labels",
             "predictions", "hardware")


def valid_value(field, value):
    def money(amount):
        try:
            number = Decimal(str(amount))
            return number.is_finite() and number >= 0
        except InvalidOperation:
            return False

    if field == "actor":
        return value == "NONE" or type(value) is int and 0 <= value < 8
    if field in ("hero_cards", "board_cards"):
        if field == "hero_cards" and value == {"status": "NOT_APPLICABLE"}:
            return True
        return (isinstance(value, list)
                and len(value) in ((2,) if field == "hero_cards" else (0, 3, 4, 5))
                and all(isinstance(card, str) and len(card) == 2
                        and card[0] in "23456789TJQKA" and card[1] in "cdhs"
                        for card in value) and len(set(value)) == len(value))
    if field in ("street_wagers", "stacks"):
        return (isinstance(value, dict) and set(value) == set(map(str, range(8)))
                and all(money(amount) or amount == {"status": "NOT_APPLICABLE"}
                        for amount in value.values()))
    if field == "actions":
        return isinstance(value, list) and all(isinstance(v, dict) for v in value)
    if field == "pot":
        return money(value)
    if field == "street":
        return value in ("preflop", "flop", "turn", "river", "settlement", "idle")
    if field == "hand":
        return isinstance(value, str) and bool(value)
    if field == "participation":
        return (isinstance(value, dict) and set(value) == set(map(str, range(8)))
                and all(v in ("active", "folded", "all_in", "waiting", "empty")
                        for v in value.values()))
    return isinstance(value, dict) and type(value.get("active")) is bool


def valid_na_context(field, value, fields):
    if field == "hero_cards" and value == {"status": "NOT_APPLICABLE"}:
        participation = fields.get("participation", {})
        street = fields.get("street", {})
        return (street.get("status") == "KNOWN" and street.get("value") == "idle"
                or participation.get("status") == "KNOWN"
                and participation.get("value", {}).get("4") in (
                    "folded", "empty", "waiting"))
    if field not in ("street_wagers", "stacks") or not isinstance(value, dict):
        return True
    context = fields.get("participation", {})
    states = context.get("value", {})
    return all(amount != {"status": "NOT_APPLICABLE"}
               or context.get("status") == "KNOWN" and isinstance(states, dict)
               and states.get(slot) in ("empty", "waiting")
               for slot, amount in value.items())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_exemption(actual, base):
    evidence = actual.get("exemption", {})
    source = evidence.get("source", {})
    path = (base / source.get("path", "")).resolve()
    return (evidence.get("reason") in ("transition", "occluded", "no_action")
            and evidence.get("source_reviewed") is True
            and isinstance(evidence.get("reviewer"), str) and bool(evidence["reviewer"])
            and path.is_file() and digest(path) == source.get("sha256"))


def template():
    """Explicit empty requirements; the resulting document cannot pass."""
    return {"schema_version": 1, "scope": "AA8", "freeze_id": None,
            "frozen_before_predictions": False,
            "artifacts": {key: [] for key in ARTIFACTS},
            "development_sessions": [], "development_intervals": [],
            "hands": [], "hardware_session": None,
            "freeze_manifest": None, "prediction_started_at_utc": None,
            "geometry": {"seat_count": 8, "hero_slot": 4,
                         "slot_order": "clockwise_from_top"},
            "policy": {"hardware_min_seconds": None,
                       "require_independent_hardware_session": False}}


def evaluate(manifest, base):
    """Validate local hashes and compare bound truth/prediction records.

    This is evidence bookkeeping, not authentication of human annotations.
    All hashes must be obtained independently before evaluation. Evidence
    collectors own accurate timing, session identity and coverage annotation.
    """
    issues, bound = [], {}

    def require(condition, reason):
        if not condition:
            issues.append(reason)

    require(manifest.get("schema_version") == 1, "schema_version")
    require(manifest.get("scope") == "AA8", "scope")
    require(manifest.get("geometry") == template()["geometry"], "AA8_geometry_binding")
    require(bool(manifest.get("freeze_id")), "missing_freeze")
    require(manifest.get("frozen_before_predictions") is True,
            "freeze_not_before_prediction")
    frozen = manifest.get("freeze_manifest") or {}
    freeze_path = (base / frozen.get("path", "")).resolve()
    if not freeze_path.is_file() or digest(freeze_path) != frozen.get("sha256"):
        issues.append("freeze_manifest_missing_or_hash_mismatch")
    else:
        freeze_data = json.loads(freeze_path.read_text(encoding="utf-8"))
        issues.extend(validate_freeze(freeze_data, freeze_path.parent,
                                      manifest.get("prediction_started_at_utc")))
        require(bool(manifest.get("prediction_started_at_utc")),
                "prediction_start_timestamp_missing")
        require(freeze_data.get("id") == manifest.get("freeze_id"),
                "freeze_id_mismatch")
        for kind in ("implementation", "model", "parameters"):
            expected_files = {(str((freeze_path.parent / row["path"]).resolve()),
                               row["sha256"]) for row in freeze_data.get("files", [])
                              if row.get("kind") == kind}
            current_files = {(str((base / row["path"]).resolve()), row["sha256"])
                             for row in manifest.get("artifacts", {}).get(kind, [])}
            require(expected_files == current_files, "frozen_artifact_binding:" + kind)
    for kind in ARTIFACTS:
        entries = manifest.get("artifacts", {}).get(kind, [])
        require(bool(entries), "missing_artifact:" + kind)
        bound[kind] = []
        for item in entries:
            path = (base / item.get("path", "")).resolve()
            if not path.is_file() or digest(path) != item.get("sha256"):
                issues.append("artifact_hash:" + kind)
                continue
            bound[kind].append(path)

    def records(kind):
        rows = []
        for path in bound[kind]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, list):
                    raise ValueError("array required")
                rows.extend(value)
            except (ValueError, UnicodeError):
                issues.append("invalid_records:" + kind)
        return rows

    gold, predicted = records("labels"), records("predictions")
    hands = manifest.get("hands", [])
    require(records("dataset") == hands, "dataset_hand_binding")
    require(bool(hands), "no_complete_holdout_hands")
    hand_ids, accepted = set(), {}
    for hand in hands:
        hid = hand.get("id")
        require(isinstance(hid, str) and bool(hid) and hid not in hand_ids,
                "duplicate_or_missing_hand")
        hand_ids.add(hid)
        require(hand.get("role") == "holdout", "not_holdout:" + str(hid))
        require(hand.get("complete") is True, "incomplete_hand:" + str(hid))
        require(hand.get("labels_reviewed") is True,
                "unreviewed_labels:" + str(hid))
        require(hand.get("tuning_exposed") is False,
                "tuning_exposure:" + str(hid))
        start, end = hand.get("first_frame"), hand.get("last_frame")
        valid = (type(start) is int and type(end) is int and 0 <= start <= end)
        require(valid, "invalid_interval:" + str(hid))
        require(bool(hand.get("session")), "missing_session:" + str(hid))
        if valid:
            for dev in manifest.get("development_intervals", []):
                if dev["session"] == hand.get("session"):
                    require(end < dev["first_frame"] or start > dev["last_frame"],
                            "development_overlap:" + str(hid))
            for other in accepted.values():
                if other["session"] == hand.get("session"):
                    require(end < other["first_frame"] or start > other["last_frame"],
                            "holdout_overlap:" + str(hid))
            accepted[hid] = hand

    def index(rows, kind):
        result = {}
        for row in rows:
            key = (row.get("hand"), row.get("frame"))
            require(key not in result, "duplicate_row:" + kind)
            hand = accepted.get(key[0])
            valid = (hand is not None and type(key[1]) is int
                     and hand["first_frame"] <= key[1] <= hand["last_frame"])
            require(valid, "row_outside_hand:" + kind)
            result[key] = row
        return result

    truth, prediction = index(gold, "labels"), index(predicted, "predictions")
    expected = {(hid, frame) for hid, hand in accepted.items()
                for frame in range(hand["first_frame"], hand["last_frame"] + 1)}
    require(set(truth) == expected and bool(expected), "incomplete_gold_frames")
    require(set(prediction) == expected and bool(expected),
            "incomplete_prediction_frames")
    for hid, hand in accepted.items():
        frames = hand.get("action_opportunity_frames", [])
        valid = (isinstance(frames, list) and bool(frames)
                 and all(type(frame) is int and hand["first_frame"] <= frame
                         <= hand["last_frame"] for frame in frames))
        require(valid and frames == sorted(set(frames)),
                "opportunity_registry_missing_or_invalid:" + hid)
        require(hand.get("opportunities_reviewed") is True,
                "opportunity_review_missing:" + hid)
        actual_frames = sorted(
            frame for (hand_id, frame), label in truth.items()
            if hand_id == hid and label.get("decision_opportunity") is True)
        require(frames == actual_frames, "opportunity_registry_gold_mismatch:" + hid)
    coverage = {field: {"total": len(expected), "known_gold": 0,
                        "automatic_known": 0, "matched": 0, "required_total": 0,
                        "required_matched": 0, "exempt_total": 0,
                        "exempt_abstained": 0} for field in FIELDS}
    scenarios = set()
    for key in expected:
        label, pred = truth.get(key, {}), prediction.get(key, {})
        opportunity = label.get("decision_opportunity")
        require(type(opportunity) is bool, "opportunity_label_missing")
        actor_gold = label.get("fields", {}).get("actor", {})
        actions_gold = label.get("fields", {}).get("actions", {})
        positive_moment = (actor_gold.get("status") == "KNOWN"
                           and type(actor_gold.get("value")) is int
                           or actions_gold.get("status") == "KNOWN"
                           and bool(actions_gold.get("value")))
        require(not positive_moment or opportunity is True,
                "known_action_moment_excluded")
        for name, record in (("labels", label), ("predictions", pred)):
            fields = record.get("fields", {})
            hero = fields.get("hero_cards", {}).get("value")
            board = fields.get("board_cards", {}).get("value")
            if isinstance(hero, list) and isinstance(board, list):
                require(not set(hero) & set(board), "card_collision:" + name)
            street = fields.get("street", {}).get("value")
            expected_cards = {"idle": 0, "preflop": 0, "flop": 3, "turn": 4, "river": 5}
            if street in expected_cards and isinstance(board, list):
                require(len(board) == expected_cards[street],
                        "board_street_mismatch:" + name)
        scenarios.update(label.get("scenarios", []))
        for field in FIELDS:
            actual = label.get("fields", {}).get(field, {})
            output = pred.get("fields", {}).get(field, {})
            require(actual.get("status") in ("KNOWN", "UNKNOWN") and "value" in actual,
                    "field_gold_record_missing:" + field)
            require(output.get("status") in ("KNOWN", "UNKNOWN") and "value" in output,
                    "field_prediction_record_missing:" + field)
            known = (actual.get("status") == "KNOWN" and "value" in actual
                     and valid_value(field, actual["value"])
                     and valid_na_context(field, actual["value"],
                                          label.get("fields", {})))
            auto = (output.get("status") == "KNOWN" and "value" in output
                    and valid_value(field, output["value"])
                    and valid_na_context(field, output["value"], pred.get("fields", {}))
                    and output.get("origin") == "automatic")
            count = coverage[field]
            requested = actual.get("required", True)
            require(type(requested) is bool, "invalid_required_flag:" + field)
            exempt = requested is False
            if exempt:
                require(valid_exemption(actual, base), "invalid_exemption:" + field)
                require(opportunity is False, "action_opportunity_exempted:" + field)
            # An invalid mask cannot erase a required opportunity or its denominator.
            required = not exempt or opportunity is not False
            match = known and auto and actual["value"] == output["value"]
            if output.get("status") == "KNOWN":
                require(auto, "invalid_or_manual_known_prediction:" + field)
                require(known and match,
                        "known_prediction_unverified_or_wrong:" + field)
            count["required_total"] += int(required)
            count["required_matched"] += int(required and match)
            count["exempt_total"] += int(not required)
            count["exempt_abstained"] += int(not required
                                             and output.get("status") == "UNKNOWN")
            count["known_gold"] += int(known)
            count["automatic_known"] += int(auto)
            count["matched"] += int(match)
    for field, count in coverage.items():
        require(count["required_total"] > 0
                and count["required_matched"] == count["required_total"],
                "field_not_complete:" + field)
    for scenario in SCENARIOS:
        require(scenario in scenarios, "scenario_missing:" + scenario)
        if scenario in ("insurance", "mushroom", "bomb"):
            present = any(isinstance(row.get("fields", {}).get(scenario, {}).get(
                "value"), dict) and row["fields"][scenario]["value"].get(
                    "active") is False for row in gold)
            active = any(isinstance(row.get("fields", {}).get(scenario, {}).get(
                "value"), dict) and row["fields"][scenario]["value"].get(
                    "active") is True for row in gold)
            require(active, "mode_positive_missing:" + scenario)
            require(present, "mode_negative_missing:" + scenario)
    # An inactive mode must be explicitly KNOWN false. UNKNOWN is never absence.
    hardware = records("hardware")
    require(len(hardware) == 1, "hardware_record_required")
    if len(hardware) == 1:
        hw = hardware[0]
        session = hw.get("session")
        require(bool(session) and session == manifest.get("hardware_session"),
                "hardware_session_binding")
        if manifest.get("policy", {}).get("require_independent_hardware_session"):
            require(session not in manifest.get("development_sessions", [])
                    and session not in {h.get("session") for h in hands},
                    "hardware_session_not_independent")
        require(hw.get("freeze_id") == manifest.get("freeze_id"),
                "hardware_freeze_mismatch")
        for key in ("user_authorized", "target_device", "end_to_end",
                    "freshness_checked", "reconnect_tested", "recovery_verified",
                    "latency_budget_met", "no_strategy_or_game_actions"):
            require(hw.get(key) is True, "hardware_unverified:" + key)
        seconds = hw.get("duration_seconds")
        minimum = manifest.get("policy", {}).get("hardware_min_seconds")
        valid = (type(minimum) in (int, float) and 0 < minimum < float("inf"))
        require(valid, "hardware_duration_policy_unspecified")
        require(valid and type(seconds) in (int, float)
                and minimum <= seconds < float("inf"),
                "hardware_duration_below_configured_minimum")
        for key in ("unexplained_gaps", "undetected_stale_frames", "crashes"):
            require(type(hw.get(key)) is int and hw[key] == 0,
                    "hardware_failure_or_unknown:" + key)
    issues = sorted(set(issues))
    return {"schema_version": 1, "status": "PARTIAL" if issues else "PASS",
            "full_visual_acceptance": not issues, "strategy_eligible": False,
            "failures": issues, "coverage": coverage,
            "complete_holdout_hands": len(accepted),
            "required_scenarios": list(SCENARIOS),
            "verified_artifact_counts": {k: len(v) for k, v in bound.items()},
            "hardware_session_independent": bool(manifest.get("hardware_session"))
            and manifest.get("hardware_session") not in {
                *manifest.get("development_sessions", []),
                *(h.get("session") for h in hands)},
            "limitations": ["Finite evidence is not a universal accuracy claim",
                            "Artifact hashes do not authenticate annotation honesty",
                            "Vision acceptance never authorizes live strategy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    if args.template:
        print(json.dumps(template(), indent=2))
        return 0
    if not args.manifest:
        parser.error("--manifest required unless --template")
    result = evaluate(json.loads(args.manifest.read_text(encoding="utf-8")),
                      args.manifest.parent)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
