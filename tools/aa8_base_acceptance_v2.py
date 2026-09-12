"""Predeclared ten-field BASE_VISUAL gate; never grants full-feature approval."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.aa8_acceptance import valid_na_context, valid_value
from tools.aa8_holdout_plan import sha, utc, validate_freeze
from tools.aa8_holdout_predict import FIELDS, validate_rows


BASE_FIELDS = FIELDS[:10]
SPECIAL = ("insurance", "mushroom", "bomb")
UNSAFE = ("actor", "street_wagers", "actions", "street", "hand", "participation")


def declaration_template():
    return {"schema_version": 1, "declared_at_utc": None, "scope_sha256": None,
            "required_fields": list(BASE_FIELDS), "all_registered_hands_required": True,
            "special_policy": "SOURCE_REVIEWED_GUARD_REQUIRED",
            "unknown_mode_means_inactive": False, "cash_policy": "UNALLOCATED",
            "resume_requires_fresh_context": True, "hardware_min_seconds": None,
            "prediction_tuning_prohibited": True}


def evaluate_evidence(scope, declaration, registry, pool, gold, predictions, hardware):
    """Pure evidence check; on-disk hashing and freeze checks are in evaluate_files."""
    failures = []

    def check(ok, why):
        if not ok:
            failures.append(why)

    check(scope.get("id") == "aa8_base_visual_v1" and scope.get("seat_count") == 8
          and scope.get("hero_slot") == 4, "wrong_base_scope_geometry")
    check(set(scope.get("deferred_semantics", [])) == set(SPECIAL),
          "wrong_deferred_scope")
    check(scope.get("retain_mode_guards") is True
          and scope.get("unknown_mode_means_inactive") is False, "unsafe_scope")
    check(set(BASE_FIELDS) <= set(scope.get("required", []))
          and scope.get("resume_requires_fresh_context") is True
          and scope.get("ambiguous_cash_policy") == "UNALLOCATED",
          "base_scope_weakened")
    for key in ("required_fields", "all_registered_hands_required", "special_policy",
                "unknown_mode_means_inactive", "cash_policy",
                "resume_requires_fresh_context", "prediction_tuning_prohibited"):
        check(declaration.get(key) == declaration_template()[key], "declaration:" + key)
    try:
        declared = utc(declaration["declared_at_utc"])
        check(declared < utc(gold["source_review_started_at_utc"]),
              "scope_not_predeclared_before_source_review")
        check(declared < utc(gold["prediction_start_utc"]),
              "scope_not_predeclared_before_prediction")
    except (KeyError, ValueError, TypeError, AttributeError):
        check(False, "missing_valid_predeclaration_timestamps")
    check(gold.get("prediction_used_for_labels") is False, "gold_prediction_leakage")
    samples = pool["samples"]
    source = {r["global_frame"]: r for r in samples}
    check(len(samples) == len(source), "duplicate_source")
    owned = validate_rows(source, registry, pool["audit_sha256"])

    def index(rows, kind):
        result = {}
        for row in rows:
            frame = row.get("frame")
            check(frame in owned and frame not in result, "invalid_frame:" + kind)
            if frame in source:
                check(row.get("source_sha256") == source[frame]["sha256"],
                      "source_hash:" + kind)
            result[frame] = row
        check(set(result) == owned, "whole_frame_coverage:" + kind)
        return result

    labels, guessed = index(gold["labels"], "gold"), index(predictions, "predictions")
    coverage = {f: {"total": len(owned), "required": 0, "matched_required": 0,
                    "known_predictions": 0, "guarded": 0} for f in BASE_FIELDS}
    opportunities, prior_special = set(), None
    for frame in sorted(owned):
        label, pred = labels.get(frame, {}), guessed.get(frame, {})
        fields, output = label.get("fields", {}), pred.get("evaluation_fields", {})
        scene = label.get("scene")
        scene_evidence = label.get("scene_evidence", {})
        check(scene in ("supported", "special"), "scene_unknown_not_normal")
        check(scene_evidence.get("source_reviewed") is True
              and scene_evidence.get("source_sha256") == source[frame]["sha256"],
              "scene_not_source_reviewed")
        if scene == "special":
            check(scene_evidence.get("mode") in SPECIAL,
                  "special_positive_label_missing")
        opportunity = label.get("decision_opportunity")
        check(type(opportunity) is bool, "unreviewed_action_opportunity")
        actor_gold, action_gold = fields.get("actor", {}), fields.get("actions", {})
        if (actor_gold.get("status") == "KNOWN" and type(actor_gold.get("value")) is int
                or action_gold.get("status") == "KNOWN"
                and bool(action_gold.get("value"))):
            check(opportunity is True, "known_action_opportunity_excluded")
        if opportunity is True:
            opportunities.add(frame)
        receipt = pred.get("visual_scope", {})
        check(receipt.get("scope_id") == scope.get("id")
              and receipt.get("automatically_exempt_from_acceptance") is False,
              "missing_scope_receipt_or_automatic_exemption")
        check(receipt.get("unknown_cash_policy") == "UNALLOCATED",
              "unallocated_cash_policy_lost")
        if scene == "special":
            check(receipt.get("status") == "DEFERRED_MODE_OBSERVED"
                  and scene_evidence.get("mode") in receipt.get(
                      "observed_deferred_modes", []),
                  "special_guard_missed")
            prior_special = frame
        elif prior_special is not None:
            fresh = receipt.get("resume_evidence", {})
            witnesses = fresh.get("source_frames", [])
            valid = (isinstance(witnesses, list) and bool(witnesses)
                     and all(type(f) is int and prior_special < f <= frame
                             and f in source for f in witnesses))
            check(valid and fresh.get("fresh_context_verified") is True,
                  "resume_without_fresh_context")
            prior_special = None
        for field in BASE_FIELDS:
            actual, value = fields.get(field, {}), output.get(field, {})
            check(actual.get("status") in ("KNOWN", "UNKNOWN") and "value" in actual,
                  "missing_gold_field:" + field)
            check(value.get("status") in ("KNOWN", "UNKNOWN") and "value" in value,
                  "missing_prediction_field:" + field)
            known = (actual.get("status") == "KNOWN" and valid_value(
                field, actual.get("value")) and valid_na_context(
                    field, actual.get("value"), fields))
            automatic = (value.get("status") == "KNOWN"
                         and value.get("origin") == "automatic"
                         and valid_value(field, value.get("value"))
                         and valid_na_context(field, value.get("value"), output))
            matched = known and automatic and actual.get("value") == value.get("value")
            if value.get("status") == "KNOWN":
                check(matched, "known_prediction_wrong_or_unverified:" + field)
                if field == "actions":
                    check(pred.get("actions_complete_and_canonical_verified") is True,
                          "uncanonical_actions_promoted")
                if field == "street_wagers":
                    check((pred.get("causal_street_wagers_v2") or {}).get(
                        "canonical_verified") is True, "uncanonical_wagers_promoted")
                if field == "hand":
                    check((pred.get("observed_state_v2") or {}).get(
                        "authoritative_hand_boundary") is True,
                        "uncanonical_hand_promoted")
            check(type(actual.get("required", True)) is bool, "invalid_required_flag")
            optional = actual.get("required", True) is False
            if optional:
                exemption = actual.get("exemption", {})
                check(opportunity is False and exemption.get("source_reviewed") is True
                      and exemption.get("source_sha256") == source[frame]["sha256"]
                      and exemption.get("reason") in (
                          "special_ui", "transition", "occluded", "no_action"),
                      "invalid_exemption:" + field)
                if exemption.get("reason") == "special_ui":
                    check(scene == "special", "special_exemption_on_normal_frame")
            required = not optional or opportunity is True
            count = coverage[field]
            count["required"] += int(required)
            count["matched_required"] += int(required and matched)
            count["known_predictions"] += int(automatic)
            count["guarded"] += int(scene == "special"
                                    and value.get("status") == "UNKNOWN")
            if scene == "special" and field in UNSAFE:
                check(value.get("status") == "UNKNOWN",
                      "unsafe_field_in_special:" + field)
    for field, counts in coverage.items():
        check(counts["required"] > 0
              and counts["required"] == counts["matched_required"],
              "base_field_incomplete:" + field)
    for hand in registry["hands"]:
        actual = sorted(f for f in opportunities
                        if hand["first_frame"] <= f <= hand["last_frame"])
        check(hand.get("opportunities_reviewed") is True and bool(actual)
              and hand.get("action_opportunity_frames") == actual,
              "whole_hand_opportunity_registry_incomplete")
    for key in ("user_authorized", "target_device", "end_to_end", "freshness_checked",
                "reconnect_tested", "recovery_verified", "latency_budget_met"):
        check(hardware.get(key) is True, "hardware:" + key)
    minimum = declaration.get("hardware_min_seconds")
    duration = hardware.get("duration_seconds")
    check(type(minimum) in (int, float) and 0 < minimum < float("inf")
          and type(duration) in (int, float) and minimum <= duration < float("inf"),
          "hardware_duration_not_verified")
    for key in ("unexplained_gaps", "undetected_stale_frames", "crashes"):
        check(type(hardware.get(key)) is int and hardware[key] == 0, "hardware:" + key)
    failures = sorted(set(failures))
    return {"status": "BASE_VISUAL_PARTIAL" if failures else "BASE_VISUAL_PASS",
            "scope_id": scope.get("id"), "full_visual_acceptance": False,
            "strategy_eligible": False, "deferred_semantics": list(SPECIAL),
            "failures": failures, "coverage": coverage, "owned_frames": len(owned)}


def evaluate_files(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base, paths, data = manifest_path.parent, {}, {}
    for key in ("scope", "declaration", "freeze", "registry", "pool", "gold",
                "predictions", "prediction_report", "hardware"):
        item = manifest[key]
        path = (base / item["path"]).resolve()
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError("bound evidence missing/changed:" + key)
        paths[key] = path
        if key != "predictions":
            data[key] = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_freeze(data["freeze"], paths["freeze"].parent,
                             data["gold"].get("prediction_start_utc"))
    if errors:
        raise ValueError(errors)
    frozen = {str(Path(r["path"]).resolve()): r["sha256"]
              for r in data["freeze"]["files"]}
    if frozen.get(str(Path(__file__).resolve())) != sha(Path(__file__)):
        raise ValueError("base gate implementation was not frozen")
    for key in ("scope", "declaration"):
        if frozen.get(str(paths[key])) != sha(paths[key]):
            raise ValueError("scope/declaration was not in model freeze")
    if (data["declaration"]["scope_sha256"] != sha(paths["scope"])
            or data["registry"]["freeze_sha256"] != sha(paths["freeze"])
            or data["pool"]["registry_sha256"] != sha(paths["registry"])
            or data["gold"]["source_manifest_sha256"] != sha(paths["pool"])
            or data["hardware"].get("freeze_sha256") != sha(paths["freeze"])
            or data["prediction_report"].get("freeze_sha256") != sha(paths["freeze"])
            or data["prediction_report"].get("observations_sha256") != sha(
                paths["predictions"])
            or data["prediction_report"].get("registry_sha256") != sha(
                paths["registry"])
            or data["prediction_report"].get("labels_used_for_prediction") is not False
            or data["prediction_report"].get("prediction_start_utc") != (
                data["gold"].get("prediction_start_utc"))):
        raise ValueError("base evidence lineage mismatch")
    predictions = [json.loads(line) for line in paths["predictions"].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_evidence(
        data["scope"], data["declaration"], data["registry"],
        data["pool"], data["gold"], predictions, data["hardware"])
    result["evidence_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["manifest_sha256"] = sha(manifest_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--template", action="store_true")
    args = parser.parse_args()
    if args.template:
        print(json.dumps(declaration_template(), indent=2))
    elif args.manifest:
        result = evaluate_files(args.manifest)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "BASE_VISUAL_PASS" else 2)
    else:
        parser.error("--manifest or --template required")
