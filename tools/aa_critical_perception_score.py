"""Hash-bound, finite five-field comparison; never reads images or grants truth."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from poker_engine.desktop import aa_critical_perception as producer


FIELDS = ("board", "hero_participation", "participation", "all_in_seats",
          "river_first_actor")
BINDING = {"source_id", "epoch", "frame", "source_frame", "pts_seconds",
           "source_sha256"}
SCOPE = "aa-critical-perception-five-fields-v1"
COHORTS = {"development": "development_exposed",
           "verification": "unexposed_declared", "synthetic": "synthetic"}
STATES = {"ACTIVE", "FOLDED", "ALL_IN", "EMPTY", "WAITING"}
REQUIRED_STRATA = {"board": {"known"},
                   "hero_participation": {"ACTIVE", "FOLDED"},
                   "participation": {"ACTIVE", "FOLDED"},
                   "all_in_seats": {"empty", "nonempty"},
                   "river_first_actor": {"hero_first", "other_first"}}
REQUIRED_IMPLEMENTATION_PATHS = (
    "tools/aa_critical_perception_score.py",
    "src/poker_engine/desktop/aa_critical_perception.py",
    "src/poker_engine/desktop/aa_live_context_v3.py",
    "src/poker_engine/desktop/aa_live_context.py",
    "src/poker_engine/desktop/aa_reader.py",
    "tools/aa8_critical_fields_v3.py",
    "tools/aa8_state_adapter_v2.py",
    "tools/aa8_cards_v2.py",
)


def _require(ok, why):
    if not ok:
        raise ValueError(why)


def _keys(value, expected, where):
    _require(type(value) is dict and set(value) == set(expected),
             "invalid_keys:" + where)


def _text(value):
    return type(value) is str and bool(value.strip())


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value):
    _require(type(value) is str, "invalid_timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
             "timestamp_not_utc")
    _require(result <= datetime.now(timezone.utc), "timestamp_in_future")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_json_key:" + key)
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=lambda value: _require(False, "nonfinite_json"))


def _bound_file(item, base):
    _keys(item, {"path", "sha256"}, "file_ref")
    _require(_text(item["path"]) and _sha(item["sha256"]), "invalid_file_ref")
    path = (base / item["path"]).resolve()
    _require(path.is_file() and _digest(path) == item["sha256"],
             "file_hash_mismatch:" + item["path"])
    return path


def _binding(value):
    _keys(value, BINDING, "binding")
    _require(_text(value["source_id"]) and _text(value["epoch"]),
             "missing_source_or_epoch")
    _require(all(type(value[key]) is int and value[key] >= 0
                 for key in ("frame", "source_frame")), "invalid_frame")
    pts = value["pts_seconds"]
    _require(type(pts) in (int, float) and 0 <= pts < float("inf"), "invalid_pts")
    _require(_sha(value["source_sha256"]), "invalid_source_hash")
    return value["source_id"], value["frame"]


def _valid_value(field, value):
    if field == "board":
        return (type(value) is list and len(value) == 5
                and all(type(card) is str and re.fullmatch(r"[2-9TJQKA][cdhs]", card)
                        for card in value) and len(set(value)) == 5)
    if field == "hero_participation":
        return type(value) is str and value in {"ACTIVE", "FOLDED", "ALL_IN"}
    if field == "participation":
        return (type(value) is dict and set(value) == set(map(str, range(8)))
                and all(type(state) is str and state in STATES
                        for state in value.values()))
    if field == "all_in_seats":
        return (type(value) is list
                and all(type(seat) is int and 0 <= seat < 8 for seat in value)
                and value == sorted(set(value)))
    return type(value) is int and 0 <= value < 8


def _validate_review(fields):
    _keys(fields, FIELDS, "review_fields")
    for field, item in fields.items():
        _keys(item, {"status", "value", "reason"}, "review_field:" + field)
        _require(item["status"] in ("KNOWN", "UNREADABLE", "UNREVIEWED")
                 and _text(item["reason"]), "invalid_review_status:" + field)
        if item["status"] == "KNOWN":
            _require(_valid_value(field, item["value"]),
                     "invalid_review_value:" + field)
        else:
            _require(item["value"] is None, "unknown_review_has_value:" + field)
    seats = fields["participation"]
    if seats["status"] == "KNOWN":
        hero, all_in = fields["hero_participation"], fields["all_in_seats"]
        _require(hero["status"] != "KNOWN" or hero["value"] == seats["value"]["4"],
                 "review_hero_participation_contradiction")
        _require(all_in["status"] != "KNOWN" or all_in["value"] == sorted(
            int(seat) for seat, state in seats["value"].items() if state == "ALL_IN"),
            "review_all_in_contradiction")


def _strata(field, value):
    if field == "river_first_actor":
        return {"hero_first" if value == 4 else "other_first"}
    if field == "hero_participation":
        return {value}
    if field == "participation":
        return set(value.values())
    if field == "all_in_seats":
        return {"nonempty" if value else "empty"}
    return {"known"}


def _score(registry, predictions, reviews):
    """Internal comparison. Only evaluate_files supplies checked artifact receipts."""
    _keys(registry, {"schema_version", "cohort", "exposure",
                     "registered_at_utc", "rows"}, "registry")
    _require(registry["schema_version"] == 1
             and type(registry["schema_version"]) is int,
             "registry_schema")
    _require(registry["cohort"] in COHORTS
             and registry["exposure"] == COHORTS[registry["cohort"]], "cohort_exposure")
    _require(type(registry["rows"]) is list and bool(registry["rows"]),
             "empty_registry")
    owned, required_fields, previous = {}, {}, {}
    for entry in registry["rows"]:
        _keys(entry, {"binding", "required_fields"}, "registered_row")
        ref, required = entry["binding"], entry["required_fields"]
        _require(type(required) is list
                 and all(type(f) is str and f in FIELDS for f in required)
                 and required == [f for f in FIELDS if f in required],
                 "invalid_required_fields")
        key = _binding(ref)
        _require(key not in owned, "duplicate_source_row")
        prior = previous.get(ref["source_id"])
        if prior is not None:
            _require(ref["frame"] == prior["frame"] + 1
                     and ref["source_frame"] == prior["source_frame"] + 1
                     and ref["pts_seconds"] > prior["pts_seconds"],
                     "source_gap_or_reordering")
        owned[key] = ref
        required_fields[key] = required
        previous[ref["source_id"]] = ref
    _require(type(predictions["rows"]) is list and type(reviews["rows"]) is list,
             "rows_must_be_lists")
    guessed = {}
    replay = producer.CriticalPerceptionBoundary()
    for row in predictions["rows"]:
        _require(type(row) is dict, "invalid_prediction_row")
        manual_keys = {"manual_corrected", "manual_override", "human_corrected"}
        _require(not manual_keys & set(row),
                 "manual_prediction_correction")
        view = producer.checked_view(row)
        _require(view is not None, "invalid_machine_candidate_projection")
        derived = replay.observe(row)
        _require(json.dumps(view, sort_keys=True, allow_nan=False)
                 == json.dumps(derived, sort_keys=True, allow_nan=False),
                 "machine_projection_replay_mismatch")
        key = _binding(view["binding"])
        _require(key in owned and key not in guessed, "extra_or_duplicate_prediction")
        _require(view["binding"] == owned[key], "prediction_source_mismatch")
        for witness in view["window"]:
            witness_key = (key[0], witness["frame"])
            _require(witness_key in owned, "unowned_temporal_witness")
            _require(all(owned[witness_key][name] == value
                         for name, value in witness.items()),
                     "temporal_witness_mismatch")
            _require(owned[witness_key]["epoch"] == owned[key]["epoch"],
                     "temporal_witness_epoch_mismatch")
        guessed[key] = view["fields"]
    labels = {}
    for row in reviews["rows"]:
        _keys(row, {"binding", "fields"}, "review_row")
        key = _binding(row["binding"])
        _require(key in owned and key not in labels, "extra_or_duplicate_review")
        _require(row["binding"] == owned[key], "review_source_mismatch")
        _validate_review(row["fields"])
        labels[key] = row["fields"]
    _require(list(guessed) == list(owned), "prediction_coverage_or_order")
    _require(list(labels) == list(owned), "review_coverage_or_order")
    metric_names = ("total", "required_total", "correct_automatic", "correct_required",
                    "wrong_known", "unknown_required", "guarded_unknown",
                    "guarded_unreadable", "unreadable", "unreviewed", "conflict")
    counters = {field: dict.fromkeys(metric_names, 0) for field in FIELDS}
    coverage = {field: set() for field in FIELDS}
    failures, comparisons = set(), []
    for key in owned:
        for field in FIELDS:
            actual, expected = guessed[key][field], labels[key][field]
            counts = counters[field]
            counts["total"] += 1
            known = actual["status"] == "KNOWN"
            reviewed = expected["status"] == "KNOWN"
            required = field in required_fields[key]
            matched = reviewed and known and actual["value"] == expected["value"]
            counts["required_total"] += int(required)
            counts["correct_automatic"] += int(matched)
            counts["correct_required"] += int(required and matched)
            counts["wrong_known"] += int(known and not matched)
            counts["unknown_required"] += int(required and not known)
            counts["guarded_unknown"] += int(
                not required and not known and expected["status"] != "UNREVIEWED")
            counts["guarded_unreadable"] += int(
                not required and actual["status"] == "UNKNOWN"
                and expected["status"] == "UNREADABLE")
            counts["unreadable"] += int(expected["status"] == "UNREADABLE")
            counts["unreviewed"] += int(expected["status"] == "UNREVIEWED")
            counts["conflict"] += int(actual["status"] == "CONFLICT")
            if required and matched:
                coverage[field].update(_strata(field, expected["value"]))
            if known and not matched:
                failures.add("wrong_or_unverified_known:" + field)
            if required and not known:
                failures.add("required_unknown:" + field)
            if required and not reviewed:
                failures.add("required_review_unavailable:" + field)
            if expected["status"] == "UNREVIEWED":
                failures.add("unreviewed:" + field)
            comparisons.append({"source_id": key[0], "frame": key[1], "field": field,
                                "prediction": actual, "review": expected,
                                "correct_automatic": matched})
    for field in FIELDS:
        for missing in REQUIRED_STRATA[field] - coverage[field]:
            failures.add("coverage_missing:" + field + ":" + missing)
    if not counters["river_first_actor"]["guarded_unreadable"]:
        failures.add("coverage_missing:river_first_actor:guarded_unreadable")
    return {"status": "CRITICAL_PERCEPTION_PARTIAL" if failures
            else "CRITICAL_PERCEPTION_FINITE_PASS", "failures": sorted(failures),
            "metrics": counters,
            "coverage": {field: sorted(coverage[field]) for field in FIELDS},
            "owned_rows": len(owned), "comparisons": comparisons}


def evaluate_files(manifest_path):
    """Verify bytes/ownership and score a declared finite batch, not independence."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read(manifest_path)
    _keys(manifest, {"schema_version", "scope", "cohort", "declared_at_utc", "freeze",
                     "sources", "predictions", "reviews"}, "manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
             and manifest["scope"] == SCOPE and manifest["cohort"] in COHORTS,
             "manifest_scope")
    paths = {key: _bound_file(manifest[key], manifest_path.parent)
             for key in ("sources", "predictions", "reviews")}
    _require(len(set(paths.values())) == 3, "inputs_not_separate")
    data = {key: _read(path) for key, path in paths.items()}
    freeze = manifest["freeze"]
    _keys(freeze, {"frozen_at_utc", "files"}, "freeze")
    _require(type(freeze["files"]) is list, "invalid_freeze_files")
    frozen = {}
    for item in freeze["files"]:
        path = _bound_file(item, manifest_path.parent)
        _require(path not in frozen, "duplicate_frozen_artifact")
        frozen[path] = item["sha256"]
    repository = Path(__file__).resolve().parents[1]
    required = {(repository / path).resolve() for path in REQUIRED_IMPLEMENTATION_PATHS}
    required.add(paths["sources"])
    _require(required <= set(frozen), "scorer_producer_or_registry_not_frozen")
    _require(paths["predictions"] not in frozen and paths["reviews"] not in frozen,
             "outputs_in_pre_prediction_freeze")
    predictions, reviews = data["predictions"], data["reviews"]
    registry = data["sources"]
    _keys(predictions, {"schema_version", "started_at_utc", "source_manifest_sha256",
                        "producer_sha256", "rows"}, "predictions")
    _keys(reviews, {"schema_version", "started_at_utc", "source_manifest_sha256",
                    "reviewer_id", "method", "rows"}, "reviews")
    for name, item in (("predictions", predictions), ("reviews", reviews)):
        _require(type(item["schema_version"]) is int and item["schema_version"] == 1,
                 "schema:" + name)
        _require(item["source_manifest_sha256"] == manifest["sources"]["sha256"],
                 "source_manifest_binding:" + name)
    producer_hash = frozen[Path(producer.__file__).resolve()]
    _require(predictions["producer_sha256"] == producer_hash,
             "producer_version_mismatch")
    _require(reviews["method"] == "source_only" and _text(reviews["reviewer_id"]),
             "review_method_not_source_only")
    _require(registry.get("cohort") == manifest["cohort"], "manifest_cohort_mismatch")
    declared = _utc(manifest["declared_at_utc"])
    frozen_at = _utc(freeze["frozen_at_utc"])
    first_consumer = min(_utc(predictions["started_at_utc"]),
                         _utc(reviews["started_at_utc"]))
    _require(declared <= _utc(registry["registered_at_utc"]) <= frozen_at
             < first_consumer,
             "invalid_declared_chronology")
    result = _score(registry, predictions, reviews)
    result.update(schema_version=1, scope=SCOPE, declared_cohort=manifest["cohort"],
                  full_visual_acceptance=False, strategy_eligible=False,
                  advice_emitted=False, real_hand_confirmed=False,
                  independence_verified=False, source_media_verified=False,
                  media_decoded=False,
                  checked_hashes={"manifest": _digest(manifest_path),
                                  **{key: manifest[key]["sha256"] for key in paths}},
                  checked_frozen_artifacts=[{"path": str(path), "sha256": digest}
                                            for path, digest in frozen.items()],
                  declarations={"timeline": "validated_order_not_authenticated_time",
                                "exposure": registry["exposure"],
                                "reviewer_id": reviews["reviewer_id"],
                                "review_method": reviews["method"]},
                  limitations=["Finite rows only; no population accuracy claim.",
                               "Hashes bind bytes, not label honesty or source images.",
                               "Exposure and review isolation remain declarations.",
                               "A candidate match never creates a confirmed fact."])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_files(args.manifest)
    except (ValueError, KeyError, TypeError, OSError, AttributeError) as exc:
        result = {"status": "CRITICAL_PERCEPTION_BLOCKED", "failures": [str(exc)],
                  "full_visual_acceptance": False, "strategy_eligible": False,
                  "real_hand_confirmed": False, "independence_verified": False}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "CRITICAL_PERCEPTION_FINITE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
