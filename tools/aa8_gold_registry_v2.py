"""Source-bound V2 gold registration. Sparse gold never proves whole-hand acceptance."""
import argparse
import json
from pathlib import Path

from tools.aa8_acceptance import valid_na_context, valid_value
from tools.aa8_holdout_plan import sha
from tools.aa8_holdout_predict import FIELDS, validate_rows


def inspect_gold(registry, manifest, labels):
    samples = manifest["samples"]
    rows = {row["global_frame"]: row for row in samples}
    if len(rows) != len(samples):
        raise ValueError("duplicate source sample")
    owned = validate_rows(rows, registry, manifest["audit_sha256"])
    seen, failures = set(), []
    coverage = {field: {"owned": len(owned), "records": 0, "known": 0,
                        "required": 0, "reviewed_exempt": 0} for field in FIELDS}
    opportunities = set()
    for label in labels:
        frame = label.get("frame")
        if frame not in owned or frame in seen:
            raise ValueError("gold frame duplicated or outside owned hand")
        seen.add(frame)
        if label.get("source_sha256") != rows[frame]["sha256"]:
            raise ValueError("gold/source hash mismatch")
        if label.get("source_reviewed") is not True or not label.get("reviewer"):
            failures.append("missing_source_review")
        expected_hand = next(h["id"] for h in registry["hands"]
                             if h["first_frame"] <= frame <= h["last_frame"])
        if label.get("hand") != expected_hand:
            raise ValueError("gold hand identity mismatch")
        opportunity = label.get("decision_opportunity")
        if type(opportunity) is not bool:
            failures.append("action_opportunity_unreviewed")
        elif opportunity:
            opportunities.add(frame)
        fields = label.get("fields", {})
        actor, actions = fields.get("actor", {}), fields.get("actions", {})
        if (actor.get("status") == "KNOWN" and type(actor.get("value")) is int
                or actions.get("status") == "KNOWN" and bool(actions.get("value"))):
            if opportunity is not True:
                failures.append("known_action_opportunity_omitted")
        for field in FIELDS:
            count, item = coverage[field], fields.get(field, {})
            if item.get("status") not in ("KNOWN", "UNKNOWN") or "value" not in item:
                failures.append("missing_field_record:" + field)
                continue
            count["records"] += 1
            required = item.get("required", True)
            if type(required) is not bool:
                failures.append("invalid_required_flag:" + field)
                required = True
            if required:
                count["required"] += 1
            else:
                exemption = item.get("exemption") or {}
                if (opportunity is not False or exemption.get("reason") not in (
                        "transition", "occluded", "no_action")
                        or exemption.get("source_reviewed") is not True
                        or exemption.get("source_sha256") != rows[frame]["sha256"]):
                    failures.append("invalid_gold_exemption:" + field)
                else:
                    count["reviewed_exempt"] += 1
            known = (item["status"] == "KNOWN" and valid_value(field, item["value"])
                     and valid_na_context(field, item["value"], fields))
            count["known"] += int(known)
            if required and not known:
                failures.append("required_gold_unknown:" + field)
    if seen != owned:
        failures.append("sparse_gold_not_full_frame_truth")
    for field, count in coverage.items():
        if count["required"] == 0:
            failures.append("no_required_gold:" + field)
    for hand in registry["hands"]:
        planned = hand.get("action_opportunity_frames")
        actual = sorted(f for f in opportunities
                        if hand["first_frame"] <= f <= hand["last_frame"])
        if (hand.get("opportunities_reviewed") is not True
                or not planned or planned != actual):
            failures.append("whole_hand_action_opportunity_registry_incomplete")
    return {"schema_version": 2,
            "status": "PARTIAL" if failures else "GOLD_READY_NOT_PASS",
            "owned_frames": len(owned), "gold_frames": len(seen), "coverage": coverage,
            "failures": sorted(set(failures)), "predictions_read": False,
            "full_visual_acceptance": False, "strategy_eligible": False}


def register(registry_path, pool, gold_path, output):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest_path = pool / "samples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("registry_sha256") != sha(registry_path):
        raise ValueError("source pool registry binding mismatch")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if (gold.get("role") != "holdout"
            or gold.get("source_manifest_sha256") != sha(manifest_path)):
        raise ValueError("gold role/source manifest binding mismatch")
    labels = gold["labels"]
    report = inspect_gold(registry, manifest, labels)
    by_frame = {row["global_frame"]: row for row in manifest["samples"]}
    for label in labels:
        row = by_frame[label["frame"]]
        path = (pool / row["file"]).resolve()
        if not path.is_relative_to(pool.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("gold source file path/hash mismatch")
    report.update(registry_sha256=sha(registry_path), gold_sha256=sha(gold_path),
                  source_manifest_sha256=sha(manifest_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("registry", "pool", "gold", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.registry, args.pool, args.gold, args.output)))
