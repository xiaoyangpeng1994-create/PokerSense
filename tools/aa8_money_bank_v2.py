"""Broader reviewed DEVELOPMENT money bank, never holdout-directed training."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import (
    GrayAmountRecognizer, gray_glyphs,
)
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_special_modes import AUDIT
from tools.aa8_unmarked_money import pot_patch
from tools.aa_amount_candidate import stack_patch
from tools.aa_pot_candidate import colon_x, pot_mask, region


# Direct full-PNG review of ALL eight visible balances, existing development boundary.
BOUNDARY_REVIEWS = {
    5050: ("5cf571fa66a6353e5947ce6739eb6c2f71175468ff2c7e7ed227472516fadae7",
           ("0", "88", "502", "574", "510", "204", "194", "246")),
    5051: ("347f699c37f0f6d8138a0f21c6df8d693bc1ac4b93b5846baece9d7fa99b9651",
           ("0", "280", "499", "570", "504", "202", "192", "244")),
}


def reviewed_rows(cash, gold):
    if (cash.get("audit_sha256") != AUDIT or gold.get("role") != "development"
            or gold.get("independent_holdout") is not False):
        raise ValueError("source-bound development review required")
    rows = []
    for point in cash["monetary_checkpoints"]:
        rows.extend({"pool": "first", "frame": point["frame"],
                     "field": f"stack_{slot}", "value": value}
                    for slot, value in point["balances"].items())
        rows.append({"pool": "first", "frame": point["frame"],
                     "field": "pot", "value": point["pot"]})
    for kind, field in (("starting", "starting_balances"),
                        ("posted", "posted_balances"),
                        ("before_settlement", "before_settlement"),
                        ("after_settlement", "after_settlement")):
        rows.extend({"pool": "first", "frame": cash["evidence_frames"][kind],
                     "field": f"stack_{slot}", "value": value}
                    for slot, value in cash[field].items())
    for point in gold["checkpoints"]:
        if (point["role"] != "development" or point["source_audit_sha256"] != AUDIT
                or point["source_pool"] not in {"first", "second"}):
            raise ValueError("non-development checkpoint")
        base = {"pool": point["source_pool"], "frame": point["frame"],
                "review_source_sha256": point["source_sha256"]}
        for slot, value in point["fields"]["stacks"]["value"].items():
            rows.append({**base, "field": f"stack_{slot}", "value": value})
        rows.append({**base, "field": "pot", "value": point["fields"]["pot"]["value"]})
    for frame, (source_sha, balances) in BOUNDARY_REVIEWS.items():
        rows.extend({"pool": "second", "frame": frame, "field": f"stack_{slot}",
                     "value": value, "review_source_sha256": source_sha}
                    for slot, value in enumerate(balances))
    return deduplicate(rows)


def deduplicate(rows):
    unique = {}
    for row in rows:
        if (type(row["frame"]) is not int or not 0 <= row["frame"] < 9000
                or row["pool"] not in {"first", "second"}):
            raise ValueError("only early reviewed development frames")
        key = (row["pool"], row["frame"], row["field"])
        previous = unique.get(key)
        if previous is not None and previous["value"] != row["value"]:
            raise ValueError("conflicting reviewed amount")
        if previous and previous.get("review_source_sha256") and row.get(
                "review_source_sha256") not in (None, previous["review_source_sha256"]):
            raise ValueError("conflicting source hashes")
        unique[key] = {**(previous or {}), **row}
    return list(unique.values())


def labelled_glyphs(value, image_patch):
    if isinstance(value, dict) and value.get("status") == "NOT_APPLICABLE":
        return [], "not_applicable_no_numeric_display"
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError("reviewed nonnegative integer text required")
    glyphs, reason = gray_glyphs(image_patch)
    if reason:
        return [], reason
    if len(glyphs) != len(value):
        return [], f"glyph_count_{len(glyphs)}_expected_{len(value)}"
    return list(zip(value, glyphs)), None


def build(first, second, cash_path, gold_path, profile_path, output):
    cash = json.loads(cash_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text())
    if len(profile["slots"]) != 8:
        raise ValueError("AA8 layout required")
    pools = {"first": first, "second": second}
    indices = {key: inventory(path, AUDIT) for key, path in pools.items()}
    reference = load(first, indices["first"][1500])
    binary = pot_mask(region(reference))
    colon = colon_x(binary)
    if colon is None or colon < 34:
        raise ValueError("reviewed pot prefix missing")
    prefix = binary[:, colon - 34:colon + 2]
    features, labels, origins, diagnostics, patches = [], [], [], [], []
    for row in reviewed_rows(cash, gold):
        source = indices[row["pool"]].get(row["frame"])
        if source is None:
            diagnostics.append({**row, "status": "EXCLUDED",
                                "reason": "reviewed_frame_not_in_provided_pool"})
            continue
        if row.get("review_source_sha256", source["sha256"]) != source["sha256"]:
            raise ValueError("review/source hash mismatch")
        image = load(pools[row["pool"]], source)
        field = row["field"]
        image_patch = pot_patch(image, prefix) if field == "pot" else stack_patch(
            image, profile["slots"][int(field.split("_")[1])]["stack"])
        labelled, reason = labelled_glyphs(row["value"], image_patch)
        diagnostics.append({**row, "source_sha256": source["sha256"],
                            "status": "EXCLUDED" if reason else "TRAINING",
                            "reason": reason, "glyph_count": len(labelled)})
        if isinstance(row["value"], str):
            patches.append((row, image_patch))
        for position, (label, (feature, box)) in enumerate(labelled):
            features.append(feature)
            labels.append(label)
            origins.append({**row, "position": position, "digit": label,
                            "source_sha256": source["sha256"], "glyph_box": box,
                            "feature_sha256": hashlib.sha256(
                                feature.tobytes()).hexdigest()})
    bank = GrayAmountRecognizer(np.array(features), np.array(labels),
                                floor=.90, margin=.05, augment=True)
    checks = [{**row, "recognition": asdict(bank.diagnose(image_patch))}
              for row, image_patch in patches]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "bank.npz", features=np.array(features),
                        labels=np.array(labels))
    report = {
        "schema": "aa8_money_bank_v2", "role": "development", "floor": .90,
        "margin": .05, "augment": True, "independent_accuracy": False,
        "arithmetic_correction": False, "full_visual_acceptance": False,
        "bank_sha256": sha(output / "bank.npz"),
        "cash_review_sha256": sha(cash_path), "gold_sha256": sha(gold_path),
        "profile_sha256": sha(profile_path),
        "implementation_sha256": sha(Path(__file__)),
        "source_manifests": {k: sha(p / "samples.json") for k, p in pools.items()},
        "training_glyph_count": len(labels),
        "digit_counts": {c: labels.count(c) for c in "0123456789"},
        "field_diagnostics": diagnostics, "glyph_provenance": origins,
        "holdin_development_checks": checks,
        "holdin_summary": {
            "fields": len(checks),
            "matched": sum(r["recognition"]["value"] == r["value"] for r in checks),
            "unknown": sum(r["recognition"]["value"] is None for r in checks),
            "wrong": sum(r["recognition"]["value"] not in (None, r["value"])
                         for r in checks)},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"glyphs": len(labels), "holdin": report["holdin_summary"],
                      "exclusions": [r for r in diagnostics
                                     if r["status"] == "EXCLUDED"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("first", "second", "cash", "gold", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    build(args.first, args.second, args.cash, args.gold, args.profile, args.output)
