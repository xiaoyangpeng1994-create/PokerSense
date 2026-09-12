"""Score locked sparse independent labels; never declares full visual acceptance."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.aa8_checkpoint_compare import fields
from tools.aa8_action_transfer import sha


def gold_fields(item):
    """Adapt two annotation schemas without changing labels or imputing values."""
    if "fields" in item:
        result = {k: v.copy() for k, v in item["fields"].items()}
        if result.get("stacks", {}).get("status") == "KNOWN":
            result["stacks"]["value"] = {
                s: {"status": "NOT_APPLICABLE"} if isinstance(v, dict)
                and v.get("status") == "NOT_APPLICABLE" else v
                for s, v in result["stacks"]["value"].items()}
        return result
    result = {k: item.get(k, {"status": "UNKNOWN"}) for k in (
        "actor", "hero_cards", "board_cards")}
    pot = item.get("pot", {"status": "UNKNOWN"})
    result["pot"] = {**pot, "value": str(pot["value"])} if (
        pot["status"] == "KNOWN") else pot
    stacks = {}
    for seat, value in item.get("stacks", {}).items():
        if value["status"] == "KNOWN":
            stacks[seat] = str(value["value"])
        elif value["status"] == "NOT_APPLICABLE":
            stacks[seat] = {"status": "NOT_APPLICABLE"}
    result["stacks"] = {"status": "KNOWN", "value": stacks} if (
        set(stacks) == set(map(str, range(8)))) else {"status": "UNKNOWN"}
    return result


def score(gold_path, gold_sha, predictions, output):
    if sha(gold_path) != gold_sha:
        raise ValueError("gold changed after independent reviewer lock")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    report = json.loads((predictions / "report.json").read_text())
    path = predictions / "observations.jsonl"
    if (sha(path) != report["observations_sha256"]
            or report["labels_used_for_prediction"]):
        raise ValueError("predictions changed or used labels")
    rows = {r["frame"]: r for r in (
        json.loads(line) for line in path.read_text().splitlines())}
    if gold.get("role") != "holdout":
        raise ValueError("independent holdout labels required")
    items = gold.get("checkpoints", gold.get("rows", []))
    if not items:
        raise ValueError("no independently reviewed checkpoints")
    counts, comparisons, seen = Counter(), [], set()
    for item in items:
        frame = item.get("frame", item.get("source_frame"))
        if frame in seen:
            raise ValueError("duplicate checkpoint")
        seen.add(frame)
        row = rows[frame]
        expected_sha = item.get("source_sha256", item.get("source_png_sha256"))
        if expected_sha != row["source_sha256"] or not row["scored_whole_hand_frame"]:
            raise ValueError("gold source or hand ownership mismatch")
        actual_fields = fields(row)
        expected_fields = gold_fields(item)
        for key, actual in actual_fields.items():
            expected = expected_fields.get(key, {"status": "UNKNOWN"})
            if expected["status"] != "KNOWN":
                counts[key + ":unreviewed"] += 1
                continue
            matched = actual == expected["value"]
            counts[key + ":reviewed"] += 1
            counts[key + ":matched"] += int(matched)
            comparisons.append({"frame": frame, "field": key,
                                "expected": expected["value"], "actual": actual,
                                "matched": matched})
    result = {"gold_sha256": gold_sha, "prediction_report_sha256": sha(
        predictions / "report.json"),
        "predictions_sha256": report["observations_sha256"],
        "checkpoint_count": len(seen),
        "whole_hand_frame_count": report["scored_frames"],
        "metrics": dict(counts), "comparisons": comparisons,
        "scope": "sparse_reviewed_fields_only_not_whole_hand_accuracy",
        "full_visual_acceptance": False, "strategy_eligible": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"metrics": dict(counts), "failures": [r for r in comparisons
                                                            if not r["matched"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("gold", "predictions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gold-sha", required=True)
    args = parser.parse_args()
    score(args.gold, args.gold_sha, args.predictions, args.output)
