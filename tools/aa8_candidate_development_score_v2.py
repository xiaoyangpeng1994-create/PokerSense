"""Score integrated development observations, never independent acceptance."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.aa8_action_transfer import sha
from tools.aa8_checkpoint_compare import fields
from tools.aa8_state_adapter_v2 import compare_reviews


def run(predictions, gold_path, reviews, output):
    report = json.loads((predictions / "report.json").read_text())
    path = predictions / "observations.jsonl"
    if sha(path) != report["observations_sha256"]:
        raise ValueError("prediction hash mismatch")
    rows = {}
    counts, reasons = Counter(), Counter()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["frame"] in rows:
            raise ValueError("duplicate frame")
        rows[row["frame"]] = row
        ledger = row["causal_street_wagers_v2"]
        counts[ledger["status"]] += 1
        if ledger.get("reason"):
            reasons[ledger["reason"]] += 1
    if len(rows) != report["frames"]:
        raise ValueError("frame count mismatch")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if (gold.get("role") != "development"
            or report.get("independent_holdout") is not False):
        raise ValueError("development scorer must not consume holdout")
    metrics, comparisons = Counter(), []
    for point in gold["checkpoints"]:
        row = rows[point["frame"]]
        if row["source_sha256"] != point["source_sha256"]:
            raise ValueError("gold source mismatch")
        for name, actual in fields(row).items():
            expected = point["fields"][name]
            if expected["status"] != "KNOWN":
                continue
            match = actual == expected["value"]
            metrics[name + ":total"] += 1
            metrics[name + ":matched"] += int(match)
            comparisons.append({"frame": point["frame"], "field": name,
                                "actual": actual, "expected": expected["value"],
                                "match": match})
    result = {"frames": len(rows), "gold_sha256": sha(gold_path),
              "prediction_sha256": sha(path), "metrics": dict(metrics),
              "comparisons": comparisons,
              "actions": compare_reviews(report["actions"], reviews),
              "ledger_coverage_not_accuracy": dict(counts),
              "ledger_unknown_reasons": dict(reasons),
              "unallocated_positive_cash": rows[max(rows)]["observed_state_v2"].get(
                  "unallocated_positive_cash", []),
              "full_visual_acceptance": False, "independent_holdout": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("predictions", "gold", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    args = parser.parse_args()
    run(args.predictions, args.gold, args.review, args.output)
