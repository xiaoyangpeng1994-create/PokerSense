"""Compare candidate dealer/ledger/action-line against DEVELOPMENT checks."""

import argparse
import json
from pathlib import Path

from tools.aa8_action_transfer import sha


def values(row):
    ledger = row.get("hand_ledger_v2") or {}
    state = row.get("observed_state_v2") or {}
    return {
        "epoch": state.get("observed_epoch"),
        "dealer": row.get("dealer_seat"),
        "commitment_total": ledger.get("observed_total"),
        "unallocated_difference": ledger.get("unallocated_difference"),
        "action_line": row.get("action_line"),
    }


def compare(rows, gold):
    results = []
    for checkpoint in gold["checkpoints"]:
        row = rows[checkpoint["frame"]]
        if row.get("source_sha256") != checkpoint["source_sha256"]:
            raise ValueError("prediction/gold source mismatch")
        actual = values(row)
        for field in (
            "epoch", "dealer", "commitment_total", "unallocated_difference",
            "action_line",
        ):
            results.append({
                "frame": checkpoint["frame"],
                "field": field,
                "expected": checkpoint[field],
                "actual": actual[field],
                "match": actual[field] == checkpoint[field],
            })
    return results


def run(predictions, gold_path, output):
    report = json.loads((predictions / "report.json").read_text(encoding="utf-8"))
    observations = predictions / "observations.jsonl"
    if (report.get("independent_holdout") is not False
            or sha(observations) != report.get("observations_sha256")):
        raise ValueError("intact development prediction required")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if (gold.get("role") != "development"
            or gold.get("independent_holdout") is not False):
        raise ValueError("development checks required")
    rows = {}
    with observations.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame"] in rows:
                raise ValueError("duplicate prediction frame")
            rows[row["frame"]] = row
    comparisons = compare(rows, gold)
    result = {
        "schema_version": 1,
        "comparisons": comparisons,
        "matched": sum(item["match"] for item in comparisons),
        "total": len(comparisons),
        "prediction_sha256": sha(observations),
        "gold_sha256": sha(gold_path),
        "independent_holdout": False,
        "canonical_verified": False,
        "full_visual_acceptance": False,
        "strategy_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "matched": result["matched"], "total": result["total"],
        "failures": [item for item in comparisons if not item["match"]],
    }))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.predictions, args.gold, args.output)
