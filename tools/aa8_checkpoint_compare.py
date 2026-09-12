"""Compare independent manual DEVELOPMENT checkpoints, not release acceptance."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.aa8_action_transfer import sha


def fields(row):
    n = row["board_count"]
    board = row["cards"]["board_slots"]
    cards = [] if n == 0 else board[:n] if n in (3, 4, 5) and all(board[:n]) else None
    stacks = {}
    for seat, read in row["stacks"].items():
        value = read["value"]
        state = row["participation"]["slots"][seat]["current"]
        if value is None and state in ("EMPTY_CANDIDATE", "WAITING_CANDIDATE"):
            value = {"status": "NOT_APPLICABLE"}
        stacks[seat] = value
    return {"actor": row["current_actor"], "hero_cards": row["cards"]["hero"],
            "board_cards": cards, "pot": row["pot"]["value"], "stacks": stacks}


def run(gold_path, first, second, output):
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = {}
    for folder in (first, second):
        report = json.loads((folder / "report.json").read_text())
        path = folder / "observations.jsonl"
        if sha(path) != report["observations_sha256"]:
            raise ValueError("changed predictions")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["frame"] in rows:
                raise ValueError("duplicate prediction frame")
            rows[row["frame"]] = row
    stats = Counter()
    comparisons = []
    for checkpoint in gold["checkpoints"]:
        row = rows[checkpoint["frame"]]
        if row["source_sha256"] != checkpoint["source_sha256"]:
            raise ValueError("gold/prediction source mismatch")
        values = fields(row)
        for name, actual in values.items():
            expected = checkpoint["fields"][name]
            if expected["status"] != "KNOWN":
                continue
            truth = expected["value"]
            match = actual == truth
            stats[name + ":total"] += 1
            stats[name + ":matched"] += int(match)
            comparisons.append({"frame": checkpoint["frame"], "field": name,
                                "expected": truth, "actual": actual, "match": match})
    result = {"gold_sha256": sha(gold_path), "metrics": dict(stats),
              "comparisons": comparisons, "independent_holdout": False,
              "full_visual_acceptance": False,
              "scope": "sparse_development_fields_not_full_state_or_temporal_accuracy"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"metrics": dict(stats), "failures": [r for r in comparisons
                                                           if not r["match"]]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "first", "second", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.gold, args.first, args.second, args.output)
