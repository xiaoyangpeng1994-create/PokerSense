"""Score difference-directed visual review against preserved before/after traces."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import sha256_file, verify_sha256sums
from tools.validate_wpk_card_batch import score_slots


def read_traces(paths):
    rows, hashes = {}, {}
    for run in paths:
        if verify_sha256sums(run):
            raise ValueError("inference evidence changed")
        path = run / "frames.jsonl"
        hashes[str(path)] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["source_frame"] in rows:
                raise ValueError("overlapping trace windows")
            rows[row["source_frame"]] = row
    return rows, hashes


def compare(window, before, after):
    review_path = window / "visual-review.json"
    spec = json.loads(review_path.read_text(encoding="utf-8"))
    old, old_hashes = read_traces(before)
    new, new_hashes = read_traces(after)
    counts = {"before": Counter(), "after": Counter()}
    rows = []
    for point in spec["checkpoints"]:
        index = point["source_frame"]
        if sha256_file(window / "frames" / f"{index:06d}.png") != point["image_sha256"]:
            raise ValueError("reviewed pixels changed")
        row = {"source_frame": index}
        for name, trace in (("before", old), ("after", new)):
            row[name] = {}
            for group, width in (("hero", 2), ("board", 5)):
                got = [r["card"] for r in trace[index][group]]
                score = score_slots(point[group], got, width)
                row[name][group] = score
                counts[name].update(score["counts"])
        rows.append(row)
    return {"independent_validation": False, "new_inference": False,
            "review_sha256": sha256_file(review_path), "before_traces": old_hashes,
            "after_traces": new_hashes,
            "counts": {k: dict(v) for k, v in counts.items()},
            "checkpoints": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--before", type=Path, nargs='+', required=True)
    parser.add_argument("--after", type=Path, nargs='+', required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous audit")
    report = compare(args.window, args.before, args.after)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))
