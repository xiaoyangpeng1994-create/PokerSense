"""Select an offline reject floor from diagnosed DEVELOPMENT predictions only."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import sha256_file
from tools.validate_wpk_card_batch import score_slots, summarize


def select_floor(spec: dict, result: dict) -> dict:
    if "Development" not in spec["independence_scope"]:
        raise ValueError("threshold selection is forbidden on new holdout batches")
    truth = {r["source_frame"]: r for r in spec["checkpoints"]}
    if set(truth) != {r["source_frame"] for r in result["checkpoints"]}:
        raise ValueError("calibration checkpoint mismatch")
    attempts = []
    for step in range(21):
        floor = step / 20  # declared .05 grid; no per-card or per-slot overrides
        rows = []
        for old in result["checkpoints"]:
            expected = truth[old["source_frame"]]
            row = deepcopy(old)
            for name, width in (("hero", 2), ("board", 5)):
                observed = [r["card"] if r["rank_score"] is not None
                            and r["rank_score"] > floor else None
                            for r in old[name]["reads"]]
                row[name] = score_slots(expected[name], observed, width)
            rows.append(row)
        summary = summarize(rows, spec["criteria"])
        attempts.append({"rank_floor": floor, "summary": summary})
        if summary["passed"]:
            return {"chosen_rank_floor": floor, "attempts": attempts,
                    "selection_only": True, "new_inference": False,
                    "independent_validation": False}
    raise ValueError("no reject floor satisfies the development criteria")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing calibration")
    spec_path, result_path = args.run / "frozen-spec.json", args.run / "result.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if sha256_file(spec_path) != result["spec_sha256"]:
        parser.error("result does not belong to frozen specification")
    report = select_floor(spec, result)
    report.update(spec_sha256=sha256_file(spec_path),
                  result_sha256=sha256_file(result_path),
                  candidate_npz_sha256=spec["candidate_heads"]["npz_sha256"],
                  tool_sha256=sha256_file(Path(__file__)))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["attempts"][-1], indent=2))
