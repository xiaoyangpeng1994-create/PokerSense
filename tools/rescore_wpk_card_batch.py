"""Apply explicit visual-label corrections to saved predictions, not the model.

Never overwrites the frozen first-run specification or result. The corrected
report records both original hashes and the correction log.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import sha256_file
from tools.validate_wpk_card_batch import score_slots, summarize


def rescore(spec: dict, original: dict, corrections: dict) -> dict:
    if corrections["source_sha256"] != spec["source_sha256"]:
        raise ValueError("correction source mismatch")
    revised = deepcopy(spec)
    checkpoints = {row["source_frame"]: row for row in revised["checkpoints"]}
    seen = set()
    for edit in corrections["corrections"]:
        key = (edit["source_frame"], edit["field"], edit["slot"])
        if key in seen or edit["field"] not in ("hero", "board"):
            raise ValueError("duplicate or invalid correction")
        seen.add(key)
        row = checkpoints[edit["source_frame"]]
        if row["pixel_sha256"] != edit["pixel_sha256"]:
            raise ValueError("correction pixel identity mismatch")
        if row[edit["field"]][edit["slot"]] != edit["from"]:
            raise ValueError("correction does not match original label")
        row[edit["field"]][edit["slot"]] = edit["to"]
    predictions = deepcopy(original["checkpoints"])
    for row in predictions:
        expected = checkpoints[row["source_frame"]]
        for field, width in (("hero", 2), ("board", 5)):
            reads = row[field]["reads"]
            row[field] = score_slots(
                expected[field], [r["card"] for r in reads], width)
            row[field]["reads"] = reads
    return {"evaluation_mode": "rescore_saved_predictions_only",
            "corrected_spec": revised,
            "corrections": corrections, "checkpoints": predictions,
            "summary": summarize(predictions, spec["criteria"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "result", "corrections", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing corrected results")
    spec, first, edits = [json.loads(p.read_text(encoding="utf-8"))
                          for p in (args.spec, args.result, args.corrections)]
    if first["batch_id"] != spec["batch_id"] or (
        first["spec_sha256"] != sha256_file(args.spec)
    ):
        raise ValueError(
            "saved predictions do not match the original frozen specification")
    report = rescore(spec, first, edits)
    report["original_spec_sha256"] = sha256_file(args.spec)
    report["original_result_sha256"] = sha256_file(args.result)
    report["corrections_sha256"] = sha256_file(args.corrections)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
