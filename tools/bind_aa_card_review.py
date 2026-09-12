"""Bind explicitly reviewed checkpoint values to normalized source image hashes."""

import argparse
import json
from pathlib import Path

from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def bind(window, review, output):
    if verify_sha256sums(window):
        raise ValueError("window integrity failure")
    samples = json.loads((window / "samples.json").read_text())
    truth = json.loads(review.read_text())
    if samples["source_sha256"] != truth["source_sha256"]:
        raise ValueError("review source mismatch")
    by_frame = {s["source_frame"]: s for s in samples["samples"]}
    rows = []
    seen = set()
    for checkpoint in truth["checkpoints"]:
        frame = checkpoint["frame"]
        if frame in seen or frame not in by_frame:
            raise ValueError("duplicate or unavailable reviewed frame")
        seen.add(frame)
        rows.append({**checkpoint, "normalized_file": by_frame[frame]["file"],
                     "normalized_sha256": by_frame[frame]["sha256"]})
    result = {**truth, "checkpoints": rows, "review_sha256": sha256_file(review),
              "samples_sha256": sha256_file(window / "samples.json"),
              "normalization_sha256": samples["normalization_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "truth.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_sha256sums(output)
    print(json.dumps({"bound_checkpoints": len(rows), "independent_holdout": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "review", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    bind(args.window, args.review, args.output)
