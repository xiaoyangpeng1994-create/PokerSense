"""Verify unchanged timeline outputs against hash-bound saved WPK observations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from poker_engine.state_engine.visual_timeline import VisualTimeline
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def check(baseline, output):
    if verify_sha256sums(baseline):
        raise ValueError("baseline integrity failure")
    old = json.loads((baseline / "report.json").read_text())
    engine = VisualTimeline()
    observations = baseline / "observations.jsonl"
    count = 0
    for line in observations.read_text().splitlines():
        engine.consume(json.loads(line))
        count += 1
    events_equal = [asdict(e) for e in engine.events] == old["events"]
    summary_equal = json.loads(json.dumps(engine.summary())) == old["summary"]
    result = {"frames": count, "events_equal": events_equal,
              "summary_equal": summary_equal,
              "observations_sha256": sha256_file(observations),
              "scope": "saved observations to timeline; not fresh vision inference"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    write_sha256sums(output)
    print(json.dumps(result))
    if not events_equal or not summary_equal:
        raise ValueError("timeline regression")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.baseline, args.output)
