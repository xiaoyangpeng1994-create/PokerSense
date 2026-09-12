"""Validate an existing shadow WAL and write a new immutable analysis report."""

import argparse
import json
from pathlib import Path

from poker_engine.strategy.shadow_log import analyze_shadow_wal


def run(wal, output):
    result = analyze_shadow_wal(wal)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({
        "records": result["records"],
        "statuses": result["statuses"],
        "stage_timing_summary": result["stage_timing_summary"],
        "top_optimization_queue": result["optimization_queue"][:5],
    }))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.wal, args.output)
