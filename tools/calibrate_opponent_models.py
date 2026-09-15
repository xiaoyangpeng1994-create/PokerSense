"""Audit all public decision opportunities before fitting a named opponent."""

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

from poker_engine.strategy.opponent_dataset_v1 import calibrate_opponent_dataset
from tools.analyze_terminal_multiway import encode as scalar_encode, unique_object


def encode(value):
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return scalar_encode(value)


def analyze_files(dataset_path, candidates_path):
    raw, candidate_raw = dataset_path.read_bytes(), candidates_path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    candidates = json.loads(candidate_raw, object_pairs_hook=unique_object)
    candidate_hash = hashlib.sha256(candidate_raw).hexdigest()
    return {
        "scope": "PUBLIC_OPPONENT_DATA_AUDIT_NOT_EMPIRICAL_OR_LIVE_APPROVAL",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "candidates_file_sha256": candidate_hash,
        **encode(calibrate_opponent_dataset(
            data, candidates, candidates_sha256=candidate_hash)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = analyze_files(args.dataset, args.candidates)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"opponent dataset rejected: {exc}\n")
    print(json.dumps({"scope": report["scope"], "audit": {
        k: report["audit"][k] for k in (
            "status", "source_kind", "opportunity_count", "eligible_count",
            "blockers")},
        "calibration_status": report["calibration"]["status"]
        if report["calibration"] else None,
    }, ensure_ascii=False, indent=2))
    return 0 if report["calibration"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
