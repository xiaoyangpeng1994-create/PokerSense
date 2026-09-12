"""Compare isolated V6 training/replay against immutable baseline artifacts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.capture_card_calibration.hashing import sha256_file
from tools.wpk_runtime_fingerprint import fingerprint


def compare_npz(baseline: Path, reproduced: Path) -> dict:
    with np.load(baseline, allow_pickle=False) as old, np.load(
        reproduced, allow_pickle=False,
    ) as new:
        keys_equal = set(old.files) == set(new.files)
        different = [key for key in sorted(set(old.files) & set(new.files))
                     if old[key].dtype != new[key].dtype
                     or not np.array_equal(old[key], new[key])]
        return {"baseline_sha256": sha256_file(baseline),
                "reproduced_sha256": sha256_file(reproduced),
                "file_bytes_equal": baseline.read_bytes() == reproduced.read_bytes(),
                "array_keys_equal": keys_equal, "different_arrays": different,
                "arrays_equal": keys_equal and not different}


def compare_replay(baseline: dict, reproduced: dict) -> dict:
    # Timing is expected to vary; every scored field and raw score must match.
    return {"same_frozen_spec": baseline["spec_sha256"] == reproduced["spec_sha256"],
            "summary_equal": baseline["summary"] == reproduced["summary"],
            "checkpoints_exactly_equal": (
                baseline["checkpoints"] == reproduced["checkpoints"]),
            "baseline_passed": baseline["summary"]["passed"],
            "reproduced_passed": reproduced["summary"]["passed"]}


def verify(corpus: Path, run: Path) -> dict:
    baseline_training = corpus / "training/rank_v6_candidate_frozen"
    baseline_replay = corpus / "validation/v6_batch_03/first-result.json"
    weights = compare_npz(baseline_training / "card_heads.npz",
                          run / "retrained/card_heads.npz")
    features = compare_npz(baseline_training / "training-features.npz",
                           run / "retrained/training-features.npz")
    replayed_path = run / "batch03-replay-result.json"
    old = json.loads(baseline_replay.read_text(encoding="utf-8"))
    new = json.loads(replayed_path.read_text(encoding="utf-8"))
    replay = compare_replay(old, new)
    runtime = fingerprint()
    records = [record for package in runtime["opencv_distributions"]
               for record in package["binary_record_checks"]]
    runtime_ok = (runtime["isolation_checks"]["passed"]
                  and len(runtime["opencv_distributions"]) == 1
                  and bool(records) and all(r["matches_record"] for r in records))
    pip_check = subprocess.run([sys.executable, "-I", "-m", "pip", "check"],
                               capture_output=True, text=True, check=False,
                               timeout=60)
    passed = (runtime_ok and pip_check.returncode == 0
              and weights["arrays_equal"] and weights["file_bytes_equal"]
              and features["arrays_equal"] and features["file_bytes_equal"]
              and all(replay.values()))
    return {"passed": passed, "scope": "same-host isolated V6 training and replay",
            "new_holdout": False, "production_calibration_revalidated": False,
            "runtime_checks_passed": runtime_ok, "runtime": runtime,
            "pip_check": {"exit_code": pip_check.returncode,
                          "stdout": pip_check.stdout, "stderr": pip_check.stderr},
            "weights": weights, "training_features": features, "replay": replay,
            "baseline_replay_sha256": sha256_file(baseline_replay),
            "reproduced_replay_sha256": sha256_file(replayed_path),
            "tool_sha256": sha256_file(Path(__file__))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve earlier verification")
    report = verify(args.corpus, args.run)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)
