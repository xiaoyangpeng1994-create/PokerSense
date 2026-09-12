"""Create evaluation-only supplements with honest current UTC chronology."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.aa8_holdout_plan import sha, validate_freeze


def create(freeze_path, registry_path, normalization, index, output, target=None):
    frozen = json.loads(freeze_path.read_text())
    problems = validate_freeze(frozen, freeze_path.parent)
    if problems:
        raise ValueError(problems)
    result = {"base_freeze_sha256": sha(freeze_path),
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_harness_sha256": sha(Path(__file__).with_name(
                  "aa8_holdout_dataset.py")),
              "evaluation_harness_sha256": sha(Path(__file__).with_name(
                  "aa8_holdout_predict.py")),
              "registry_sha256": sha(registry_path),
              "normalization_sha256": sha(normalization),
              "audit_index_sha256": sha(index),
              "stage": "prediction" if target else "extraction",
              "model_parameters_unchanged": True,
              "receipt_generator_sha256": sha(Path(__file__))}
    if target is not None:
        result["target_manifest_sha256"] = sha(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2))
    print(json.dumps({"stage": result["stage"], "receipt_sha256": sha(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "registry", "normalization", "index", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path)
    args = parser.parse_args()
    create(args.freeze, args.registry, args.normalization, args.index,
           args.output, args.target_manifest)
