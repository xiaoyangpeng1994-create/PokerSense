"""Recheck a diagnosed batch on current code, never as an independent holdout.

Preserves the original labels/results, snapshots current local source/config
dependencies before inference, and evaluates the corrected labels separately.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.validate_wpk_card_batch import validate


def freeze_development(repo: Path, batch: Path, output: Path,
                       heads: Path | None = None) -> Path:
    """Freeze a new development run; refuse to overwrite any previous run."""
    parent = batch / "corrected-score.json"
    spec = deepcopy(json.loads(parent.read_text(encoding="utf-8"))["corrected_spec"])
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    paths = list((repo / "src").rglob("*.py"))
    paths += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    paths += list((repo / "tools").rglob("*.py"))
    paths += [repo / "pyproject.toml"]
    for path in sorted(paths):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo).as_posix()
        digest = sha256_file(path)
        target = output / "model-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ValueError(f"snapshot changed during copy: {relative}")
        files[relative] = digest
    spec.update(
        batch_id=output.name,
        independence_scope="Development rerun of diagnosed batch 02; NOT holdout.",
        parent_corrected_score_sha256=sha256_file(parent),
        frozen_model_files=files,
        validator_sha256=sha256_file(repo / "tools/validate_wpk_card_batch.py"),
        runtime={"python": platform.python_version(), "numpy": np.__version__,
                 "opencv": cv2.__version__},
    )
    if heads is not None:
        target = output / "candidate-heads" / "card_heads.npz"
        target.parent.mkdir()
        for source, destination in (
            (heads, target), (heads.with_suffix(".json"), target.with_suffix(".json")),
        ):
            before = sha256_file(source)
            shutil.copy2(source, destination)
            if sha256_file(destination) != before:
                raise ValueError("candidate model changed during freeze")
        spec["candidate_heads"] = {
            "relative_path": target.relative_to(output).as_posix(),
            "npz_sha256": sha256_file(target),
            "json_sha256": sha256_file(target.with_suffix(".json")),
        }
    path = output / "frozen-spec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=Path, help="offline model; never installs it")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    spec_path = freeze_development(repo, args.batch, args.output, args.heads)
    result = validate(args.corpus, spec_path)
    result["independent_validation"] = False
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    write_sha256sums(args.output)
    print(json.dumps(result["summary"], indent=2))
    raise SystemExit(0 if result["summary"]["passed"] else 2)
