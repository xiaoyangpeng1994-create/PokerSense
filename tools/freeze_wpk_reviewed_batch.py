"""Freeze manually reviewed new video checkpoints before candidate inference."""

import argparse
import json
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, read_image


def freeze(corpus: Path, batch: Path, heads: Path, calibration: Path) -> Path:
    target = batch / "frozen-spec.json"
    snapshot = batch / "model-snapshot"
    if target.exists() or snapshot.exists():
        raise ValueError("preserve the first batch freeze")
    review_path = batch / "visual-review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not review["reviewer"] or not review["checkpoints"]:
        raise ValueError("visual review is required")
    summary = json.loads((batch / "summary.json").read_text(encoding="utf-8"))
    inventory = json.loads((corpus / "sources.json").read_text(encoding="utf-8"))
    source = next(row for row in inventory["sources"]
                  if row["session"] == "session_002")
    if source["sha256"] != summary["source_sha256"]:
        raise ValueError("batch/source mismatch")
    indices = set()
    for point in review["checkpoints"]:
        index = point["source_frame"]
        if index in indices or not 14500 <= index <= 17050:
            raise ValueError("checkpoint duplicates or enters development frames")
        indices.add(index)
        image = batch / "frames" / f"{index:06d}.png"
        if sha256_file(image) != point["image_sha256"]:
            raise ValueError("visually reviewed image changed")
        point["pixel_sha256"] = pixels_digest(read_image(image))
    repo = Path(__file__).resolve().parents[1]
    paths = list((repo / "src").rglob("*.py")) + list((repo / "tools").rglob("*.py"))
    paths += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    paths += [repo / "pyproject.toml"]
    files = {}
    for path in sorted(paths):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo).as_posix()
        digest = sha256_file(path)
        copy = snapshot / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        if sha256_file(copy) != digest:
            raise ValueError("model snapshot changed")
        files[relative] = digest
    candidate = batch / "candidate-heads" / "card_heads.npz"
    candidate.parent.mkdir()
    for source_path, destination in (
        (heads, candidate),
        (heads.with_suffix(".json"), candidate.with_suffix(".json")),
    ):
        digest = sha256_file(source_path)
        shutil.copy2(source_path, destination)
        if sha256_file(destination) != digest:
            raise ValueError("candidate changed during freeze")
    calibration_target = batch / "rank-calibration.json"
    calibration_digest = sha256_file(calibration)
    shutil.copy2(calibration, calibration_target)
    if sha256_file(calibration_target) != calibration_digest:
        raise ValueError("rank calibration changed during copy")
    spec = {"schema_version": 1, "batch_id": batch.name, "session": "session_002",
            "source_sha256": source["sha256"], "warmup_start_frame": 14500,
            "labels_frozen_before_model_run": True, "reviewer": review["reviewer"],
            "visual_review_sha256": sha256_file(review_path),
            "independence_scope": (
                "New segment not used to fit/select v6 rank candidate or v5 contrast. "
                "Rank trained only on session_001. Legacy suit-head training overlap "
                "and whole-hand boundaries remain unverified; not full-model holdout."),
            "criteria": {"max_wrong_accepted": 0,
                         "min_required_complete_fraction": .95},
            "frozen_model_files": files,
            "validator_sha256": sha256_file(repo / "tools/validate_wpk_card_batch.py"),
            "candidate_heads": {
                "relative_path": "candidate-heads/card_heads.npz",
                "npz_sha256": sha256_file(candidate),
                "json_sha256": sha256_file(candidate.with_suffix(".json"))},
            "rank_calibration": {"relative_path": "rank-calibration.json",
                                 "sha256": calibration_digest},
            "runtime": {"python": platform.python_version(),
                        "numpy": np.__version__, "opencv": cv2.__version__},
            "checkpoints": review["checkpoints"]}
    target.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    write_sha256sums(batch)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.corpus, args.batch, args.heads, args.calibration))
