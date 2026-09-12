"""Fit an offline rank MLP from explicitly reviewed session_001 glyphs only.

No pickle, no production overwrite, no eval data for fitting or early stopping.
Exports numpy weights and their provenance alongside the unchanged suit heads.
"""

import argparse
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import warnings

import cv2
import numpy as np

from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedSlotBuffer, GlyphNormalizer, load_card_heads,
)
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import read_image, safe_reference_path

RANKS = frozenset("23456789TJQKA")


def installed_training_inventory() -> list[str]:
    """Inventory metadata without requiring a conflicting second OpenCV wheel."""
    names = ["numpy", "scikit-learn", "scipy", "threadpoolctl", "joblib"]
    rows = [f"{name}=={version(name)}" for name in names]
    for name in ("opencv-python", "opencv-contrib-python", "opencv-python-headless",
                 "opencv-contrib-python-headless"):
        try:
            rows.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    return rows


def reviewed_samples(manifest: dict, review: dict) -> list[tuple[dict, str]]:
    """Reject missing/duplicate decisions, invalid ranks, and eval contamination."""
    samples = {r["id"]: r for r in manifest["samples"]}
    if len(samples) != len(manifest["samples"]):
        raise ValueError("duplicate sample id")
    decisions, approved = set(), []
    for rank, identifiers in review["approved"].items():
        if rank not in RANKS:
            raise ValueError("invalid reviewed rank")
        for identifier in identifiers:
            if identifier in decisions or identifier not in samples:
                raise ValueError("duplicate/unknown review decision")
            decisions.add(identifier)
            row = samples[identifier]
            if row["session"] != "session_001" or row["split"] != "train":
                raise ValueError("only session_001 training data is permitted")
            if any(not o["frame"].startswith("session_001__") for o in row["origins"]):
                raise ValueError("mixed-session glyph origins")
            approved.append((row, rank))
    for identifier, reason in review["rejected"].items():
        if identifier in decisions or identifier not in samples or not reason:
            raise ValueError("invalid rejection decision")
        decisions.add(identifier)
    if decisions != set(samples):
        raise ValueError("every glyph must be explicitly reviewed")
    if {rank for _, rank in approved} != RANKS:
        raise ValueError("training requires all 13 ranks")
    return sorted(approved, key=lambda pair: pair[0]["id"])


def augment_glyph(glyph, rng):
    """Training-only camera/shape variation, normalized with current code."""
    h, w = glyph.shape[:2]
    width = max(2, round(w * rng.uniform(.85, 1.15)))
    height = max(3, round(h * rng.uniform(.9, 1.1)))
    resized = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)
    gain, offset = rng.uniform(.4, 1), rng.uniform(0, 20)
    varied = np.clip(resized.astype(np.float32) * gain + offset, 0, 255)
    return GlyphNormalizer.normalize(varied.astype(np.uint8))


def train(dataset: Path, output: Path, base_heads: Path, *, seed=7, augmentations=24):
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from threadpoolctl import threadpool_limits

    if output.exists():
        raise ValueError("preserve previous training outputs")
    if augmentations < 0:
        raise ValueError("augmentation count cannot be negative")
    manifest_path, review_path = dataset / "manifest.json", dataset / "review.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != review["manifest_sha256"]:
        raise ValueError("dataset changed after visual review")
    samples = reviewed_samples(manifest, review)
    repo = Path(__file__).resolve().parents[1]
    for relative, expected in manifest["feature_source_files"].items():
        if sha256_file(safe_reference_path(repo, relative)) != expected:
            raise ValueError(f"feature extraction dependency changed: {relative}")
    rng = np.random.default_rng(seed)
    xs, ys, owners = [], [], []
    for row, rank in samples:
        path = safe_reference_path(dataset, row["glyph"])
        if sha256_file(path) != row["glyph_sha256"]:
            raise ValueError("reviewed glyph changed")
        glyph = read_image(path)
        original = GlyphNormalizer.normalize(glyph)
        if original is None:
            raise ValueError("approved glyph no longer normalizes")
        features = [original]
        for _ in range(augmentations):
            # Synthetic temporal fusion uses only variants of THIS training
            # glyph and the repaired registration, never neighboring hands.
            variants = [augment_glyph(glyph, rng)
                        for _ in range(int(rng.integers(1, 5)))]
            valid = [v for v in variants if v is not None]
            if valid:
                features.append(FusedSlotBuffer._registered_mean(valid))
        for canvas in features:
            xs.append(canvas.ravel().astype(np.float64) / 255)
            ys.append(rank)
            owners.append(row["id"])
    x, y = np.stack(xs), np.array(ys)
    model = MLPClassifier(hidden_layer_sizes=(64,), solver="adam", alpha=.001,
                          batch_size=128, max_iter=600, tol=1e-5,
                          random_state=seed, early_stopping=False)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    convergence_warnings = [str(w.message) for w in caught]
    output.mkdir(parents=True)
    # Save exact training arrays: numeric/string dtypes only, no object pickle.
    np.savez_compressed(output / "training-features.npz", x=x, y=y,
                        owners=np.array(owners))
    with np.load(base_heads, allow_pickle=False) as base:
        arrays = {name: base[name].copy() for name in base.files}
    arrays.update(rank__w1=model.coefs_[0], rank__b1=model.intercepts_[0],
                  rank__w2=model.coefs_[1], rank__b2=model.intercepts_[1],
                  rank__classes=model.classes_)
    weights = output / "card_heads.npz"
    np.savez_compressed(weights, **arrays)
    meta = json.loads(base_heads.with_suffix(".json").read_text(encoding="utf-8"))
    meta["heads"]["rank"] = {"classes": model.classes_.tolist(), "hidden": 64,
                             "out_activation": "softmax"}
    meta["candidate_only"] = True
    meta["rank_training"] = f"rank-v6-session001-reviewed-seed{seed}"
    weights.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    exported = load_card_heads(weights)["rank"]
    with threadpool_limits(limits=1):
        labels = [exported.predict((row.reshape(40, 40) * 255))[0] for row in x]
        reference = model.predict(x)
    if labels != reference.tolist():
        raise ValueError("exported inference differs from sklearn")
    for name in ("manifest.json", "review.json"):
        shutil.copy2(dataset / name, output / name)
    shutil.copy2(Path(__file__), output / "train_wpk_rank_head.py")
    report = {"candidate_only": True, "independent_validation": False,
              "dataset_path": str(dataset.resolve()),
              "feature_mode": "reviewed static glyphs + synthetic same-glyph fusion",
              "actual_continuous_training_features": False,
              "train_session": "session_001", "excluded_session": "session_002",
              "hand_boundaries_verified": False, "unique_glyphs": len(samples),
              "reviewed_rank_counts": dict(Counter(rank for _, rank in samples)),
              "augmented_samples": len(y), "seed": seed,
              "augmentations_per_glyph": augmentations,
              "parameters": model.get_params(),
              "iterations": model.n_iter_,
              "convergence_warnings": convergence_warnings,
              "training_accuracy_not_validation": float(np.mean(reference == y)),
              "numpy_export_label_parity": True,
              "weights_sha256": sha256_file(weights),
              "base_weights_sha256": sha256_file(base_heads),
              "manifest_sha256": sha256_file(manifest_path),
              "review_sha256": sha256_file(review_path),
              "trainer_sha256": sha256_file(Path(__file__)),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "opencv": cv2.__version__, "sklearn": sklearn.__version__,
                          "scipy": version("scipy"),
                          "threadpoolctl": version("threadpoolctl")},
              "production_calibration_revalidated": False}
    (output / "training-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    requirements = installed_training_inventory()
    (output / "requirements-runtime.txt").write_text(
        "# Installed distribution inventory, not proof of loaded-module versions.\n"
        "# Check wpk_runtime_fingerprint for overlapping OpenCV distributions.\n"
        + "\n".join(requirements) + "\n", encoding="utf-8")
    write_sha256sums(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-heads", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.output, args.base_heads), indent=2))
