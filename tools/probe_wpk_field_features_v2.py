"""Versioned offline feature experiment; old outputs/configs are never changed."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.probe_wpk_field_candidate import candidate_score, save_image
from tools.probe_wpk_observation_fields import field_record, score_read
from tools.wpk_field_candidate import ActionCandidate, crop, stack_table
from tools.wpk_field_features_v2 import DigitCandidate, FeatureActionCandidate
from tools.wpk_video_dataset import pixels_digest, read_image


def run(baseline, later_window, followup, output):
    if output.exists():
        raise ValueError("preserve previous feature experiments")
    for folder in (baseline, later_window):
        if verify_sha256sums(folder):
            raise ValueError("input manifest failure")
    repo = Path(__file__).resolve().parents[1]
    input_paths = [baseline / "review.json", baseline / "bound-samples.json", followup]
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    review = json.loads((baseline / "review.json").read_text())
    late_review = json.loads(followup.read_text())
    summary = json.loads((later_window / "summary.json").read_text())
    if review["source_sha256"] != late_review["source_sha256"] or (
        summary["source_sha256"] != review["source_sha256"]
    ):
        raise ValueError("different source videos")
    rows = json.loads((baseline / "bound-samples.json").read_text())
    samples = {s["source_frame"]: s for s in rows}
    later = json.loads((later_window / "samples.json").read_text())
    later_ids = {p["source_frame"] for p in late_review["checkpoints"]}
    for sample in later:
        if sample["source_frame"] in later_ids:
            path = (later_window / sample["image"]).resolve()
            if not path.is_relative_to(later_window.resolve()):
                raise ValueError("source path escaped window")
            samples[sample["source_frame"]] = {**sample, "path": str(path)}
    for sample in samples.values():
        path = Path(sample["path"])
        if sha256_file(path) != sample["image_sha256"] or (
            pixels_digest(read_image(path)) != sample["pixel_sha256"]
        ):
            raise ValueError("source pixels changed")
    recipe = repo / "tests/fixtures/wpk_reference_hands/field_candidate_v1.json"
    profile = json.loads(recipe.read_text())
    folder = repo / "configs/vision" / CAPTURE_CARD_PLATFORM
    templates = {p.stem: read_image(p) for p in (folder / "action_glyph").glob("*.png")}
    for row in profile["new_templates"]:
        image = read_image(Path(samples[row["source_frame"]]["path"]))
        templates[row["action"]] = crop(image, row["rect"])
    output.mkdir(parents=True)
    original_hashes = json.loads((baseline / "source-hashes.json").read_text())
    dependencies = [repo / p for p in original_hashes
                    if p.startswith(("src/", "configs/"))]
    if any(sha256_file(p) != original_hashes[p.relative_to(repo).as_posix()]
           for p in dependencies):
        raise ValueError("production dependencies changed from baseline")
    dependencies += [recipe, followup, *[repo / "tools" / name for name in (
        "wpk_field_candidate.py", "wpk_field_features_v2.py",
        "probe_wpk_field_features_v2.py",
        "probe_wpk_field_candidate.py", "probe_wpk_observation_fields.py",
        "wpk_video_dataset.py")]]
    hashes = {}
    for path in dependencies:
        relative = path.resolve().relative_to(repo)
        digest = sha256_file(path)
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ValueError("snapshot changed")
        hashes[relative.as_posix()] = digest
    shutil.copy2(baseline / "review.json", output / "review.json")
    (output / "source-hashes.json").write_text(json.dumps(hashes, indent=2))
    (output / "bound-samples.json").write_text(
        json.dumps(list(samples.values()), indent=2))
    action_v1 = ActionCandidate(profile, templates)
    action_v2 = FeatureActionCandidate(profile, templates)
    digits = DigitCandidate({p.stem: read_image(p)
                             for p in (folder / "stack_digit").glob("*.png")})
    for label, template in digits.templates.items():
        save_image(output / "digit-features" / f"{label}.png", template)
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    tight = stack_table(table, profile)
    predictions = {}
    for index, sample in sorted(samples.items()):
        image = read_image(Path(sample["path"]))
        stamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
        frame = Frame(index, stamp, "offline-field-feature",
                      WindowRect(0, 0, 498, 1080),
                      image, 498, 1080)
        stacks = vision._recognize_stacks(frame, tight, stamp)
        predictions[index] = {
            "action_v1": action_v1.recognize(image),
            "action_v2": action_v2.recognize(image),
            "stack_v1": {str(s.slot_id): field_record(s.field) for s in stacks},
            "stack_v2": {str(r["slot"]): digits.recognize(crop(image, r["stack"]))
                         for r in profile["slots"]},
        }
    counts, checks = {}, []
    followup_group = late_review.get("group", "prior_followup")
    if followup_group == "development":
        raise ValueError("followup must remain a separate group")
    for group, truth in (("development", review), (followup_group, late_review)):
        counts[group] = {key: Counter() for key in
                         ("action_v1", "action_v2", "stack_v1", "stack_v2")}
        for point in truth["checkpoints"]:
            index = point["source_frame"]
            for slot in range(8):
                for field in counts[group]:
                    expected = point["actions" if field.startswith("action")
                                     else "stacks"][slot]
                    read = predictions[index][field][str(slot)]
                    result = (score_read(expected, read, money=True)
                              if field == "stack_v1"
                              else candidate_score(expected, read))
                    counts[group][field][result] += 1
                    checks.append({"group": group, "source_frame": index, "slot": slot,
                                   "field": field, "expected": expected,
                                   "result": result, "read": read})
    report = {"offline_only": True, "release_eligible": False, "events_emitted": 0,
              "independent_holdout": False, "frames_processed": len(predictions),
              "counts": counts, "checks": checks, "predictions": predictions,
              "action_gate": {"floor": .85, "margin": .10},
              "digit_gate": {"floor": .85, "margin": .08},
              "note": "Uncalibrated feature metrics; same old templates",
              "source_review_sha256": sha256_file(baseline / "review.json")}
    if any(sha256_file(repo / p) != h for p, h in hashes.items()):
        raise ValueError("implementation changed during run")
    if any(sha256_file(Path(p)) != h for p, h in input_hashes.items()):
        raise ValueError("review or input index changed during run")
    if any(sha256_file(Path(s["path"])) != s["image_sha256"] for s in samples.values()):
        raise ValueError("source changed during run")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "later-window", "followup", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.baseline, args.later_window, args.followup, args.output)
    print(json.dumps(result, indent=2))
