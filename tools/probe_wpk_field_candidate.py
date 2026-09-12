"""Freeze and measure offline action/stack candidates against the saved baseline."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.amount_recognizer import segment_characters
from poker_engine.perceptual.vision.roi import extract_roi
from poker_engine.perceptual.vision.table_map import ROIKind
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.probe_wpk_observation_fields import field_record, score_read
from tools.wpk_field_candidate import (
    ActionCandidate, crop, stack_table, validate_profile,
)
from tools.wpk_video_dataset import pixels_digest, read_image


def save_image(path, picture):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", picture)[1].tobytes())


def candidate_score(expected, result):
    return score_read(expected, {
        "status": "valid" if result["accepted_candidate"] else "unknown",
        "value": result["value"],
    })


def run(baseline, profile_path, output):
    if output.exists():
        raise ValueError("preserve all previous candidate outputs")
    if verify_sha256sums(baseline):
        raise ValueError("baseline integrity failed")
    repo = Path(__file__).resolve().parents[1]
    baseline_report = json.loads((baseline / "report.json").read_text())
    review = json.loads((baseline / "review.json").read_text())
    samples = json.loads((baseline / "bound-samples.json").read_text())
    profile = json.loads(profile_path.read_text())
    validate_profile(profile)
    # Require the baseline's actual production dependencies, not just its label.
    baseline_hashes = json.loads((baseline / "source-hashes.json").read_text())
    for relative, digest in baseline_hashes.items():
        if relative.startswith(("src/", "configs/")):
            if sha256_file(repo / relative) != digest:
                raise ValueError("production dependency differs from baseline")
    indexed = {}
    for sample in samples:
        path = Path(sample["path"])
        if (sha256_file(path) != sample["image_sha256"]
                or pixels_digest(read_image(path)) != sample["pixel_sha256"]):
            raise ValueError("source frame changed")
        index = sample["source_frame"]
        if (index in indexed
                and indexed[index]["pixel_sha256"] != sample["pixel_sha256"]):
            raise ValueError("duplicate frame disagrees")
        indexed[index] = sample
    output.mkdir(parents=True)
    files = [Path(__file__), repo / "tools/wpk_field_candidate.py",
             repo / "tools/probe_wpk_observation_fields.py", profile_path]
    files += [repo / p for p in baseline_hashes if p.startswith(("src/", "configs/"))]
    hashes = {}
    for path in files:
        relative = path.resolve().relative_to(repo)
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(path)
        shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ValueError("snapshot changed while copying")
        hashes[relative.as_posix()] = digest
    (output / "source-hashes.json").write_text(json.dumps(hashes, indent=2))
    shutil.copy2(baseline / "review.json", output / "review.json")
    shutil.copy2(baseline / "bound-samples.json", output / "bound-samples.json")
    folder = repo / "configs/vision" / CAPTURE_CARD_PLATFORM / "action_glyph"
    templates = {p.stem: read_image(p) for p in folder.glob("*.png")}
    training_frames = set()
    for row in profile["new_templates"]:
        sample = indexed[row["source_frame"]]
        training_frames.add(row["source_frame"])
        templates[row["action"]] = crop(read_image(Path(sample["path"])), row["rect"])
    recognizer = ActionCandidate(profile, templates)
    for action, mask in recognizer.templates.items():
        save_image(output / "templates" / f"{action}.png", mask)
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    tight_table = stack_table(table, profile)
    rows = {}
    for index, sample in sorted(indexed.items()):
        picture = read_image(Path(sample["path"]))
        timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
            milliseconds=sample["container_pts_ms"])
        frame = Frame(index, timestamp, "offline-field-candidate",
                      WindowRect(0, 0, 498, 1080), picture, 498, 1080)
        stacks = vision._recognize_stacks(frame, tight_table, timestamp)
        rows[index] = {
            "source_frame": index, "actions": recognizer.recognize(picture),
            "stacks": {str(s.slot_id): field_record(s.field) for s in stacks},
        }
    counts = {"actions": Counter(), "stacks": Counter(),
              "action_template_source_frames": Counter(),
              "action_same_hand_other_frames": Counter()}
    checks, diagnostics, tiles = [], [], []
    baseline_rows = {r["source_frame"]: r for r in baseline_report["rows"]}
    for point in review["checkpoints"]:
        index = point["source_frame"]
        picture = read_image(Path(indexed[index]["path"]))
        stamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
        frame = Frame(index, stamp, "crop-diagnostic",
                      WindowRect(0, 0, 498, 1080), picture, 498, 1080)
        for slot in range(8):
            action = rows[index]["actions"][str(slot)]
            verdict = candidate_score(point["actions"][slot], action)
            counts["actions"][verdict] += 1
            group = ("action_template_source_frames" if index in training_frames else
                     "action_same_hand_other_frames")
            counts[group][verdict] += 1
            checks.append({"source_frame": index, "slot": slot, "field": "action",
                           "expected": point["actions"][slot], "verdict": verdict,
                           "candidate": action})
            stack = rows[index]["stacks"][str(slot)]
            verdict = score_read(point["stacks"][slot], stack, money=True)
            counts["stacks"][verdict] += 1
            checks.append({"source_frame": index, "slot": slot, "field": "stack",
                           "expected": point["stacks"][slot], "verdict": verdict,
                           "candidate": stack,
                           "baseline": baseline_rows[index]["stacks"][str(slot)]})
            original = next(r for r in table.rois
                            if r.kind is ROIKind.STACK and r.slot_id == slot)
            old_crop = extract_roi(frame, original)
            new_crop = crop(picture, profile["slots"][slot]["stack"])
            details = {}
            for label, patch in (("original", old_crop), ("tight", new_crop)):
                text, score = vision._stack_amount._decode(patch)
                details[label] = {"decoded_candidate_not_accepted": text,
                                  "score": score, "crop_shape": list(patch.shape),
                                  "glyph_shapes": [list(c.shape)
                                                   for c in segment_characters(patch)]}
                path = output / "stack-crops" / f"{index}-{slot}-{label}.png"
                save_image(path, patch)
            diagnostics.append({"source_frame": index, "slot": slot, **details})
            if index == 8500:
                tile = np.zeros((100, 480, 3), dtype=np.uint8)
                cv2.putText(tile, f"slot {slot}: original | tight", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                for left, patch in ((5, old_crop), (250, new_crop)):
                    resized = cv2.resize(patch, None, fx=2, fy=2,
                                         interpolation=cv2.INTER_NEAREST)
                    h, w = resized.shape[:2]
                    tile[28:28 + h, left:left + w] = resized
                tiles.append(tile)
    save_image(output / "stack-geometry-review.png", np.concatenate(tiles))
    report = {"candidate_only": True, "production_changed": False,
              "release_eligible": False, "events_emitted": 0,
              "baseline_report_sha256": sha256_file(baseline / "report.json"),
              "source_frames_processed": len(rows), "counts": counts,
              "template_source_frames": sorted(training_frames),
              "independence": "same-hand development including template-source frames",
              "rows": list(rows.values()), "checks": checks,
              "stack_diagnostics": diagnostics}
    if any(sha256_file(repo / p) != digest for p, digest in hashes.items()):
        raise ValueError("source changed during candidate probe")
    if any(sha256_file(Path(s["path"])) != s["image_sha256"] for s in indexed.values()):
        raise ValueError("source image changed during probe")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.profile, args.output), indent=2))
