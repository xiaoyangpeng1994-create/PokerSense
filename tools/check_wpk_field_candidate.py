"""Check a frozen offline candidate on pre-reviewed later frames, without tuning."""

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
from tools.probe_wpk_field_candidate import candidate_score
from tools.probe_wpk_observation_fields import field_record, score_read
from tools.wpk_field_candidate import ActionCandidate, stack_table
from tools.wpk_video_dataset import pixels_digest, read_image


def run(candidate, window, review_path, output):
    if output.exists():
        raise ValueError("do not overwrite earlier checks")
    for folder in (candidate, window):
        if verify_sha256sums(folder):
            raise ValueError("evidence integrity failure")
    repo = Path(__file__).resolve().parents[1]
    source_hashes = json.loads((candidate / "source-hashes.json").read_text())
    for relative, digest in source_hashes.items():
        if sha256_file(repo / relative) != digest:
            raise ValueError("frozen candidate changed")
    review_hash = sha256_file(review_path)
    review = json.loads(review_path.read_text())
    summary = json.loads((window / "summary.json").read_text())
    if summary["source_sha256"] != review["source_sha256"]:
        raise ValueError("different source video")
    samples = {s["source_frame"]: s for s in json.loads(
        (window / "samples.json").read_text())}
    profile = json.loads((candidate / "source-snapshot/tests/fixtures/"
                          "wpk_reference_hands/field_candidate_v1.json").read_text())
    frames = [p["source_frame"] for p in review["checkpoints"]]
    if len(frames) != len(set(frames)) or not frames:
        raise ValueError("duplicate or absent review frames")
    if set(frames) & {r["source_frame"] for r in profile["new_templates"]}:
        raise ValueError("template source is not a followup")
    for point in review["checkpoints"]:
        if len(point["actions"]) != 8 or len(point["stacks"]) != 8:
            raise ValueError("review requires all eight slots")
    output.mkdir(parents=True)
    shutil.copy2(review_path, output / "review.json")
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    masks = {p.stem: read_image(p) for p in (candidate / "templates").glob("*.png")}
    reader = ActionCandidate(profile, masks)
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    tight = stack_table(table, profile)
    rows, bound = {}, []
    for index in sorted(frames):
        sample = samples[index]
        path = (window / sample["image"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("image escaped window")
        image = read_image(path)
        if (sha256_file(path) != sample["image_sha256"]
                or pixels_digest(image) != sample["pixel_sha256"]):
            raise ValueError("followup image changed")
        bound.append(sample)
        stamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
        frame = Frame(index, stamp, "offline-static-followup",
                      WindowRect(0, 0, 498, 1080), image, 498, 1080)
        rows[index] = {"actions": reader.recognize(image)}
        for name, geometry in (("baseline_stacks", table), ("candidate_stacks", tight)):
            stacks = vision._recognize_stacks(frame, geometry, stamp)
            rows[index][name] = {str(s.slot_id): field_record(s.field) for s in stacks}
    counts = {k: Counter() for k in ("actions", "baseline_stacks", "candidate_stacks")}
    checks = []
    for point in review["checkpoints"]:
        index = point["source_frame"]
        for slot in range(8):
            for kind in counts:
                read = rows[index][kind][str(slot)]
                expected = point["actions" if kind == "actions" else "stacks"][slot]
                verdict = (candidate_score(expected, read) if kind == "actions" else
                           score_read(expected, read, money=True))
                counts[kind][verdict] += 1
                checks.append({"source_frame": index, "slot": slot, "kind": kind,
                               "expected": expected, "read": read, "verdict": verdict})
    report = {"candidate_only": True, "release_eligible": False,
              "independent_project_holdout": False, "thresholds_retuned": False,
              "candidate_report_sha256": sha256_file(candidate / "report.json"),
              "review_sha256": review_hash, "counts": counts, "checks": checks,
              "bound_samples": bound,
              "scope": "static later-frame transfer; prior card-development footage",
              "zero_semantics": "numeric screen text, not proof of playable balance"}
    if sha256_file(review_path) != review_hash:
        raise ValueError("review changed during check")
    if any(sha256_file(repo / p) != h for p, h in source_hashes.items()):
        raise ValueError("candidate changed during check")
    for sample in bound:
        if sha256_file(window / sample["image"]) != sample["image_sha256"]:
            raise ValueError("input image changed during check")
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "window", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.candidate, args.window, args.review, args.output)
    print(json.dumps(result, indent=2))
