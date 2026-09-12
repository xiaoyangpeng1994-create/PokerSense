"""Evaluate reviewed session001 gray bank on source-bound session002 observations."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.probe_wpk_hero_layout import bind_windows, review_points
from tools.wpk_field_candidate import crop
from tools.wpk_gray_amount import load_reviewed_bank
from tools.wpk_hero_balance_layout import HeroBalanceLayout, LOCATIONS, DIGIT_RECTS
from tools.wpk_video_dataset import read_image


def verdict(expected, value):
    if expected is None:
        return "correct_rejection" if value is None else "false_accept"
    return "abstain" if value is None else "correct" if value == expected else "wrong"


def run(bank, windows, reviews, profile_path, output):
    if output.exists():
        raise ValueError("preserve previous evaluations")
    input_files = [*reviews, profile_path, bank / "inventory.json",
                   bank / "proposals.npz", bank / "visual-review.json"]
    input_hashes = {str(path): sha256_file(path) for path in input_files}
    truth = [(p.stem, json.loads(p.read_text())) for p in reviews]
    sources = {data["source_sha256"] for _, data in truth}
    if len(sources) != 1:
        raise ValueError("one evaluation source required")
    samples = bind_windows(windows, sources.pop())
    profile = json.loads(profile_path.read_text())
    for _, data in truth:
        if any(p["source_frame"] not in samples for p in review_points(data)):
            raise ValueError("missing evaluation frame")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    dependencies = list((repo / "src").rglob("*.py"))
    dependencies += list((repo / "tools").rglob("*.py"))
    dependencies += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    source_hashes = {}
    for path in dependencies:
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo)
        digest = sha256_file(path)
        destination = output / "source-snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256_file(destination) != digest:
            raise ValueError("snapshot changed")
        source_hashes[str(relative)] = digest
    for index, path in enumerate(input_files):
        shutil.copy2(path, output / f"input-{index}-{path.name}")
    (output / "source-hashes.json").write_text(json.dumps(source_hashes, indent=2))
    (output / "input-hashes.json").write_text(json.dumps(input_hashes, indent=2))
    (output / "bound-samples.json").write_text(
        json.dumps(list(samples.values()), indent=2))
    readers = {"plain": load_reviewed_bank(bank),
               "augmented": load_reviewed_bank(bank, augment=True)}
    _, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    selector = None
    if {10700, 11000} <= samples.keys():
        pairs = (("lower", 11000), ("raised", 10700))
        templates = {key: crop(read_image(Path(samples[index]["path"])), LOCATIONS[key])
                     for key, index in pairs}
        selector = HeroBalanceLayout(templates)
    checks, groups, predictions, timings = [], {}, {}, []
    for index, sample in sorted(samples.items()):
        image = read_image(Path(sample["path"]))
        layout = selector.recognize(image) if selector else None
        predictions[index] = {}
        for slot in range(8):
            rect = profile["slots"][slot]["stack"]
            if slot == 0 and layout and layout.location:
                rect = DIGIT_RECTS[layout.location]
            patch = crop(image, rect)
            started = time.perf_counter()
            read = readers["augmented"].diagnose(patch)
            timings.append(time.perf_counter() - started)
            base = vision._stack_amount.recognize(patch)
            base_value = (str(base.value.value) if base.value is not None
                          and not vision._cal("stack").should_abstain(base.raw_score)
                          else None)
            blocked = slot == 0 and layout is not None and layout.location is None
            predictions[index][str(slot)] = {
                "gray": asdict(read), "old_value": base_value,
                "value": None if blocked else read.value,
                "geometry_unknown": blocked, "rect": rect,
                "usable_stack_verified": False}
    for name, data in truth:
        groups[name] = {"old": Counter(), "gray": Counter()}
        for point in review_points(data):
            expected = ({0: point["visible_digits"]} if "visible_digits" in point else
                        dict(enumerate(point["stacks"])))
            for slot, value in expected.items():
                read = predictions[point["source_frame"]][str(slot)]
                for algorithm, prediction in (("old", read["old_value"]),
                                              ("gray", read["value"])):
                    status = verdict(value, prediction)
                    groups[name][algorithm][status] += 1
                checks.append({"group": name, "source_frame": point["source_frame"],
                               "slot": slot, "expected": value, "read": read})
    report = {"groups": groups, "checks": checks, "predictions": predictions,
              "frames_processed": len(predictions), "source_glyphs": 51,
              "augmented_vectors": len(readers["augmented"].labels),
              "training_source": "session_001_only_reviewed_glyphs",
              "candidate_only": True, "release_eligible": False,
              "per_slot_mean_ms": sum(timings) / len(timings) * 1000,
              "per_slot_max_ms": max(timings) * 1000,
              "timing_scope": "local gray matching only, not capture-to-render"}
    if any(sha256_file(Path(p)) != h for p, h in input_hashes.items()):
        raise ValueError("input changed during evaluation")
    if any(sha256_file(repo / p) != h for p, h in source_hashes.items()):
        raise ValueError("source changed during evaluation")
    if samples != bind_windows(windows, truth[0][1]["source_sha256"]):
        raise ValueError("source images changed")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("window", "review"):
        parser.add_argument("--" + name, type=Path, required=True, action="append")
    args = parser.parse_args()
    result = run(args.bank, args.window, args.review, args.profile, args.output)
    print(json.dumps(result, indent=2))
