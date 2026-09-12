"""Source-bound offline Hero balance geometry audit, with no event promotion."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.probe_wpk_field_candidate import save_image
from tools.probe_wpk_observation_fields import score_read
from tools.wpk_hero_balance_layout import (
    DIGIT_RECTS, LOCATIONS, HeroBalanceLayout, crop, read_hero_balance,
)
from tools.wpk_video_dataset import pixels_digest, read_image


def review_points(review):
    points = list(review.get("checkpoints", []))
    for interval in review.get("intervals", []):
        start, end = interval["start"], interval["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("invalid reviewed interval")
        points.extend({**interval, "source_frame": index}
                      for index in range(start, end + 1))
    ids = [point["source_frame"] for point in points]
    if not points or len(ids) != len(set(ids)):
        raise ValueError("missing or duplicate truth frames")
    return points


def bind_windows(windows, expected_source):
    samples = {}
    for folder in windows:
        if verify_sha256sums(folder):
            raise ValueError("window integrity failure")
        summary = json.loads((folder / "summary.json").read_text())
        if summary["source_sha256"] != expected_source:
            raise ValueError("different video source")
        for sample in json.loads((folder / "samples.json").read_text()):
            path = (folder / sample["image"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("escaped input path")
            if sha256_file(path) != sample["image_sha256"] or (
                pixels_digest(read_image(path)) != sample["pixel_sha256"]
            ):
                raise ValueError("source pixels changed")
            index = sample["source_frame"]
            fields = ("pixel_sha256", "container_pts_ms")
            if index in samples and any(samples[index][key] != sample[key]
                                        for key in fields):
                raise ValueError("overlapping samples disagree")
            samples[index] = {**sample, "path": str(path)}
    return samples


def run(windows, review_path, output):
    if output.exists():
        raise ValueError("do not overwrite earlier layout audits")
    review_hash = sha256_file(review_path)
    review = json.loads(review_path.read_text())
    points = review_points(review)
    samples = bind_windows(windows, review["source_sha256"])
    required = set(review["templates"].values()) | {
        p["source_frame"] for p in points}
    if not required <= samples.keys():
        raise ValueError("required reviewed frame absent")
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    shutil.copy2(review_path, output / "review.json")
    files = list((repo / "src").rglob("*.py"))
    files += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    files += list((repo / "tools").rglob("*.py"))
    hashes = {}
    for path in files:
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo)
        digest = sha256_file(path)
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ValueError("snapshot changed")
        hashes[relative.as_posix()] = digest
    (output / "source-hashes.json").write_text(json.dumps(hashes, indent=2))
    (output / "bound-samples.json").write_text(
        json.dumps(list(samples.values()), indent=2))
    templates = {key: crop(read_image(Path(samples[index]["path"])), LOCATIONS[key])
                 for key, index in review["templates"].items()}
    selector = HeroBalanceLayout(templates, review["floor"])
    for key, feature in selector.templates.items():
        save_image(output / "templates" / f"{key}.png", feature)
    _, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    rows, tiles = {}, []
    for index, sample in sorted(samples.items()):
        picture = read_image(Path(sample["path"]))
        read = read_hero_balance(
            picture, selector, vision._stack_amount, vision._cal("stack"))
        old = vision._stack_amount.recognize(crop(picture, DIGIT_RECTS["lower"]))
        old_valid = old.value is not None and not vision._cal("stack").should_abstain(
            old.raw_score)
        rows[index] = {"source_frame": index,
                       "container_pts_ms": sample["container_pts_ms"],
                       **read, "fixed_lower": {
                           "status": "valid" if old_valid else "unknown",
                           "value": str(old.value.value) if old_valid else None}}
        if 10530 <= index <= 10580 or 10860 <= index <= 10890:
            tile = np.zeros((132, 108, 3), np.uint8)
            cv2.putText(tile, str(index), (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 255, 255), 1)
            tile[22:62, 4:104] = picture[860:900, 199:299]
            tile[66:94, 10:98] = crop(picture, LOCATIONS["raised"])
            tile[98:126, 10:98] = crop(picture, LOCATIONS["lower"])
            tiles.append(tile)
    sheet = np.zeros((((len(tiles) + 9) // 10) * 132, 1080, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // 10) * 132, (i % 10) * 108
        sheet[y:y + 132, x:x + 108] = tile
    save_image(output / "transition-review.png", sheet)
    counts = {key: Counter() for key in ("layout", "selected_amount", "fixed_amount")}
    checks = []
    for point in points:
        read = rows[point["source_frame"]]
        result = ("correct" if read["location"] == point["location"] else
                  "abstain" if read["location"] is None else "wrong")
        counts["layout"][result] += 1
        verdict = score_read(point["visible_digits"], read, money=True)
        counts["selected_amount"][verdict] += 1
        counts["fixed_amount"][score_read(
            point["visible_digits"], read["fixed_lower"], money=True)] += 1
        checks.append({**point, "layout_result": result, "read": read})
    report = {"counts": counts, "checks": checks, "rows": list(rows.values()),
              "frames_processed": len(rows), "review_sha256": review_hash,
              "candidate_only": True, "release_eligible": False,
              "events_emitted": 0, "participation_inferred": False,
              "independent_holdout": False,
              "dense_truth_reviewed": bool(review.get("intervals"))}
    if sha256_file(review_path) != review_hash:
        raise ValueError("review changed")
    if any(sha256_file(repo / p) != h for p, h in hashes.items()):
        raise ValueError("source changed")
    if samples != bind_windows(windows, review["source_sha256"]):
        raise ValueError("evidence changed")
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, action="append", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.window, args.review, args.output), indent=2))
