"""Source-bound pixel/field parity audit, never an action-event or advice emitter.

Uses the production VisionEngine without changing gates. Raw action candidates
are reported separately, never promoted. Reviewed screen truth is used only
after inference; logical pot values must not replace displayed digits.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import platform
import shutil

import cv2
import numpy as np

from poker_engine.core.enums import ActionType
from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.roi import extract_roi
from poker_engine.perceptual.vision.table_map import ROIKind
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import pixels_digest, read_image


def validate_review(review):
    points = review["checkpoints"]
    if not points or review["slot_to_source_seat"] != list(range(8)):
        raise ValueError("this reviewed layout requires eight explicit identity slots")
    seen = set()
    for point in points:
        frame = point["source_frame"]
        if type(frame) is not int or frame < 0 or frame in seen:
            raise ValueError("invalid or duplicate source frame")
        seen.add(frame)
        if point["window"] not in ("hand", "cash"):
            raise ValueError("unknown window")
        if len(point["stacks"]) != 8 or len(point["actions"]) != 8:
            raise ValueError("all eight slots need an explicit value or negative")
        for value in [point["pot"], *point["stacks"]]:
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("money must be a decimal string or null")
                number = Decimal(value)
                if not number.is_finite() or number < 0:
                    raise ValueError("invalid money")
        for action in point["actions"]:
            if action is not None:
                ActionType(action)


def score_read(expected, read, *, money=False):
    """Never give a rejected candidate credit as a usable observation."""
    accepted = read["status"] == "valid" and read["value"] is not None
    if expected is None:
        return "false_accept" if accepted else "correct_rejection"
    if not accepted:
        return "abstain"
    actual = read["value"]
    matches = Decimal(actual) == Decimal(expected) if money else actual == expected
    return "correct" if matches else "wrong"


def field_record(field):
    value = field.value
    return {
        "value": str(value.value) if value is not None else None,
        "status": field.validation_status.value,
        "raw_score": field.evidence.get("raw_score"),
        "confidence": field.confidence,
    }


def read_fields(vision, table, frame):
    """No review values enter this function; use the unmodified production path."""
    obs = vision.process(frame, table)
    candidates = {}
    for roi in table.rois:
        if roi.kind is ROIKind.ACTION:
            rec = vision._action.recognize(extract_roi(frame, roi), roi.slot_id)
            candidates[str(roi.slot_id)] = {
                "value": rec.value.value if rec.value is not None else None,
                "raw_score": rec.raw_score, "runner_up_score": rec.runner_up_score,
                "not_calibrated_not_an_event": True,
            }
    return {
        "source_frame": frame.frame_seq,
        "pot": field_record(obs.pot),
        "stacks": {str(s.slot_id): field_record(s.field) for s in obs.slot_stacks},
        "actions": {str(s.slot_id): field_record(s.field) for s in obs.slot_actions},
        "raw_action_candidates": candidates,
    }


def bind_samples(review, windows):
    """Verify cached extraction bytes/pixels, not a new raw-video decode claim."""
    result = {}
    identities = {}
    for name, folder in windows.items():
        problems = verify_sha256sums(folder)
        if problems:
            raise ValueError(f"window integrity failure: {problems}")
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary["source_sha256"] != review["source_sha256"]:
            raise ValueError("window belongs to another source")
        samples = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
        for sample in samples:
            frame = sample["source_frame"]
            pts = sample["container_pts_ms"]
            if type(frame) is not int or frame < 0 or not math.isfinite(pts) or pts < 0:
                raise ValueError("invalid source frame or container PTS")
            identity = sample["pixel_sha256"], pts
            if frame in identities and identities[frame] != identity:
                raise ValueError("overlapping windows disagree on pixels or PTS")
            identities[frame] = identity
            path = (folder / sample["image"]).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("sample path escaped window")
            if sha256_file(path) != sample["image_sha256"]:
                raise ValueError("sample image hash mismatch")
            if pixels_digest(read_image(path)) != sample["pixel_sha256"]:
                raise ValueError("sample pixel hash mismatch")
            if (name, frame) in result:
                raise ValueError("duplicate sample")
            result[name, frame] = {**sample, "path": str(path), "window": name}
    for point in review["checkpoints"]:
        if (point["window"], point["source_frame"]) not in result:
            raise ValueError("missing reviewed source frame")
    return result


def compare(review, rows):
    totals = defaultdict(Counter)
    checks, mismatches = [], []
    for point in review["checkpoints"]:
        row = rows[point["source_frame"]]
        for kind in ("pot", "stacks", "actions"):
            expected = {None: point[kind]} if kind == "pot" else dict(
                enumerate(point[kind]))
            for slot, value in expected.items():
                read = row[kind] if slot is None else row[kind].get(str(slot), {
                    "status": "unknown", "value": None, "reason": "missing_roi",
                })
                verdict = score_read(value, read, money=kind != "actions")
                totals[kind][verdict] += 1
                checks.append({"source_frame": point["source_frame"],
                               "field": kind, "slot": slot, "expected": value,
                               "observed": read, "verdict": verdict})
        if "ledger_pot" in point and point["pot"] != point["ledger_pot"]:
            mismatches.append({"source_frame": point["source_frame"],
                               "displayed_pot": point["pot"],
                               "logical_pot": point["ledger_pot"],
                               "eligible_for_atomic_state_match": False})
    return {"counts": dict(totals), "checks": checks,
            "display_vs_ledger_mismatches": mismatches}


def run(review_path, windows, output):
    if output.exists():
        raise ValueError("preserve prior observation reports")
    review_hash = sha256_file(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review)
    samples = bind_samples(review, windows)
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    shutil.copy2(review_path, output / "review.json")
    hashes = {}
    sources = list((repo / "src").rglob("*.py"))
    sources += list((repo / "tools").rglob("*.py"))
    sources += [p for p in (repo / "configs").rglob("*") if p.is_file()]
    sources += [repo / "pyproject.toml"]
    for source in sorted(sources):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(repo)
        digest = sha256_file(source)
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != digest:
            raise ValueError("source changed during snapshot")
        hashes[relative.as_posix()] = digest
    (output / "source-hashes.json").write_text(json.dumps(hashes, indent=2))
    (output / "bound-samples.json").write_text(
        json.dumps(list(samples.values()), indent=2), encoding="utf-8")
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    folder = repo / "configs/vision" / CAPTURE_CARD_PLATFORM
    calibration = json.loads((folder / "calibration.json").read_text())
    action_slots = {roi.slot_id for roi in table.rois if roi.kind is ROIKind.ACTION}
    template_actions = {p.stem for p in (folder / "action_glyph").glob("*.png")}
    gaps = {
        "action_slots_missing": sorted(set(range(8)) - action_slots),
        "action_calibration_present": "action" in calibration,
        "action_templates_missing": sorted(
            {"fold", "check", "call", "bet", "raise", "all_in"} - template_actions),
    }
    rows = {}
    seen_pixels = {}
    for sample in sorted(samples.values(), key=lambda s: s["source_frame"]):
        index = sample["source_frame"]
        if index in rows:
            if sample["pixel_sha256"] != seen_pixels[index]:
                raise ValueError("overlapping windows disagree on pixels")
            continue
        seen_pixels[index] = sample["pixel_sha256"]
        picture = read_image(Path(sample["path"]))
        if pixels_digest(picture) != sample["pixel_sha256"]:
            raise ValueError("sample changed before inference")
        height, width = picture.shape[:2]
        stamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
            milliseconds=sample["container_pts_ms"])
        frame = Frame(index, stamp, "offline-observation-audit",
                      WindowRect(0, 0, width, height), picture, width, height)
        rows[index] = read_fields(vision, table, frame)
    # Only now score against screen truth; no logical cash or later cards input.
    report = {**compare(review, rows), "frames_processed": len(rows),
              "review_sha256": review_hash, "source_sha256": review["source_sha256"],
              "capture_to_state_verified": False, "release_eligible": False,
              "independent_holdout": False, "actions_emitted": 0,
              "configuration_gaps": gaps,
              "runtime": {"python": platform.python_version(),
                          "numpy": np.__version__, "opencv": cv2.__version__},
              "scope": "Cached source-bound samples; not continuous raw-video replay",
              "rows": list(rows.values())}
    if sha256_file(review_path) != review_hash or any(
        sha256_file(repo / p) != digest for p, digest in hashes.items()
    ):
        raise ValueError("review or source changed during measurement")
    if samples != bind_samples(review, windows):
        raise ValueError("input windows changed during measurement")
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_sha256sums(output)
    keys = ("frames_processed", "counts", "display_vs_ledger_mismatches",
            "configuration_gaps")
    return {key: report[key] for key in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--hand-window", type=Path, required=True)
    parser.add_argument("--cash-window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = {"hand": args.hand_window, "cash": args.cash_window}
    print(json.dumps(run(args.review, windows, args.output), indent=2))
