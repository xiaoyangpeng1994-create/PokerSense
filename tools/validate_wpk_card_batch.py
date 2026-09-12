"""Replay a frozen, visually labeled WPK card batch without changing the model.

Candidate-only: keep production's calibration gate closed. Report wrong
accepted cards separately from abstentions, absent-slot errors and coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time

import cv2

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, _resource_root, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, load_card_heads,
)
from tools.capture_card_calibration.hashing import sha256_file
from tools.wpk_video_dataset import pixels_digest


def score_slots(expected: list[str], observed: list[str | None], width: int) -> dict:
    if len(expected) > width or len(observed) != width:
        raise ValueError("invalid checkpoint slot dimensions")
    truth = expected + [None] * (width - len(expected))
    counts = Counter()
    errors = []
    for index, (want, got) in enumerate(zip(truth, observed)):
        if want is not None:
            counts["positive_slots"] += 1
        else:
            counts["absent_slots"] += 1
        if got is None:
            key = "abstain_on_positive" if want is not None else "correct_absence"
            counts[key] += 1
        elif got == want:
            counts["correct_accepted"] += 1
        else:
            counts["wrong_accepted"] += 1
            if want is None:
                counts["false_positive_on_absent"] += 1
            errors.append({"slot": index, "expected": want, "observed": got})
    return {"counts": dict(counts), "complete_match": truth == observed,
            "errors": errors}


def summarize(rows: list[dict], criteria: dict) -> dict:
    fields = {}
    all_pass = True
    for name in ("hero", "board"):
        counts = Counter()
        required = complete = 0
        for row in rows:
            counts.update(row[name]["counts"])
            if row[f"require_{name}_complete"]:
                required += 1
                complete += row[name]["complete_match"]
        coverage = complete / required if required else None
        passed = counts["wrong_accepted"] <= criteria["max_wrong_accepted"]
        if coverage is None or coverage < criteria["min_required_complete_fraction"]:
            passed = False
        all_pass = all_pass and passed
        accepted = counts["correct_accepted"] + counts["wrong_accepted"]
        fields[name] = {"counts": dict(counts), "required_checkpoints": required,
                        "complete_required_checkpoints": complete, "coverage": coverage,
                        "accepted_precision": counts["correct_accepted"] / accepted
                        if accepted else None, "passed": passed}
    return {"fields": fields, "passed": all_pass,
            "production_calibration_revalidated": False}


class RecordingRecognizer:
    def __init__(self, delegate, floor: float):
        self.delegate = delegate
        self.floor = floor
        self.reads = {}

    def begin_frame(self, *args):
        begin = getattr(self.delegate, "begin_frame", None)
        if callable(begin):
            begin(*args)

    def reset(self):
        reset = getattr(self.delegate, "reset", None)
        if callable(reset):
            reset()
        self.reads.clear()

    def recognize(self, image, card_model=None):
        result = self.delegate.recognize(image, card_model)
        if isinstance(card_model, tuple):
            accepted = result.value is not None and result.raw_score >= self.floor
            self.reads[card_model] = {
                "card": str(result.value[0]) if accepted else None,
                "raw_score": result.raw_score,
                "rank_score": result.slots[0].rank_score if result.slots else None,
                "suit_score": result.slots[0].suit_score if result.slots else None,
            }
        return result


def checked_candidate_heads(spec: dict, base: Path, default: Path) -> Path:
    """An offline override must be copied into the frozen run and hash-bound."""
    candidate = spec.get("candidate_heads")
    if candidate is None:
        return default
    relative = Path(candidate["relative_path"])
    path = (base / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError("candidate heads escape frozen run")
    for key, file in (("npz_sha256", path), ("json_sha256", path.with_suffix(".json"))):
        if sha256_file(file) != candidate[key]:
            raise ValueError("candidate head artifact changed")
    return path


def checked_rank_floor(spec: dict, base: Path, default: float) -> float:
    artifact = spec.get("rank_calibration")
    if artifact is None:
        return default
    relative = Path(artifact["relative_path"])
    path = (base / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError("rank calibration escapes frozen run")
    if sha256_file(path) != artifact["sha256"]:
        raise ValueError("rank calibration changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["candidate_npz_sha256"] != spec["candidate_heads"]["npz_sha256"]:
        raise ValueError("rank calibration is for another model")
    floor = report["chosen_rank_floor"]
    if (isinstance(floor, bool) or not isinstance(floor, (int, float))
            or not math.isfinite(floor) or not default <= floor <= 1):
        raise ValueError("invalid calibrated rank floor")
    return float(floor)


def validate(corpus: Path, spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_digest = sha256_file(spec_path)
    repo = _resource_root()
    for relative, expected in spec["frozen_model_files"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo.resolve()) or sha256_file(path) != expected:
            raise ValueError(f"frozen model changed: {relative}")
    if sha256_file(Path(__file__)) != spec["validator_sha256"]:
        raise ValueError("validator changed after batch freeze")
    inventory = json.loads((corpus / "sources.json").read_text(encoding="utf-8"))
    source = next(row for row in inventory["sources"]
                  if row["session"] == spec["session"])
    if source["sha256"] != spec["source_sha256"]:
        raise ValueError("source hash mismatch")
    path = Path(source["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        source["size_bytes"], source["mtime_ns"]
    ):
        raise ValueError("source changed")
    folder = repo / "configs/vision" / CAPTURE_CARD_PLATFORM
    cal = json.loads((folder / "calibration.json")
                     .read_text(encoding="utf-8"))["card_fused"]
    config = NormalizationConfig.from_json(
        (folder / "normalization.json").read_text(encoding="utf-8"))
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    heads_path = checked_candidate_heads(
        spec, spec_path.parent, folder / "card_heads.npz")
    rank_floor = checked_rank_floor(spec, spec_path.parent, cal["rank_floor"])
    # Explicit offline candidate instance; no production acceptance flag changes.
    recorder = RecordingRecognizer(FusedCardRecognizerAdapter(FusedCardRecognizer(
        load_card_heads(heads_path), rank_floor=rank_floor,
        suit_floor=cal["suit_floor"])), floor=cal["suit_floor"])
    vision._card = recorder
    checkpoints = {row["source_frame"]: row for row in spec["checkpoints"]}
    if len(checkpoints) != len(spec["checkpoints"]) or not checkpoints:
        raise ValueError("empty or duplicate checkpoints")
    warmup = spec["warmup_start_frame"]
    if warmup > min(checkpoints) or warmup < 0:
        raise ValueError("invalid warmup")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("cannot open frozen video")
    rows = []
    started = time.monotonic()
    try:
        for index in range(max(checkpoints) + 1):
            if index < warmup:
                if not capture.grab():
                    raise ValueError("source ended during skip")
                continue
            ok, raw = capture.read()
            if not ok:
                raise ValueError("source ended before checkpoints")
            picture = normalize(raw, config)
            timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(
                milliseconds=capture.get(cv2.CAP_PROP_POS_MSEC))
            recorder.reads.clear()
            obs = vision.process(Frame(
                index, timestamp, "frozen-batch-offline",
                WindowRect(0, 0, 498, 1080), picture, 498, 1080), table)
            if index in checkpoints:
                expected = checkpoints[index]
                if pixels_digest(picture) != expected["pixel_sha256"]:
                    raise ValueError(f"checkpoint pixel mismatch: {index}")
                result = {"source_frame": index,
                          "participation": expected["participation"],
                          "require_hero_complete": expected["require_hero_complete"],
                          "require_board_complete": expected["require_board_complete"]}
                for name, width in (("hero", 2), ("board", 5)):
                    reads = [recorder.reads.get((name, i), {"card": None})
                             for i in range(width)]
                    result[name] = score_slots(
                        expected[name], [r["card"] for r in reads], width)
                    result[name]["reads"] = reads
                result["observation_status"] = {
                    "hero": obs.hero_cards.validation_status.value,
                    "board": obs.board_cards.validation_status.value,
                    "street": obs.street.validation_status.value,
                }
                rows.append(result)
                print(json.dumps({
                    "stage": "checkpoint", "frame": index,
                    "hero_match": result["hero"]["complete_match"],
                    "board_match": result["board"]["complete_match"]}), flush=True)
            if (index - warmup) % 500 == 0:
                elapsed = round(time.monotonic() - started, 1)
                print(json.dumps({"stage": "progress", "frame": index,
                                  "elapsed_s": elapsed}), flush=True)
    finally:
        capture.release()
    if sha256_file(spec_path) != spec_digest:
        raise ValueError("specification changed during evaluation")
    checked_candidate_heads(spec, spec_path.parent, folder / "card_heads.npz")
    checked_rank_floor(spec, spec_path.parent, cal["rank_floor"])
    for relative, expected in spec["frozen_model_files"].items():
        if sha256_file(repo / relative) != expected:
            raise ValueError("model changed during evaluation")
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("source changed during evaluation")
    return {"batch_id": spec["batch_id"], "spec_sha256": spec_digest,
            "candidate_only": True, "independence_scope": spec["independence_scope"],
            "summary": summarize(rows, spec["criteria"]), "checkpoints": rows,
            "elapsed_s": round(time.monotonic() - started, 3)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("result already exists; preserve the first frozen evaluation")
    report = validate(args.corpus, args.spec)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(0 if report["summary"]["passed"] else 2)
