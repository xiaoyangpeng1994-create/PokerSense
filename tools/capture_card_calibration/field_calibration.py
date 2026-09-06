# -*- coding: utf-8 -*-
"""Stage B6: measure a production field against labels and land its
``MeasuredCalibration`` block in ``configs/vision/<platform>/calibration.json``.

This is the missing link between the offline readers and production: the
``card_fused`` block was written by hand; every other field needs a tool
that turns "labels + recognizer" into an auditable measurement:

1. Run the **production** recognizer over every labeled frame (production
   ROI geometry, production segmentation, production scoring).
2. Score it against the confirmed ``VALID`` labels: correct / false VALID /
   UNKNOWN-on-positive; plus true negatives where the field is absent
   (e.g. an EMPTY seat has no stack pill).
3. Derive the ``MeasuredCalibration`` contract:
   ``floor``   = lowest raw score among correct reads,
   ``ceiling`` = highest raw score among non-correct observations,
   and require ``ceiling < floor`` (strictly separable) with
   ``false_valid == 0`` — the guide's zero-false-VALID rule.
4. With ``--write``, merge the block into ``calibration.json`` (backing the
   file up first). Without ``--write`` it only prints the report.

The measurement is honest only when templates were built WITHOUT the
measured frames: pass ``--eval-frames`` (a hand-isolated hold-out from
``stack_auto.hold_out_split``) to measure on frames the templates never
saw.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .dataset import read_frames_jsonl
from .production_assets import (
    PLATFORM_ID,
    RoiRect,
    iter_amount_targets,
    load_table_map_rois,
)
from .schema import FrameLabel, LabelStatus, Occupancy


@dataclass(frozen=True)
class ReadRecord:
    """One recognizer read scored against the labels."""

    frame: str
    slot_id: int | None
    kind: str  # "correct" | "false_valid" | "unknown_positive" | "negative"
    raw_score: float
    read_value: str | None
    truth_value: str | None


@dataclass(frozen=True)
class FieldMeasurement:
    """The numbers behind one field's calibration block."""

    field: str
    records: tuple[ReadRecord, ...]
    positives: int
    correct: int
    false_valid: int
    unknown_positive: int
    negatives: int
    floor: float | None
    ceiling: float | None
    details: dict = _dc_field(default_factory=dict)

    @property
    def separable(self) -> bool:
        return (
            self.floor is not None
            and self.ceiling is not None
            and self.ceiling < self.floor
            and self.false_valid == 0
        )

    def wilson_lower_bound(self) -> float:
        from poker_engine.perceptual.vision.calibration import (
            wilson_lower_bound_95,
        )

        return wilson_lower_bound_95(self.correct, self.positives)

    def to_block(self, source: str) -> dict[str, object]:
        """Render the ``MeasuredCalibration``-compatible JSON block."""
        if not self.separable:
            raise ValueError(
                f"field {self.field} is not separable: floor={self.floor} "
                f"ceiling={self.ceiling} false_valid={self.false_valid}"
            )
        return {
            "samples": self.positives,
            "correct": self.correct,
            "readable_score_floor": round(self.floor, 6),
            "unreadable_score_ceiling": round(self.ceiling, 6),
            "source": source,
        }

    def summary(self) -> dict[str, object]:
        return {
            "field": self.field,
            "positives": self.positives,
            "correct": self.correct,
            "false_valid": self.false_valid,
            "unknown_positive": self.unknown_positive,
            "negatives": self.negatives,
            "floor": self.floor,
            "ceiling": self.ceiling,
            "separable": self.separable,
            "wilson_lower_bound": (
                round(self.wilson_lower_bound(), 6) if self.positives else 0.0
            ),
            **self.details,
        }


def _read_frame(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def load_png_templates(directory: Path) -> dict[str, np.ndarray]:
    """Load ``<char>.png`` templates (unicode-safe) for offline scoring."""
    import cv2

    templates: dict[str, np.ndarray] = {}
    for path in sorted(directory.glob("*.png")):
        img = cv2.imdecode(
            np.frombuffer(path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if img is not None:
            templates[path.stem] = img
    if not templates:
        raise ValueError(f"no templates found in {directory}")
    return templates


def build_amount_recognizer(templates_dir: Path, min_score: float = 0.8):
    """Construct the production recognizer over the exported templates."""
    from poker_engine.perceptual.vision.amount_recognizer import (
        DigitTemplateSet,
        TemplateAmountRecognizer,
    )

    return TemplateAmountRecognizer(
        DigitTemplateSet(
            templates=load_png_templates(templates_dir),
            version=f"{PLATFORM_ID}-measure",
        ),
        min_score=min_score,
    )


def measure_amount_field(
    labels: Sequence[FrameLabel],
    frames_dir: Path,
    rois: Mapping[str, RoiRect | dict[int, RoiRect]],
    field: str,
    recognizer,
    *,
    eval_frames: set[str] | None = None,
) -> FieldMeasurement:
    """Score an amount field (pot / stack) against the confirmed labels.

    Positives are confirmed ``VALID`` labels. Negatives (stack only) are
    ``EMPTY`` seats — no pill, so the recognizer must abstain or score low.
    ``eval_frames`` restricts the measurement to a hand-isolated hold-out.
    """
    records: list[ReadRecord] = []
    frame_cache: dict[str, np.ndarray | None] = {}

    def frame_image(name: str) -> np.ndarray | None:
        if name not in frame_cache:
            path = frames_dir / name
            frame_cache[name] = _read_frame(path) if path.is_file() else None
        return frame_cache[name]

    for label, slot_id, text in iter_amount_targets(labels, field):
        if eval_frames is not None and label.frame not in eval_frames:
            continue
        img = frame_image(label.frame)
        if img is None:
            continue
        if field == "pot":
            rect = rois["pot"]
            assert isinstance(rect, RoiRect)
        else:
            bucket = rois["stack"]
            assert isinstance(bucket, dict)
            if slot_id not in bucket:
                continue
            rect = bucket[slot_id]
            assert isinstance(rect, RoiRect)
        rec = recognizer.recognize(rect.crop(img))
        read = None if rec.value is None else str(rec.value.value)
        if read is None:
            kind = "unknown_positive"
        elif read == text:
            kind = "correct"
        else:
            kind = "false_valid"
        records.append(ReadRecord(
            label.frame, slot_id, kind, float(rec.raw_score), read, text,
        ))

    negatives = 0
    if field == "stack":
        bucket = rois["stack"]
        assert isinstance(bucket, dict)
        for label in labels:
            if not label.stable or label.scene.value != "table":
                continue
            if eval_frames is not None and label.frame not in eval_frames:
                continue
            img = frame_image(label.frame)
            if img is None:
                continue
            for slot in label.slots:
                if not (
                    slot.occupancy.status is LabelStatus.VALID
                    and slot.occupancy.value == Occupancy.EMPTY.value
                ):
                    continue
                if slot.slot_id not in bucket:
                    continue
                rect = bucket[slot.slot_id]
                assert isinstance(rect, RoiRect)
                rec = recognizer.recognize(rect.crop(img))
                negatives += 1
                records.append(ReadRecord(
                    label.frame, slot.slot_id, "negative",
                    float(rec.raw_score),
                    None if rec.value is None else str(rec.value.value),
                    None,
                ))

    positives = sum(
        1 for r in records
        if r.kind in ("correct", "false_valid", "unknown_positive")
    )
    correct = sum(1 for r in records if r.kind == "correct")
    false_valid = sum(1 for r in records if r.kind == "false_valid")
    unknown = sum(1 for r in records if r.kind == "unknown_positive")
    correct_scores = [r.raw_score for r in records if r.kind == "correct"]
    other_scores = [
        r.raw_score for r in records if r.kind != "correct"
    ]
    return FieldMeasurement(
        field=field,
        records=tuple(records),
        positives=positives,
        correct=correct,
        false_valid=false_valid,
        unknown_positive=unknown,
        negatives=negatives,
        floor=min(correct_scores) if correct_scores else None,
        ceiling=max(other_scores) if other_scores else 0.0,
    )


def update_calibration_json(
    repo_root: Path,
    blocks: Mapping[str, Mapping[str, object]],
) -> Path:
    """Merge field blocks into calibration.json, backing up first."""
    path = (
        repo_root
        / "configs"
        / "vision"
        / PLATFORM_ID
        / "calibration.json"
    )
    backup = path.with_suffix(".json.bak")
    shutil.copyfile(path, backup)
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, block in blocks.items():
        data[name] = dict(block)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return backup


# --- CLI -------------------------------------------------------------------


def _load_frame_set(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--field", choices=("pot", "stack"), required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--eval-frames", type=Path, default=None)
    parser.add_argument("--write", action="store_true",
                        help="merge the block into calibration.json")
    args = parser.parse_args()

    labels = read_frames_jsonl(args.root / "labels" / "frames.jsonl")
    rois = load_table_map_rois(args.repo)
    recognizer = build_amount_recognizer(args.templates, args.min_score)
    eval_frames = (
        _load_frame_set(args.eval_frames) if args.eval_frames else None
    )
    measurement = measure_amount_field(
        labels,
        args.root / "normalized" / "frames",
        rois,
        args.field,
        recognizer,
        eval_frames=eval_frames,
    )
    report = measurement.summary()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write:
        source = (
            f"field_calibration.py on {args.root.name} "
            f"({'eval hold-out' if eval_frames else 'all labeled frames'}); "
            f"zero false VALID, Wilson 95% lower bound "
            f"{report['wilson_lower_bound']}"
        )
        block = measurement.to_block(source)
        # The engine reads the pot amount under its generic field name
        # "amount" (see live.load_measured_calibrations); a block keyed
        # "pot" would be invisible to production and force UNKNOWN.
        engine_name = "amount" if args.field == "pot" else args.field
        backup = update_calibration_json(args.repo, {engine_name: block})
        if engine_name != args.field:
            path = (
                args.repo / "configs" / "vision" / PLATFORM_ID
                / "calibration.json"
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            stale = data.pop(args.field, None)
            if stale is not None:
                path.write_text(
                    json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"removed stale {args.field!r} key")
        print(f"written: {engine_name} -> calibration.json (backup {backup})")
    return 0 if measurement.separable else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FieldMeasurement",
    "ReadRecord",
    "build_amount_recognizer",
    "load_png_templates",
    "measure_amount_field",
    "update_calibration_json",
]
