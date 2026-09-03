"""Land capture-card geometry measurements into the production config tree.

Stage L (guide section 13) promotes *accepted* measurements into
``configs/platform/`` and ``configs/vision/``. This tool does the mechanical
half of that promotion: it re-reads the Stage E ROI measurement CSV and the
normalized frames, derives per-card sub-ROIs, and writes the production
``TableMap`` and ``HeroSlotLayout`` JSON through the *real* production schema
classes, so anything that fails production validation fails here, at authoring
time, instead of at load time.

Measurements provenance
-----------------------
- The 20-ROI ``TableMap`` comes from the Stage E ROI measurement CSV
  (``labels/roi_measurements.csv``). Each ROI (hero_cards / board_cards / pot /
  stacks / actions / actor) is measured from real normalized frames and
  cross-checked stable across session_001 (heads-up) and session_002 (8-max).
- The per-card sub-ROIs (hero 2-card, board 3-card) are *not* stored in the CSV
  (the CSV only holds parent-ROI boxes). This tool instead measures them
  directly from the normalized frames by detecting bright card columns within
  each parent ROI, using the frame indices supplied via ``--hero-frame`` /
  ``--board-frame`` (or defaults that point at verified card-present frames).
  This keeps the landed slots tied to evidence, not to guesswork.

Scope — what this tool CAN and CANNOT honestly land:

- CAN: the 20-ROI ``TableMap`` and the 2-slot ``HeroSlotLayout`` (measured:
  two cards occupying ~[0.0,0.46] and ~[0.52,0.99] of the hero ROI).
- CANNOT (and therefore does NOT write): ``BoardSlotLayout`` (the schema
  requires exactly 5 slots, but the measured material contains only 3-card
  FLOP boards with no reserved 5th slot — there is no 5-card RIVER frame, so a
  5-slot layout would be fabricated), ``dealer_slot_layout`` and
  ``empty_slot_layout`` (measurements only cover a subset of slots), and
  ``calibration.json`` (no recognition-accuracy evidence exists).

Failure-closed philosophy: geometry that is genuinely measured can be landed,
but anything without evidence is left absent so the production loader falls
back to its uncalibrated path and reads UNKNOWN rather than inheriting a
fabricated value.

Usage (from the repo root, with the capture-card dataset checked out at the
path given by ``--root``):

    python tools/land_capture_card_configs.py \
        --root ../capture_card_calibration_20260903 \
        --platform-id wepoker_android_capture_card \
        --layout-id \
        phone_samsung_galaxy_s25_ultra__card_ugreen__uvc_1920x1080_30__canvas_498x1080__v1
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig
from poker_engine.perceptual.vision.card_layout import (
    CardSubROI,
    HeroSlotLayout,
    hero_layout_to_dict,
)
from poker_engine.perceptual.vision.table_map import TableMap

from tools.capture_card_calibration.dataset import RoiMeasurement
from tools.capture_card_calibration.geometry import build_table_map_draft
from tools.capture_card_calibration.layout_id import is_valid_layout_id

_HERO_FRAME_REF = 180   # a verified hero-cards-present frame in the set
_BOARD_FRAME_REF = 297  # a verified 3-card FLOP frame in the set
_CARD_BRIGHTNESS = 150  # gray threshold above which a column counts as card white


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {what}: {path}")


def _read_measurements(csv_path: Path) -> list[RoiMeasurement]:
    rows: list[RoiMeasurement] = []
    with csv_path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                RoiMeasurement(
                    field=record["field"],
                    slot_id=(int(record["slot_id"]) if record["slot_id"] else None),
                    x0=int(record["x0"]),
                    y0=int(record["y0"]),
                    x1=int(record["x1"]),
                    y1=int(record["y1"]),
                    source_frame=record["source_frame"],
                    notes=record["notes"],
                )
            )
    return rows


def _canvas_from_normalization(root: Path) -> tuple[int, int]:
    path = root / "normalization" / "normalization.json"
    _require_file(path, "normalization config")
    config = NormalizationConfig.from_json(path.read_text(encoding="utf-8"))
    if config.output_size is None:
        raise SystemExit("normalization.json lacks an output_size; cannot size canvas")
    return tuple(config.output_size)  # type: ignore[return-value]


def _table_map_roi(
    table_map: TableMap, kind: str
) -> tuple[int, int, int, int]:
    """Return pixel (x0, y0, x1, y1) box for a global TableMap ROI."""
    roi = next(r for r in table_map.rois if r.kind.value == kind)
    w, h = table_map.reference_size
    return (
        int(roi.x * w),
        int(roi.y * h),
        int((roi.x + roi.width) * w),
        int((roi.y + roi.height) * h),
    )


def _card_column_segments(crop_gray: np.ndarray) -> list[tuple[int, int]]:
    """Detect bright card columns within a parent-ROI crop.

    A column is "on card" when >40% of its pixels are brighter than
    ``_CARD_BRIGHTNESS``. Contiguous on-columns separated by <=1px are split at
    their gaps; segments narrower than 3px are dropped as noise.
    """
    mask = (crop_gray > _CARD_BRIGHTNESS).astype(float)
    col = mask.mean(axis=0)
    on = col > 0.4
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(on):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(on)))
    return [(a, b) for a, b in segments if b - a > 3]


def _measure_card_slots(
    root: Path,
    table_map: TableMap,
    os_path: Path,
    *,
    parent_kind: str,
    frame_ref: int,
    expected: int,
) -> list[CardSubROI]:
    """Measure per-card sub-ROIs (fractions of the parent ROI).

    Reads the ``frame_ref``-th normalized PNG, crops the parent ROI, detects the
    card columns, and expresses each card's box as a fraction of the parent ROI.
    """
    frames = sorted(os_path.glob("*.png"))
    if not frames:
        raise SystemExit(f"no normalized frames under {os_path}")
    if not (0 <= frame_ref < len(frames)):
        raise SystemExit(
            f"frame_ref {frame_ref} out of range (0..{len(frames) - 1})"
        )
    x0, y0, x1, y1 = _table_map_roi(table_map, parent_kind)
    parent_w = x1 - x0

    image = cv2.imdecode(
        np.fromfile(str(frames[frame_ref]), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise SystemExit(f"cannot read {frames[frame_ref]}")
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    segments = _card_column_segments(gray)
    if len(segments) != expected:
        raise SystemExit(
            f"{parent_kind}: expected {expected} card columns in frame "
            f"{frames[frame_ref].name}, detected {len(segments)} "
            f"({segments})"
        )
    slots = []
    for a, b in segments:
        slots.append(
            CardSubROI(
                x=round(a / parent_w, 4),
                y=0.0,
                width=round((b - a) / parent_w, 4),
                height=1.0,
            )
        )
    return slots


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _cmd_land(args: argparse.Namespace) -> int:
    root = Path(args.root)
    measurements = _read_measurements(root / "labels" / "roi_measurements.csv")
    counters = Counter(m.field for m in measurements)
    print(f"read {len(measurements)} ROI measurements; fields={dict(counters)}")

    canvas = _canvas_from_normalization(root)
    print(f"canvas (from normalization.json): {canvas}")
    if not is_valid_layout_id(args.layout_id):
        raise SystemExit(f"invalid layout_id: {args.layout_id}")

    table_map = build_table_map_draft(
        measurements,
        platform_id=args.platform_id,
        layout_id=args.layout_id,
        canvas=canvas,
    )
    # Validate through the production schema before writing.
    TableMap.from_json(table_map.to_json())

    frames_dir = root / "normalized" / "frames"
    hero_slots = _measure_card_slots(
        root,
        table_map,
        frames_dir,
        parent_kind="hero_cards",
        frame_ref=args.hero_frame,
        expected=2,
    )
    hero_layout = HeroSlotLayout(
        layout_id=args.layout_id, version=1, slots=tuple(hero_slots)
    )
    HeroSlotLayout(
        layout_id=hero_layout.layout_id,
        version=hero_layout.version,
        slots=hero_layout.slots,
    )

    # Board slots: measured as 3 cards, but the production schema requires 5.
    # We measure them (for the record) but deliberately do NOT write a
    # board_slot_layout.json, because a 5-slot file would have to fabricate the
    # last two slot positions — a failure-closed violation.
    board_segments = None
    try:
        board_slots = _measure_card_slots(
            root,
            table_map,
            frames_dir,
            parent_kind="board_cards",
            frame_ref=args.board_frame,
            expected=3,
        )
        board_segments = [
            (round(s.x, 4), round(s.x + s.width, 4)) for s in board_slots
        ]
        print(
            f"board 3-slot measured (x-range): {board_segments}; "
            "NOT written — BoardSlotLayout requires 5 slots and no 5-card "
            "RIVER frame was measured, so a 5-slot file would be fabricated."
        )
    except SystemExit as exc:
        print(f"board measurement skipped: {exc}")

    config_root = Path(args.config_root)
    platform_dir = config_root / "platform"
    vision_dir = config_root / "vision" / args.platform_id

    table_map_path = _write_json(
        platform_dir / f"{args.platform_id}__{args.layout_id}.json",
        table_map.to_dict(),
    )
    hero_layout_path = _write_json(
        vision_dir / "hero_slot_layout.json", hero_layout_to_dict(hero_layout)
    )

    print("\nlanded:")
    print(f"  table_map      -> {table_map_path}")
    print(f"  hero layout    -> {hero_layout_path}")
    print(
        "\nNOT written (no evidence / fabricated would violate failure-closed):"
    )
    for name in ("board_slot_layout.json", "dealer_slot_layout.json",
                 "empty_slot_layout.json", "calibration.json"):
        print(f"  - {name}")
    print(
        f"\nREMINDER: layout_id embeds the capture-card model placeholder "
        f"'{args.layout_id.split('__card_')[1].split('__')[0]}'; regenerate it "
        "once the real capture-card model is known."
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="capture-card calibration dataset root")
    parser.add_argument("--config-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs",
                        help="repo configs/ directory (default: repo configs/)")
    parser.add_argument("--platform-id", default="wepoker_android_capture_card")
    parser.add_argument(
        "--layout-id",
        default=(
            "phone_samsung_galaxy_s25_ultra__card_ugreen__"
            "uvc_1920x1080_30__canvas_498x1080__v1"
        ),
        help="section-6 layout id (embeds phone/card/uvc/canvas)",
    )
    parser.add_argument("--hero-frame", type=int, default=_HERO_FRAME_REF,
                        help="normalized frame index with a readable hero hand")
    parser.add_argument("--board-frame", type=int, default=_BOARD_FRAME_REF,
                        help="normalized frame index with a 3-card flop board")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _cmd_land(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_cmd_land", "main"]
