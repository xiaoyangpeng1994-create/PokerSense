# -*- coding: utf-8 -*-
"""Stage B4: export production template assets from the private dataset.

The production ``VisionEngine`` for the capture-card platform degrades every
field whose template directory is missing (``live.py`` falls back to a
placeholder that can never clear a gate). This module turns the confirmed
labels in the private dataset into the committed PNG assets the production
recognizers consume:

- ``digit/``       -> pot amount templates (``TemplateAmountRecognizer``)
- ``stack_digit/`` -> per-seat stack templates (same recognizer, own set)

Method (mirrors the guide's failure-closed rules):

- Crops are taken with the **production** ROI geometry (the platform
  TableMap JSON), so templates see exactly what the live pipeline sees.
- Segmentation uses the **production** ``segment_characters``; a confirmed
  value contributes only when the segmented glyph count equals its digit
  count (we never invent a glyph, and merged glyphs never become templates).
- The template for each character is the pixel-wise mean of its normalized
  28x28 crops (the same grid ``_match_char`` uses), which is more robust
  than any single sample.
- Hand-isolated: pass ``frame_ids`` to restrict which frames contribute, so
  templates can be built on the train split while calibration is measured
  on the held-out eval split (no leakage).
- Assets are written with ``imencode`` + ``tofile`` so non-ASCII checkout
  paths work.

This module is import-safe without OpenCV — readers import cv2 lazily.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .dataset import read_frames_jsonl
from .schema import FrameLabel, LabelStatus

#: Repo-relative platform config for the capture-card platform.
PLATFORM_ID = "wepoker_android_capture_card"
LAYOUT_ID = (
    "phone_samsung_galaxy_s25_ultra__card_ugreen__uvc_1920x1080_30"
    "__canvas_498x1080__v1"
)

_NORM = (28, 28)  # must match amount_recognizer._NORM


@dataclass(frozen=True)
class RoiRect:
    """A normalized production ROI (x, y, width, height in 0..1)."""

    x: float
    y: float
    width: float
    height: float

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Pixel crop using the same floor() rounding as production roi.py."""
        height, width = img.shape[:2]
        x0 = int(self.x * width)
        y0 = int(self.y * height)
        x1 = int((self.x + self.width) * width)
        y1 = int((self.y + self.height) * height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI collapses to zero size")
        return img[y0:y1, x0:x1]


def load_table_map_rois(repo_root: Path) -> dict[str, RoiRect | dict[int, RoiRect]]:
    """Load the committed production ROIs for the capture-card platform.

    Returns ``{"pot": RoiRect, "stack": {slot_id: RoiRect},
    "action": {slot_id: RoiRect}, "actor": RoiRect, ...}``.
    """
    path = (
        repo_root
        / "configs"
        / "platform"
        / f"{PLATFORM_ID}__{LAYOUT_ID}.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    rois: dict[str, RoiRect | dict[int, RoiRect]] = {}
    for entry in data["rois"]:
        rect = RoiRect(
            x=float(entry["x"]),
            y=float(entry["y"]),
            width=float(entry["width"]),
            height=float(entry["height"]),
        )
        kind = entry["kind"]
        slot_id = entry.get("slot_id")
        if slot_id is None:
            rois[kind] = rect
        else:
            bucket = rois.setdefault(kind, {})
            assert isinstance(bucket, dict)
            bucket[int(slot_id)] = rect
    return rois


def _read_frame(path: Path) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def _write_png(path: Path, img: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"failed to encode {path}")
    encoded.tofile(str(path))


def _normalize_char(img: np.ndarray) -> np.ndarray:
    """Letterbox a glyph crop onto the production 28x28 grid (grayscale).

    Must mirror ``amount_recognizer._normalize_char`` exactly, including the
    corner-median background canvas (a fixed white canvas washes out
    white-on-dark glyphs).
    """
    import cv2

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    corners = np.array(
        [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    ).reshape(-1)
    background = int(np.median(corners))
    scale = min(_NORM[0] / h, _NORM[1] / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full(_NORM, background, dtype=resized.dtype)
    y0 = (_NORM[0] - nh) // 2
    x0 = (_NORM[1] - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


# --- digit crop collection -------------------------------------------------


def iter_amount_targets(
    labels: Sequence[FrameLabel],
    field: str,
) -> tuple[tuple[FrameLabel, int | None, str], ...]:
    """Yield ``(label, slot_id, value_text)`` for every confirmed target.

    ``field="pot"`` takes the frame-level pot value (``slot_id=None``);
    ``field="stack"`` takes every per-slot confirmed stack. Only ``VALID``
    labels on stable table frames contribute — a confirmed value is the only
    honest source for a template.
    """
    targets: list[tuple[FrameLabel, int | None, str]] = []
    for label in labels:
        if not label.stable or label.scene.value != "table":
            continue
        if field == "pot":
            if label.pot.status is LabelStatus.VALID:
                targets.append((label, None, str(int(label.pot.value))))
        elif field == "stack":
            for slot in label.slots:
                if slot.stack.status is LabelStatus.VALID:
                    targets.append(
                        (label, slot.slot_id, str(int(slot.stack.value)))
                    )
        else:
            raise ValueError(f"unsupported amount field: {field!r}")
    return tuple(targets)


def collect_digit_crops(
    labels: Sequence[FrameLabel],
    frames_dir: Path,
    rois: Mapping[str, RoiRect | dict[int, RoiRect]],
    field: str,
    *,
    frame_ids: set[str] | None = None,
) -> dict[str, list[np.ndarray]]:
    """Collect per-character glyph crops keyed by character label.

    A confirmed value contributes only when production segmentation yields
    exactly one glyph per character (count match); otherwise the whole read
    is dropped (never invent a glyph).
    """
    from poker_engine.perceptual.vision.amount_recognizer import (
        segment_characters,
    )

    crops: dict[str, list[np.ndarray]] = {}
    frame_cache: dict[str, np.ndarray | None] = {}
    for label, slot_id, text in iter_amount_targets(labels, field):
        if frame_ids is not None and label.frame not in frame_ids:
            continue
        path = frames_dir / label.frame
        if not path.is_file():
            continue
        if label.frame not in frame_cache:
            frame_cache[label.frame] = _read_frame(path)
        img = frame_cache[label.frame]
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
        crop = rect.crop(img)
        chars = segment_characters(crop)
        if len(chars) != len(text):
            continue
        for char_img, char_label in zip(chars, text):
            crops.setdefault(char_label, []).append(_normalize_char(char_img))
    return crops


def build_mean_templates(
    crops: Mapping[str, Sequence[np.ndarray]],
    *,
    min_samples: int = 1,
) -> dict[str, np.ndarray]:
    """Build the per-character mean template from normalized crops."""
    templates: dict[str, np.ndarray] = {}
    for char, samples in sorted(crops.items()):
        if len(samples) < min_samples:
            continue
        stacked = np.stack(samples).astype(np.float64)
        templates[char] = np.rint(stacked.mean(axis=0)).astype(np.uint8)
    return templates


#: Canonical enclosed-region counts for this platform's digit font. Samples
#: whose topology contradicts the canonical count are mislabeled or broken
#: (e.g. a ``5`` glyph filed under ``6`` by a wrong frame label) and must not
#: become the template.
CANONICAL_HOLE_COUNTS: Mapping[str, int] = {
    "0": 1,
    "6": 1,
    "8": 2,
    "9": 1,
}


def build_medoid_templates(
    crops: Mapping[str, Sequence[np.ndarray]],
    *,
    min_samples: int = 1,
) -> dict[str, np.ndarray]:
    """Build per-character templates from the real crop closest to the mean.

    A pixel-wise mean blurs anti-aliased glyphs and destroys their hole
    topology (a mean ``0``/``6``/``9`` loses its enclosed region, a mean
    ``8`` loses one of two), which silently disables the recognizer's
    topology filter and even injects phantom holes. The medoid is an actual
    glyph, so its topology is real. For digits with a canonical hole count
    the medoid is chosen among topology-correct samples only, which also
    purges cross-labeled glyphs from wrong frame labels.
    """
    from poker_engine.perceptual.vision.amount_recognizer import (
        _glyph_hole_count,
    )

    templates: dict[str, np.ndarray] = {}
    for char, samples in sorted(crops.items()):
        if len(samples) < min_samples:
            continue
        canonical = CANONICAL_HOLE_COUNTS.get(char)
        pool = list(samples)
        if canonical is not None:
            correct = [s for s in pool if _glyph_hole_count(s) == canonical]
            if correct:
                pool = correct
        stacked = np.stack(pool).astype(np.float64)
        mean = stacked.mean(axis=0)
        distances = ((stacked - mean) ** 2).sum(axis=(1, 2))
        templates[char] = pool[int(np.argmin(distances))]
    return templates


def export_digit_assets(
    labels: Sequence[FrameLabel],
    frames_dir: Path,
    rois: Mapping[str, RoiRect | dict[int, RoiRect]],
    field: str,
    out_dir: Path,
    *,
    frame_ids: set[str] | None = None,
    min_samples: int = 1,
) -> dict[str, int]:
    """Export ``<char>.png`` templates for one amount field.

    Returns ``{char: sample_count}`` for the characters actually written.
    """
    crops = collect_digit_crops(
        labels, frames_dir, rois, field, frame_ids=frame_ids
    )
    templates = build_medoid_templates(crops, min_samples=min_samples)
    counts: dict[str, int] = {}
    for char, template in templates.items():
        _write_png(out_dir / f"{char}.png", template)
        counts[char] = len(crops[char])
    return counts


# --- CLI -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="private dataset root (contains labels/ and normalized/)")
    parser.add_argument("--repo", type=Path, required=True,
                        help="PokerSense repo root (configs/ lives here)")
    parser.add_argument("--field", choices=("pot", "stack"), required=True)
    parser.add_argument("--frames", type=Path, default=None,
                        help="optional file with one frame id per line "
                             "(train split restriction)")
    args = parser.parse_args()

    labels = read_frames_jsonl(args.root / "labels" / "frames.jsonl")
    frames_dir = args.root / "normalized" / "frames"
    rois = load_table_map_rois(args.repo)
    frame_ids = None
    if args.frames is not None:
        frame_ids = {
            line.strip()
            for line in args.frames.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    out_dir = (
        args.repo
        / "configs"
        / "vision"
        / PLATFORM_ID
        / ("digit" if args.field == "pot" else "stack_digit")
    )
    counts = export_digit_assets(
        labels, frames_dir, rois, args.field, out_dir, frame_ids=frame_ids
    )
    print(json.dumps({"field": args.field, "out": str(out_dir),
                      "chars": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LAYOUT_ID",
    "PLATFORM_ID",
    "RoiRect",
    "build_mean_templates",
    "build_medoid_templates",
    "collect_digit_crops",
    "export_digit_assets",
    "iter_amount_targets",
    "load_table_map_rois",
]
