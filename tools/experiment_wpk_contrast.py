"""Static development experiment; does not edit or enable production models.

Batch 02 is now a diagnosed development set. Repeating one crop is not a
continuous-video test or an independent validation of this preprocessing.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, _resource_root, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.engine import _crop_slot
from poker_engine.perceptual.vision.roi import extract_roi
from poker_engine.perceptual.vision.fused_card_adapter import FusedCardRecognizerAdapter
from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, GlyphNormalizer, load_card_heads,
)
from tools.validate_wpk_card_batch import score_slots, summarize
from tools.wpk_video_dataset import pixels_digest, read_image


def contrast_glyph(glyph):
    if glyph is None or glyph.size == 0:
        return None
    gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY) if glyph.ndim == 3 else glyph
    low, high = np.percentile(gray, (5, 95))
    if high - low < 10:
        return None
    return np.clip((gray.astype(np.float32) - low) * 255 / (high - low),
                   0, 255).astype(np.uint8)


def experiment(batch: Path) -> dict:
    spec = json.loads((batch / "corrected-score.json")
                      .read_text(encoding="utf-8"))["corrected_spec"]
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    folder = _resource_root() / "configs/vision" / CAPTURE_CARD_PLATFORM
    heads = load_card_heads(folder / "card_heads.npz")
    original = GlyphNormalizer.normalize

    def adjusted(glyph, geometry=None):
        return original(contrast_glyph(glyph))

    rows = []
    GlyphNormalizer.normalize = staticmethod(adjusted)
    try:
        for point in spec["checkpoints"]:
            index = point["source_frame"]
            image = read_image(batch / "frames" / f"{index:06d}.png")
            if pixels_digest(image) != point["pixel_sha256"]:
                raise ValueError("checkpoint changed")
            frame = Frame(index, datetime.now(timezone.utc), "static-experiment",
                          WindowRect(0, 0, 498, 1080), image, 498, 1080)
            row = {"source_frame": index,
                   "require_hero_complete": point["require_hero_complete"],
                   "require_board_complete": point["require_board_complete"]}
            for group, kind, width in (("hero", "hero_cards", 2),
                                       ("board", "board_cards", 5)):
                roi = next(r for r in table.rois if r.kind.value == kind)
                base = extract_roi(frame, roi)
                layout = (
                    vision._hero_layout if group == "hero" else vision._board_layout
                )
                predicted = []
                for slot in range(width):
                    crop = _crop_slot(base, layout.slots[slot])
                    adapter = FusedCardRecognizerAdapter(FusedCardRecognizer(heads))
                    for _ in range(10):
                        result = adapter.recognize(crop, (group, slot))
                    predicted.append(str(result.value[0]) if result.value is not None
                                     and result.raw_score >= 0.3 else None)
                row[group] = score_slots(point[group], predicted, width)
                row[group]["predicted"] = predicted
            rows.append(row)
    finally:
        GlyphNormalizer.normalize = staticmethod(original)
    return {"experiment": "5/95 percentile contrast normalization",
            "production_model_changed": False, "static_only": True,
            "independent_validation": False, "checkpoints": rows,
            "summary": summarize(rows, spec["criteria"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing experiment results")
    report = experiment(args.batch)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
