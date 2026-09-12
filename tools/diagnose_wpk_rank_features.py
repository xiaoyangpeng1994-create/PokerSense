"""Inspect 5/6 features on reviewed development frames without fitting a model.

Writes private glyph-only evidence and predictions; labels always come from
the explicitly corrected visual review, never from model output.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.desktop.live import (
    CAPTURE_CARD_LAYOUT, CAPTURE_CARD_PLATFORM, load_calibration,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.corner_glyph_recognizer import (
    DEFAULT_GEOMETRY, isolate_glyph, locate_card_face,
)
from poker_engine.perceptual.vision.engine import _crop_slot
from poker_engine.perceptual.vision.fused_card_recognizer import (
    GlyphNormalizer, load_card_heads,
)
from poker_engine.perceptual.vision.roi import extract_roi
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, read_image


def diagnose(batch: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("preserve previous diagnostic evidence")
    spec_path = batch / "corrected-score.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))["corrected_spec"]
    repo = Path(__file__).resolve().parents[1]
    model = repo / "configs/vision" / CAPTURE_CARD_PLATFORM / "card_heads.npz"
    head = load_card_heads(model)["rank"]
    table, vision = load_calibration(CAPTURE_CARD_PLATFORM, CAPTURE_CARD_LAYOUT)
    rows, canvases, tiles = [], [], []
    for point in spec["checkpoints"]:
        index = point["source_frame"]
        image = read_image(batch / "frames" / f"{index:06d}.png")
        if pixels_digest(image) != point["pixel_sha256"]:
            raise ValueError("source pixels changed")
        frame = Frame(index, datetime.now(timezone.utc), "rank-diagnostic",
                      WindowRect(0, 0, 498, 1080), image, 498, 1080)
        for group in ("hero", "board"):
            roi = next(r for r in table.rois if r.kind.value == f"{group}_cards")
            base = extract_roi(frame, roi)
            layout = vision._hero_layout if group == "hero" else vision._board_layout
            for slot, truth in enumerate(point[group]):
                if truth[0] not in "56":
                    continue
                crop = _crop_slot(base, layout.slots[slot])
                card = locate_card_face(crop)
                glyph = isolate_glyph(card, DEFAULT_GEOMETRY.rank_band)
                canvas = GlyphNormalizer.normalize(glyph)
                if canvas is None:
                    raise ValueError(f"unreadable diagnostic glyph: {index}/{slot}")
                label, margin = head.predict(canvas)
                row = {"source_frame": index, "group": group, "slot": slot,
                       "truth": truth, "predicted_rank": label,
                       "rank_margin": margin, "glyph_shape": list(glyph.shape),
                       "normalized_sha256": pixels_digest(canvas),
                       "width_sensitivity": []}
                # Sensitivity only: none of these interventions becomes a
                # selected classifier or changes live crop geometry.
                for dw in (-1, 1):
                    varied = cv2.resize(glyph, (glyph.shape[1] + dw, glyph.shape[0]),
                                        interpolation=cv2.INTER_AREA)
                    feature = GlyphNormalizer.normalize(varied)
                    prediction = head.predict(feature) if feature is not None else None
                    row["width_sensitivity"].append(
                        {"delta_px": dw, "prediction": prediction})
                rows.append(row)
                canvases.append(canvas)
                tile = np.full((170, 370, 3), 240, np.uint8)
                cv2.putText(tile, f"{index} {group}{slot} {truth} -> {label}",
                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
                display_width = int(round(120 * glyph.shape[1] / glyph.shape[0]))
                enlarged = cv2.resize(glyph, (display_width, 120),
                                      interpolation=cv2.INTER_NEAREST)
                tile[35:155, 10:10 + display_width] = enlarged
                norm_image = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                tile[35:155, 140:260] = cv2.resize(
                    norm_image, (120, 120), interpolation=cv2.INTER_NEAREST)
                tiles.append(tile)
    report = {"independent_validation": False, "static_only": True,
              "model_changed": False, "head_sha256": sha256_file(model),
              "diagnostic_tool_sha256": sha256_file(Path(__file__)),
              "normalizer_source_sha256": sha256_file(repo / (
                  "src/poker_engine/perceptual/vision/fused_card_recognizer.py")),
              "review_sha256": sha256_file(spec_path), "rows": rows,
              "pairwise_feature_mae": [[float(np.mean(np.abs(a - b)))
                                        for b in canvases] for a in canvases]}
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if tiles:
        ok, encoded = cv2.imencode(".png", np.concatenate(tiles, axis=0))
        if not ok:
            raise ValueError("cannot encode glyph sheet")
        (output / "glyphs.png").write_bytes(encoded.tobytes())
    write_sha256sums(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.batch, args.output), indent=2))
