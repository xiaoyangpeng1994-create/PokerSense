"""Read-only development comparison of uniform photometric card transforms.

Does not select a crop by predicted identity, change heads, or promote a variant.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.vision.fused_card_recognizer import (
    FusedCardRecognizer, FusedSlotBuffer, load_card_heads,
)
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa_card_visibility import face_card_support, locate_face_card


VARIANTS = ("identity", "gaussian_035", "gaussian_050", "gaussian_070", "median3")


def transform_card(crop, variant):
    if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8 or (
            crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 3):
        raise ValueError("valid card crop required")
    if variant == "identity":
        return crop.copy()
    if variant.startswith("gaussian_") and variant in VARIANTS:
        sigma = {"gaussian_035": .35, "gaussian_050": .5, "gaussian_070": .7}[variant]
        return cv2.GaussianBlur(crop, (3, 3), sigma)
    if variant == "median3":
        return cv2.medianBlur(crop, 3)
    raise ValueError("unregistered transform")


def diagnose(crop, heads):
    h, w = crop.shape[:2]
    buffer = FusedSlotBuffer((0, 0, w, h), min_glyphs=1)
    if not buffer.ingest(crop):
        return {"value": None, "reason": "glyph_extraction_failed"}
    glyphs = buffer.latest_glyphs()
    result = FusedCardRecognizer(heads, rank_floor=.5, suit_floor=.3).recognize_fused(
        *glyphs)
    return {"value": str(result.value[0]) if result.value else None,
            "rank": heads["rank"].predict(glyphs[0]),
            "suit": heads["suit_red" if glyphs[2] else "suit_black"].predict(glyphs[1]),
            "reason": "single_current_frame_diagnostic"}


def run(source, target, heads_path, output):
    audit = json.loads((source / "samples.json").read_text())["audit_sha256"]
    heads = load_card_heads(heads_path)
    rows_out = []
    for pool, frames in ((source, (1470, 1980, 2070, 2400, 2820, 3060, 3180)),
                         (target, (4890, 4950))):
        rows = inventory(pool, audit)
        for frame in frames:
            image = load(pool, rows[frame])
            for group, xs, y in (("hero", (193, 250), 938),
                                 ("board", (110, 166, 223, 279, 335), 473)):
                for slot, x in enumerate(xs):
                    rect = (x, y, 53, 78)
                    if group == "board":
                        rect, _ = locate_face_card(image, rect)
                    elif not face_card_support(image, rect)[0]:
                        rect = None
                    if rect is None:
                        continue
                    x0, y0, w, h = rect
                    crop = image[y0:y0 + h, x0:x0 + w]
                    predictions = {v: diagnose(transform_card(crop, v), heads)
                                   for v in VARIANTS}
                    rows_out.append({"frame": frame, "group": group, "slot": slot,
                                     "rect": rect, "sha256": rows[frame]["sha256"],
                                     "predictions": predictions})
    output.mkdir(parents=True, exist_ok=False)
    report = {"rows": rows_out, "heads_sha256": sha(heads_path),
              "implementation_sha256": sha(Path(__file__)),
              "source_manifest_sha256": sha(source / "samples.json"),
              "target_manifest_sha256": sha(target / "samples.json"),
              "rank_floor": .5, "suit_floor": .3,
              "model_changed": False, "production_changed": False,
              "independent_holdout": False, "variant_promoted": None}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{k: r[k] for k in ("frame", "predictions")}
                      for r in rows_out if r["group"] == "hero" and r["slot"] == 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "heads", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.target, args.heads, args.output)
