"""Positive empty AA8 wager display evidence; never converts absence into zero.

The caller must establish supported scene and no obstructing overlay. Even then
the observation describes only this display rectangle, not street contribution.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_continuous_state import WAGERS
from tools.aa_visual_candidate import canvas_ok


def patch_visibility(patch, coin, *, scene_supported=False, unobstructed=False):
    result = {"status": "UNKNOWN", "value": None,
              "street_wager_zero_verified": False, "strategy_eligible": False}
    if (not isinstance(patch, np.ndarray) or patch.dtype != np.uint8
            or patch.ndim != 3 or patch.shape[2] != 3
            or min(patch.shape[:2]) < 17):
        return {**result, "reason": "unsupported_patch"}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    felt = cv2.inRange(hsv, (35, 100, 40), (95, 255, 255)) > 0
    neutral_bright = (hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= 120)
    dark = hsv[:, :, 2] < 40
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    matches = cv2.matchTemplate(gray, coin, cv2.TM_CCOEFF_NORMED)
    coin_score = float(matches.max())
    edges = cv2.Canny(gray, 50, 100) > 0
    result["metrics"] = {"felt_fraction": float(felt.mean()),
                         "neutral_bright_pixels": int(neutral_bright.sum()),
                         "dark_pixels": int(dark.sum()),
                         "coin_score": coin_score,
                         "edge_fraction": float(edges.mean())}
    if scene_supported is not True or unobstructed is not True:
        return {**result, "reason": "scene_or_occlusion_not_established"}
    if coin_score >= .85:
        return {**result, "status": "VISIBLE_COIN_CANDIDATE",
                "reason": "coin_present_even_if_amount_ocr_fails"}
    # Strict positive felt coverage; there is no tolerance for clipped/faint ink.
    if felt.all() and not neutral_bright.any() and not dark.any():
        return {**result, "status": "VISIBLE_EMPTY_CANDIDATE",
                "reason": "entire_rect_positive_felt_no_coin_or_neutral_ink"}
    return {**result, "reason": "non_felt_pixels_without_identified_coin"}


class AA8WagerVisibility:
    def __init__(self, reference1500):
        if not canvas_ok(reference1500):
            raise ValueError("AA8 reference canvas required")
        self.coin = cv2.cvtColor(reference1500[604:621, 392:409], cv2.COLOR_BGR2GRAY)
        if float(self.coin.std()) < 3:
            raise ValueError("nonflat source coin required")

    def read(self, image, *, scene_supported=False, unobstructed=False):
        if not canvas_ok(image):
            return {str(s): {"status": "UNKNOWN", "value": None,
                             "reason": "unsupported_canvas",
                             "street_wager_zero_verified": False,
                             "strategy_eligible": False} for s in range(8)}
        return {str(s): {**patch_visibility(
            image[y:y + h, x:x + w], self.coin,
            scene_supported=scene_supported, unobstructed=unobstructed), "rect": rect}
                for s, rect in enumerate(WAGERS) for x, y, w, h in (rect,)}


def run(source, target, output):
    audit = json.loads((source / "samples.json").read_text())["audit_sha256"]
    first = inventory(source, audit)
    second = inventory(target, audit)
    reader = AA8WagerVisibility(load(source, first[1500]))
    observations = []
    for pool, rows, frames in ((source, first, (1500, 2400, 2700)),
                               (target, second, (4755,))):
        for frame in frames:
            observations.append({"frame": frame, "sha256": rows[frame]["sha256"],
                                 "wager_visibility": reader.read(
                                     load(pool, rows[frame]), scene_supported=True,
                                     unobstructed=True)})
    report = {"observations": observations, "coin_source": first[1500],
              "source_manifest_sha256": sha(source / "samples.json"),
              "target_manifest_sha256": sha(target / "samples.json"),
              "implementation_sha256": sha(Path(__file__)),
              "scene_and_occlusion_gate_source": "manual reviewed four checkpoints",
              "zero_values_injected": False, "exact_street_price_verified": False,
              "independent_holdout": False, "full_visual_acceptance": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps([{"frame": r["frame"], "statuses": {
        s: v["status"] for s, v in r["wager_visibility"].items()}}
        for r in observations]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.target, args.output)
