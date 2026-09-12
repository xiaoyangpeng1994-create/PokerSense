"""AA8 bounded wager geometry candidate, independent of frozen V1 readers."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa8_continuous_state import WAGERS
from tools.aa8_action_transfer import inventory, load, sha
from tools.aa8_wager_visibility import patch_visibility
from tools.aa_pot_candidate import pot_mask
from tools.aa_visual_candidate import canvas_ok


# Hero chips occupy y814:831; a gray preselection button begins near y833.
# x292 also excludes the adjacent name. This remains a development candidate.
WAGERS_V2 = tuple((292, 808, 78, 24) if s == 4 else rect
                  for s, rect in enumerate(WAGERS))


def digit_bounds(binary):
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    digits = [v for v in stats[1:] if v[4] >= 6 and 7 <= v[3] <= 20 and v[2] <= 16]
    if not digits:
        return None
    top = min(v[1] for v in digits)
    bottom = max(v[1] + v[3] for v in digits)
    for x, y, w, h, area in stats[1:]:
        if area >= 6 and 7 <= h <= 20 and w <= 16:
            continue
        # Do not silently erase decimal/currency punctuation near the digit line.
        if area >= 2 and y + h > top - 1 and y < bottom + 2:
            return None
    return (min(v[0] for v in digits), top,
            max(v[0] + v[2] for v in digits), bottom)


class AA8WagerReaderV2:
    def __init__(self, bank, reference1500, *, coin_smoothing=False):
        if type(coin_smoothing) is not bool:
            raise ValueError("explicit boolean coin_smoothing required")
        self.coin_smoothing = coin_smoothing
        if not canvas_ok(reference1500):
            raise ValueError("source reference canvas required")
        self.bank = bank
        self.coin = cv2.cvtColor(reference1500[604:621, 392:409], cv2.COLOR_BGR2GRAY)
        if self.coin.std() < 3:
            raise ValueError("nonflat reviewed coin required")

    def read_label(self, patch, slot):
        if (type(slot) is not int or slot not in range(8)
                or not isinstance(patch, np.ndarray) or patch.dtype != np.uint8
                or patch.ndim != 3 or patch.shape[2] != 3
                or patch.shape[:2] != (WAGERS_V2[slot][3], WAGERS_V2[slot][2])):
            return {"value": None, "reason": "unsupported_wager_patch"}
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        locator, coin = gray, self.coin
        if self.coin_smoothing:
            locator = cv2.GaussianBlur(gray, (3, 3), .5)
            coin = cv2.GaussianBlur(self.coin, (3, 3), .5)
        matches = cv2.matchTemplate(locator, coin, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(matches)
        unknown = {"value": None, "coin_score": float(score), "coin_unique": False,
                   "coin_locator_preprocessing": "gaussian_050" if self.coin_smoothing
                   else None,
                   "coin_template_sha256": hashlib.sha256(coin.tobytes()).hexdigest()}
        if score < .85:
            return {**unknown, "reason": "no_unique_coin_prefix"}
        ys, xs = np.where(matches >= .85)
        if int(xs.max() - xs.min()) > 4 or int(ys.max() - ys.min()) > 4:
            return {**unknown, "reason": "multiple_coin_candidates"}
        unknown["coin_unique"] = True
        # Inspect bounded context beyond the coin; the numeric component must
        # still have clear edges. A pill-border speck is not a taller digit.
        top_y, bottom_y = max(0, y - 3), min(patch.shape[0], y + 20)
        strip = patch[top_y:bottom_y]
        text = strip[:, :x] if slot in (1, 2, 3) else strip[:, x + 18:]
        bounds = digit_bounds(pot_mask(text))
        if bounds is None:
            return {**unknown, "reason": "no_clean_digit_line_or_punctuation"}
        left, top, right, bottom = map(int, bounds)
        if left == 0 or right >= text.shape[1] or top == 0 or bottom >= text.shape[0]:
            return {**unknown, "reason": "clipped_wager_digits"}
        digits = cv2.copyMakeBorder(text[top:bottom, left:right], 2, 2, 2, 2,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return {**unknown, **asdict(self.bank.diagnose(digits)),
                "vertical_strip": [top_y, bottom_y],
                "digit_bounds_in_text": [left, top, right, bottom]}

    def read(self, image, *, scene_supported=False, unobstructed=False):
        if not canvas_ok(image):
            return {"wagers": dict.fromkeys(map(str, range(8))),
                    "visibility": {}, "diagnostics": {}}
        diagnostics, visibility = {}, {}
        for s, (x, y, w, h) in enumerate(WAGERS_V2):
            patch = image[y:y + h, x:x + w]
            diagnostics[str(s)] = self.read_label(patch, s)
            visibility[str(s)] = patch_visibility(
                patch, self.coin, scene_supported=scene_supported,
                unobstructed=unobstructed)
            if (scene_supported is True and unobstructed is True
                    and diagnostics[str(s)].get("coin_unique") is True):
                visibility[str(s)] = {
                    **visibility[str(s)], "status": "VISIBLE_COIN_CANDIDATE",
                    "reason": "unique_v2_coin_locator_even_if_digits_unknown",
                    "coin_locator_preprocessing": "gaussian_050" if self.coin_smoothing
                    else None, "coin_score": diagnostics[str(s)]["coin_score"]}
        return {"wagers": {s: v["value"] for s, v in diagnostics.items()},
                "visibility": visibility, "diagnostics": diagnostics,
                "geometry": WAGERS_V2, "canonical_verified": False}


def run(first, second, bank_path, output, coin_smoothing=False):
    from poker_engine.perceptual.vision.gray_amount_recognizer import (
        GrayAmountRecognizer,
    )
    audit = json.loads((first / "samples.json").read_text())["audit_sha256"]
    sources, others = inventory(first, audit), inventory(second, audit)
    with np.load(bank_path, allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    reader = AA8WagerReaderV2(bank, load(first, sources[1500]),
                              coin_smoothing=coin_smoothing)
    observations = []
    for frame in (1320, 1500, 2070, 2400, 3390, 3400, 4755, 4890):
        pool, rows = (first, sources) if frame in sources else (second, others)
        observations.append({"frame": frame, "sha256": rows[frame]["sha256"],
                             "read": reader.read(load(pool, rows[frame]),
                                                 scene_supported=True,
                                                 unobstructed=True)})
    output.mkdir(parents=True, exist_ok=False)
    report = {"observations": observations, "bank_sha256": sha(bank_path),
              "implementation_sha256": sha(Path(__file__)),
              "first_manifest_sha256": sha(first / "samples.json"),
              "second_manifest_sha256": sha(second / "samples.json"),
              "scene_gate_manually_reviewed": True, "independent_holdout": False,
              "coin_smoothing_opt_in": coin_smoothing,
              "canonical_verified": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames": len(observations), "canonical_verified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "bank", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--coin-smoothing", action="store_true")
    args = parser.parse_args()
    run(args.first, args.second, args.bank, args.output, args.coin_smoothing)
