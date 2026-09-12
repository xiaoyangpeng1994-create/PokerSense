"""AA displayed pot: require the literal prefix and a unique colon before OCR."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa_visual_candidate import AASceneCandidate, canvas_ok
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import read_image


def pot_mask(image):
    return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                       (0, 0, 120), (179, 80, 255))


def colon_candidates(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    dots = [(x, y, w, h) for x, y, w, h, area in stats[1:]
            if 1 <= w <= 3 and 1 <= h <= 3 and area >= 2]
    candidates = {int(x) for x, y, w, h in dots
                  if any(abs(x - xx) <= 1 and 4 <= yy - y <= 8
                         for xx, yy, ww, hh in dots)}
    return sorted(candidates)


def colon_x(mask):
    candidates = colon_candidates(mask)
    return candidates[0] if len(candidates) == 1 else None


def prefix_score(candidate, reference):
    first = cv2.GaussianBlur(candidate.astype(np.float32), (3, 3), .7)
    second = cv2.GaussianBlur(reference.astype(np.float32), (3, 3), .7)
    return float(cv2.matchTemplate(np.pad(first, 2), second,
                                   cv2.TM_CCOEFF_NORMED).max())


def region(image):
    return image[279:300, 198:302]


class AAPotCandidate:
    def __init__(self, reference, bank, scene, *, prefix_reference=None):
        if not canvas_ok(reference):
            raise ValueError("AA canvas required")
        mask = pot_mask(region(reference))
        x = colon_x(mask)
        if x is None or x < 34:
            raise ValueError("reviewed pot prefix missing")
        self.prefix = mask[:, x - 34:x + 2].copy()
        self.prefixes = [self.prefix]
        if prefix_reference is not None:
            if not canvas_ok(prefix_reference):
                raise ValueError("AA prefix canvas required")
            extra = pot_mask(region(prefix_reference))
            extra_x = colon_x(extra)
            if extra_x is None or extra_x < 34:
                raise ValueError("additional reviewed prefix missing")
            self.prefixes.append(extra[:, extra_x - 34:extra_x + 2].copy())
        self.bank, self.scene = bank, scene

    def recognize(self, image):
        result = {"value": None, "reason": "unsupported_scene", "raw_text": None}
        if (not canvas_ok(image) or
                self.scene.recognize(image)["scene"] != "AA_TABLE_CANDIDATE"):
            return result
        patch = region(image)
        mask = pot_mask(patch)
        matches = [(x, max(prefix_score(mask[:, x - 34:x + 2], p)
                           for p in self.prefixes))
                   for x in colon_candidates(mask) if x >= 34]
        matches = [(x, score) for x, score in matches if score >= .90]
        if len(matches) != 1:
            return {**result, "reason": "no_unique_pot_colon"}
        x, score = matches[0]
        # Leave two background pixels around the number. Prefix pixels cannot
        # enter the crop and be guessed as digits; decimals still fail closed.
        ys, xs = np.where(mask[:, x + 3:] > 0)
        if not len(xs):
            return {**result, "reason": "no_digits"}
        left, right = int(xs.min()) + x + 3, int(xs.max()) + x + 4
        top, bottom = int(ys.min()), int(ys.max()) + 1
        if right + 2 > patch.shape[1]:
            return {**result, "reason": "clipped_digits"}
        if top < 2 or bottom + 2 > patch.shape[0]:
            return {**result, "reason": "clipped_digits"}
        if left - 2 <= x + 1:
            # A verified blank separator is enough; padding must not copy the
            # adjacent colon into an otherwise complete first digit.
            if left < x + 3 or np.any(mask[:, left - 1]):
                return {**result, "reason": "clipped_digits"}
            background = tuple(int(v) for v in patch[top, left - 1])
            digits = cv2.copyMakeBorder(patch[top:bottom, left:right], 2, 2, 2, 2,
                                        cv2.BORDER_CONSTANT, value=background)
        else:
            digits = patch[top - 2:bottom + 2, left - 2:right + 2]
        read = self.bank.diagnose(digits)
        return {**asdict(read), "prefix_score": score,
                "money_semantics": "displayed_total_not_side_pot_ledger"}


def run(window, truth_dir, exploration, bank_dir, profile_path, output):
    for folder in (window, truth_dir, exploration, bank_dir):
        if verify_sha256sums(folder):
            raise ValueError("input integrity failure")
    truth = json.loads((truth_dir / "truth.json").read_text())
    if truth["samples_sha256"] != sha256_file(window / "samples.json"):
        raise ValueError("truth does not bind this window")
    bank_report = json.loads((bank_dir / "report.json").read_text())
    if (bank_report["source_sha256"] != truth["source_sha256"] or
            bank_report["profile_sha256"] != sha256_file(profile_path)):
        raise ValueError("bank source or geometry mismatch")
    profile = json.loads(profile_path.read_text())
    reference = read_image(exploration / "frame_000000.png")
    with np.load(bank_dir / "bank.npz", allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    prefix_reference = read_image(window / "frames/frame_000720.png")
    reader = AAPotCandidate(reference, bank, AASceneCandidate(reference, profile),
                            prefix_reference=prefix_reference)
    rows = []
    for checkpoint in truth["checkpoints"]:
        path = (window / checkpoint["normalized_file"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("sample escapes window")
        if sha256_file(path) != checkpoint["normalized_sha256"]:
            raise ValueError("reviewed sample hash mismatch")
        rows.append({"frame": checkpoint["frame"], "expected": checkpoint["pot"],
                     **reader.recognize(read_image(path))})
    output.mkdir(parents=True, exist_ok=False)
    report = {"rows": rows, "independent_holdout": False, "release_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "truth", "exploration", "bank", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.window, args.truth, args.exploration, args.bank, args.profile, args.output)
