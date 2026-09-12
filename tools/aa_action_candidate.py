"""AA visible fold/all-in text candidate, NOT a betting event or turn detector."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa_visual_candidate import AASceneCandidate, canvas_ok, validate_layout
from tools.capture_card_calibration.hashing import verify_sha256sums, write_sha256sums
from tools.wpk_video_dataset import read_image


def text_patch(image, avatar, label):
    x, y, width, _ = avatar
    if label == "fold":
        return image[y + 23:y + 48, x + width // 2 - 22:x + width // 2 + 22]
    return image[y + 20:y + 48, x + width // 2 - 32:x + width // 2 + 32]


def text_mask(patch, label):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    limits = ((0, 0, 140), (179, 80, 255)) if label == "fold" else (
        (18, 90, 140), (40, 255, 255))
    return cv2.inRange(hsv, *limits)


class AAActionCandidate:
    def __init__(self, reference, profile, scene, *, smooth=False):
        validate_layout(profile)
        if not canvas_ok(reference):
            raise ValueError("AA canvas required")
        self.profile, self.scene = profile, scene
        self.smooth = smooth
        self.floor = .90 if smooth else .85
        self.templates = {}
        for label, slot in (("fold", 1), ("all_in", 2)):
            patch = text_patch(reference, profile["slots"][slot]["avatar"], label)
            mask = text_mask(patch, label)
            if np.count_nonzero(mask) < 25:
                raise ValueError("reviewed action glyph missing")
            self.templates[label] = mask

    def recognize(self, image):
        output = {}
        supported = (canvas_ok(image) and
                     self.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE")
        for row in self.profile["slots"]:
            scores = {}
            if supported:
                for label, template in self.templates.items():
                    patch = text_patch(image, row["avatar"], label)
                    mask = text_mask(patch, label)
                    padded = np.pad(mask, 2)
                    score = 0.
                    for dy in range(5):
                        for dx in range(5):
                            shifted = padded[dy:dy + mask.shape[0],
                                             dx:dx + mask.shape[1]]
                            denom = (np.count_nonzero(template)
                                     + np.count_nonzero(shifted))
                            intersection = np.count_nonzero(
                                (template > 0) & (shifted > 0))
                            ratio = 2 * intersection / denom if denom else 0.
                            score = max(score, ratio)
                    if self.smooth:
                        first = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), .7)
                        second = cv2.GaussianBlur(
                            template.astype(np.float32), (3, 3), .7)
                        score = max(0., float(cv2.matchTemplate(
                            np.pad(first, 2), second, cv2.TM_CCOEFF_NORMED).max()))
                    scores[label] = score
            accepted = [label for label, score in scores.items() if score >= self.floor]
            output[str(row["slot"])] = {
                "value": accepted[0] if len(accepted) == 1 else None,
                "scores": scores, "observational_only": True,
                "current_actor": "UNKNOWN"}
        return output


def run(exploration, profile_path, output, *, smooth=False):
    if verify_sha256sums(exploration):
        raise ValueError("exploration integrity failure")
    profile = json.loads(profile_path.read_text())
    queue = json.loads((exploration / "review_queue.json").read_text())
    if queue["source_sha256"] != (
            "2638c3ea894fa6a342947b9c4a06b5d7746a6512360f4ef7d79a387fe89da82f"):
        raise ValueError("action template sources are frozen AA frames")
    reader = AAActionCandidate(
        read_image(exploration / "frame_029700.png"), profile,
        AASceneCandidate(read_image(exploration / "frame_000000.png"), profile),
        smooth=smooth)
    rows = []
    for sample in queue["frames"]:
        path = (exploration / sample["candidate_file"]).resolve()
        if not path.is_relative_to(exploration.resolve()):
            raise ValueError("sample escapes exploration")
        rows.append({"source_frame": sample["source_frame_index"],
                     "actions": reader.recognize(read_image(path))})
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps({
        "source_sha256": queue["source_sha256"], "rows": rows,
        "template_source_frame": 29700, "floor": reader.floor, "smooth": smooth,
        "independent_holdout": False, "release_eligible": False}, indent=2))
    write_sha256sums(output)
    for row in rows:
        if row["source_frame"] in (0, 17100, 29700):
            print(json.dumps(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exploration", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--smooth", action="store_true")
    args = parser.parse_args()
    run(args.exploration, args.profile, args.output, smooth=args.smooth)
