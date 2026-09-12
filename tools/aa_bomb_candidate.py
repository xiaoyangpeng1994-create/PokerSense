"""Source-bound visible lucky-bomb title detection, not inferred betting rules."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.aa_visual_candidate import canvas_ok
from tools.aa_data_separation import assert_training_frames
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import read_image


class AABombTitleCandidate:
    def __init__(self, reference):
        if not canvas_ok(reference):
            raise ValueError("AA reference canvas required")
        patch = cv2.cvtColor(reference[497:602, 63:441], cv2.COLOR_BGR2GRAY)
        self.templates = []
        for scale in np.arange(.80, 1.251, .025):
            template = cv2.resize(patch, None, fx=.5 * scale, fy=.5 * scale,
                                  interpolation=cv2.INTER_AREA)
            if float(template.std()) < 15:
                raise ValueError("non-flat title reference required")
            self.templates.append(template)

    def recognize(self, image):
        result = {"critical_hit_animation": None, "score": None,
                  "reason": "unsupported_canvas", "strategy_eligible": False}
        if not canvas_ok(image):
            return result
        search = cv2.resize(cv2.cvtColor(image[350:700, 10:488], cv2.COLOR_BGR2GRAY),
                            None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        if float(search.std()) < 15:
            return {**result, "reason": "no_positive_title_evidence"}
        score = max(float(cv2.matchTemplate(search, template,
                                            cv2.TM_CCOEFF_NORMED).max())
                    for template in self.templates)
        return {**result, "critical_hit_animation": True if score >= .80 else None,
                "score": score, "reason": "visible_title_candidate" if score >= .80
                else "no_positive_title_evidence"}


def run(window, output):
    if verify_sha256sums(window):
        raise ValueError("window integrity failure")
    samples = json.loads((window / "samples.json").read_text())
    reservation_path = Path(__file__).resolve().parents[1] / (
        "configs/reproduction/aa_holdout_reservations_v1.json")
    reservations = json.loads(reservation_path.read_text())
    if samples["source_sha256"] != reservations["source_sha256"]:
        raise ValueError("wrong source")
    assert_training_frames([r["source_frame"] for r in samples["samples"]],
                           reservations)
    reference_path = window / "frames/frame_008640.png"
    reader = AABombTitleCandidate(read_image(reference_path))
    rows = []
    for sample in samples["samples"]:
        path = (window / sample["file"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("sample escapes source")
        rows.append({"source_frame": sample["source_frame"],
                     **reader.recognize(read_image(path))})
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "rows": rows, "reference_sha256": sha256_file(reference_path),
        "source_sha256": samples["source_sha256"], "floor": .80,
        "independent_holdout": False, "full_animation_recall_verified": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps([r for r in rows if r["critical_hit_animation"] is True]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.window, args.output)
