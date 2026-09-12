"""AA source-bound grayscale stack candidate; visible text is not playable cash."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import (
    GrayAmountRecognizer, gray_glyphs,
)
from tools.aa_visual_candidate import AASceneCandidate, canvas_ok, validate_layout
from tools.aa_data_separation import assert_training_frames
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import read_image


def stack_patch(image, rect):
    x, y, width, height = rect
    return image[y + 2:y + height - 2, x + 10:x + width - 10]


class AAAmountCandidate:
    def __init__(self, bank, profile, scene):
        validate_layout(profile)
        self.bank, self.profile, self.scene = bank, profile, scene

    def recognize(self, image):
        supported = (canvas_ok(image) and
                     self.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE")
        values = {}
        for slot in self.profile["slots"]:
            result = {"value": None, "reason": "unsupported_scene", "scores": (),
                      "raw_text": None}
            if supported:
                result = asdict(self.bank.diagnose(stack_patch(image, slot["stack"])))
            values[str(slot["slot"])] = result
        return {"stacks": values, "participation": "UNKNOWN",
                "strategy_eligible": False, "money_semantics": "displayed_text_only"}


def run(exploration, profile_path, output, *, window=None, truth_dir=None):
    if (window is None) != (truth_dir is None):
        raise ValueError("window and bound truth are required together")
    if verify_sha256sums(exploration):
        raise ValueError("exploration hash mismatch")
    queue = json.loads((exploration / "review_queue.json").read_text())
    if queue["source_sha256"] != (
            "2638c3ea894fa6a342947b9c4a06b5d7746a6512360f4ef7d79a387fe89da82f"):
        raise ValueError("manual stack labels are source-specific")
    profile = json.loads(profile_path.read_text())
    validate_layout(profile)
    reference = read_image(exploration / "frame_000000.png")
    # Whole-number labels reviewed from source frame0 before glyph extraction.
    labels = {1: "1671", 2: "2160", 3: "857", 4: "3526", 5: "728",
              6: "979", 7: "2175", 8: "451"}
    features, targets, origins = [], [], []
    for slot, text in labels.items():
        rect = profile["slots"][slot]["stack"]
        glyphs, reason = gray_glyphs(stack_patch(reference, rect))
        if reason or len(glyphs) != len(text):
            raise ValueError(f"review glyph segmentation at slot{slot}: {reason}")
        for digit, (feature, box) in zip(text, glyphs):
            features.append(feature)
            targets.append(digit)
            origins.append({"source_frame": 0, "slot": slot, "text": text,
                            "digit": digit, "glyph_box_in_inset_crop": box})
    if window is not None:
        if verify_sha256sums(window) or verify_sha256sums(truth_dir):
            raise ValueError("additional source integrity failure")
        truth = json.loads((truth_dir / "truth.json").read_text())
        if (truth["source_sha256"] != queue["source_sha256"] or
                truth["samples_sha256"] != sha256_file(window / "samples.json")):
            raise ValueError("additional truth is not source-bound")
        for checkpoint in truth["checkpoints"]:
            if checkpoint["frame"] not in (1200, 1320, 1740):
                continue
            path = (window / checkpoint["normalized_file"]).resolve()
            if not path.is_relative_to(window.resolve()):
                raise ValueError("additional sample escapes source window")
            if sha256_file(path) != checkpoint["normalized_sha256"]:
                raise ValueError("additional sample hash mismatch")
            rect = profile["slots"][profile["hero_slot"]]["stack"]
            glyphs, reason = gray_glyphs(stack_patch(read_image(path), rect))
            text = checkpoint["cash"]
            if reason or len(glyphs) != len(text):
                raise ValueError("additional reviewed glyph segmentation failed")
            for digit, (feature, box) in zip(text, glyphs):
                features.append(feature)
                targets.append(digit)
                origins.append({"source_frame": checkpoint["frame"], "slot": 5,
                                "text": text, "digit": digit,
                                "normalized_sha256": checkpoint["normalized_sha256"],
                                "glyph_box_in_inset_crop": box})
    reservations = json.loads((Path(__file__).resolve().parents[1] /
                               "configs/reproduction/aa_holdout_reservations_v1.json")
                              .read_text())
    if reservations["source_sha256"] != queue["source_sha256"]:
        raise ValueError("reservation source mismatch")
    assert_training_frames([r["source_frame"] for r in origins], reservations)
    bank = GrayAmountRecognizer(features, targets, augment=True)
    reader = AAAmountCandidate(bank, profile, AASceneCandidate(reference, profile))
    rows = []
    for sample in queue["frames"]:
        path = (exploration / sample["candidate_file"]).resolve()
        if not path.is_relative_to(exploration.resolve()):
            raise ValueError("sample escapes exploration")
        rows.append({"source_frame": sample["source_frame_index"],
                     **reader.recognize(read_image(path))})
    # Later already-exposed development checkpoint, not independent acceptance.
    validation = ["138", "1109", "0", "168", "1054", "318", "166", "1225", "140"]
    candidate = next(r for r in rows if r["source_frame"] == 29700)
    comparisons = [{"slot": i, "expected": wanted,
                    "got": candidate["stacks"][str(i)]["value"]}
                   for i, wanted in enumerate(validation)]
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "bank.npz", features=np.asarray(features),
             labels=np.array(targets))
    report = {"source_sha256": queue["source_sha256"], "origins": origins,
              "template_frame_sha256": sha256_file(exploration / "frame_000000.png"),
              "profile_sha256": sha256_file(profile_path), "rows": rows,
              "comparisons": comparisons, "independent_holdout": False,
              "release_eligible": False, "floor": .90, "margin": .05}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps(comparisons))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exploration", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--window", type=Path)
    parser.add_argument("--truth", type=Path)
    args = parser.parse_args()
    run(args.exploration, args.profile, args.output,
        window=args.window, truth_dir=args.truth)
