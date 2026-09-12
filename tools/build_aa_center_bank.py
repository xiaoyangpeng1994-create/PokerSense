"""Add source-reviewed centre-font prototypes without lowering OCR thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayRead, gray_glyphs
from tools.aa_pot_evidence import AACenterAmountCandidate
from tools.aa_visual_candidate import AASceneCandidate
from tools.aa_data_separation import assert_training_frames
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import pixels_digest, read_image


class CropCollector:
    def __init__(self):
        self.patch = None

    def diagnose(self, patch):
        self.patch = patch.copy()
        return GrayRead(None, None, "training_crop_only", ())


def build(window, base_bank, profile_path, output, *, include_fourteen=False):
    if verify_sha256sums(window) or verify_sha256sums(base_bank):
        raise ValueError("input integrity failure")
    source = json.loads((window / "samples.json").read_text())
    base = json.loads((base_bank / "report.json").read_text())
    if (base["source_sha256"] != source["source_sha256"] or
            base["profile_sha256"] != sha256_file(profile_path)):
        raise ValueError("bank source or geometry mismatch")
    reservations = json.loads((Path(__file__).resolve().parents[1] /
                               "configs/reproduction/aa_holdout_reservations_v1.json")
                              .read_text())
    if reservations["source_sha256"] != source["source_sha256"]:
        raise ValueError("wrong source")
    # Reviewed from full images, not copied from the title (720 title120/centre16).
    reviewed = {0: "16", 1050: "240", 8700: "120", 23640: "90", 32880: "329"}
    if include_fourteen:
        reviewed[1380] = "14"
    assert_training_frames(list(reviewed), reservations)
    profile = json.loads(profile_path.read_text())
    reference = read_image(window / "frames/frame_000000.png")
    collector = CropCollector()
    extractor = AACenterAmountCandidate(
        collector, AASceneCandidate(reference, profile),
        reference=reference if include_fourteen else None)
    with np.load(base_bank / "bank.npz", allow_pickle=False) as data:
        features, labels = list(data["features"]), list(data["labels"])
    origins = []
    for frame, text in reviewed.items():
        path = window / f"frames/frame_{frame:06d}.png"
        collector.patch = None
        extractor.recognize(read_image(path))
        if collector.patch is None:
            raise ValueError(f"no valid number crop at {frame}")
        glyphs, reason = gray_glyphs(collector.patch)
        if reason or len(glyphs) != len(text):
            raise ValueError(f"glyph segmentation mismatch at {frame}")
        for digit, (feature, box) in zip(text, glyphs):
            features.append(feature)
            labels.append(digit)
            origins.append({"frame": frame, "text": text, "digit": digit,
                            "glyph_box": box,
                            "source_png_sha256": sha256_file(path),
                            "crop_pixels_sha256": pixels_digest(collector.patch)})
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "bank.npz", features=np.asarray(features),
             labels=np.array(labels))
    report = {"source_sha256": source["source_sha256"],
              "profile_sha256": sha256_file(profile_path),
              "base_bank_sha256": sha256_file(base_bank / "bank.npz"),
              "include_fourteen": include_fourteen,
              "added_glyphs": origins, "floor": .90, "margin": .05,
              "independent_holdout": False, "release_eligible": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps({"added_glyphs": len(origins), "total": len(labels)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "base-bank", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--include-fourteen", action="store_true")
    args = parser.parse_args()
    build(args.window, args.base_bank, args.profile, args.output,
          include_fourteen=args.include_fourteen)
