"""Render the eight-slot draft against its pre-recording reference only."""

import argparse
import json
from pathlib import Path

import cv2

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.aa_visual_candidate import render_geometry, table_map
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import read_image


def render(reference_path, profile_path, output):
    profile = json.loads(profile_path.read_text())
    digest = sha256_file(reference_path)
    if digest != profile["geometry_reference_sha256"]:
        raise ValueError("geometry reference differs from reviewed source")
    normalization = NormalizationConfig(
        rotate_degrees=0, source_size=(1920, 1080),
        crop_after_rotation=(711, 0, 1209, 1080), output_size=(498, 1080),
        version="aa8-phone-candidate-v1")
    canvas = normalize(read_image(reference_path), normalization)
    mapping = table_map(profile)
    output.mkdir(parents=True, exist_ok=False)
    for name, image in (("canvas.png", canvas),
                        ("overlay.png", render_geometry(canvas, profile))):
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("PNG encoding failed")
        (output / name).write_bytes(encoded.tobytes())
    (output / "normalization.candidate.json").write_text(normalization.to_json())
    (output / "table_map.candidate.json").write_text(mapping.to_json())
    (output / "report.json").write_text(json.dumps({
        "reference_sha256": digest, "profile_sha256": sha256_file(profile_path),
        "physical_slots": 8, "hero_slot": 4, "recognition_validated": False,
        "source_is_pre_recording_reference": True}, indent=2))
    write_sha256sums(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.reference, args.profile, args.output)
