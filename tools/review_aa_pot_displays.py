"""Source-bound paired-display checks, not full pot-ledger acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.aa_pot_candidate import AAPotCandidate
from tools.aa_pot_evidence import AACenterAmountCandidate, reconcile_pot_displays
from tools.aa_visual_candidate import AASceneCandidate
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import read_image


def run(window, bank_dir, profile_path, output):
    if verify_sha256sums(window) or verify_sha256sums(bank_dir):
        raise ValueError("input integrity failure")
    samples = json.loads((window / "samples.json").read_text())
    bank_report = json.loads((bank_dir / "report.json").read_text())
    if (samples["source_sha256"] != bank_report["source_sha256"] or
            bank_report["profile_sha256"] != sha256_file(profile_path)):
        raise ValueError("wrong source or geometry")
    profile = json.loads(profile_path.read_text())
    ref = read_image(window / "frames/frame_000000.png")
    scene = AASceneCandidate(ref, profile)
    with np.load(bank_dir / "bank.npz", allow_pickle=False) as data:
        bank = GrayAmountRecognizer(data["features"], data["labels"], augment=True)
    title = AAPotCandidate(ref, bank, scene, prefix_reference=read_image(
        window / "frames/frame_000720.png"))
    center = AACenterAmountCandidate(bank, scene)
    rows = []
    for frame in (0, 720, 8700, 23640, 32880, 40650):
        path = window / f"frames/frame_{frame:06d}.png"
        image = read_image(path)
        first, second = title.recognize(image), center.recognize(image)
        rows.append({"frame": frame, "png_sha256": sha256_file(path),
                     "title_read": first, "center_read": second,
                     "result": reconcile_pot_displays(first["value"], second["value"])})
    output.mkdir(parents=True, exist_ok=False)
    report = {"source_sha256": samples["source_sha256"], "rows": rows,
              "release_eligible": False, "independent_holdout": False}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    write_sha256sums(output)
    print(json.dumps([{k: r[k] for k in ("frame", "result")} for r in rows]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("window", "bank", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.window, args.bank, args.profile, args.output)
