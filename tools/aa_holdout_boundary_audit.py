"""Boundary annotation of reserved intervals; never runs or trains recognizers."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.aa_data_separation import validate_reservations
from tools.aa_window_sheet import field_strip
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)


def run(audit, exploration, reservation_path, output):
    if verify_sha256sums(audit) or verify_sha256sums(exploration):
        raise ValueError("input integrity failure")
    source = json.loads((audit / "report.json").read_text())
    reservation = json.loads(reservation_path.read_text())
    validate_reservations(reservation)
    if reservation["source_sha256"] != source["source_sha256"]:
        raise ValueError("reservation source mismatch")
    original = Path(source["source"])
    before = original.stat()
    if (before.st_size, before.st_mtime_ns) != (
            source["source_size_bytes"], source["source_mtime_ns"]):
        raise ValueError("source changed")
    config_path = exploration / "normalization.candidate.json"
    config = NormalizationConfig.from_json(config_path.read_text())
    intervals = reservation["reserved_inclusive_intervals"]
    requested = {i: sorted(set(range(a, b + 1, 60)) | {b})
                 for i, (a, b) in enumerate(intervals)}
    selected = {frame: group for group, frames in requested.items() for frame in frames}
    frozen_index = [json.loads(line) for line in
                    (audit / "decode_index.jsonl").read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    tiles = {i: [] for i in requested}
    evidence = []
    cap = cv2.VideoCapture(str(original))
    try:
        for frame in range(max(selected) + 1):
            ok, raw = cap.read()
            if not ok:
                raise ValueError("early decode end")
            if frame % 3000 == 0:
                print(f"Boundary-only decode {frame}/{max(selected)}", flush=True)
            if frame not in selected:
                continue
            stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if abs(stamp - frozen_index[frame]["opencv_pos_msec"]) > .001:
                raise ValueError("source frame time mismatch")
            image = normalize(raw, config)
            path = output / f"frame_{frame:06d}.png"
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise ValueError("PNG encoding failure")
            path.write_bytes(encoded.tobytes())
            tiles[selected[frame]].append(field_strip(image, frame))
            evidence.append({"frame": frame, "file": path.name,
                             "sha256": sha256_file(path),
                             "role": "holdout_boundary_only"})
    finally:
        cap.release()
    after = original.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during annotation extraction")
    for group, images in tiles.items():
        for start in range(0, len(images), 8):
            success, encoded = cv2.imencode(".png", np.vstack(images[start:start + 8]))
            if not success:
                raise ValueError("sheet encoding failure")
            path = output / f"group_{group}_sheet_{start // 8}.png"
            path.write_bytes(encoded.tobytes())
    (output / "report.json").write_text(json.dumps({
        "source_sha256": source["source_sha256"],
        "reservation_sha256": sha256_file(reservation_path),
        "normalization_sha256": sha256_file(config_path), "evidence": evidence,
        "purpose": "boundary_annotation_only_not_model_tuning_or_scoring",
        "complete_hand_verified": False}, indent=2))
    write_sha256sums(output)
    print(json.dumps({"boundary_samples": len(evidence)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "exploration", "reservations", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.exploration, args.reservations, args.output)
