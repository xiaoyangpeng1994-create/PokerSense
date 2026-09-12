"""Extract bounded sequential AA windows, checking original audit anchors.

This does not seek by nominal FPS or assign hands automatically. The saved
window is development exposure and keeps non-table/folded/waiting frames.
"""

import argparse
import json
from pathlib import Path

import cv2

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.capture_card_calibration.hashing import (
    sha256_file, verify_sha256sums, write_sha256sums,
)
from tools.wpk_video_dataset import pixels_digest
from tools.aa_data_separation import validate_reservations


def extract(audit, exploration, output, *, end=1800, step=15, reservations=None):
    if type(end) is not int or end < 0 or type(step) is not int or step < 1:
        raise ValueError("invalid frame window")
    if verify_sha256sums(audit) or verify_sha256sums(exploration):
        raise ValueError("input manifest integrity failure")
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    exclusions = []
    if reservations is not None:
        reserved = json.loads(reservations.read_text())
        validate_reservations(reserved)
        if reserved["source_sha256"] != report["source_sha256"]:
            raise ValueError("reservation source mismatch")
        exclusions = reserved["reserved_inclusive_intervals"]
    if end >= report["decoded_frames"]:
        raise ValueError("window beyond audited source")
    source = Path(report["source"])
    before = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
            report["source_size_bytes"], report["source_mtime_ns"]):
        raise ValueError("source attributes changed; revalidation needed")
    config_path = exploration / "normalization.candidate.json"
    config = NormalizationConfig.from_json(config_path.read_text())
    anchors = {s["source_frame_index"]: s for s in report["samples"]}
    index = [json.loads(line) for line in
             (audit / "decode_index.jsonl").read_text().splitlines()][:end + 1]
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    cap = cv2.VideoCapture(str(source))
    samples = []
    checked = []
    try:
        if not cap.isOpened():
            raise ValueError("source not opened")
        for frame_id in range(end + 1):
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"early decode termination at {frame_id}")
            stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if abs(stamp - index[frame_id]["opencv_pos_msec"]) > .001:
                raise ValueError("decoder time diverges from frozen sequential index")
            if frame_id in anchors:
                if pixels_digest(raw) != anchors[frame_id]["pixels_sha256"]:
                    raise ValueError("decoded anchor pixels differ")
                checked.append(frame_id)
            if frame_id % 3000 == 0:
                print(f"Source-bound sampling: {frame_id}/{end}", flush=True)
            if frame_id % step and frame_id != end:
                continue
            if any(start <= frame_id <= stop for start, stop in exclusions):
                continue
            canvas = normalize(raw, config)
            path = output / "frames" / f"frame_{frame_id:06d}.png"
            ok, encoded = cv2.imencode(".png", canvas)
            if not ok:
                raise ValueError("PNG encoding failed")
            path.write_bytes(encoded.tobytes())
            samples.append({"source_frame": frame_id, "opencv_pos_msec": stamp,
                            "file": path.relative_to(output).as_posix(),
                            "sha256": sha256_file(path), "hand_id": None,
                            "participation": "UNKNOWN", "split": "development"})
    finally:
        cap.release()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during extraction")
    result = {"source_sha256": report["source_sha256"],
              "normalization_sha256": sha256_file(config_path),
              "sequential_decode_start": 0, "end": end, "sampling_step": step,
              "checked_raw_anchors": checked, "samples": samples,
              "excluded_reserved_intervals": exclusions,
              "complete_hand_truth": False, "independent_holdout": False}
    (output / "samples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_sha256sums(output)
    print(json.dumps({"saved": len(samples), "checked_raw_anchors": checked}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "exploration", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--end", type=int, default=1800)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--exclude-reservations", type=Path)
    args = parser.parse_args()
    extract(args.audit, args.exploration, args.output, end=args.end, step=args.step,
            reservations=args.exclude_reservations)
