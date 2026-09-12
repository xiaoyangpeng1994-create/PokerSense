"""Offline sequential AA source inventory; never trains or labels gameplay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from tools.capture_card_calibration.hashing import sha256_file
from tools.wpk_video_dataset import pixels_digest, probe_video


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def audit(source: Path, output: Path, every: int = 900) -> dict:
    if every <= 0:
        raise ValueError("sample interval must be positive")
    source = source.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    frames = output / "exploration_raw"
    frames.mkdir()
    before = signature(source)
    started = time.monotonic()
    print("Hashing original source (read only)", flush=True)
    source_hash = sha256_file(source)
    if signature(source) != before:
        raise ValueError("source changed during hashing")
    probe = probe_video(source)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("source cannot be opened")
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    anomalies = 0
    previous = None
    samples = []
    try:
        with (output / "decode_index.jsonl").open("x", encoding="utf-8") as log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                monotonic = previous is None or stamp > previous
                anomalies += int(not monotonic)
                record = {"source_frame_index": count,
                          "opencv_pos_msec": stamp,
                          "time_strictly_increasing": monotonic,
                          "width": frame.shape[1], "height": frame.shape[0]}
                log.write(json.dumps(record) + "\n")
                if count % every == 0:
                    path = frames / f"frame_{count:06d}.png"
                    success, encoded = cv2.imencode(".png", frame)
                    if not success:
                        raise ValueError("PNG encoding failed")
                    path.write_bytes(encoded.tobytes())
                    samples.append({**record,
                                    "file": path.relative_to(output).as_posix(),
                                    "sha256": sha256_file(path),
                                    "pixels_sha256": pixels_digest(frame),
                                    "participation": "UNKNOWN",
                                    "exposure": "exploration_not_holdout"})
                    print(f"Decoded {count + 1}/{declared} frames", flush=True)
                previous = stamp
                count += 1
    finally:
        cap.release()
    unchanged = signature(source) == before
    report = {"schema_version": 1, "platform": "aa_android_capture_card",
              "source": str(source), "source_sha256": source_hash,
              "source_size_bytes": before[0], "source_mtime_ns": before[1],
              "source_unchanged": unchanged, "probe": probe,
              "declared_frames": declared, "decoded_frames": count,
              "count_matches_header": count == declared and count > 0,
              "nonincreasing_decoder_times": anomalies,
              "time_basis": "OpenCV POS_MSEC; not verified phone time or latency",
              "termination": "read_false; EOF vs corruption not independently verified",
              "elapsed_seconds": time.monotonic() - started,
              "samples": samples, "normalization_verified": False,
              "independent_acceptance_ready": False,
              "vision_ready_for_strategy": False}
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not unchanged or not report["count_matches_header"]:
        raise ValueError("source integrity/count check failed; evidence retained")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=900)
    args = parser.parse_args()
    report = audit(args.source, args.output, args.every)
    summary = {key: value for key, value in report.items()
               if key not in {"samples", "probe"}}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
