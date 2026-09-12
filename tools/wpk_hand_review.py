"""Extract dense, source-indexed review windows from frozen WPK recordings.

Decode sequentially from frame zero, validate against existing pixel anchors,
and preserve full-resolution PNGs. No recognition output becomes ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import pixels_digest, render_contact_sheets, review_page


def extract_window(corpus: Path, output: Path, start: int, end: int, step: int,
                   *, selected_frames: set[int] | None = None) -> dict:
    corpus, output = corpus.resolve(), output.resolve()
    if output == corpus or not output.is_relative_to(corpus):
        raise ValueError("window output must be a child of the new review corpus")
    if output.exists() and any(output.iterdir()):
        raise ValueError("window output must be new or empty")
    if start < 0 or end < start or step < 1:
        raise ValueError("invalid frame window")
    if selected_frames is not None and (
        not selected_frames or any(type(i) is not int or not start <= i <= end
                                   for i in selected_frames)
    ):
        raise ValueError("selected frames must be nonempty integers within window")
    inventory = json.loads((corpus / "sources.json").read_text(encoding="utf-8"))
    source = next(row for row in inventory["sources"]
                  if row["session"] == "session_002")
    path = Path(source["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
        raise ValueError("source no longer matches frozen inventory")
    old_dataset = path.parents[2]
    config_path = old_dataset / "normalization/normalization.json"
    config = NormalizationConfig.from_json(config_path.read_text(encoding="utf-8"))
    anchors = json.loads(
        (corpus / "session_002/samples.json").read_text(encoding="utf-8"))
    anchor_hashes = {row["source_frame"]: row["pixel_sha256"] for row in anchors}
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("cannot open frozen source")
    samples = []
    verified_anchors = []
    started = time.monotonic()
    try:
        for index in range(end + 1):
            wanted = index >= start and (
                index in selected_frames if selected_frames is not None else
                (index - start) % step == 0 or index in anchor_hashes or index == end)
            if wanted:
                ok, raw = capture.read()
            else:
                ok = capture.grab()
            if not ok:
                raise ValueError(f"source ended before requested frame {index}")
            if not wanted:
                continue
            picture = normalize(raw, config)
            digest = pixels_digest(picture)
            if index in anchor_hashes:
                if digest != anchor_hashes[index]:
                    raise ValueError(f"source pixel anchor mismatch at frame {index}")
                verified_anchors.append(index)
            ok, encoded = cv2.imencode(".png", picture)
            if not ok:
                raise ValueError("PNG encoding failed")
            relative = f"frames/{index:06d}.png"
            (output / relative).write_bytes(encoded.tobytes())
            samples.append({
                "source_frame": index,
                "container_pts_ms": float(capture.get(cv2.CAP_PROP_POS_MSEC)),
                "image": relative, "pixel_sha256": digest,
                "image_sha256": sha256_file(output / relative),
                "owner_presence": "UNKNOWN", "hand_participation": "UNKNOWN",
                "hero_turn": "UNKNOWN", "review_state": "unreviewed",
            })
    finally:
        capture.release()
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during extraction")
    summary = {
        "session": "session_002", "source_sha256": source["sha256"],
        "normalization_sha256": sha256_file(config_path),
        "requested_frame_window": [start, end],
        "sample_step": step if selected_frames is None else None,
        "selected_source_frames": sorted(selected_frames)
        if selected_frames is not None else None,
        "extracted_frames": len(samples), "verified_pixel_anchors": verified_anchors,
        "elapsed_s": round(time.monotonic() - started, 3),
        "complete_hand_verified": False, "golden_eligible": False,
    }
    for name, payload in (("summary.json", summary), ("samples.json", samples)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    (output / "review.html").write_text(
        review_page(samples, f"session_002 · {start}–{end}"), encoding="utf-8")
    render_contact_sheets(output)
    write_sha256sums(output)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--indices", help="comma-separated exact source frames")
    args = parser.parse_args()
    selected = {int(i) for i in args.indices.split(',')} if args.indices else None
    print(json.dumps(extract_window(args.corpus, args.output, args.start_frame,
                                    args.end_frame, args.step,
                                    selected_frames=selected), indent=2))
