#!/usr/bin/env python3
"""Extract a private, deduplicated LDPlayer calibration frame set.

The output is deliberately a local working dataset, not a repository asset.
Each retained PNG is the normalized 1440x2560 Android game canvas and the
manifest records its source timestamp, hashes, crop, and selection settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractionSettings:
    sample_seconds: float = 1.0
    host_toolbar_pixels: int = 48
    hash_distance: int = 5
    expected_width: int = 1440
    expected_height: int = 2560


def normalize_frame(frame: np.ndarray, settings: ExtractionSettings) -> np.ndarray:
    """Remove the LDPlayer host toolbar and validate the Android canvas."""
    if frame is None or frame.size == 0:
        raise ValueError("empty video frame")
    top = settings.host_toolbar_pixels
    if top < 0 or top >= frame.shape[0]:
        raise ValueError("host toolbar crop is outside the frame")
    canvas = frame[top:, :] if top else frame
    height, width = canvas.shape[:2]
    if (width, height) != (settings.expected_width, settings.expected_height):
        raise ValueError(
            f"normalized frame is {width}x{height}; expected "
            f"{settings.expected_width}x{settings.expected_height}"
        )
    return canvas


def difference_hash(image: np.ndarray, size: int = 16) -> int:
    """Return a deterministic perceptual dHash as an integer."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def should_keep(candidate_hash: int, retained: list[int], distance: int) -> bool:
    """Keep a frame only when it differs from every retained visual state."""
    if distance < 0:
        raise ValueError("hash distance must be non-negative")
    return all(hamming_distance(candidate_hash, prior) > distance for prior in retained)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_video(
    video_path: Path,
    output_dir: Path,
    settings: ExtractionSettings,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("video has invalid FPS or frame count")

    output_dir.mkdir(parents=True, exist_ok=True)
    interval = max(1, round(fps * settings.sample_seconds))
    retained_hashes: list[int] = []
    entries: list[dict] = []
    frame_index = 0
    sampled = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval:
                frame_index += 1
                continue
            sampled += 1
            canvas = normalize_frame(frame, settings)
            visual_hash = difference_hash(canvas)
            if should_keep(visual_hash, retained_hashes, settings.hash_distance):
                encoded_ok, encoded = cv2.imencode(".png", canvas)
                if not encoded_ok:
                    raise RuntimeError(f"PNG encoding failed at frame {frame_index}")
                payload = encoded.tobytes()
                digest = hashlib.sha256(payload).hexdigest()
                timestamp_ms = round(frame_index * 1000.0 / fps)
                filename = f"{timestamp_ms:010d}_{digest[:12]}.png"
                (output_dir / filename).write_bytes(payload)
                retained_hashes.append(visual_hash)
                entries.append({
                    "file": filename,
                    "timestamp_ms": timestamp_ms,
                    "source_frame": frame_index,
                    "sha256": digest,
                    "dhash": f"{visual_hash:064x}",
                })
            frame_index += 1
    finally:
        capture.release()

    manifest = {
        "schema_version": 1,
        "source": {
            "name": video_path.name,
            "sha256": file_sha256(video_path),
            "fps": fps,
            "frame_count": frame_count,
        },
        "normalization": {
            "crop": [0, settings.host_toolbar_pixels,
                     settings.expected_width, settings.expected_height],
            "output_size": [settings.expected_width, settings.expected_height],
        },
        "selection": {
            "sample_seconds": settings.sample_seconds,
            "hash_distance": settings.hash_distance,
            "sampled_frames": sampled,
            "retained_frames": len(entries),
        },
        "frames": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--host-toolbar-pixels", type=int, default=48)
    parser.add_argument("--hash-distance", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = ExtractionSettings(
        sample_seconds=args.sample_seconds,
        host_toolbar_pixels=args.host_toolbar_pixels,
        hash_distance=args.hash_distance,
    )
    manifest = extract_video(args.video, args.out, settings)
    print(
        f"retained {manifest['selection']['retained_frames']} of "
        f"{manifest['selection']['sampled_frames']} sampled frames in {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
