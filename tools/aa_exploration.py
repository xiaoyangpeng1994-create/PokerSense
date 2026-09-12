"""Generate source-bound AA crop candidates, not production calibration."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.perceptual.capture.normalization import NormalizationConfig, normalize
from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.wpk_video_dataset import read_image


def bright_bounds(image: np.ndarray) -> list[int] | None:
    """Diagnostic only: dark menus can shrink this interval without crop drift."""
    active = np.flatnonzero((image.max(axis=2) > 20).mean(axis=0) > 0.5)
    return [int(active[0]), int(active[-1]) + 1] if len(active) else None


def build(audit: Path, output: Path) -> dict:
    report = json.loads((audit / "report.json").read_text(encoding="utf-8"))
    if not report["source_unchanged"] or not report["count_matches_header"]:
        raise ValueError("source audit incomplete")
    config = NormalizationConfig(
        rotate_degrees=0, source_size=(1920, 1080),
        crop_after_rotation=(711, 0, 1209, 1080), output_size=(498, 1080),
        version="aa-exploration-candidate-v1")
    output.mkdir(parents=True, exist_ok=False)
    (output / "normalization.candidate.json").write_text(
        config.to_json(), encoding="utf-8")
    rows = []
    tiles = []
    for sample in report["samples"]:
        path = (audit / sample["file"]).resolve()
        if not path.is_relative_to(audit.resolve()):
            raise ValueError("sample escapes source audit")
        if sha256_file(path) != sample["sha256"]:
            raise ValueError("sample hash mismatch")
        raw = read_image(path)
        canvas = normalize(raw, config)
        name = path.name
        success, encoded = cv2.imencode(".png", canvas)
        if not success:
            raise ValueError("PNG encoding failed")
        (output / name).write_bytes(encoded.tobytes())
        rows.append({"source_frame_index": sample["source_frame_index"],
                     "source_png_sha256": sample["sha256"],
                     "bright_bounds_diagnostic": bright_bounds(raw),
                     "candidate_file": name, "scene": "UNKNOWN",
                     "critical_hit_enabled": "UNKNOWN",
                     "critical_hit_triggered": "UNKNOWN",
                     "squid_enabled": "UNKNOWN", "squid_triggered": "UNKNOWN",
                     "participation": "UNKNOWN", "split": "exploration"})
        tile = np.zeros((384, 166, 3), dtype=np.uint8)
        tile[24:] = cv2.resize(canvas, (166, 360))
        cv2.putText(tile, str(sample["source_frame_index"]), (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    for start in range(0, len(tiles), 12):
        batch = tiles[start:start + 12]
        batch += [np.zeros_like(tiles[0])] * (12 - len(batch))
        sheet = np.vstack([np.hstack(batch[i:i + 6]) for i in (0, 6)])
        success, encoded = cv2.imencode(".png", sheet)
        if not success:
            raise ValueError("contact sheet encoding failed")
        (output / f"contact_{start // 12:02d}.png").write_bytes(encoded.tobytes())
    result = {"source_sha256": report["source_sha256"],
              "normalization_status": "CANDIDATE_NOT_PRODUCTION",
              "full_frame_boundary_verified": False,
              "vision_ready_for_strategy": False, "frames": rows}
    (output / "review_queue.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sha256sums(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.audit, args.output)
