"""Private, source-indexed field strips for human AA hand-boundary review."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.capture_card_calibration.hashing import verify_sha256sums, write_sha256sums
from tools.wpk_video_dataset import read_image


def field_strip(image, frame):
    if image.shape != (1080, 498, 3):
        raise ValueError("AA normalized canvas required")
    strip = np.zeros((112, 498, 3), dtype=np.uint8)
    strip[28:111, :114] = image[936:1019, 192:306]
    strip[28:109, 118:398] = image[472:553, 109:389]
    strip[28:50, 405:481] = image[914:936, 211:287]
    strip[57:80, 399:498] = image[278:301, 200:299]
    cv2.putText(strip, f"source {frame} | hero / board / cash / pot", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return strip


def build(window, output, stride=4):
    if type(stride) is not int or stride < 1:
        raise ValueError("positive review stride required")
    if verify_sha256sums(window):
        raise ValueError("window integrity failure")
    data = json.loads((window / "samples.json").read_text())
    selected = data["samples"][::stride]
    output.mkdir(parents=True, exist_ok=False)
    strips = []
    for row in selected:
        path = (window / row["file"]).resolve()
        if not path.is_relative_to(window.resolve()):
            raise ValueError("sample escapes window")
        strips.append(field_strip(read_image(path), row["source_frame"]))
    for start in range(0, len(strips), 12):
        sheet = np.vstack(strips[start:start + 12])
        ok, encoded = cv2.imencode(".png", sheet)
        if not ok:
            raise ValueError("PNG encoding failed")
        (output / f"sheet_{start // 12:02d}.png").write_bytes(encoded.tobytes())
    (output / "review_index.json").write_text(json.dumps({
        "source_sha256": data["source_sha256"],
        "selected_frames": [r["source_frame"] for r in selected],
        "hand_boundaries_verified": False, "split": "development"}, indent=2))
    write_sha256sums(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    build(args.window, args.output, args.stride)
