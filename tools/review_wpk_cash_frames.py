"""Private source-size Hero/seat6 cash crops for the current WPK reference layout."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.wpk_video_dataset import read_image


def render(window):
    target = window / "cash-review.png"
    if target.exists():
        raise ValueError("preserve earlier review")
    samples = json.loads((window / "samples.json").read_text(encoding="utf-8"))
    sheet = np.full((((len(samples) + 5) // 6) * 115, 6 * 240, 3), 235, np.uint8)
    for i, row in enumerate(samples):
        picture = read_image(window / row["image"])
        if picture.shape != (1080, 498, 3):
            raise ValueError("unsupported review canvas")
        x, y = (i % 6) * 240, (i // 6) * 115
        cv2.putText(sheet, f'{row["source_frame"]}  Hero / seat6', (x + 2, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        sheet[y + 27:y + 79, x:x + 125] = picture[1028:1080, 185:310]
        sheet[y + 27:y + 97, x + 130:x + 233] = picture[465:535, 395:498]
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise ValueError("cannot encode cash review")
    target.write_bytes(encoded.tobytes())
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.window))
