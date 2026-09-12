"""Contact sheet for visual hand-boundary review, with no classifier output."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.probe_wpk_field_candidate import save_image
from tools.wpk_video_dataset import read_image


def render(window, output):
    if output.exists():
        raise ValueError("preserve prior boundary sheets")
    samples = json.loads((window / "samples.json").read_text())
    sheet = np.zeros((((len(samples) + 7) // 8) * 240, 8 * 160, 3), np.uint8)
    for index, sample in enumerate(samples):
        picture = read_image(window / sample["image"])
        y, x = index // 8 * 240, index % 8 * 160
        cv2.putText(sheet, str(sample["source_frame"]), (x + 3, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        strips = [picture[205:245, 200:310], picture[850:905, 205:293],
                  picture[920:1010, 190:310], picture[1035:1075, 205:295]]
        top = y + 22
        for patch in strips:
            height = int(patch.shape[0] * .8)
            resized = cv2.resize(patch, (int(patch.shape[1] * .8), height))
            sheet[top:top + height, x + 5:x + 5 + resized.shape[1]] = resized
            top += height + 3
    save_image(output, sheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.window, args.output)
