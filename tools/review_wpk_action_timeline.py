"""Render raw action/stack strips for independent visual timeline annotation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.probe_wpk_field_candidate import save_image
from tools.wpk_field_candidate import crop
from tools.wpk_video_dataset import read_image


def render(window, profile_path, output, start=1197, end=1595, step=5):
    if output.exists():
        raise ValueError("preserve prior review pages")
    profile = json.loads(profile_path.read_text())
    output.mkdir(parents=True)
    frames = sorted(set(range(start, end + 1, step)) | {end})
    for begin in range(0, len(frames), 12):
        selected = frames[begin:begin + 12]
        sheet = np.zeros((len(selected) * 110, 864, 3), np.uint8)
        for row_index, index in enumerate(selected):
            image = read_image(window / "frames" / f"{index:06d}.png")
            y = row_index * 110
            cv2.putText(sheet, str(index), (4, y + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (255, 255, 255), 1)
            sheet[y + 19:y + 40, 4:68] = image[284:305, 217:281]
            board = cv2.resize(image[477:558, 109:390], (150, 43))
            sheet[y + 46:y + 89, 4:154] = board
            for slot in profile["slots"]:
                x = 160 + slot["slot"] * 88
                cv2.putText(sheet, f'S{slot["slot"]}', (x, y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
                top = y + 18
                for key in ("badge", "avatar", "stack"):
                    patch = crop(image, slot[key])
                    h, w = patch.shape[:2]
                    sheet[top:top + h, x:x + w] = patch
                    top += h + 2
        save_image(output / f"page-{begin // 12:02d}.png", sheet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("window", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    render(args.window, args.profile, args.output)
