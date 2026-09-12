"""Inspect already extracted AA8 development crops without changing labels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from tools.aa8_action_reader import TEMPLATES, mask, similarity
from tools.aa_visual_candidate import validate_layout
from tools.wpk_video_dataset import read_image


def run(pool, profile, output):
    cases = [(1590, 5, "call"), (2310, 3, "check"), (2490, 5, "call"),
             (2700, 2, "fold"), (2790, 3, "fold"), (3030, 5, "check"),
             (3120, 1, "all_in")]
    layout = json.loads(profile.read_text())
    validate_layout(layout)
    if len(layout["slots"]) != 8:
        raise ValueError("eight-slot layout required")
    manifest = json.loads((pool / "samples.json").read_text())
    rows = {r["global_frame"]: r for r in manifest["samples"]}

    def load(frame):
        row = rows[frame]
        path = (pool / row["file"]).resolve()
        if (row["role"] != "development" or not path.is_relative_to(pool.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("untrusted or non-development frame")
        return read_image(path)

    sheet = Image.new("RGB", (800, 160 * len(cases)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for i, (frame, slot, label) in enumerate(cases):
        source = load(frame)
        ref_frame, ref_slot = TEMPLATES[label]
        ref = load(ref_frame)
        a, b = (layout["slots"][s]["avatar"] for s in (slot, ref_slot))
        masks = [mask(source, a, label), mask(ref, b, label)]
        y = i * 160
        draw.text((5, y + 4), f"{frame} slot{slot} {label} "
                  f"score={similarity(*masks):.4f}; source / template", fill="black")
        for j, (img, box) in enumerate(((source, a), (ref, b))):
            x0, y0, w, h = box
            crop = img[y0 - 42:y0 + h, max(0, x0 - 2):x0 + w]
            sheet.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                        (5 + j * 100, y + 30))
            sheet.paste(Image.fromarray(masks[j]).resize((272, 100)),
                        (220 + j * 285, y + 40))
    output.mkdir(parents=True, exist_ok=False)
    sheet.save(output / "diagnostics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pool", "profile", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.pool, args.profile, args.output)
