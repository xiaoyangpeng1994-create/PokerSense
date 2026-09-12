"""Render offline action feature alternatives; never changes production templates."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from poker_engine.core.enums import ActionType
from poker_engine.perceptual.vision.action_recognizer import (
    TemplateActionGlyphRecognizer,
)
from tools.probe_wpk_field_candidate import save_image
from tools.wpk_field_candidate import crop
from tools.wpk_video_dataset import read_image


def feature(picture, action, mode):
    if mode.endswith("_blur"):
        return cv2.GaussianBlur(feature(picture, action, mode[:-5]), (3, 3), .7)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    if mode == "original":
        return TemplateActionGlyphRecognizer._mask(picture, ActionType(action))
    if mode == "white110":
        return cv2.inRange(hsv, (0, 0, 110), (180, 100, 255))
    value = hsv[:, :, 2]
    highpass = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
    mask = TemplateActionGlyphRecognizer._mask(picture, ActionType(action))
    return cv2.bitwise_and(mask, (highpass > 20).astype(np.uint8) * 255)


def run(window, profile_path, output):
    if output.exists():
        raise ValueError("preserve previous diagnostics")
    profile = json.loads(profile_path.read_text())
    modes = ("original_blur", "white110_blur", "tophat_blur")
    templates = {}
    for row in profile["new_templates"]:
        picture = read_image(window / "frames" / f'{row["source_frame"]:06d}.png')
        templates[row["action"]] = crop(picture, row["rect"])
    tiles, scores = [], []
    for frame, slot, action in ((8500, 2, "fold"), (8500, 5, "fold"),
                                (8600, 6, "all_in"), (8700, 0, "all_in")):
        image = read_image(window / "frames" / f"{frame:06d}.png")
        patch = crop(image, profile["slots"][slot]["avatar"])
        tile = np.zeros((150, 650, 3), np.uint8)
        cv2.putText(tile, f"{frame}/{slot} {action}: original white110 tophat",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tile[30:90, :140] = cv2.resize(patch, (140, 60))
        for index, mode in enumerate(modes):
            mask = feature(patch, action, mode)
            template = feature(templates[action], action, mode)
            score = float(cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED).max())
            scores.append({"frame": frame, "slot": slot, "mode": mode, "score": score})
            tile[30:90, 150 + index * 160:290 + index * 160] = cv2.cvtColor(
                cv2.resize(mask, (140, 60)), cv2.COLOR_GRAY2BGR)
        tiles.append(tile)
    output.mkdir(parents=True)
    save_image(output / "features.png", np.concatenate(tiles))
    (output / "scores.json").write_text(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("window", "profile", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.window, args.profile, args.output), indent=2))
