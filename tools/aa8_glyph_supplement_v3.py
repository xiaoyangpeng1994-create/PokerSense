"""Offline development-only text competition; never a legal-action provider."""

import cv2
import numpy as np

from tools.aa8_action_reader import similarity
from tools.aa_visual_candidate import canvas_ok


def white_text(image, avatar):
    x, y, width, _ = avatar
    patch = image[y + 23:y + 48, x + width // 2 - 26:x + width // 2 + 26]
    return cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                       (0, 0, 140), (179, 95, 255))


def resolve(base, scores, floor=.90, margin=.04):
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, best = ordered[0]
    if best < floor:
        return base
    if best - ordered[1][1] < margin:
        return base  # Supplemental evidence is inconclusive; keep baseline.
    if label == "muck":
        return None
    if label == "all_in":
        return "all_in" if base in (None, "all_in") else None
    return base


class GlyphSupplementV3:
    """References are hash-verified by the caller and remain private training.

    Apply only to scene-supported, non-suspended frames. Muck is suppressed,
    not relabelled as a betting action. The original reader remains unchanged.
    """

    def __init__(self, profile, references):
        self.profile = profile
        if set(references) != {"fold", "muck", "all_in"}:
            raise ValueError("three explicit competing references required")
        self.templates = {}
        for label, (image, slot) in references.items():
            if not canvas_ok(image) or type(slot) is not int or not 0 <= slot < 8:
                raise ValueError("invalid reference image or slot")
            template = white_text(image, profile["slots"][slot]["avatar"])
            if np.count_nonzero(template) < 25:
                raise ValueError("missing positive text reference")
            self.templates[label] = template

    def recognize(self, image, base, *, supported):
        if supported is not True or not canvas_ok(image):
            return {str(i): None for i in range(8)}
        result = dict(base)
        for row in self.profile["slots"]:
            slot = str(row["slot"])
            value = white_text(image, row["avatar"])
            scores = {k: similarity(value, t) for k, t in self.templates.items()}
            result[slot] = resolve(base.get(slot), scores)
        return result
