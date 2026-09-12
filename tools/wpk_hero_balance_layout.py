"""Offline positive-evidence Hero balance location selector, not participation AI.

Two independent locations may be visible during a transition: conflict abstains.
Never guesses from missing buttons/cards, and never carries a previous balance.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


LOCATIONS = {
    "lower": (205, 1043, 88, 28),
    "raised": (205, 924, 88, 28),
}
DIGIT_RECTS = {"lower": (219, 1046, 60, 21), "raised": (219, 927, 60, 21)}


def crop(image, rect):
    x, y, width, height = rect
    return image[y:y + height, x:x + width]


def pill_edges(patch):
    """Use only left/right capsule ends, never the numeric text in the centre."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 60)
    edges[:, 13:-13] = 0
    return edges


@dataclass(frozen=True)
class LayoutRead:
    location: str | None
    status: str
    lower_score: float
    raised_score: float
    participation: str = "UNKNOWN"


def choose_layout(lower, raised, floor=.70):
    if (any(not math.isfinite(v) or not 0 <= v <= 1 for v in (lower, raised))
            or not math.isfinite(floor) or not 0 < floor <= 1):
        raise ValueError("invalid layout scores or floor")
    accepted = [key for key, score in (("lower", lower), ("raised", raised))
                if score >= floor]
    if len(accepted) == 2:
        return LayoutRead(None, "CONFLICT", lower, raised)
    if not accepted:
        return LayoutRead(None, "UNKNOWN", lower, raised)
    return LayoutRead(accepted[0], "CANDIDATE", lower, raised)


class HeroBalanceLayout:
    def __init__(self, templates, floor=.70):
        if (set(templates) != set(LOCATIONS) or isinstance(floor, bool)
                or not math.isfinite(floor) or not 0 < floor <= 1):
            raise ValueError("both layout templates and a bounded floor required")
        self.templates = {}
        self.floor = floor
        for key, patch in templates.items():
            if patch.shape != (28, 88, 3):
                raise ValueError("wrong template dimensions")
            feature = pill_edges(patch)
            if np.count_nonzero(feature) < 20:
                raise ValueError("no positive capsule edge template")
            feature.setflags(write=False)
            self.templates[key] = feature

    def recognize(self, image):
        if image is None or image.shape != (1080, 498, 3):
            return LayoutRead(None, "UNKNOWN", 0, 0)
        scores = {}
        for key, rect in LOCATIONS.items():
            edges = pill_edges(crop(image, rect))
            template = self.templates[key]
            # Symmetric pixel F1 penalizes both missing and extra edge structure.
            intersection = np.count_nonzero((edges > 0) & (template > 0))
            denominator = np.count_nonzero(edges) + np.count_nonzero(template)
            scores[key] = float(2 * intersection / denominator) if denominator else 0.0
        return choose_layout(scores["lower"], scores["raised"], self.floor)


def read_hero_balance(image, selector, recognizer, calibrator):
    """Same-image composition only; unknown/conflicting location never calls OCR."""
    layout = selector.recognize(image)
    result = {"location": layout.location, "layout_status": layout.status,
              "lower_score": layout.lower_score, "raised_score": layout.raised_score,
              "value": None, "status": "unknown", "raw_score": None,
              "participation": "UNKNOWN", "usable_stack_verified": False,
              "production_valid": False}
    if layout.location is None:
        return result
    read = recognizer.recognize(crop(image, DIGIT_RECTS[layout.location]))
    result["raw_score"] = read.raw_score
    if read.value is not None and not calibrator.should_abstain(read.raw_score):
        result.update(value=str(read.value.value), status="valid")
    return result
