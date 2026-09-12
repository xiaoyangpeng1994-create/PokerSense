"""Offline eight-slot, two-zone action candidate and stack-geometry experiment.

Not imported by the live pipeline. Floors are development engineering choices,
not calibrated probabilities. Recognized text is never emitted as a new event.
"""

from dataclasses import replace
from copy import deepcopy
import math

import cv2
import numpy as np

from poker_engine.core.enums import ActionType
from poker_engine.perceptual.vision.action_recognizer import (
    TemplateActionGlyphRecognizer,
)
from poker_engine.perceptual.vision.table_map import ROI, ROIKind


def validate_profile(profile):
    if profile["canvas"] != [498, 1080]:
        raise ValueError("candidate has only this explicitly reviewed canvas")
    for key in ("action_floor", "action_margin"):
        value = profile[key]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("invalid candidate gate")
    if [row["slot"] for row in profile["slots"]] != list(range(8)):
        raise ValueError("exactly eight ordered slot definitions required")
    for row in profile["slots"]:
        for kind in ("badge", "avatar", "stack"):
            validate_rect(row[kind], profile["canvas"])
    if (len(profile["new_templates"]) != 2 or
            {row["action"] for row in profile["new_templates"]} != {"fold", "all_in"}):
        raise ValueError("fold and all-in need explicit reviewed sources")
    for row in profile["new_templates"]:
        validate_rect(row["rect"], profile["canvas"])
        if type(row["source_frame"]) is not int or row["source_frame"] < 0:
            raise ValueError("invalid template source frame")


def validate_rect(rect, canvas):
    if len(rect) != 4 or any(type(v) is not int for v in rect):
        raise ValueError("pixel rect requires four integers")
    x, y, width, height = rect
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("invalid pixel rectangle")
    if x + width > canvas[0] or y + height > canvas[1]:
        raise ValueError("pixel rectangle outside canvas")


def crop(image, rect):
    x, y, width, height = rect
    return image[y:y + height, x:x + width]


def choose_action(scores, floor, margin):
    """Two plausible actions abstain; the strongest match alone is insufficient."""
    if not scores or any(not math.isfinite(v) or not 0 <= v <= 1
                         for v in scores.values()):
        raise ValueError("invalid action scores")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best, value = ordered[0]
    runner = ordered[1][1] if len(ordered) > 1 else 0.0
    accepted = value >= floor and value - runner >= margin
    return {"value": best if accepted else None, "accepted_candidate": accepted,
            "best_label": best, "score": value, "runner_up": runner,
            "scores": scores, "not_an_event": True, "production_valid": False}


class ActionCandidate:
    def __init__(self, profile, templates):
        validate_profile(profile)
        required = {"fold", "all_in", "bet", "call", "check", "raise"}
        if set(templates) != required:
            raise ValueError("all six observed action classes required")
        self.profile = deepcopy(profile)
        self.templates = {}
        for action, template in templates.items():
            mask = TemplateActionGlyphRecognizer._mask(template, ActionType(action))
            if mask.size == 0 or np.std(mask) == 0:
                raise ValueError("degenerate action template")
            mask = mask.copy()
            mask.setflags(write=False)
            self.templates[action] = mask

    def recognize(self, image):
        if image is None or image.shape != (1080, 498, 3):
            raise ValueError("wrong canvas; do not rescale unreviewed geometry")
        result = {}
        for row in self.profile["slots"]:
            scores = {}
            for action, template in self.templates.items():
                zone = "avatar" if action in ("fold", "all_in") else "badge"
                mask = TemplateActionGlyphRecognizer._mask(
                    crop(image, row[zone]), ActionType(action))
                if any(a > b for a, b in zip(template.shape, mask.shape)):
                    scores[action] = 0.0
                else:
                    match = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
                    scores[action] = min(1.0, max(0.0, float(np.max(match))))
            result[str(row["slot"])] = choose_action(
                scores, self.profile["action_floor"], self.profile["action_margin"])
        return result


def stack_table(table, profile):
    """Change only offline stack rectangles, retaining recognizer and gates."""
    validate_profile(profile)
    rois = tuple(r for r in table.rois if r.kind is not ROIKind.STACK)
    for row in profile["slots"]:
        x, y, width, height = row["stack"]
        rois += (ROI(ROIKind.STACK, x / 498, y / 1080, width / 498,
                     height / 1080, row["slot"]),)
    return replace(table, rois=rois)
