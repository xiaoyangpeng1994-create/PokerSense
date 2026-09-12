"""Offline feature candidates: blur-stable action masks and white-on-dark digits.

No production imports this module. Numeric text is not a playable-balance claim.
Unsupported punctuation, cropped glyphs and ambiguous matches abstain.
"""

import math

import cv2
import numpy as np

from poker_engine.perceptual.vision.amount_recognizer import _column_spans
from tools.wpk_field_candidate import ActionCandidate, choose_action, crop


def colour_mask(patch, action):
    """Raw grayscale pixels must not bypass the yellow all-in colour condition."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if action == "all_in":
        return cv2.inRange(hsv, (15, 90, 130), (45, 255, 255))
    return cv2.inRange(hsv, (0, 0, 145), (180, 100, 255))


def action_feature(patch, action):
    mask = colour_mask(patch, action)
    if action == "all_in":
        value = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]
        highpass = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_and(mask, (highpass > 20).astype(np.uint8) * 255)
    return cv2.GaussianBlur(mask, (3, 3), .7)


class FeatureActionCandidate(ActionCandidate):
    def __init__(self, profile, templates, *, variants=None,
                 badge_scale_tolerance=False):
        super().__init__(profile, templates)
        # Committed badge templates are binary; only avatar actions need colour
        # source crops for glow removal. Never re-threshold a saved soft feature.
        for action in self.templates:
            if action in ("fold", "all_in"):
                feature = action_feature(templates[action], action)
            else:
                feature = cv2.GaussianBlur(self.templates[action], (3, 3), .7)
            feature.setflags(write=False)
            self.templates[action] = feature
        self.variants = {key: [value] for key, value in self.templates.items()}
        for action, patches in (variants or {}).items():
            if action not in self.variants:
                raise ValueError("unknown action variant class")
            for patch in patches:
                feature = action_feature(patch, action)
                if np.std(feature) < 1:
                    raise ValueError("empty variant feature")
                feature.setflags(write=False)
                self.variants[action].append(feature)
        if badge_scale_tolerance:
            for action in ("call", "check", "bet", "raise"):
                expanded = []
                for feature in self.variants[action]:
                    for x in (.95, 1., 1.05):
                        for y in (.95, 1., 1.05):
                            variant = cv2.resize(feature, None, fx=x, fy=y)
                            variant.setflags(write=False)
                            expanded.append(variant)
                self.variants[action] = expanded

    def recognize(self, image):
        if image is None or image.shape != (1080, 498, 3):
            raise ValueError("wrong canvas")
        result = {}
        for row in self.profile["slots"]:
            scores = {}
            for action, templates in self.variants.items():
                zone = "avatar" if action in ("fold", "all_in") else "badge"
                patch = crop(image, row[zone])
                mask = colour_mask(patch, action)
                if action == "all_in":
                    mask = action_feature(patch, action)
                else:
                    mask = cv2.GaussianBlur(mask, (3, 3), .7)
                scores[action] = 0.0
                for template in templates:
                    if any(a > b for a, b in zip(template.shape, mask.shape)):
                        continue
                    match = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
                    score = min(1.0, max(0.0, float(match.max())))
                    scores[action] = max(scores[action], score)
            result[str(row["slot"])] = choose_action(
                scores, self.profile["action_floor"], self.profile["action_margin"])
        return result


def white_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def normalized_digit(mask):
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("empty glyph")
    glyph = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = glyph.shape
    scale = min(24 / width, 24 / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    canvas = np.zeros((28, 28), np.uint8)
    x, y = (28 - w) // 2, (28 - h) // 2
    canvas[y:y + h, x:x + w] = cv2.resize(glyph, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(canvas, (3, 3), .7)


def template_mask(image):
    """Remove the committed template's uniform letterbox before thresholding.

    Existing 28x28 PNGs can have a lighter constant frame than their original
    glyph background. Treating that frame as ink corrupts Otsu/shape matching.
    This only trims a spatially uniform exterior, not arbitrary glyph strokes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ys, xs = np.where(gray != gray[0, 0])
    if not len(xs):
        raise ValueError("constant template")
    inner = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return white_mask(inner)


class DigitCandidate:
    def __init__(self, templates, floor=.85, margin=.08):
        if any(isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 1
               for v in (floor, margin)):
            raise ValueError("invalid digit gate")
        if set(templates) != set("0123456789"):
            raise ValueError("all ten digits required")
        self.templates = {key: normalized_digit(template_mask(value))
                          for key, value in templates.items()}
        for feature in self.templates.values():
            feature.setflags(write=False)
        self.floor, self.margin = floor, margin

    def recognize(self, patch):
        unknown = {"value": None, "accepted_candidate": False,
                   "production_valid": False}
        if patch is None or patch.size == 0:
            return {**unknown, "reason": "empty_crop"}
        binary = white_mask(patch)
        if np.any(binary[0]) or np.any(binary[-1]) or np.any(binary[:, 0]) or np.any(
            binary[:, -1]
        ):
            return {**unknown, "reason": "edge_ink_or_clipped_glyph"}
        spans = _column_spans(binary)
        if not 1 <= len(spans) <= 5:
            return {**unknown, "reason": "unsupported_glyph_count"}
        digits = []
        for start, end in spans:
            glyph = binary[:, start:end]
            ys, xs = np.where(glyph > 0)
            if ys.max() - ys.min() + 1 < 7:
                return {**unknown, "reason": "punctuation_or_speck"}
            feature = normalized_digit(glyph)
            scores = {key: min(1.0, max(0.0, float(cv2.matchTemplate(
                feature, template, cv2.TM_CCOEFF_NORMED).max())))
                for key, template in self.templates.items()}
            digits.append(choose_action(scores, self.floor, self.margin))
        accepted = all(row["accepted_candidate"] for row in digits)
        text = "".join(row["best_label"] for row in digits)
        if len(text) > 1 and text.startswith("0"):
            accepted = False
        return {"value": text if accepted else None, "accepted_candidate": accepted,
                "raw_text": text, "digits": digits, "production_valid": False,
                "reason": "candidate" if accepted else "ambiguous_digit"}
