"""Action recognizer (per-seat, seat-rendered action label/state).

Consumes a per-seat ACTION ROI image (with slot_id) and recognizes the
seat-rendered action label: FOLD/CHECK/CALL/BET/RAISE/ALL_IN/UNKNOWN.

This is OBSERVED player action, NOT Hero clickable buttons. If a platform does
not expose per-seat action visual signal, the caller must raise a Platform
Visual Gap — this module never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from poker_engine.core.enums import ActionType

from .protocols import ActionRecognition, freeze_templates


@dataclass(frozen=True)
class ActionTemplateSet:
    templates: Mapping[ActionType, np.ndarray]
    version: str

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("templates must be non-empty")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        object.__setattr__(self, "templates", freeze_templates(self.templates))


class TemplateActionRecognizer:
    """Recognize a seat action label via template matching."""

    def __init__(self, templates: ActionTemplateSet, min_score: float = 0.5) -> None:
        self._templates = templates
        self._min_score = min_score

    def recognize(self, roi_image: np.ndarray, slot_id: int) -> ActionRecognition:
        if roi_image is None or roi_image.size == 0:
            return ActionRecognition(value=None, raw_score=0.0)

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        scores = []
        for action, tmpl in self._templates.templates.items():
            if tmpl.ndim == 3:
                tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            else:
                tmpl_gray = tmpl
            if tmpl_gray.shape[0] > gray.shape[0] or tmpl_gray.shape[1] > gray.shape[1]:
                tmpl_gray = cv2.resize(tmpl_gray, (gray.shape[1], gray.shape[0]))
            res = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
            _, maxval, _, _ = cv2.minMaxLoc(res)
            score = max(0.0, maxval)
            scores.append((float(score), action))

        scores.sort(reverse=True, key=lambda item: item[0])
        best_score, best = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0

        # A low best score means "no action rendered" -> abstain from guessing.
        if best_score < self._min_score:
            return ActionRecognition(
                value=None, raw_score=float(best_score),
                runner_up_score=float(runner_up),
            )
        return ActionRecognition(
            value=best, raw_score=float(best_score),
            runner_up_score=float(runner_up),
        )


class TemplateActionGlyphRecognizer:
    """Match stable action text glyphs while ignoring avatar/background art.

    WePoker renders completed actions either as white text on a colored pill
    or, for all-in, as yellow text over the avatar. Templates are committed as
    small binary glyph masks, so no player avatar or private screenshot is
    retained and changing avatars cannot change the match.
    """

    def __init__(self, templates: ActionTemplateSet, min_score: float = 0.5) -> None:
        self._templates = templates
        self._min_score = min_score

    @staticmethod
    def _mask(image: np.ndarray, action: ActionType) -> np.ndarray:
        if image.ndim == 2:
            return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
        # Committed templates are already binary masks but OpenCV loads PNGs
        # as three identical channels by default. Preserve that mask instead
        # of trying to interpret it as colored table pixels.
        if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(
            image[:, :, 1], image[:, :, 2]
        ):
            gray = image[:, :, 0]
            return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if action is ActionType.ALL_IN:
            return cv2.inRange(hsv, (15, 90, 130), (45, 255, 255))
        return cv2.inRange(hsv, (0, 0, 145), (180, 100, 255))

    def recognize(self, roi_image: np.ndarray, slot_id: int) -> ActionRecognition:
        if roi_image is None or roi_image.size == 0:
            return ActionRecognition(value=None, raw_score=0.0)
        scores = []
        for action, template in self._templates.templates.items():
            mask = self._mask(roi_image, action)
            template_mask = self._mask(template, action)
            if (
                template_mask.shape[0] > mask.shape[0]
                or template_mask.shape[1] > mask.shape[1]
            ):
                scores.append((0.0, action))
                continue
            result = cv2.matchTemplate(
                mask, template_mask, cv2.TM_CCOEFF_NORMED
            )
            _, maximum, _, _ = cv2.minMaxLoc(result)
            scores.append((float(max(0.0, maximum)), action))
        scores.sort(reverse=True, key=lambda item: item[0])
        best_score, best = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        value = best if best_score >= self._min_score else None
        return ActionRecognition(value, best_score, runner_up)


__all__ = [
    "ActionTemplateSet", "TemplateActionGlyphRecognizer",
    "TemplateActionRecognizer",
]
