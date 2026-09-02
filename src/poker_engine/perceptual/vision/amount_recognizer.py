"""Amount recognizer via OpenCV multi-character template matching (no float).

MVP: preprocess -> connected-component segmentation -> left-to-right sort ->
per-character template match -> assemble numeric string -> validate ->
ChipAmount.

PaddleOCR can later implement the same AmountRecognizer protocol without
changing VisionEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import InvalidOperation
from typing import Mapping

import cv2
import numpy as np

from poker_engine.core.value_objects import ChipAmount

from .protocols import AmountRecognition, freeze_templates


@dataclass(frozen=True)
class DigitTemplateSet:
    """Character -> template image (digits 0-9 plus '.' and ',', optional)."""

    templates: Mapping[str, np.ndarray]
    version: str

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("templates must be non-empty")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a non-empty str")
        object.__setattr__(self, "templates", freeze_templates(self.templates))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _template_gray(tmpl: np.ndarray) -> np.ndarray:
    if tmpl.ndim == 3:
        return cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    return tmpl


def _minority_foreground(gray: np.ndarray) -> np.ndarray:
    """Return a binary glyph mask for either light-on-dark or dark-on-light."""
    _, light = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    dark = cv2.bitwise_not(light)
    return light if int((light > 0).sum()) < int((dark > 0).sum()) else dark


def _glyph_hole_count(img: np.ndarray) -> int:
    """Count stable enclosed regions in one segmented character glyph.

    Correlation alone can score Android's anti-aliased ``8`` closer to ``3``.
    Their topology is unambiguous, so templates with a different number of
    enclosed regions are excluded whenever an equal-topology candidate exists.
    Tiny compression specks are ignored.
    """
    foreground = _minority_foreground(_to_gray(img))
    contours, hierarchy = cv2.findContours(
        foreground, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return sum(
        1
        for index, contour in enumerate(contours)
        if hierarchy[0][index][3] >= 0 and cv2.contourArea(contour) >= 2.0
    )


_NORM = (28, 28)  # fixed normalized grid for single-char matching


def _normalize_char(img: np.ndarray) -> np.ndarray:
    """Resize a character image to the fixed grid, preserving aspect ratio.

    Letterboxed so a narrow '1' and a wide '8' keep their shape proportions.
    """
    h, w = img.shape[:2]
    scale = min(_NORM[0] / h, _NORM[1] / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full(_NORM, 255, dtype=resized.dtype)
    y0 = (_NORM[0] - nh) // 2
    x0 = (_NORM[1] - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _match_char(char_img: np.ndarray, templates: Mapping[str, np.ndarray]):
    """Match a single character crop, returning (label, score in [0,1]).

    Both the crop and every template are normalized to a fixed grid (aspect
    ratio preserved), so digit shape is compared independently of source size.
    """
    gray = _normalize_char(_to_gray(char_img))
    char_holes = _glyph_hole_count(char_img)
    matching_topology = {
        label for label, tmpl in templates.items()
        if _glyph_hole_count(tmpl) == char_holes
    }
    candidates = (
        matching_topology if matching_topology else set(templates)
    )
    best = None
    best_score = -1.0
    for label, tmpl in templates.items():
        if label not in candidates:
            continue
        tg = _normalize_char(_template_gray(tmpl))
        res = cv2.matchTemplate(gray, tg, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, _ = cv2.minMaxLoc(res)
        score = float(max(0.0, maxval))
        if score > best_score:
            best_score = score
            best = label
    return best, best_score


def segment_characters(roi_image: np.ndarray) -> list[np.ndarray]:
    """Return character crops sorted left-to-right (deterministic).

    Uses vertical projection (column ink mass) to find character spans, then
    trims each span to its ink bounding box (both axes) so normalized matching
    compares comparable shapes.
    """
    gray = _to_gray(roi_image)
    # Support both black-on-light desktop text and white-on-dark Android UI.
    # In a tight amount ROI the glyphs are the minority class; selecting the
    # smaller foreground also avoids making the whole dark banner one glyph.
    binary = _minority_foreground(gray)
    col_ink = (binary > 0).sum(axis=0)  # ink mass per column

    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for x, ink in enumerate(col_ink):
        has_ink = int(ink) > 0
        if has_ink and not in_span:
            in_span = True
            start = x
        elif not has_ink and in_span:
            in_span = False
            spans.append((start, x))
    if in_span:
        spans.append((start, len(col_ink)))

    crops: list[np.ndarray] = []
    for x0, x1 in spans:
        if x1 - x0 <= 0:
            continue
        seg = binary[:, x0:x1]
        ys, xs = np.where(seg > 0)
        if len(xs) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        crops.append(gray[y0:y1, x0:x1])
    return crops


class TemplateAmountRecognizer:
    """Recognize an amount region as a ChipAmount (or None when unreadable)."""

    def __init__(self, templates: DigitTemplateSet, min_score: float = 0.5) -> None:
        self._templates = templates
        self._min_score = min_score

    def recognize(self, roi_image: np.ndarray) -> AmountRecognition:
        if roi_image is None or roi_image.size == 0:
            return AmountRecognition(value=None, raw_score=0.0)

        text, score = self._decode(roi_image)
        if text is None or score < self._min_score:
            return AmountRecognition(value=None, raw_score=score)

        try:
            amount = ChipAmount(text)
        except (InvalidOperation, ValueError):
            return AmountRecognition(value=None, raw_score=score)
        return AmountRecognition(value=amount, raw_score=score)

    def _decode(self, roi_image: np.ndarray):
        # Single-character fast path: match the whole ROI first. If it matches
        # a single template strongly, use it (avoids fragile small-crop splits).
        whole_label, whole_score = _match_char(
            roi_image, self._templates.templates
        )
        if whole_label is not None and whole_score >= 0.8:
            return whole_label, whole_score

        chars = segment_characters(roi_image)
        if not chars:
            return (
                (whole_label, whole_score)
                if whole_label is not None
                else (None, whole_score)
            )

        out: list[str] = []
        scores: list[float] = []
        for crop in chars:
            label, score = _match_char(crop, self._templates.templates)
            if label is None:
                return None, 0.0
            out.append(label)
            scores.append(score)

        text = "".join(out)
        # aggregate raw score = min per-char score (weakest char dominates)
        raw = float(min(scores)) if scores else 0.0
        return text, raw


def build_identity_templates() -> DigitTemplateSet:
    """Build a trivial digit template set (digits + '.') for tests."""
    templates: dict[str, np.ndarray] = {}
    for ch in "0123456789.":
        img = np.full((24, 40, 3), 255, dtype=np.uint8)
        cv2.putText(img, ch, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        templates[ch] = img
    return DigitTemplateSet(templates=templates, version="v1-identity")


__all__ = [
    "DigitTemplateSet",
    "TemplateAmountRecognizer",
    "build_identity_templates",
    "segment_characters",
]
