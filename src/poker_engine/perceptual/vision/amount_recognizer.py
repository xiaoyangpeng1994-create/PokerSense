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
    The canvas takes the crop's own background (corner-median) value: on a
    fixed white canvas a white-on-dark glyph would wash into the background,
    destroying both correlation signal and hole topology (2026-09-06).
    """
    h, w = img.shape[:2]
    corners = np.array(
        [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    ).reshape(-1)
    background = int(np.median(corners))
    scale = min(_NORM[0] / h, _NORM[1] / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full(_NORM, background, dtype=resized.dtype)
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
    # Count holes on the same normalized grid the templates live on, not on
    # the raw segment: at raw size (e.g. a 15x10 thin-font ribbon "8") Otsu
    # binarization collapses enclosed counters, and a 0-hole reading would
    # exclude the true 2-hole template from the candidate set below — the
    # 2026-09-06 "184 -> 134" false VALID. The 28x28 letterbox upscales the
    # glyph enough to keep counters closed, and both sides of the topology
    # gate then see the same grid (capsule-font digits count identically
    # either way; verified on the capture-card dataset).
    char_holes = _glyph_hole_count(gray)
    matching_topology = {
        label for label, tmpl in templates.items()
        if _glyph_hole_count(_normalize_char(_template_gray(tmpl)))
        == char_holes
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


def _column_spans(binary: np.ndarray) -> list[tuple[int, int]]:
    """Find [x0, x1) ink spans via vertical projection (column ink mass)."""
    col_ink = (binary > 0).sum(axis=0)
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
    return spans


def _single_degenerate_blob(
    spans: list[tuple[int, int]], width: int
) -> bool:
    """True when segmentation collapsed into one ROI-filling blob."""
    return (
        not spans
        or (len(spans) == 1 and spans[0][1] - spans[0][0] >= 0.9 * width)
    )


def _remove_background_strips(binary: np.ndarray) -> np.ndarray:
    """Zero full-bleed edge strips so they cannot bridge glyph spans.

    A background sliver inside a loose ROI (bright seat edge above a dark
    pill, pill border at the ROI boundary) shows up as rows/columns that are
    >= 80% ink and touch the image edge. Glyphs never span 80% of the ROI
    width, so stripping only edge-connected full-bleed bands is safe.
    """
    out = binary.copy()
    height, width = out.shape
    for y in range(height):
        if (out[y] > 0).mean() >= 0.8:
            out[y] = 0
        else:
            break
    for y in range(height - 1, -1, -1):
        if (out[y] > 0).mean() >= 0.8:
            out[y] = 0
        else:
            break
    for x in range(width):
        if (out[:, x] > 0).mean() >= 0.8:
            out[:, x] = 0
        else:
            break
    for x in range(width - 1, -1, -1):
        if (out[:, x] > 0).mean() >= 0.8:
            out[:, x] = 0
        else:
            break
    return out


def segment_characters(roi_image: np.ndarray) -> list[np.ndarray]:
    """Return character crops sorted left-to-right (deterministic).

    Uses vertical projection (column ink mass) to find character spans, then
    trims each span to its ink bounding box (both axes) so normalized matching
    compares comparable shapes.

    Robustness rules (capture-card pills, 2026-09-06):

    - **Background-strip removal**: full-bleed edge bands (bright seat sliver
      above a dark pill, pill border) are zeroed so they cannot bridge all
      glyph columns into one blob.
    - **Polarity fallback**: when the minority foreground still degenerates
      into a single ROI-filling blob, retry with the inverted binary and keep
      it when it segments cleanly.
    - **Cluster isolation**: spans separated from the main cluster by a gap
      far wider than the tightest inter-character gap are detached junk
      (dealer badge, timer specks); keep the largest cluster.
    - **Speck filter**: 1-2 px compression specks are dropped.
    - **Oversize filter**: a span much taller *and* wider than the median
      glyph (a circular badge next to the pill) is dropped.
    Filters never empty the span list — when everything would be dropped the
    unfiltered spans are returned, so a borderline read becomes UNKNOWN
    downstream instead of silently inventing digits.
    """
    gray = _to_gray(roi_image)
    # Support both black-on-light desktop text and white-on-dark Android UI.
    # In a tight amount ROI the glyphs are the minority class; selecting the
    # smaller foreground also avoids making the whole dark banner one glyph.
    binary = _remove_background_strips(_minority_foreground(gray))
    width = gray.shape[1]
    spans = _column_spans(binary)
    if _single_degenerate_blob(spans, width):
        inverted = _remove_background_strips(cv2.bitwise_not(binary))
        alt = _column_spans(inverted)
        if alt and not _single_degenerate_blob(alt, width):
            binary, spans = inverted, alt

    items: list[tuple[int, int, int, int, np.ndarray]] = []
    for x0, x1 in spans:
        if x1 - x0 <= 0:
            continue
        seg = binary[:, x0:x1]
        ys, xs = np.where(seg > 0)
        if len(xs) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        crop = gray[y0:y1, x0:x1]
        items.append((x0, x1, x1 - x0, y1 - y0, crop))

    if len(items) > 1:
        # Detached-cluster isolation: a gap far wider than the tightest
        # inter-character gap splits off detached junk (badges, specks).
        gaps = [items[i + 1][0] - items[i][1] for i in range(len(items) - 1)]
        tightest = min(gaps)
        split_at = max(2.5 * tightest, 10)
        clusters: list[list[int]] = [[0]]
        for i, gap in enumerate(gaps):
            if gap >= split_at:
                clusters.append([])
            clusters[-1].append(i + 1)
        if len(clusters) > 1:
            biggest = max(clusters, key=len)
            if len(biggest) < len(items):
                items = [items[i] for i in biggest]

    if len(items) > 1:
        non_speck = [c for c in items if not (c[2] <= 2 and c[3] <= 3)]
        if non_speck:
            items = non_speck

    if len(items) > 1:
        heights = sorted(c[3] for c in items)
        widths = sorted(c[2] for c in items)
        med_h = heights[len(heights) // 2]
        med_w = widths[len(widths) // 2]
        non_oversized = [
            c
            for c in items
            if not (c[3] > 1.2 * med_h and c[2] > 1.5 * med_w)
        ]
        if non_oversized:
            items = non_oversized

    return [crop for _, _, _, _, crop in items]


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
        # Single-character fast path: match the whole ROI first, but only
        # when the ROI has a plausible single-glyph aspect ratio. A wide
        # multi-digit pill can spuriously correlate with one template above
        # 0.8 (observed on capture-card highlighted pills, 2026-09-06), so
        # wide ROIs must always go through segmentation. All production
        # amount ROIs are >= 2.27 aspect; a genuine single-glyph ROI is < 2.
        height, width = roi_image.shape[:2]
        whole_label = None
        whole_score = 0.0
        if width <= 2.0 * height:
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
