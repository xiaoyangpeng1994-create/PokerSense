# -*- coding: utf-8 -*-
"""Read per-seat ground truth (occupancy / stack / dealer) from a single
normalized table frame for the capture-card platform.

This module implements the *pixel reading* half of stage F section 9 for
the seat fields. It takes a BGR frame (already normalized to the canvas)
and reads, for each of the ``SLOT_COUNT`` visual slots:

- ``occupancy`` -> ``OCCUPIED`` if an avatar picture or a readable stack
  pill is present, ``EMPTY`` only on the positive "+" empty-button signal,
  and ``UNKNOWN`` whenever neither side has positive evidence (a dimmed
  avatar or an ambiguous band is never guessed);
- ``stack``     -> the integer chip count in the dark-green stack pill;
- ``dealer``    -> whether a circular white "D" badge sits under the slot.

Design principles (mirroring the guide's failure-closed philosophy):

- Every read returns a confident ``VALID`` value or ``UNKNOWN``. A slot
  with no readable pixels is never guessed. In particular an EMPTY claim
  requires the positive "+" cross evidence: ``map_snapshot_candidate``
  mutates seat state on an EMPTY read, so a false EMPTY is a wrong state
  transition, not a safe abstain.
- Detection is threshold-driven on luminance: the white-on-green stack
  digits, the "+" empty button and the "D" badge are all bright against
  the green felt captured via the UVC card. AVATAR occupancy is decided by
  chroma saturation, which distinguishes a picture from the uniform felt.
- It does NOT reuse any coordinate, ROI or threshold from the LDPlayer or
  H5 platforms (guide rules 1 and 2). All geometry is expressed in the
  0..1 normalized canvas space and must be calibrated on this platform.

Import-safe without OpenCV: only the reader functions import cv2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import SLOT_COUNT
from .schema import FieldValue, LabelStatus, Occupancy

__all__ = [
    "HERO_SLOT",
    "SLOT_LAYOUT_MULTI",
    "SLOT_LAYOUT_HEADS",
    "SLOT_LAYOUT_S002",
    "DigitMatch",
    "build_digit_templates",
    "classify_digit",
    "read_seat_fields",
    "read_slot",
    "split_stack_digits",
]


# Slot id -> physical seat. Slot 0 is the hero seat (bottom-centre). This
# mirrors the slot_id convention already used for this platform in
# ``labels/roi_measurements.csv``: 0=hero bottom, 1=LL, 2=LM, 3=UL,
# 4=Top, 5=UR, 6=RM, 7=LR.
#
# (cx, cy, w, h) are normalized canvas coordinates of the **stack pill**
# (the dark-green rounded pill carrying the white chip count). They were
# measured on the full 8-handed frame
# ``session_001__t_00010500__f_000315``. The avatar sits directly above
# the pill; the "+" empty button sits where the pill would be for an empty
# slot; the "D" dealer badge sits under the name row.
SLOT_LAYOUT_MULTI: dict[int, dict[str, float]] = {
    0: dict(cx=0.50, cy=0.865, w=0.20, h=0.032),   # hero bottom
    1: dict(cx=0.12, cy=0.640, w=0.18, h=0.030),   # LL
    2: dict(cx=0.10, cy=0.462, w=0.18, h=0.030),   # LM
    3: dict(cx=0.10, cy=0.301, w=0.18, h=0.030),   # UL
    4: dict(cx=0.50, cy=0.211, w=0.20, h=0.032),   # Top
    5: dict(cx=0.90, cy=0.301, w=0.18, h=0.030),   # UR
    6: dict(cx=0.90, cy=0.462, w=0.18, h=0.030),   # RM
    7: dict(cx=0.88, cy=0.640, w=0.18, h=0.030),   # LR
}

# Head-up / two-handed layout. Only the hero (0) and the opponent seat
# opposite (top, 4) are normally occupied on a 2-handed capture; the ring
# is kept complete so the geometry stays coherent.
SLOT_LAYOUT_HEADS: dict[int, dict[str, float]] = dict(SLOT_LAYOUT_MULTI)

# session_002 layout. This capture uses the *same* 8-max ring as session_001
# (seats 1-7 and the side columns align with SLOT_LAYOUT_MULTI), but the hero
# stack pill sits lower because the hero hand is revealed nudge the name/type
# row down. Measured on ``session_002__t_00042700``: hero pill at
# (0.500, 0.949); side seats at (0.096/0.095, 0.466/0.640) and
# (0.905/0.907, 0.466/0.640) — identical to SLOT_LAYOUT_MULTI. Only slot 0
# differs.
SLOT_LAYOUT_S002: dict[int, dict[str, float]] = dict(SLOT_LAYOUT_MULTI)
SLOT_LAYOUT_S002[0] = dict(cx=0.50, cy=0.978, w=0.20, h=0.026)

# Dealer badge: a compact white disc with a black "D" core. It sits just
# off the stack pill (to the side for edge seats, below for top/bottom).
# Detection is: a near-round low-saturation bright blob whose centre holds
# dark pixels (the letter D). A cyan "chip" ring has the same low-sat
# brightness but always carries the same dark fraction, so the discriminator
# is the *low-saturation white* area (a pure-white disc is far larger than a
# cyan ring).
_DEALER_BLOB_W = (14, 24)
_DEALER_BLOB_H = (14, 24)
_DEALER_BLOB_ASQUARE = 5     # |w-h| tolerance (near round)
_DEALER_MIN_AREA = 120
_DEALER_WHITE_LOWSAT_MIN = 0.20   # pure-white disc area ratio
_DEALER_BLACK_MIN = 0.05          # black "D" core area ratio
_DEALER_SCAN_MULT = 1.1           # search window (pill height multiplier)
_DEALER_VAL_MIN = 180
_DEALER_SAT_MAX = 60
_DEALER_BLACK_MAX = 70

# Digit segmentation parameters (measured on the reference frame). Stack
# digits are a consistent height (~14px at the 498-x-1080 canvas); a "D"
# dealer badge that bleeds into the ROI is taller (~17px) and is dropped
# by the height window. Aspect ratio is intentionally NOT constrained:
# a narrow "1" is ~2.8 h/w, the same as a "D", so only height separates
# them.
_DIGIT_MIN_AREA = 25
_DIGIT_H = (11, 16)

#: Maximum width of a connected white component that can still be a single
#: stack digit. On this canvas a real digit is 9-11px ("1" as narrow as 5px,
#: a wide "2"/"9" up to ~12px); nothing legitimate is wider than ~13px. The
#: UI *merges two adjacent digits* when it draws a multi-digit value with them
#: touching (e.g. a "194" whose "9" and "4" touch, or "198" with "9"+"8"
#: touching), and ``cv2.dilate(2,2)`` bridges that run into one component.
#: Measured on the reference frames, that bridged blob is 21-22px — so a
#: component wider than ``_DIGIT_MAX_W`` (=20) is a merge, not a digit, and
#: the 20px threshold was validated on the private corpus (every real merge
#: lands at 21px or more).
#:
#: A blob at least this wide is a *reliability signal*, not a discard rule
#: for the child pixels: the failure-closed path must never trust the
#: remaining partial glyphs as a complete digit sequence (a merged "198" must
#: never be read as a confident "1"). Callers that can tolerate an UNKNOWN
#: read (``stack_auto``) use ``split_stack_digits`` and fail closed when a
#: merge is detected; ``_split_digits`` keeps the historical discard behaviour
#: so existing callers/tests are unchanged.
_DIGIT_MAX_W = 20

# White luminance floor: digits and badges are bright (>150), the green
# felt sits well below 120.
_WHITE_MIN = 150

# Avatar occupancy: a picture is saturated (>80 chroma over 12% of pixels);
# the "+" button and felt are not.
_AVATAR_SAT = 80

#: Luminance spread floor for the avatar picture. A real avatar is a dense,
#: high-contrast picture (std well above 18); a bare dark empty-slot disc is
#: nearly uniform. Below the floor we make NO occupancy claim (UNKNOWN):
#: before this rule a dimmed/"away" avatar was misread as EMPTY, which is a
#: wrong positive claim, not a fail-closed abstain.
_AVATAR_STD_MIN = 18.0

#: Hero visual slot. The hero seat has NO avatar disc: the band above its
#: pill holds the hero cards and the white hand-type label (e.g. "对子"),
#: whose text false-fires the "+" cross detector (measured on the private
#: corpus: 54 of 62 occupancy misses were hero slots read EMPTY while
#: occupied). Hero occupancy therefore never uses the avatar/cross path.
HERO_SLOT = 0
_AVATAR_FRAC = 0.12


def _imread(path: Path | str) -> np.ndarray:
    import cv2

    return cv2.imdecode(
        np.frombuffer(Path(path).read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def _roi(img: np.ndarray, cx: float, cy: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]

    def f(v: float) -> int:
        return int(round(v))

    try:
        return img[
            f((cy - h / 2) * H) : f((cy + h / 2) * H),
            f((cx - w / 2) * W) : f((cx + w / 2) * W),
        ]
    except IndexError:
        return np.zeros((0, 0, 3), dtype=img.dtype)


def _white_mask(crop: np.ndarray, *, floor: int = _WHITE_MIN) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, floor, 255, cv2.THRESH_BINARY)
    return th


def _digits_from_crop(crop: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Split a stack-pill crop into x-sorted digit masks.

    Returns ``(glyphs, had_merge)`` where ``had_merge`` is True when a
    connected white component exceeded a single digit's width. ``glyphs`` is
    every digit-height component that fits a single digit (``w <= _DIGIT_MAX_W``),
    in x order; the over-wide merged blob is **not** returned (its pixels
    cannot be trusted as one digit — e.g. a merged "194" must not surface as a
    confident "1").
    """
    import cv2

    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 3:
        return [], False
    th = _white_mask(crop)
    th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(th, 8)
    chars: list[tuple[int, int, np.ndarray]] = []
    had_merge = False
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < _DIGIT_MIN_AREA or not (_DIGIT_H[0] <= h <= _DIGIT_H[1]):
            continue
        if w > _DIGIT_MAX_W:
            # A >20px white blob is two or more digits that the UI drew
            # touching (dilated into one component). Mark it; do not return
            # its pixels as a single glyph.
            had_merge = True
            continue
        mask = (lab[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        chars.append((x, w, mask))
    chars.sort(key=lambda c: c[0])
    return [c[-1] for c in chars], had_merge


def split_stack_digits(crop: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Split a stack-pill crop, reporting whether a digit merge was detected.

    This is the failure-closed entry point for the auto-reader. It behaves
    exactly like ``_split_digits`` but additionally returns ``had_merge``: a
    call (e.g. ``stack_auto._read_stack_pill``) must treat any read from a
    crop with ``had_merge is True`` as unreliable — a merged multi-digit value
    would otherwise be truncated into a confident-looking single digit (the
    "198 -> 1" failure), which violates the never-guess rule.
    """
    return _digits_from_crop(crop)


def _split_digits(crop: np.ndarray) -> list[np.ndarray]:
    """Return x-sorted digit masks (uint8 0/255) from a stack-pill crop.

    A tiny dilation merges a digit's own broken strokes (e.g. the crossbar
    of a "7"); connected components are then filtered to digit-like size
    and consistent height (a "D" dealer badge is taller and is dropped).
    """
    glyphs, _ = _digits_from_crop(crop)
    return glyphs


def _norm(mask: np.ndarray, size: tuple[int, int] = (20, 28)) -> np.ndarray:
    import cv2

    m = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.float32)


@dataclass(frozen=True)
class DigitMatch:
    """One normalized digit classified against a template library.

    ``best`` / ``best_dist`` are the winning digit (the template sample the
    text glyph is closest to) and its mean-square error. ``runner_up`` /
    ``runner_up_dist`` are the second-best *different* digit and its MSE. The
    confidence margin (``margin = runner_up_dist - best_dist``) measures how
    decisively the winner beats the runner-up: a large margin means a clean
    read, a small one means the two digits are confusable (e.g. 6/8, 9/7,
    1/7 — the capture-card digits are all bright white on dark green and
    9x-1x 3x look-alikes occur). ``recognized`` is True only when the winner
    is a real digit; the runner-up is undefined (``None``) when there is no
    competing digit.
    """

    best: str
    best_dist: float
    runner_up: str | None
    runner_up_dist: float | None

    @property
    def margin(self) -> float:
        """Positive gap between runner-up and best MSE; ``inf`` if no runner-up."""
        if self.runner_up_dist is None:
            return float("inf")
        return self.runner_up_dist - self.best_dist


def classify_digit(
    v: np.ndarray, templates: dict[str, list[np.ndarray]]
) -> DigitMatch:
    """Classify one normalized glyph, returning the full winner/runner-up pair.

    The MSE distance is *unnormalized* but comparable across candidates for a
    single query; only the *ordering* and the *gap* matter for the gate, never
    the raw value (which varies with glyph brightness/size).
    """
    best_label, best_dist = None, float("inf")
    runner_label, runner_dist = None, float("inf")
    for d, samples in templates.items():
        for s in samples:
            dist = float(np.mean((v - s) ** 2))
            if dist < best_dist:
                if best_label is not None and best_label != d:
                    # The current best demotes to runner-up.
                    runner_label, runner_dist = best_label, best_dist
                best_label, best_dist = d, dist
            elif dist < runner_dist and d != best_label:
                runner_label, runner_dist = d, dist
    return DigitMatch(
        best=best_label if best_label is not None else "?",
        best_dist=best_dist,
        runner_up=runner_label,
        runner_up_dist=runner_dist,
    )


def _classify(
    v: np.ndarray, templates: dict[str, list[np.ndarray]]
) -> tuple[str, float]:
    """Backward-compatible wrapper returning the legacy ``(best, best_dist)``."""
    match = classify_digit(v, templates)
    return (match.best, match.best_dist)


def _find_dealer(crop: np.ndarray) -> bool:
    """True if a near-round white disc with a black "D" core is present.

    The D badge is a bright, low-saturation white disc; the black letter D
    sits at its centre. A cyan "chip" ring (a common look-alike) has the
    same dark fraction but far less low-saturation white, so requiring a
    minimum pure-white area cleanly separates the two.
    """
    import cv2

    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 3:
        return False
    th = _white_mask(crop)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(th, 8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (_DEALER_MIN_AREA <= area):
            continue
        if not (
            _DEALER_BLOB_W[0] <= w <= _DEALER_BLOB_W[1]
            and _DEALER_BLOB_H[0] <= h <= _DEALER_BLOB_H[1]
        ):
            continue
        if abs(w - h) > _DEALER_BLOB_ASQUARE:
            continue
        sub_val = val[y : y + h, x : x + w]
        sub_sat = sat[y : y + h, x : x + w]
        sub_gray = gray[y : y + h, x : x + w]
        # Pure-white disc: bright + low saturation.
        white = ((sub_val > _DEALER_VAL_MIN) & (sub_sat < _DEALER_SAT_MAX)).mean()
        # Black letter D core.
        black = (sub_gray < _DEALER_BLACK_MAX).mean()
        if white >= _DEALER_WHITE_LOWSAT_MIN and black >= _DEALER_BLACK_MIN:
            return True
    return False


def _has_white_cross(band: np.ndarray, *, floor: int = 150) -> bool:
    """True if the band holds the empty-slot "+" button.

    The "+" is a white cross (one horizontal stroke and one vertical
    stroke meeting at the centre) on a dark disc. As a connected
    component it is a single near-square blob whose white pixels hug the
    horizontal and vertical centre-lines (the four corners stay dark), so
    its *fill ratio* (white / bounding-box area) is low (~0.2) and its
    centre is bright while the corners are dark. An avatar is a dense
    picture and never produces such a sparse, cross-shaped blob.
    """
    import cv2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, floor, 255, cv2.THRESH_BINARY)
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(th, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (12 <= w <= 34 and 12 <= h <= 34 and 0.7 <= w / h <= 1.4):
            continue
        box = w * h
        fill = area / box
        if not (0.05 <= fill <= 0.42):
            continue
        sub = (lab[y : y + h, x : x + w] == i).astype(np.uint8)
        # Cross: center row/column are lit, the four corners are dark.
        mid_y, mid_x = h // 2, w // 2
        centre = sub[mid_y, mid_x]
        corners = (sub[0, 0], sub[0, -1], sub[-1, 0], sub[-1, -1])
        # At least the centre pixel must be white and corners mostly dark.
        if centre == 1 and sum(corners) <= 2:
            return True
    return False


def _avatar_picture_std(band: np.ndarray) -> float:
    """Luminance spread of an avatar band (0.0 for an empty band)."""
    import cv2

    if band is None or band.size == 0 or min(band.shape[:2]) < 3:
        return 0.0
    return float(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).std())


def build_digit_templates(
    reference: np.ndarray,
    layout: dict[int, dict[str, float]],
    slot_counts: dict[int, str],
) -> dict[str, list[np.ndarray]]:
    """Build a 0-9 digit library from a known reference frame.

    ``slot_counts`` maps slot_id -> the ground-truth chip count visible in
    ``reference`` (only used to supervise extraction). The reference ring
    must cover all ten digits for a complete library.
    """
    templates: dict[str, list[np.ndarray]] = {}
    for slot_id, row in layout.items():
        exp = slot_counts.get(slot_id)
        if not exp:
            continue
        masks = _split_digits(_roi(reference, row["cx"], row["cy"], row["w"], row["h"]))
        if len(masks) != len(exp):
            continue
        for mask, d in zip(masks, exp):
            templates.setdefault(d, []).append(_norm(mask))
    return templates


def read_slot(
    slot_id: int,
    img: np.ndarray,
    layout: dict[int, dict[str, float]],
    templates: dict[str, list[np.ndarray]],
) -> dict[str, FieldValue]:
    """Read occupancy/stack/dealer for one slot (``VALID`` or ``UNKNOWN``)."""
    row = layout[slot_id]
    H, W = img.shape[:2]
    pill = _roi(img, row["cx"], row["cy"], row["w"], row["h"])

    # Occupancy: an avatar sits above the pill. Approximate band = 2.2x pill.
    ph = pill.shape[0]
    avatar_top = max(0, int((row["cy"] - row["h"] / 2) * H) - int(ph * 2.2))
    pill_top = int((row["cy"] - row["h"] / 2) * H)
    pill_left = int((row["cx"] - row["w"] / 2) * W)
    pill_right = int((row["cx"] + row["w"] / 2) * W)
    avatar = img[avatar_top:pill_top, max(0, pill_left - 4):min(W, pill_right + 4)]

    # Stack first: a readable chip count is the strongest occupancy signal
    # (only a seated player has a stack number). Empty slots have none.
    stack = FieldValue.unknown()
    if templates:
        masks = _split_digits(pill)
        if masks:
            digits = "".join(_classify(_norm(m), templates)[0] for m in masks)
            if digits.isdigit():
                stack = FieldValue.valid(int(digits))

    # Occupancy, failure-closed:
    # - a readable stack pill is positive OCCUPIED evidence (only a seated
    #   player carries a chip count);
    # - EMPTY requires the positive "+" cross signal — never inferred from
    #   the mere absence of a picture (a dimmed "away" avatar is not an
    #   empty seat);
    # - the hero slot has no avatar disc at all (its band holds the hero
    #   cards and the white hand-type label, which false-fires the cross
    #   detector), so hero occupancy comes from the stack pill only;
    # - anything ambiguous stays UNKNOWN.
    if slot_id == HERO_SLOT:
        if stack.status is LabelStatus.VALID:
            occupancy = FieldValue.valid(Occupancy.OCCUPIED)
        else:
            occupancy = FieldValue.unknown()
    elif stack.status is LabelStatus.VALID:
        occupancy = FieldValue.valid(Occupancy.OCCUPIED)
    elif avatar.size and _has_white_cross(avatar):
        occupancy = FieldValue.valid(Occupancy.EMPTY)
    elif avatar.size:
        if _avatar_picture_std(avatar) > _AVATAR_STD_MIN:
            occupancy = FieldValue.valid(Occupancy.OCCUPIED)
        else:
            occupancy = FieldValue.unknown()
    else:
        occupancy = FieldValue.unknown()

    # Dealer: the badge hugs the pill (to the right for edge seats, below
    # for top/bottom seats), so we only extend right and down — never up,
    # which would otherwise sweep the top status-bar / gear icons.
    s = int(ph * _DEALER_SCAN_MULT)
    win_y0 = pill_top
    win_y1 = min(H, pill_top + ph + s)
    win_x0 = max(0, pill_left - s // 3)
    win_x1 = min(W, pill_right + s)
    win = img[win_y0:win_y1, win_x0:win_x1]
    dealer = FieldValue.valid(_find_dealer(win)) if win.size else FieldValue.unknown()

    return {"occupancy": occupancy, "stack": stack, "dealer": dealer}


def read_seat_fields(
    img: np.ndarray,
    layout: dict[int, dict[str, float]],
    templates: dict[str, list[np.ndarray]],
    *,
    slots: Sequence[int] | None = None,
) -> dict[int, dict[str, FieldValue]]:
    """Read seat fields for the requested (default all) slots."""
    out: dict[int, dict[str, FieldValue]] = {}
    for slot_id in (slots if slots is not None else range(SLOT_COUNT)):
        out[slot_id] = read_slot(slot_id, img, layout, templates)
    return out
