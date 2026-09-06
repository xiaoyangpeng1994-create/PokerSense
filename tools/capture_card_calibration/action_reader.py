# -*- coding: utf-8 -*-
"""Read per-seat `completed_action` badges from a single normalized table
frame for the capture-card platform.

This module implements the *pixel reading* half of stage F section 9 for the
action badge field. It takes a BGR frame (already normalized to the canvas)
and reads, for each of the ``SLOT_COUNT`` visual slots, the action badge that
sits just above the seat avatar:

- ``CALL``  -> a blue rounded pill, white text "跟注"
- ``RAISE`` -> an orange rounded pill, white text "加注"
- ``BET``   -> a rounded pill, white text "下注"
- ``CHECK`` -> a green rounded pill, white text "让牌"
- ``FOLD``  -> a dimmed dark disc with white text "弃牌" (no coloured pill)

Design principles (mirroring the guide's failure-closed philosophy and the
other readers in this toolkit):

- Every read returns a confident ``VALID`` value or ``UNKNOWN``; a seat with
  no readable badge is never guessed. In particular a *coloured* badge is the
  only positive evidence for a non-FOLD action: a dimmed disc (fold / wait
  state) is never turned into an action it did not perform.
- **Colour is only a family hint, never a verdict.** The two saturated
  families are the blue CALL pill and the orange RAISE/BET pill, plus the
  green CHECK pill. But BET shares its orange hue with RAISE *and* its green
  hue with CHECK (the "下注" pill is orange on some frames and green on
  others — measured on the hero slot it flips between the two across the
  same session). The white glyph inside the pill is therefore matched against
  per-action text templates and the *template* decides the action, not the
  colour.
- **A badge pill is a fixed, measured shape** (66 x 20 px on the 498 x 1080
  canvas, normalised to w=0.1325, h=0.0185). We search only for that
  wide-and-short oval and reject anything taller or rounder — this is what
  keeps a round avatar, the dealer or the wait-state disc from being
  mistaken for an action badge. The dark-disc states (弃牌 / 等待审核 /
  等待中) carry *no* such pill, so they never fire a coloured verdict.
- It does NOT reuse any coordinate, ROI or threshold from the LDPlayer or H5
  platforms (guide rules 1 and 2). All geometry is expressed in the 0..1
  normalized canvas space and was calibrated on this capture card; the per
  session slot layouts come from ``seat_reader``.

Import-safe without OpenCV: only the reader functions import cv2.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import SLOT_COUNT
from .schema import CompletedAction, FieldValue, LabelStatus

__all__ = [
    "ACTION_LABELS",
    "ActionRead",
    "_build_templates_from_measure_file",
    "read_action_badge",
    "read_action_fields",
]


#: Chinese labels drawn inside each badge pill. These are the glyphs the text
#: templates are built from; the mapping is the ground truth for the
#: template library (a "跟注" pill is always CALL, a "下注" pill always BET).
#: BET shares the orange/green pills with RAISE/CHECK, so the *label* is the
#: only thing that separates them.
ACTION_LABELS: dict[str, str] = {
    "CALL": "跟注",
    "RAISE": "加注",
    "BET": "下注",
    "CHECK": "让牌",
}


# --- colour families ---------------------------------------------------------

#: Hue ranges (OpenCV HSV, H in [0,179]) per action family. These are
#: *family* gates: blue = CALL, orange = RAISE/BET, green = CHECK. They set
#: which colour mask to scan, but never which action to report.
_HUE_CALL = (95, 135)    # blue pill   (measured 104-105)
_HUE_ORANGE = (5, 25)    # RAISE + BET (measured 13)
_HUE_GREEN = (35, 90)    # CHECK pill  (measured 75)
_SAT_MIN = 90
_VAL_MIN = 120

#: For the green (CHECK / green-BET) family the pill must be a *bright*
#: green. The felt shares the hue but is dark (Value ~51-200 measured on the
#: corpus); the pill is ~250. Requiring V >= 180 keeps felt out.
_GREEN_VAL_MIN = 180


# --- badge pill geometry (measured on session_002, 498 x 1080 canvas) -------

#: The badge pill is a wide, short oval. It was measured across the corpus:
#: every confirmed action pill is exactly 66 x 20 px (normalised w=0.1325,
#: h=0.0185, aspect ~3.3). The search window is a tight box around that so a
#: round avatar disc (w ~ h) or a tall element never matches.
_PILL_MIN_W = 54
_PILL_MAX_W = 74
_PILL_MIN_H = 16
_PILL_MAX_H = 24
_PILL_MIN_AREA = 200
#: Aspect ratio window for the wide-short pill (66/20 = 3.3). Tight enough to
#: reject the round avatar (ar ~1) and the felt edge blobs (ar 0.2-2.9).
_PILL_ASPECT = (2.7, 4.6)

#: Vertical band, in fraction of canvas height, scanned above the stack pill
#: for the action badge. The badge sits above the avatar, which sits above the
#: stack pill; on this platform the badge centre is ~0.92 of the pill centre
#: height for the ring slots (measured offsets -92 to -102 px to the stack
#: pill) and ~0.78 for the hero slot.
_BAND_UP = 0.24
_BAND_DOWN = 0.02
_BAND_SIDE = 0.20


# --- text-template matching ---------------------------------------------------

#: Text-template match threshold (normalised cross-correlation). A pill's
#: white glyph is compared against the per-action average template; the action
#: whose template scores highest wins, but only if that score clears this
#: floor meaningfully (recall the best-score bias: the max over all templates
#: always picks something, so a low best score means the glyph is not a clean
#: action word and must fail closed). Measured NCC samples: CALL own 0.899 /
#: best-other 0.567; RAISE own 0.931 / best-other 0.487; BET own 0.962 /
#: best-other 0.608; CHECK own 1.000 / best-other -0.052. A best-score floor
#: of 0.62 keeps every confirmed action while it screens the residual noisy
#: reads.
_TEMPLATE_TOP1_MIN = 0.62

#: Resize shape for a normalised glyph before correlating.
_GLYPH_W = 36
_GLYPH_H = 16

#: White text pixels inside a pill: bright and low saturation.
_TEXT_VAL_MIN = 140
_TEXT_SAT_MAX = 90


# --- data ---------------------------------------------------------------------


def _default_templates() -> dict[str, np.ndarray]:
    """Empty placeholders (keyed by action) until templates are supplied."""
    return {
        act: np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.float32) for act in ACTION_LABELS
    }


def _build_templates_from_measure_file(
    measure_path: Path, frames_dir: Path
) -> dict[str, np.ndarray]:
    """Build per-action average glyph templates from ``_mining`` measure results.

    This is the calibration-time helper: it reads ``action_badge_measure.json``
    (produced by the private measurement probe), crops each pill, extracts the
    white text glyph, and averages the glyphs per action. The averaged
    template is what :func:`read_action_badge` correlates against at runtime.

    Returns a dict action -> (normalised average template). The input measure
    JSON must have the ``kind == "pill"`` records with a ``px`` pixel bbox.
    """
    recs = json.loads(measure_path.read_text(encoding="utf-8"))
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in recs:
        if rec.get("kind") != "pill":
            continue
        import cv2

        frame = frames_dir / rec["frame"]
        img = cv2.imdecode(
            np.frombuffer(frame.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        x0, y0, x1, y1 = rec["px"]
        crop = img[y0:y1, x0:x1]
        g = _glyph(crop)
        if g is not None:
            groups[rec["action"]].append(g)
    return {act: np.mean(gs, axis=0) for act, gs in groups.items() if gs}


# --- glyph extraction ----------------------------------------------------------

def _glyph(crop: np.ndarray) -> np.ndarray | None:
    """Return the normalised white-text glyph mask of a pill crop, or None."""
    import cv2

    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = (
        (hsv[:, :, 2] > _TEXT_VAL_MIN) & (hsv[:, :, 1] < _TEXT_SAT_MAX)
    ).astype(np.uint8) * 255
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return None
    sub = m[y0 : y1 + 1, x0 : x1 + 1]
    resized = cv2.resize(sub, (_GLYPH_W, _GLYPH_H), interpolation=cv2.INTER_AREA)
    return (resized > 127).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two flat glyph vectors."""
    aa = a - a.mean()
    bb = b - b.mean()
    d = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float((aa * bb).sum() / d) if d > 0 else 0.0


# --- candidate read -------------------------------------------------------------

@dataclass(frozen=True)
class ActionRead:
    """The action badge read for one seat.

    ``status`` is ``VALID`` with ``value`` a :class:`CompletedAction` when a
    badge pill was found and its text glyph matched a single action template
    confidently. ``UNKNOWN`` otherwise — including when the seat shows a
    dimmed disc (fold / wait state) or no badge at all, which are *never*
    turned into an invented action.
    """

    status: LabelStatus
    value: CompletedAction | None

    @classmethod
    def unknown(cls) -> "ActionRead":
        return cls(status=LabelStatus.UNKNOWN, value=None)

    @classmethod
    def valid(cls, value: CompletedAction) -> "ActionRead":
        return cls(status=LabelStatus.VALID, value=value)


# --- colour mask -------------------------------------------------------------

def _colour_mask(crop: np.ndarray, family: str) -> np.ndarray:
    """Binary mask (0/255) of the badge-pill pixels of one colour family."""
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if family == "blue":
        mask = (h >= _HUE_CALL[0]) & (h <= _HUE_CALL[1])
    elif family == "orange":
        mask = (h >= _HUE_ORANGE[0]) & (h <= _HUE_ORANGE[1])
    elif family == "green":
        mask = (h >= _HUE_GREEN[0]) & (h <= _HUE_GREEN[1])
        mask &= v >= _GREEN_VAL_MIN
    else:
        mask = np.zeros(h.shape, dtype=bool)
    mask &= s >= _SAT_MIN
    mask &= v >= _VAL_MIN
    return mask.astype(np.uint8) * 255


def _find_pill(crop: np.ndarray, family: str) -> tuple[int, int, int, int] | None:
    """Locate the widest-short badge pill of a colour family in a crop.

    Returns ``(x, y, w, h)`` in crop pixel coords, or ``None`` (fail closed).
    Only a blob that matches the measured badge shape is accepted; a round
    avatar or a felt edge is rejected by the aspect / size windows.
    """
    import cv2

    mask = _colour_mask(crop, family)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < _PILL_MIN_AREA:
            continue
        if not (_PILL_MIN_W <= w <= _PILL_MAX_W and _PILL_MIN_H <= h <= _PILL_MAX_H):
            continue
        ar = w / h if h else 0.0
        if not (_PILL_ASPECT[0] <= ar <= _PILL_ASPECT[1]):
            continue
        # Badge sits above the avatar: prefer the topmost matching pill.
        if best is None or y < best[1]:
            best = (x, y, w, h)
    return best


# --- main reader -------------------------------------------------------------

def read_action_badge(
    slot_id: int,
    img: np.ndarray,
    layout: dict[int, dict[str, float]],
    templates: dict[str, np.ndarray] | None = None,
) -> FieldValue:
    """Read the completed-action badge for one slot (``VALID`` or ``UNKNOWN``).

    Order of evidence (fail-closed at each step):

    1. Scan the band above the stack pill for a badge pill of any colour
       family. If *no* pill matches the measured shape, the seat is not
       showing a coloured action badge — it is either a dimmed disc (fold /
       wait) or an ordinary avatar, and we return ``UNKNOWN``. A FOLD is
       *intentionally* not reported as an action here: this reader detects
       the *coloured* badge only, and ``FOLD`` is the absence of it. (Callers
       that need to distinguish fold from wait can combine this with the seat
       dark-disc evidence; the badge reader never invents a FOLD.)
    2. For each colour family that produced a pill, take its crop and match
       the white glyph against the per-action text templates. The action
       whose template scores highest wins. Because BET shares its pill colour
       with RAISE and CHECK, the *colour family* only narrows the search; the
       *text template* decides the reported action.
    3. If no template score clears the confidence floor, return ``UNKNOWN``.
    """
    if templates is None:
        templates = _default_templates()
    row = layout[slot_id]
    H, W = img.shape[:2]
    pill_cx = row["cx"] * W
    pill_cy = row["cy"] * H
    y0 = max(0, int(pill_cy - H * _BAND_UP))
    y1 = min(H, int(pill_cy + H * _BAND_DOWN))
    x0 = max(0, int(pill_cx - W * _BAND_SIDE))
    x1 = min(W, int(pill_cx + W * _BAND_SIDE))
    band = img[y0:y1, x0:x1]
    if band.size == 0:
        return FieldValue.unknown()

    # Family -> candidate actions sharing that family's pill colour.
    family_actions: dict[str, list[str]] = {
        "blue": ["CALL"],
        "orange": ["RAISE", "BET"],
        "green": ["CHECK", "BET"],
    }

    best: tuple[str, float] | None = None
    seen_families: set[str] = set()
    for family, actions in family_actions.items():
        loc = _find_pill(band, family)
        if loc is None:
            continue
        seen_families.add(family)
        px, py, pw, ph = loc
        crop = img[y0 + py : y0 + py + ph, x0 + px : x0 + px + pw]
        g = _glyph(crop)
        if g is None:
            continue
        for act in actions:
            tmpl = templates.get(act)
            if tmpl is None or np.allclose(tmpl, 0):
                continue
            score = _ncc(g, tmpl)
            if best is None or score > best[1]:
                best = (act, score)

    if best is None or best[1] < _TEMPLATE_TOP1_MIN:
        return FieldValue.unknown()
    try:
        action = CompletedAction(best[0])
    except ValueError:
        return FieldValue.unknown()
    return FieldValue.valid(action)


def read_action_fields(
    img: np.ndarray,
    layout: dict[int, dict[str, float]],
    templates: dict[str, np.ndarray] | None = None,
    *,
    slots: Sequence[int] | None = None,
) -> dict[int, dict[str, FieldValue]]:
    """Read action badges for the requested (default all) slots."""
    out: dict[int, dict[str, FieldValue]] = {}
    for slot_id in (slots if slots is not None else range(SLOT_COUNT)):
        out[slot_id] = {
            "completed_action": read_action_badge(slot_id, img, layout, templates)
        }
    return out
