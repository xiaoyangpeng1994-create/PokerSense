"""Tests for stage F action-badge reading (pure rules, fail closed).

These tests are hardware-free and private-data-free: they synthesize a tiny
badge pill (a 66x20 rounded box in the correct colour) with the correct white
action text (跟注 / 加注 / 下注 / 让牌) drawn on top of a solid green felt,
then verify the whole read path — colour-family gate, strict pill-shape gate,
and the text-template match that separates same-colour actions (RAISE vs BET,
CHECK vs BET).

The failure-closed contract is asserted directly:

- a dimmed dark disc (fold / wait state) has *no* coloured pill and must be
  read as ``UNKNOWN`` — never invented as an action;
- a round avatar or a felt-edge blob that is not the measured 66x20 pill is
  rejected by the shape window;
- a pill whose text glyph does not clearly match any action template is
  ``UNKNOWN`` (never guessed);
- same-colour actions are separated by the text template, not the colour.
"""

from __future__ import annotations

import cv2
import numpy as np

from tools.capture_card_calibration import action_reader as ar
from tools.capture_card_calibration.schema import LabelStatus

# --- synthetic badge rendering ---------------------------------------------

#: Canvas is a square green felt so a slot band is easy to lay out.
_CANVAS = 400
_FELT = np.array((80, 160, 70), dtype=np.uint8)  # BGR green felt
_PILL_W, _PILL_H = 66, 20
_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Colour (BGR) of each family, drawn so the hue lands in the reader's gate.
_COLOR = {
    "CALL": (255, 140, 10),    # blue pill   (BGR)
    "RAISE": (0, 110, 255),    # orange pill
    "BET": (0, 110, 255),      # orange pill (some BET frames)
    "CHECK": (120, 230, 0),    # green pill
}

#: Synthetic glyph label per action. ``cv2.putText`` cannot render CJK (it
#: draws a "?"), so the synthetic pills use a distinct ASCII marker per action.
#: These only exercise the *template-matching mechanism* (colour family +
#: shape gate + NCC); the *real* Chinese-label distinction is validated on the
#: measured corpus (see the module docstring's NCC numbers), where the glyphs
#: 跟注 / 加注 / 下注 / 让牌 correlate own > 0.89 and best-other < 0.61.
_GLYPH_LABEL = {
    "CALL": "CA",
    "RAISE": "RA",
    "BET": "BE",
    "CHECK": "CH",
}


def _felt(w: int = _CANVAS, h: int = _CANVAS) -> np.ndarray:
    return np.tile(_FELT, (h, w, 1))


def _pill_image(action: str, *, colour: tuple | None = None) -> np.ndarray:
    """A felt frame with one badge pill at a known, centered location."""
    img = _felt()
    cx, cy = _CANVAS // 2, _CANVAS // 2
    x0, y0 = cx - _PILL_W // 2, cy - _PILL_H // 2
    x1, y1 = x0 + _PILL_W, y0 + _PILL_H
    col = colour if colour is not None else _COLOR[action]
    cv2.rectangle(img, (x0, y0), (x1, y1), col, thickness=-1)
    # white marker centred in the pill (ASCII so cv2 can render it)
    label = _GLYPH_LABEL[action]
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
    tx = cx - tw // 2
    ty = cy + th // 2
    cv2.putText(img, label, (tx, ty), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _disc_frame() -> np.ndarray:
    """A dimmed dark disc (fold / wait state): no coloured pill."""
    img = _felt()
    cx, cy = _CANVAS // 2, _CANVAS // 2
    cv2.circle(img, (cx, cy), 30, (20, 20, 30), thickness=-1)
    cv2.putText(img, "X", (cx - 5, cy + 6), _FONT, 0.5, (220, 220, 220), 1)
    return img


#: A layout where the badge pill lands exactly at the canvas centre band.
_LAYOUT: dict[int, dict[str, float]] = {
    0: dict(cx=0.5, cy=0.5, w=0.2, h=0.05),
}


def _templates() -> dict[str, np.ndarray]:
    """Build per-action templates from one synthetic pill each.

    The template must come from the *exact* pill bbox (the same crop the
    runtime uses) so the glyph normalisation matches. Each synthetic pill is
    placed centred in the frame and located with ``_find_pill`` on the
    reader's scan band, then its crop is glyph-extracted — mirroring
    ``_build_templates_from_measure_file``.
    """
    groups: dict[str, list[np.ndarray]] = {}
    _layout = {0: dict(cx=0.5, cy=0.5, w=0.2, h=0.05)}
    for act in ar.ACTION_LABELS:
        img = _pill_image(act)
        H, W = img.shape[:2]
        row = _layout[0]
        pcy = row["cy"] * H
        pcx = row["cx"] * W
        y0 = int(pcy - H * ar._BAND_UP)
        y1 = int(pcy + H * ar._BAND_DOWN)
        x0 = int(pcx - W * ar._BAND_SIDE)
        x1 = int(pcx + W * ar._BAND_SIDE)
        band = img[y0:y1, x0:x1]
        # The synthetic pill colour per action maps to a family.
        fam = "blue" if act == "CALL" else ("green" if act == "CHECK" else "orange")
        loc = ar._find_pill(band, fam)
        if loc is None:
            continue
        px, py, pw, ph = loc
        crop = img[y0 + py : y0 + py + ph, x0 + px : x0 + px + pw]
        g = ar._glyph(crop)
        if g is not None:
            groups.setdefault(act, []).append(g)
    return {act: np.mean(gs, axis=0) for act, gs in groups.items() if gs}


# --- helpers -----------------------------------------------------------------

def _read(action: str, colour: tuple | None = None) -> str | None:
    img = _pill_image(action, colour=colour)
    tmpl = _templates()
    fv = ar.read_action_badge(0, img, _LAYOUT, tmpl)
    return fv.value.value if fv.status is LabelStatus.VALID else None


# --- tests -------------------------------------------------------------------

def test_call_blue_returns_call():
    assert _read("CALL") == "CALL"


def test_raise_orange_returns_raise():
    assert _read("RAISE") == "RAISE"


def test_check_green_returns_check():
    assert _read("CHECK") == "CHECK"


def test_bet_orange_returns_bet_not_raise():
    # Same orange family as RAISE but the text is 下注 -> BET.
    assert _read("BET") == "BET"


def test_bet_green_returns_bet_not_check():
    # The hero slot draws the 下注 pill green on some frames (measured: BET
    # flips orange/green across the session); the text must still read BET.
    assert _read("BET", colour=_COLOR["CHECK"]) == "BET"


def test_dark_disc_fails_closed():
    img = _disc_frame()
    fv = ar.read_action_badge(0, img, _LAYOUT, _templates())
    assert fv.status is LabelStatus.UNKNOWN
    assert fv.value is None


def test_no_badge_fails_closed():
    img = _felt()
    fv = ar.read_action_badge(0, img, _LAYOUT, _templates())
    assert fv.status is LabelStatus.UNKNOWN


def test_round_avatar_rejected_by_shape():
    # A saturated round disc (avatar) must NOT be read as a pill.
    img = _felt()
    cv2.circle(img, (_CANVAS // 2, _CANVAS // 2), 40, _COLOR["CALL"], thickness=-1)
    fv = ar.read_action_badge(0, img, _LAYOUT, _templates())
    assert fv.status is LabelStatus.UNKNOWN


def test_empty_templates_fail_closed():
    img = _pill_image("CALL")
    fv = ar.read_action_badge(0, img, _LAYOUT, ar._default_templates())
    assert fv.status is LabelStatus.UNKNOWN


def test_read_action_fields_shapes():
    img = _pill_image("RAISE")
    out = ar.read_action_fields(img, _LAYOUT, _templates(), slots=[0])
    assert 0 in out
    assert set(out[0].keys()) == {"completed_action"}


def test_unknown_value_never_carries_value():
    fv = ar.read_action_badge(0, _felt(), _LAYOUT, _templates())
    assert fv.status is LabelStatus.UNKNOWN
    assert fv.value is None
