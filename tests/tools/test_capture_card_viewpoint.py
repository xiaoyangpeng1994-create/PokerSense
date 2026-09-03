# -*- coding: utf-8 -*-
"""Tests for the viewpoint evidence extractor (viewpoint.py).

The extractor deliberately does *not* auto-classify — it surfaces explainable
signals so the owner can confirm by eye which frames are their own play. The
unit tests pin three things:

1. The low-level pixel signals (white card fraction, coloured action band).
2. The conservative verdict mapping (``LIVE``/``SPECTATE``/``UNKNOWN``),
   including that a bare felt with no signal stays ``UNKNOWN`` (fail-closed),
   that a non-table scene stays ``UNKNOWN``, and that the three-button action
   band is the primary LIVE anchor.
3. The self-contained HTML contact sheet.

These use synthetic BGR frames and require OpenCV — they are the *only*
viewpoint tests that touch pixels; everything else is import-safe and
byte-based. No private dataset and no real frames are used.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.capture_card_calibration.schema import Scene
from tools.capture_card_calibration.viewpoint import (
    Viewpoint,
    hero_signal,
    render_viewpoint_report,
    extract_evidence,
)

# A canvas with the normalized aspect ratio (tall/portrait). Kept small so the
# suite stays fast; the signal bands are expressed as fractions of width/height.
_H, _W = 200, 92


def _blank_bgr() -> np.ndarray:
    """A green felt frame: no cards, no buttons, no avatar."""
    frame = np.zeros((_H, _W, 3), dtype=np.uint8)
    frame[:, :] = (60, 120, 70)  # BGR green felt
    return frame


def _with_revealed_cards(base: np.ndarray) -> np.ndarray:
    """Paint a bright white block in the hero-card zone (y 0.855-0.940)."""
    frame = base.copy()
    y0, y1 = int(0.855 * _H), int(0.940 * _H)
    x0, x1 = int(0.28 * _W), int(0.72 * _W)
    frame[y0:y1, x0:x1] = (250, 250, 250)
    return frame


def _with_action_buttons(base: np.ndarray) -> np.ndarray:
    """Paint a bright saturated band in the action zone (y 0.795-0.865)."""
    frame = base.copy()
    y0, y1 = int(0.795 * _H), int(0.865 * _H)
    x0, x1 = int(0.12 * _W), int(0.88 * _W)
    # A saturated bright disc/band: high value (V) and high saturation (S).
    frame[y0:y1, x0:x1] = (30, 200, 230)  # BGR -> HSV: bright, saturated
    return frame


def _cv2_available() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - environment dependent
        return False


_cv2 = pytest.mark.skipif(not _cv2_available(), reason="OpenCV is not installed")


# --- low-level signals -----------------------------------------------------

@_cv2
def test_hero_signal_blank_felt_is_low() -> None:
    assert hero_signal(_blank_bgr()) < 0.12


@_cv2
def test_hero_signal_with_revealed_cards_is_high() -> None:
    frame = _with_revealed_cards(_blank_bgr())
    assert hero_signal(frame) >= 0.12


@_cv2
def test_hero_signal_zero_on_empty_band() -> None:
    # A zero-height frame (band clipped to nothing) must not crash and returns 0.
    tiny = np.zeros((0, 0, 3), dtype=np.uint8)
    assert hero_signal(tiny) == 0.0


# --- verdict mapping (fail-closed) -----------------------------------------

@_cv2
def test_bare_felt_stays_unknown() -> None:
    """A table with no cards, no buttons, no hero — the owner's own idle table
    and a spectated table look identical from the felt alone -> UNKNOWN."""
    evidence = extract_evidence(_blank_bgr())
    assert evidence.verdict is Viewpoint.UNKNOWN
    assert evidence.confidence == "LOW"


@_cv2
def test_bare_felt_hero_occupied_is_weak_live() -> None:
    evidence = extract_evidence(_blank_bgr(), hero_occupied=True)
    assert evidence.verdict is Viewpoint.LIVE
    assert evidence.confidence == "LOW"


@_cv2
def test_revealed_cards_leans_live() -> None:
    evidence = extract_evidence(_with_revealed_cards(_blank_bgr()))
    assert evidence.verdict is Viewpoint.LIVE
    assert evidence.confidence == "MEDIUM"


@_cv2
def test_action_buttons_is_primary_live_anchor() -> None:
    evidence = extract_evidence(_with_action_buttons(_blank_bgr()))
    assert evidence.verdict is Viewpoint.LIVE
    assert evidence.has_action_button is True
    assert evidence.action_colored_frac >= 0.03


@_cv2
def test_buttons_and_cards_is_high_confidence_live() -> None:
    frame = _with_action_buttons(_with_revealed_cards(_blank_bgr()))
    evidence = extract_evidence(frame)
    assert evidence.verdict is Viewpoint.LIVE
    assert evidence.confidence == "HIGH"


@_cv2
def test_buttons_and_occupied_hero_is_high_confidence_live() -> None:
    frame = _with_action_buttons(_blank_bgr())
    evidence = extract_evidence(frame, hero_occupied=True)
    assert evidence.verdict is Viewpoint.LIVE
    assert evidence.confidence == "HIGH"


@_cv2
def test_confirmed_empty_hero_seat_is_spectate() -> None:
    evidence = extract_evidence(_blank_bgr(), hero_occupied=False)
    assert evidence.verdict is Viewpoint.SPECTATE
    assert evidence.confidence == "MEDIUM"


@_cv2
def test_non_table_scene_is_unknown_even_when_live_signals_present() -> None:
    frame = _with_action_buttons(_with_revealed_cards(_blank_bgr()))
    evidence = extract_evidence(frame, scene=Scene.MENU)
    assert evidence.verdict is Viewpoint.UNKNOWN
    assert evidence.confidence == "LOW"


@_cv2
def test_scene_accepts_string_value() -> None:
    evidence = extract_evidence(_blank_bgr(), scene="table")
    assert evidence.scene is Scene.TABLE


# --- evidence payload ------------------------------------------------------

@_cv2
def test_evidence_to_dict_is_machine_readable() -> None:
    evidence = extract_evidence(
        _with_action_buttons(_blank_bgr()), hero_occupied=True
    )
    data = evidence.to_dict()
    assert data["verdict"] == "LIVE"
    assert data["hero_occupied"] is True
    assert "live_signals" in data
    assert data["live_signals"] >= 2


def test_viewpoint_str_is_value() -> None:
    assert str(Viewpoint.LIVE) == "LIVE"
    assert Viewpoint("SPECTATE") is Viewpoint.SPECTATE
    assert Viewpoint("UNKNOWN") is Viewpoint.UNKNOWN


# --- live_signals counter --------------------------------------------------

@_cv2
def test_live_signals_zero_on_bare_felt() -> None:
    assert extract_evidence(_blank_bgr()).live_signals == 0


@_cv2
def test_live_signals_counts_all_present() -> None:
    frame = _with_action_buttons(_with_revealed_cards(_blank_bgr()))
    evidence = extract_evidence(frame, hero_occupied=True)
    assert evidence.live_signals == 3


# --- HTML contact sheet ----------------------------------------------------

def test_render_viewpoint_report_is_self_contained() -> None:
    entries = [
        {
            "frame": "a.png",
            "verdict": "LIVE",
            "confidence": "HIGH",
            "image": "iVBORw0KGgo=",
            "meta": "session_001 · hand_0000",
        },
        {
            "frame": "b.png",
            "verdict": "SPECTATE",
            "confidence": "MEDIUM",
            "image": "iVBORw0KGgo=",
        },
    ]
    page = render_viewpoint_report(entries, title="t", summary="s")
    assert "<!DOCTYPE html>" in page
    assert "data:image/png;base64," in page
    assert "LIVE" in page and "SPECTATE" in page
    assert "<link" not in page and "<script" not in page
    assert "http://" not in page and "https://" not in page


def test_render_viewpoint_report_escapes_frame_name() -> None:
    entries = [
        {
            "frame": "<script>alert(1)</script>.png",
            "verdict": "UNKNOWN",
            "confidence": "LOW",
            "image": "iVBORw0KGgo=",
        }
    ]
    page = render_viewpoint_report(entries, title="t", summary="s")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
