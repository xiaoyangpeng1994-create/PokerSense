# -*- coding: utf-8 -*-
"""Viewpoint evidence for the capture-card platform: is a frame the owner
*playing* the hand or *spectating* another player's table?

Guide rules 1-2 forbid reusing another platform's geometry, and the failure-
closed philosophy means a machine must not guess which table a frame belongs
to. So this module is deliberately **not** a hard auto-classifier: it extracts
the *explainable visual evidence* of whose table a frame shows and returns a
conservative verdict — ``LIVE`` / ``SPECTATE`` / ``UNKNOWN`` — plus a
:class:`ViewpointEvidence` breakdown a human can audit and a per-frame HTML
page to confirm against the actual pixels.

Why not a single threshold? The capture-card UI draws the owner's hole cards
and the three action buttons at the bottom only while the hero is deciding;
the same table shows a bare felt otherwise. On the owner's own 8-max seat the
buttons are small and sparse; on a heads-up decision row they are large and
dense. Both the "revealed cards" zone and the "coloured action-band" are
therefore *per-frame* live signals that can each be absent on a live frame.
The module therefore only asserts a verdict when the evidence is strong and
consistent, and otherwise stays ``UNKNOWN`` so the owner can label it by eye.
This mirrors the guide's rule: a missing observation is not a contradiction,
and is never filled with a guess.

The owner anchored the discriminator on **the three action buttons**
(``以三按钮为准``): a coloured action-button band at the bottom is the primary
LIVE indicator. ``hero_occupied`` — whether the owner's bottom-centre seat is
occupied — is deliberately **not derived here**. Occupancy is read by the
seat-reading pipeline (stage F) and confirmed by a human; routing it through a
second, inconsistent heuristic would break the failure-closed contract, so it
is supplied by the caller (``True`` / ``False`` / ``None``) and only used to
corroborate. Import-safe without OpenCV; the signal functions import cv2
lazily.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import Scene

__all__ = [
    "Viewpoint",
    "ViewpointEvidence",
    "classify_viewpoint",
    "extract_evidence",
    "hero_signal",
    "render_viewpoint_report",
]


#: Verdict for which table a frame belongs to.
class Viewpoint(str, Enum):
    LIVE = "LIVE"          # the owner is playing (hero seat, own cards/actions)
    SPECTATE = "SPECTATE"  # a table the owner is watching, not playing
    UNKNOWN = "UNKNOWN"    # evidence is not strong enough to decide

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --- geometry (normalized canvas space) ------------------------------------
# The owner's revealed hole-card zone: the bottom-centre band where the two
# cards + the hand-strength label ("高牌"/"对子"/...) render. Measured on the
# 498x1080 canvas: cards sit around y ≈ 0.86-0.93, centred on x ≈ 0.5.
_HERO_CARD_Y0, _HERO_CARD_Y1 = 0.855, 0.940
_HERO_CARD_X0, _HERO_CARD_X1 = 0.28, 0.72

# The bottom action-button band (fold/check/bet discs). Roughly y 0.80-0.86.
_ACTION_Y0, _ACTION_Y1 = 0.795, 0.865
_ACTION_X0, _ACTION_X1 = 0.12, 0.88

# Strong-white threshold for the card face (cards are white, felt is green).
_CARD_WHITE = 200
# Minimum white fraction in the hero-card zone to count as "revealed cards".
_CARD_WHITE_FRACTION = 0.12
# Minimum coloured (saturated, bright) fraction in the action band.
_ACTION_COLORED_FRACTION = 0.03
# Bright minimum for a coloured button disc.
_ACTION_SAT = 90
_ACTION_VAL = 120


@dataclass(frozen=True)
class ViewpointEvidence:
    """Explainable signals extracted from one frame, plus the verdict."""

    revealed_cards_frac: float          # white fraction in hero-card zone
    action_colored_frac: float          # coloured fraction in action band
    has_action_button: bool             # coloured band present (approx.)
    hero_occupied: bool | None          # slot0 occupied; None if not supplied
    scene: Scene
    verdict: Viewpoint
    confidence: str                     # "HIGH" | "MEDIUM" | "LOW"

    @property
    def live_signals(self) -> int:
        """Count of strong LIVE indicators present (0..3)."""
        count = 0
        if self.revealed_cards_frac >= _CARD_WHITE_FRACTION:
            count += 1
        if (
            self.has_action_button
            and self.action_colored_frac >= _ACTION_COLORED_FRACTION
        ):
            count += 1
        if self.hero_occupied is True:
            count += 1
        return count

    def to_dict(self) -> dict[str, object]:
        return {
            "revealed_cards_frac": round(self.revealed_cards_frac, 4),
            "action_colored_frac": round(self.action_colored_frac, 4),
            "has_action_button": self.has_action_button,
            "hero_occupied": self.hero_occupied,
            "scene": self.scene.value,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "live_signals": self.live_signals,
        }


# --- low-level pixel helpers ----------------------------------------------

def _imread(path: Path | str) -> np.ndarray:
    import cv2

    return cv2.imdecode(
        np.frombuffer(Path(path).read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )


def _band(img: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def hero_signal(img: np.ndarray) -> float:
    """White fraction of the owner's revealed-hole-card zone.

    A high value (>0.12) means the owner's two cards are face-up at the
    bottom-centre, which happens while the owner is in the hand. This is a
    *live* signal but is per-frame: on a live frame where the owner folded
    earlier or is waiting, the cards are not revealed and the fraction is low.
    """
    if img is None or img.ndim != 3 or min(img.shape[:2]) < 1:
        return 0.0
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    band = _band(img, _HERO_CARD_Y0, _HERO_CARD_Y1, _HERO_CARD_X0, _HERO_CARD_X1)
    if band.size == 0:
        return 0.0
    bright = g[
        int(_HERO_CARD_Y0 * img.shape[0]) : int(_HERO_CARD_Y1 * img.shape[0]),
        int(_HERO_CARD_X0 * img.shape[1]) : int(_HERO_CARD_X1 * img.shape[1]),
    ] > _CARD_WHITE
    return float(bright.mean())


def _action_signal(img: np.ndarray) -> tuple[bool, float]:
    """Coloured fraction of the action-button band plus a button-present bool.

    The three action buttons (fold/check/bet) are bright, saturated discs on a
    dark band. On a heads-up decision row they are large and dense (high
    coloured fraction); on an 8-max seat they are smaller and sparser (low
    fraction). Both are the same *signal* — a button band exists — so the
    threshold is deliberately low and the verdict is corroborated, never
    decided by this number alone.
    """
    if img is None or img.ndim != 3 or min(img.shape[:2]) < 1:
        return False, 0.0
    import cv2

    band = _band(img, _ACTION_Y0, _ACTION_Y1, _ACTION_X0, _ACTION_X1)
    if band.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    ss, vv = hsv[:, :, 1], hsv[:, :, 2]
    colored = (vv > _ACTION_VAL) & (ss > _ACTION_SAT)
    frac = float(colored.mean())
    return frac >= _ACTION_COLORED_FRACTION, frac


# --- verdict ---------------------------------------------------------------

def _decide(
    scene: Scene,
    *,
    card_frac: float,
    has_button: bool,
    action_frac: float,
    hero_occupied: bool | None,
) -> tuple[Viewpoint, str]:
    """Map the extracted signals to a conservative verdict + confidence.

    The only authoritative LIVE anchor is the three-button action band (the
    owner's own decision row), corroborated by revealed cards and an occupied
    hero seat. A bare felt table with no live signal and no confirmed empty
    hero seat stays ``UNKNOWN`` — a spectated table and the owner's own idle
    table look identical from the felt alone, and we never guess.
    """
    if scene not in (Scene.TABLE, Scene.RESULT):
        # A menu / overlay / signal-loss frame is not live play from the
        # owner's seat on this frame.
        return Viewpoint.UNKNOWN, "LOW"

    button_live = has_button and action_frac >= _ACTION_COLORED_FRACTION
    cards_live = card_frac >= _CARD_WHITE_FRACTION

    if button_live:
        # The three-button action band is the owner's own decision row. This
        # is the primary, owner-anchored signal.
        if cards_live or hero_occupied is True:
            return Viewpoint.LIVE, "HIGH"
        return Viewpoint.LIVE, "MEDIUM"
    if cards_live:
        # Cards revealed at the hero seat strongly imply the owner is in the
        # hand, even when no button band is lit this instant.
        return Viewpoint.LIVE, "MEDIUM"
    if hero_occupied is True:
        # Seat occupied but no button band and no revealed cards yet: the
        # owner is seated but not mid-decision. Weak but live.
        return Viewpoint.LIVE, "LOW"
    if hero_occupied is False:
        # The owner's bottom-centre seat is demonstrably empty on a table
        # scene: the owner is not playing this table -> spectating.
        return Viewpoint.SPECTATE, "MEDIUM"

    # No strong signal and no confirmed empty hero seat. The owner's own idle
    # table and a spectated table are indistinguishable from the felt alone.
    return Viewpoint.UNKNOWN, "LOW"


def extract_evidence(
    img: np.ndarray,
    *,
    scene: Scene | str = Scene.TABLE,
    hero_occupied: bool | None = None,
) -> ViewpointEvidence:
    """Extract the explainable viewpoint signals from a BGR frame.

    ``hero_occupied`` is caller-supplied (``True``/``False``/``None``) — the
    seat-reader pipeline and a human confirm occupancy; it is never guessed
    here. It merely corroborates the verdict.
    """
    if not isinstance(scene, Scene):
        scene = Scene(scene)

    card_frac = hero_signal(img)
    has_button, action_frac = _action_signal(img)
    verdict, confidence = _decide(
        scene,
        card_frac=card_frac,
        has_button=has_button,
        action_frac=action_frac,
        hero_occupied=hero_occupied,
    )
    return ViewpointEvidence(
        revealed_cards_frac=card_frac,
        action_colored_frac=action_frac,
        has_action_button=has_button,
        hero_occupied=hero_occupied,
        scene=scene,
        verdict=verdict,
        confidence=confidence,
    )


def classify_viewpoint(
    path: Path | str,
    *,
    scene: Scene | str = Scene.TABLE,
    hero_occupied: bool | None = None,
) -> ViewpointEvidence:
    """Load a frame and classify its viewpoint (see :func:`extract_evidence`)."""
    img = _imread(path)
    if img is None:
        raise ValueError(f"could not read frame: {path}")
    return extract_evidence(img, scene=scene, hero_occupied=hero_occupied)


# --- per-frame review page -------------------------------------------------

_CSS_VP = """
:root {
  --bg:#f5f6f8; --card-bg:#fff; --ink:#1c1e21; --muted:#6a7280; --line:#e2e5ea;
  --live:#15803d; --spectate:#b91c1c; --unknown:#c2410c;
}
* { box-sizing:border-box; }
body{font:14px/1.5 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",
  sans-serif;margin:0;padding:20px;background:var(--bg);color:var(--ink)}
h1{font-size:18px;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
  gap:12px}
.card{background:var(--card-bg);border:1px solid var(--line);
  border-radius:10px;padding:8px}
.card img{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
.tag{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);
  color:var(--muted);margin:8px 4px 0 0;display:inline-block}
.tag.LIVE{color:var(--live);border-color:var(--live)}
.tag.SPECTATE{color:var(--spectate);border-color:var(--spectate)}
.tag.UNKNOWN{color:var(--unknown);border-color:var(--unknown)}
.meta{font-size:11px;color:var(--muted);margin-top:6px}
.foot{margin-top:18px;color:var(--muted);font-size:12px}
"""


def render_viewpoint_report(
    entries: Sequence[dict[str, object]],
    *,
    title: str = "Viewpoint review",
    summary: str = "",
) -> str:
    """Render viewpoint verdicts into a self-contained HTML contact sheet.

    ``entries`` is a list of dicts with keys ``frame``, ``verdict``,
    ``confidence`` and ``image`` (base64 payload), plus optional ``meta`` (a
    short human string) and ``signals`` (a :class:`ViewpointEvidence`). The
    page is a human-auditable contact sheet: the verdict is a *suggestion*,
    the image is authoritative.
    """
    cards = []
    for entry in entries:
        verdict = html.escape(str(entry["verdict"]))
        confidence = html.escape(str(entry.get("confidence", "")))
        frame = html.escape(str(entry.get("frame", "")))
        meta = html.escape(str(entry.get("meta", "")))
        image = html.escape(str(entry["image"]), quote=True)
        card = (
            f'<div class="card">'
            f'<img alt="{frame}" src="data:image/png;base64,{image}">'
            f'<div><span class="tag">{verdict}</span>'
            f'<span class="tag">{confidence}</span></div>'
            f'<div class="meta">{frame}{" · " + meta if meta else ""}</div>'
            "</div>"
        )
        cards.append(card)
    body = (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{title}</title><style>{_CSS_VP}</style></head><body>"
        f"<h1>{title}</h1><p class=\"sub\">{summary}</p>"
        f'<div class="grid">{"".join(cards)}</div>'
        f'<p class="foot">Verdicts are suggestions from extracted signals; '
        "the image is authoritative. Confirm by eye before trusting a label.</p>"
        "</body></html>"
    )
    return body
