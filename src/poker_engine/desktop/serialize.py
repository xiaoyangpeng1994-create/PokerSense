"""JSON wire contract for the desktop UI.

One function, one direction (engine -> UI). This must stay in sync with
``ui/app.js``'s expected shape:

    {
      "frame_seq": int,
      "state": {"hand_id": str, "state_version": int, "street": str,
                "hero_cards": [str], "board_cards": [str], "pot": str},
      "equity": {"win_rate": float, "tie_rate": float},
      "confidence": {"overall_confidence": float, "field_status": [[str, str], ...]},
    }

Money is stringified (project-wide convention, see docs/serialization.md);
everything else is plain JSON-safe primitives so the frontend needs no
knowledge of the engine's internal value objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from poker_engine.realtime.analysis import RealtimeAnalysis
from poker_engine.strategy.advice import Advice, mark_stale

from .advice_view import advice_to_view


@dataclass(frozen=True)
class DesktopFrame:
    """One atomic UI update: table analysis plus optional strategy Advice."""

    analysis: RealtimeAnalysis
    advice: Advice | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, RealtimeAnalysis):
            raise TypeError("analysis must be a RealtimeAnalysis")
        if self.advice is not None and not isinstance(self.advice, Advice):
            raise TypeError("advice must be an Advice or None")


def analysis_to_dict(
    analysis: RealtimeAnalysis,
    advice: Advice | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    if not isinstance(analysis, RealtimeAnalysis):
        raise TypeError("analysis must be a RealtimeAnalysis")
    state = analysis.state
    payload = {
        "frame_seq": analysis.frame_seq,
        "state": {
            "hand_id": state.hand_id,
            "state_version": state.state_version,
            "street": state.street.value,
            "hero_cards": [str(c) for c in state.hero_cards],
            "board_cards": [str(c) for c in state.board_cards],
            "pot": str(state.pot.value),
        },
        "equity": {
            "win_rate": analysis.equity.win_rate,
            "tie_rate": analysis.equity.tie_rate,
        },
        "confidence": {
            "overall_confidence": analysis.confidence.overall_confidence,
            "field_status": [list(pair) for pair in analysis.confidence.field_status],
        },
    }
    if advice is not None:
        if (
            advice.hand_id != state.hand_id
            or advice.state_version != state.state_version
        ):
            advice = mark_stale(
                advice,
                reason="analysis_identity_mismatch",
                now=now,
            )
        payload["advice"] = advice_to_view(advice, now=now)
    return payload


def desktop_frame_to_dict(
    frame: DesktopFrame | RealtimeAnalysis,
    *,
    now: datetime | None = None,
) -> dict:
    """Serialize new atomic frames while retaining legacy analysis streams."""
    if isinstance(frame, DesktopFrame):
        return analysis_to_dict(frame.analysis, frame.advice, now=now)
    if isinstance(frame, RealtimeAnalysis):
        return analysis_to_dict(frame, now=now)
    raise TypeError("frame must be a DesktopFrame or RealtimeAnalysis")


__all__ = [
    "DesktopFrame",
    "advice_to_view",
    "analysis_to_dict",
    "desktop_frame_to_dict",
]
