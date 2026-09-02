"""JSON-safe, fail-closed Advice view model for the companion UI."""

from __future__ import annotations

from datetime import datetime

from poker_engine.core._freeze import _require_aware_dt, utc_now
from poker_engine.strategy.advice import Advice, AdviceStatus, mark_stale


def advice_to_view(
    advice: Advice,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Create a presentation contract without moving policy into JavaScript."""
    if not isinstance(advice, Advice):
        raise TypeError("advice must be an Advice")
    now = now or utc_now()
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime")
    _require_aware_dt(now)
    value = advice
    if now >= advice.expires_at and advice.status is not AdviceStatus.STALE:
        value = mark_stale(advice, reason="expired_advice", now=now)
    show_actions = value.status is AdviceStatus.READY
    actions = []
    if show_actions:
        for action, probability in sorted(
            value.action_probabilities.items(),
            key=lambda item: (-item[1], item[0].value),
        ):
            actions.append({
                "action": action.value,
                "probability": float(probability),
                "probability_exact": str(probability),
                "sizes": [
                    str(size.value)
                    for size in value.recommended_sizes.get(action, ())
                ],
                "ev": (
                    str(value.action_ev[action].value)
                    if action in value.action_ev else None
                ),
                "preferred": action is value.preferred_action,
            })
    return {
        "status": value.status.value,
        "show_actions": show_actions,
        "actions": actions,
        "strategy_source": value.strategy_source,
        "strategy_version": value.strategy_version,
        "match_kind": value.match_kind.value if value.match_kind else None,
        "state_match_score": value.state_match_score,
        "match_dimensions": [
            {
                "name": item.name,
                "requested": item.requested,
                "matched": item.matched,
                "distance": str(item.distance),
                "maximum_distance": str(item.maximum_distance),
                "score": item.score,
            }
            for item in value.match_dimensions
        ],
        "confidence": value.confidence,
        "ev_gap": (
            value.ev_gap.value.to_eng_string()
            if value.ev_gap is not None else None
        ),
        "rejection_reasons": list(value.rejection_reasons),
        "gate_results": [
            {
                "name": item.name,
                "status": item.status.value,
                "reasons": list(item.reasons),
            }
            for item in value.gate_results
        ],
        "missing_inputs": list(value.missing_inputs),
        "assumptions": list(value.assumptions),
        "evidence": list(value.evidence),
        "input_provenance": [
            {
                "field_name": item.field_name,
                "source": item.source.value,
                "status": item.status.value,
                "confidence": item.confidence,
            }
            for item in sorted(
                value.input_provenance, key=lambda item: item.field_name
            )
        ],
        "evidence_chain_id": value.evidence_chain_id,
        "evidence_complete": value.evidence_complete,
        "missing_evidence": list(value.missing_evidence),
        "expires_at": value.expires_at.isoformat(),
        "identity": {
            "hand_id": value.hand_id,
            "state_version": value.state_version,
            "request_id": value.request_id,
            "player_count": value.player_count,
            "active_player_count": value.active_player_count,
        },
    }


__all__ = ["advice_to_view"]
