"""Test-only confidence profile injection (NOT a config change).

The Frozen ConfidenceGate thresholds (hero/board >= 0.995, street >= 0.999,
...) are the REAL Golden-data acceptance standard. Synthetic realtime tests
cannot reach those (template-match calibrated confidence tops out at 0.9), so
they inject a RELAXED gate to verify pipeline *connectivity* only.

Audit note (kept deliberately visible):
  Realtime synthetic integration uses a relaxed confidence profile ONLY to
  verify state pipeline connectivity (Frame -> Vision -> Observation ->
  Orchestrator -> StateEngine -> Memory). Production/Frozen Golden acceptance
  thresholds remain unchanged (see src/poker_engine/confidence/gate.py).

The 0.995-vs-0.9 gap is NOT solved here; it belongs to the Real Golden Data
Calibration Phase (a later, separate effort).
"""

from __future__ import annotations

from poker_engine.confidence.gate import ConfidenceGate

# Relaxed thresholds derived from measured Vision calibrated output (0.9 max
# for hero/board/street/pot/bet; action is 0 and stays gated out of scope).
# These are TEST-ONLY and deliberately looser than Frozen production values.
_RELAXED_THRESHOLDS = {
    "hero_cards": 0.8,
    "board_cards": 0.8,
    "street": 0.8,
    "pot": 0.8,
    "stacks": 0.8,
    "bet_size": 0.8,
    "action": 0.99,  # action recognition is not exercised in synthetic tests
}


def relaxed_confidence_gate() -> ConfidenceGate:
    """A ConfidenceGate with test-only relaxed thresholds.

    Use ONLY in synthetic realtime integration tests. Never in production.
    """
    return ConfidenceGate(thresholds=_RELAXED_THRESHOLDS)


__all__ = ["relaxed_confidence_gate"]
