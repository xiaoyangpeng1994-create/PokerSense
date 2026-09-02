"""Orchestrator-layer exceptions."""

from __future__ import annotations

from poker_engine.core.errors import PokerEngineError


class OrchestratorError(PokerEngineError):
    """Raised for orchestrator lifecycle / wiring errors.

    Examples:
    - process_observation() called with no active hand.
    - hand mismatch between memory and observation.
    """


__all__ = ["OrchestratorError"]
