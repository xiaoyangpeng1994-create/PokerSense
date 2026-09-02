"""Confidence Gate exceptions."""

from __future__ import annotations

from poker_engine.core.errors import PokerEngineError


class ConfidenceGateError(PokerEngineError):
    """Raised for invalid threshold configuration."""


__all__ = ["ConfidenceGateError"]
