"""State Engine errors."""

from __future__ import annotations

from poker_engine.core.errors import PokerEngineError


class StateEngineError(PokerEngineError):
    """Base error for State Engine contract violations.

    Raised for programmer/contract errors (e.g. context.previous_state does
    not match the explicit previous_state argument), NOT for domain
    observation conflicts (those become an invalid ValidationResult).
    """


__all__ = ["StateEngineError"]
