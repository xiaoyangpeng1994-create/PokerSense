"""Memory-layer exceptions for Hand Memory.

These extend the core errors without polluting the frozen ``core/errors.py``.
"""

from __future__ import annotations

from poker_engine.core.errors import PokerEngineError


class HandMemoryError(PokerEngineError):
    """Base class for all Hand Memory errors."""


class HandNotFoundError(HandMemoryError):
    """Raised when operating on a hand_id that does not exist."""


class HandConflictError(HandMemoryError):
    """Raised when a conflicting write is detected.

    Examples:
    - Same state_version with a different PokerState.
    - Same hand_id but a different start (initial_state / started_at mismatch).
    - A duplicate event that is not value-identical to an existing one.
    """


class HandLifecycleError(HandMemoryError):
    """Raised on an invalid lifecycle transition.

    Examples:
    - Appending state/event to a completed hand.
    - Completing a hand that is already completed.
    - Completing a hand with no recorded PokerState.
    """


__all__ = [
    "HandMemoryError",
    "HandNotFoundError",
    "HandConflictError",
    "HandLifecycleError",
]
