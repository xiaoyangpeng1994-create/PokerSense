"""Domain-layer exceptions for Poker Intelligence Engine.

Task 1B introduces the first error used by the value objects.
Additional domain errors (ConfidenceGateError, StaleResultError, ...)
are added in later tasks.
"""


class PokerEngineError(Exception):
    """Base class for all Poker Engine domain errors."""


class InvalidStateError(PokerEngineError):
    """Raised when a value/state violates a domain invariant.

    Examples:
    - Constructing a ChipAmount with a negative value.
    - Adding a ChipDelta to a ChipAmount that would go below zero.
    """


class SerializationError(PokerEngineError):
    """Raised when serialization/deserialization fails.

    Examples:
    - Unsupported or unknown ``__type__`` tag.
    - Unsupported ``schema_version``.
    - Money value that is not a valid finite string.
    - A free mapping key collides with the reserved ``__type__`` key.
    """
