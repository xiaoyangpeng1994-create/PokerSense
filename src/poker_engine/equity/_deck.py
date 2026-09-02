"""Shared card-deck helpers for equity estimators."""

from __future__ import annotations

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card


def full_deck() -> tuple[Card, ...]:
    """Return the full 52-card deck in a deterministic order."""
    return tuple(
        Card(rank, suit)
        for rank in Rank
        for suit in Suit
    )


def remaining_deck(used: tuple[Card, ...]) -> tuple[Card, ...]:
    """Return the deck with ``used`` cards removed (used must be distinct)."""
    used_set = set(used)
    if len(used_set) != len(used):
        raise ValueError("used cards must be distinct")
    return tuple(c for c in full_deck() if c not in used_set)
