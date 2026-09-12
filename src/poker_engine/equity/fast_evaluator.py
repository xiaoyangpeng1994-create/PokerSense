"""Direct 5–7 card evaluation, cross-checked against subset enumeration.

Returns the same strength tuple as ``evaluator.evaluate``. Evaluates rank and
suit counts directly instead of enumerating 21 five-card subsets per player.
The reference implementation remains unchanged as an independent oracle.
"""

from __future__ import annotations

from collections import Counter

from poker_engine.core.value_objects import Card


def _straight_high(values: set[int]) -> int | None:
    if 14 in values:
        values = values | {1}
    for high in range(14, 4, -1):
        if all(value in values for value in range(high - 4, high + 1)):
            return high
    return None


def evaluate_fast(cards: tuple[Card, ...] | list[Card]) -> tuple:
    """Rank a legal, distinct 5–7 card holding using reference tuple ordering."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate_fast requires 5..7 cards")
    if len(set(cards)) != len(cards):
        raise ValueError("cards must be distinct")
    counts = Counter(card.rank_value for card in cards)
    suited: dict[object, list[int]] = {}
    for card in cards:
        suited.setdefault(card.suit, []).append(card.rank_value)
    flush = next((values for values in suited.values() if len(values) >= 5), None)
    if flush is not None:
        straight = _straight_high(set(flush))
        if straight is not None:
            return (8, straight)
    quads = sorted((value for value, n in counts.items() if n == 4), reverse=True)
    if quads:
        quad = quads[0]
        return (7, quad, max(value for value in counts if value != quad))
    trips = sorted((value for value, n in counts.items() if n >= 3), reverse=True)
    pairs = sorted((value for value, n in counts.items() if n >= 2), reverse=True)
    if trips:
        remaining_pairs = [value for value in pairs if value != trips[0]]
        if remaining_pairs:
            return (6, trips[0], remaining_pairs[0])
    if flush is not None:
        return (5, *sorted(flush, reverse=True)[:5])
    straight = _straight_high(set(counts))
    if straight is not None:
        return (4, straight)
    if trips:
        kickers = sorted((v for v in counts if v != trips[0]), reverse=True)[:2]
        return (3, trips[0], *kickers)
    if len(pairs) >= 2:
        kicker = max(v for v in counts if v not in pairs[:2])
        return (2, pairs[0], pairs[1], kicker)
    if pairs:
        kickers = sorted((v for v in counts if v != pairs[0]), reverse=True)[:3]
        return (1, pairs[0], *kickers)
    return (0, *sorted(counts, reverse=True)[:5])
