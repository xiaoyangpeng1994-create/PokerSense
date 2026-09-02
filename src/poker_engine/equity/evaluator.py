"""Hand evaluator: rank a 5-card combination, pick the best 5 of up to 7.

Pure, deterministic, no float money, no third-party deps. Used by both the
exact (enumeration) and Monte Carlo equity estimators so they share one source
of truth for "which hand wins".

The strength of a hand is encoded as a comparable tuple
``(category, t1, t2, t3, t4, t5)`` where category is 8 (straight flush) ... 0
(high card) and the trailing values are tie-breakers (rank ints 2..14).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from poker_engine.core.value_objects import Card

# Category codes (higher == stronger).
_STRAIGHT_FLUSH = 8
_FOUR_KIND = 7
_FULL_HOUSE = 6
_FLUSH = 5
_STRAIGHT = 4
_THREE_KIND = 3
_TWO_PAIR = 2
_ONE_PAIR = 1
_HIGH_CARD = 0

# The lowest straight is A-2-3-4-5 (the "wheel"); in that case A acts as 1.
# The rank_value ints for those ranks (A=14, 5, 4, 3, 2).
_WHEEL = (14, 5, 4, 3, 2)


def _evaluate_5(cards: list[Card]) -> tuple:
    """Evaluate exactly 5 cards -> (category, t1..t5) comparable tuple."""
    values = sorted((c.rank_value for c in cards), reverse=True)
    suits = [c.suit for c in cards]

    is_flush = len(set(suits)) == 1

    # straight detection (with wheel A-2-3-4-5 special case)
    unique = sorted(set(values), reverse=True)
    is_straight = False
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        elif set(unique) == set(_WHEEL):
            # wheel: A,5,4,3,2 -> high card is 5
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        return (_STRAIGHT_FLUSH, straight_high)

    counts = Counter(values)
    # groups sorted by (count desc, rank desc)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    if groups[0][1] == 4:
        quad = groups[0][0]
        kicker = groups[1][0]
        return (_FOUR_KIND, quad, kicker)

    if groups[0][1] == 3 and groups[1][1] == 2:
        trips = groups[0][0]
        pair = groups[1][0]
        return (_FULL_HOUSE, trips, pair)

    if is_flush:
        return (_FLUSH, *values[:5])

    if is_straight:
        return (_STRAIGHT, straight_high)

    if groups[0][1] == 3:
        trips = groups[0][0]
        kickers = sorted((g[0] for g in groups[1:]), reverse=True)
        return (_THREE_KIND, trips, *kickers[:2])

    if groups[0][1] == 2 and groups[1][1] == 2:
        hi_pair = groups[0][0]
        lo_pair = groups[1][0]
        kicker = groups[2][0]
        return (_TWO_PAIR, hi_pair, lo_pair, kicker)

    if groups[0][1] == 2:
        pair = groups[0][0]
        kickers = sorted((g[0] for g in groups[1:]), reverse=True)
        return (_ONE_PAIR, pair, *kickers[:3])

    return (_HIGH_CARD, *values[:5])


def evaluate(cards: list[Card] | tuple[Card, ...]) -> tuple:
    """Evaluate 5, 6, or 7 cards -> best 5-card hand strength tuple.

    For 6 or 7 cards, all 5-card subsets are considered and the strongest is
    returned (a Hold'em hand uses 2 hole + up to 5 board cards).
    """
    lst = list(cards)
    n = len(lst)
    if n < 5 or n > 7:
        raise ValueError(f"evaluate requires 5..7 cards, got {n}")
    if n == 5:
        return _evaluate_5(lst)
    best = None
    for combo in combinations(lst, 5):
        s = _evaluate_5(list(combo))
        if best is None or s > best:
            best = s
    return best


def compare(a_cards, b_cards) -> int:
    """Return -1 if a loses, 0 if tie, 1 if a wins (both 5..7 cards)."""
    sa = evaluate(a_cards)
    sb = evaluate(b_cards)
    if sa > sb:
        return 1
    if sa < sb:
        return -1
    return 0


__all__ = ["evaluate", "compare"]
