"""Opponent range equity: hero vs a SET of possible opponent holdings.

A "range" is a collection of concrete two-card holdings an opponent may hold,
represented as a tuple of (card, card) tuples. Duplicate entries express
weighting (a holding appearing twice counts twice). This is the simplest
correct form: no need for a separate probability field, exactly the composite
used to move from "known exact opponent" toward "real decision under
uncertainty".

Two estimators mirror the known-card estimators:
  - :func:`enumeration_range_equity` — EXACT: enumerate board + every opponent
    holding combination, average by frequency.
  - (Monte Carlo lives in :mod:`montecarlo`.)

Invariant (the self-check anchor): a range containing a SINGLE holding must
match the known-card enumerator for that exact opponent — i.e. range estimation
degenerates to exact estimation when the range is a singleton.
"""

from __future__ import annotations

from itertools import product

from poker_engine.core.value_objects import Card

from ._deck import remaining_deck
from .calculator import EquityResult
from .enumeration import _MAX_BOARD
from .evaluator import compare

# A range is an ordered tuple of two-card holdings (duplicates = weighting).
Range = tuple[tuple[Card, Card], ...]


def _validate_range(opponent_range: Range, name: str) -> None:
    if not opponent_range:
        raise ValueError(f"{name} range must be non-empty")
    for holding in opponent_range:
        if len(holding) != 2:
            raise ValueError(f"{name} range holdings must each be 2 cards")
        if holding[0] == holding[1]:
            raise ValueError(f"{name} range holding must be two distinct cards")


def enumeration_range_equity(
    hero: tuple[Card, ...],
    opponent_ranges: tuple[Range, ...],
    board: tuple[Card, ...] = (),
) -> EquityResult:
    """Exact hero equity against one or more opponent RANGES, by enumeration.

    Every holding in every opponent's range is combined with every possible
    board completion and every other opponent's holding, and each concrete
    matchup is evaluated once. Results are frequency-averaged, so a holding
    listed twice is weighted twice.
    """
    if len(hero) != 2:
        raise ValueError("hero must have exactly 2 hole cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board must have 0, 3, 4, or 5 cards")
    for i, rng in enumerate(opponent_ranges):
        _validate_range(rng, f"opponent[{i}]")

    need = _MAX_BOARD - len(board)

    win = tie = loss = 0
    share_sum = 0.0
    total = 0

    # Enumerate every concrete opponent assignment across all ranges.
    for opp_holding_combo in product(*opponent_ranges):
        opp_cards: list[Card] = [c for holding in opp_holding_combo for c in holding]
        known = list(hero) + opp_cards + list(board)
        if len(set(known)) != len(known):
            # A concrete assignment that clashes with itself/hero/board is
            # impossible under a real deal — skip (it contributes no weight).
            continue

        deck = remaining_deck(tuple(known))
        for board_rest in _combinations(deck, need):
            full_board = tuple(board) + tuple(board_rest)
            hero_hand = list(hero) + list(full_board)
            lost = False
            tie_count = 0
            for holding in opp_holding_combo:
                r = compare(hero_hand, list(holding) + list(full_board))
                if r == -1:
                    lost = True
                    break
                if r == 0:
                    tie_count += 1
            if lost:
                loss += 1
            elif tie_count > 0:
                tie += 1
                share_sum += 1.0 / (tie_count + 1)
            else:
                win += 1
                share_sum += 1.0
            total += 1

    if total == 0:
        return EquityResult(win=0.0, tie=0.0, loss=0.0, equity=0.0, samples=0)
    equity = share_sum / total
    return EquityResult(
        win=win / total,
        tie=tie / total,
        loss=loss / total,
        equity=equity,
        samples=total,
    )


def _combinations(deck, r):
    from itertools import combinations

    return combinations(deck, r)


__all__ = ["Range", "enumeration_range_equity"]
