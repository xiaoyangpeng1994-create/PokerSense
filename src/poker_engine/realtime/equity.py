"""Pluggable equity strategies for the realtime pipeline.

The realtime layer needs "hero vs random opponent range" equity. Two strategies:

  - :class:`MonteCarloRandomRangeEquity` — fast Monte Carlo over the random
    range (default for realtime; bounded latency).
  - :class:`ExactRandomRangeEquity` — exact enumeration over the random range
    (slower; used to self-check that Monte Carlo converges to the exact value).

The exact enumeration can also take an explicit opponent range (the Task 9
range-equity capability), which is the bridge to the future opponent model.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

from poker_engine.core.state import PokerState
from poker_engine.equity._deck import remaining_deck
from poker_engine.equity.montecarlo import MonteCarloEquity
from poker_engine.equity.range import enumeration_range_equity
from poker_engine.core.value_objects import Card

from .analysis import EquitySnapshot


class EquityStrategy(Protocol):
    """Compute a hero equity snapshot from a canonical state."""

    def compute(self, state: PokerState) -> EquitySnapshot:
        ...


def _random_range(state: PokerState) -> tuple[tuple[Card, Card], ...]:
    """All distinct two-card holdings not in hero/board (the random range)."""
    hero = tuple(state.hero_cards)
    board = tuple(state.board_cards)
    used = set(hero) | set(board)
    deck = remaining_deck(tuple(used))
    return tuple(combinations(deck, 2))


class MonteCarloRandomRangeEquity:
    """Hero equity vs a random opponent range, via Monte Carlo (realtime)."""

    def __init__(self, trials: int = 2000, seed: int = 0) -> None:
        self._mc = MonteCarloEquity(trials=trials, seed=seed)

    def compute(self, state: PokerState) -> EquitySnapshot:
        hero = tuple(state.hero_cards)
        board = tuple(state.board_cards)
        if len(hero) != 2:
            return EquitySnapshot(win_rate=0.0, tie_rate=0.0)
        rng = _random_range(state)
        res = self._mc.estimate_range(hero, (rng,), board)
        return EquitySnapshot(win_rate=res.win, tie_rate=res.tie)


class ExactRandomRangeEquity:
    """Hero equity vs a random opponent range, via exact enumeration.

    Slower than Monte Carlo; use to verify convergence / for offline analysis.
    May be expensive on the flop (large random range x board completions).
    """

    def compute(self, state: PokerState) -> EquitySnapshot:
        hero = tuple(state.hero_cards)
        board = tuple(state.board_cards)
        if len(hero) != 2:
            return EquitySnapshot(win_rate=0.0, tie_rate=0.0)
        rng = _random_range(state)
        res = enumeration_range_equity(hero, (rng,), board)
        return EquitySnapshot(win_rate=res.win, tie_rate=res.tie)


__all__ = [
    "EquityStrategy",
    "MonteCarloRandomRangeEquity",
    "ExactRandomRangeEquity",
]
