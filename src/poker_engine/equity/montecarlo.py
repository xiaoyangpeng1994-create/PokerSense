"""Monte Carlo equity via random board completion (seeded, reproducible).

Same contract as enumeration, but instead of enumerating every board it draws
a fixed number of random completions. For a known opponent (hole cards given)
this converges to the exact enumeration result; for unknown opponents it can
also deal random opponent hole cards from the remaining deck.
"""

from __future__ import annotations

import random

from poker_engine.core.value_objects import Card

from ._deck import remaining_deck
from .calculator import EquityEstimator, EquityResult
from .evaluator import compare
from .range import Range, _validate_range

_MAX_BOARD = 5


class MonteCarloEquity(EquityEstimator):
    """Seeded Monte Carlo equity estimator."""

    def __init__(self, trials: int = 10000, seed: int | None = 42):
        if trials <= 0:
            raise ValueError("trials must be positive")
        self._trials = trials
        self._seed = seed

    def _sample(self, rng: random.Random, deck, count):
        return rng.sample(list(deck), count)

    def estimate(
        self,
        hero: tuple[Card, ...],
        opponents: tuple[tuple[Card, ...], ...],
        board: tuple[Card, ...] = (),
    ) -> EquityResult:
        if len(hero) != 2:
            raise ValueError("hero must have exactly 2 hole cards")
        for op in opponents:
            if len(op) != 2:
                raise ValueError("each opponent must have exactly 2 hole cards")
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("board must have 0, 3, 4, or 5 cards")

        known = list(hero) + [c for op in opponents for c in op] + list(board)
        if len(set(known)) != len(known):
            raise ValueError("duplicate card in hero/opponents/board")

        need = _MAX_BOARD - len(board)
        deck = remaining_deck(tuple(known))
        rng = random.Random(self._seed)

        win = tie = loss = 0
        share_sum = 0.0
        for _ in range(self._trials):
            board_rest = self._sample(rng, deck, need)
            full_board = tuple(board) + tuple(board_rest)
            hero_hand = list(hero) + list(full_board)
            hero_wins_all = True
            tie_count = 0
            for op in opponents:
                r = compare(hero_hand, list(op) + list(full_board))
                if r == -1:
                    hero_wins_all = False
                    break
                if r == 0:
                    tie_count += 1
            if not hero_wins_all:
                loss += 1
            elif tie_count > 0:
                tie += 1
                share_sum += 1.0 / (tie_count + 1)
            else:
                win += 1
                share_sum += 1.0

        total = self._trials
        equity = share_sum / total
        return EquityResult(
            win=win / total,
            tie=tie / total,
            loss=loss / total,
            equity=equity,
            samples=total,
        )

    def estimate_range(
        self,
        hero: tuple[Card, ...],
        opponent_ranges: tuple[Range, ...],
        board: tuple[Card, ...] = (),
    ) -> EquityResult:
        """Monte Carlo hero equity against opponent RANGES.

        Each trial draws one concrete holding per opponent range (uniform over
        the range's holdings), then completes the board randomly and evaluates
        once. Averaged over ``trials``, this converges to the exact range
        enumeration.
        """
        if len(hero) != 2:
            raise ValueError("hero must have exactly 2 hole cards")
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("board must have 0, 3, 4, or 5 cards")
        for i, rng in enumerate(opponent_ranges):
            _validate_range(rng, f"opponent[{i}]")

        rng = random.Random(self._seed)
        win = tie = loss = 0
        share_sum = 0.0

        for _ in range(self._trials):
            # draw one concrete holding per opponent range (uniform, with
            # replacement so ranges with duplicate holdings weight accordingly).
            holdings = tuple(rng.choice(list(rng_)) for rng_ in opponent_ranges)
            opp_cards = [c for holding in holdings for c in holding]
            known = list(hero) + opp_cards + list(board)

            # If the drawn holdings clash (with each other, hero, or board),
            # the deal is impossible; redraw the whole trial until valid.
            attempts = 0
            while len(set(known)) != len(known) and attempts < 1000:
                holdings = tuple(rng.choice(list(rng_)) for rng_ in opponent_ranges)
                opp_cards = [c for holding in holdings for c in holding]
                known = list(hero) + opp_cards + list(board)
                attempts += 1
            if len(set(known)) != len(known):
                # degenerate ranges that can never form a legal deal — skip.
                continue

            need = _MAX_BOARD - len(board)
            deck = remaining_deck(tuple(known))
            board_rest = self._sample(rng, deck, need)
            full_board = tuple(board) + tuple(board_rest)
            hero_hand = list(hero) + list(full_board)

            lost = False
            tie_count = 0
            for holding in holdings:
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

        total = win + tie + loss
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


__all__ = ["MonteCarloEquity"]
