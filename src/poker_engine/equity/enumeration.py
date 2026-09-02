"""Exact equity via full enumeration of remaining board cards.

Known hole cards (hero + every opponent) and a partial board are completed by
enumerating ALL possible remaining board cards; each completion is evaluated
once with the shared hand evaluator. This is the exact reference against which
Monte Carlo is validated.
"""

from __future__ import annotations

from itertools import combinations

from poker_engine.core.value_objects import Card

from ._deck import remaining_deck
from .calculator import EquityEstimator, EquityResult
from .evaluator import compare

# Number of board cards by street (for validation).
_MAX_BOARD = 5


class EnumerationEquity(EquityEstimator):
    """Exact equity estimator (brute-force board completion)."""

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

        # all known cards must be distinct
        known = list(hero) + [c for op in opponents for c in op] + list(board)
        if len(set(known)) != len(known):
            raise ValueError("duplicate card in hero/opponents/board")

        need = _MAX_BOARD - len(board)
        deck = remaining_deck(tuple(known))

        win = tie = loss = 0
        share_sum = 0.0
        for board_rest in combinations(deck, need):
            full_board = tuple(board) + tuple(board_rest)
            hero_hand = list(hero) + list(full_board)
            # compare against every opponent: hero loses if ANY opponent beats
            # it; otherwise it wins outright or ties with a subset.
            lost = False
            tie_count = 0  # opponents beating hero? no; tying with hero.
            for op in opponents:
                r = compare(hero_hand, list(op) + list(full_board))
                if r == -1:
                    lost = True
                    break
                if r == 0:
                    tie_count += 1
            if lost:
                loss += 1
            elif tie_count > 0:
                # hero splits the pot with the tie_count opponents that tied
                # plus itself: 1 / (tie_count + 1) share.
                tie += 1
                share_sum += 1.0 / (tie_count + 1)
            else:
                win += 1
                share_sum += 1.0

        total = win + tie + loss
        equity = share_sum / total if total else 0.0
        return EquityResult(
            win=win / total if total else 0.0,
            tie=tie / total if total else 0.0,
            loss=loss / total if total else 0.0,
            equity=equity,
            samples=total,
        )


__all__ = ["EnumerationEquity"]
