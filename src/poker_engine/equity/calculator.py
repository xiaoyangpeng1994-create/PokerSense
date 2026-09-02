"""Equity calculation interface + result value object.

Pure domain logic: given hero hole cards, opponent hole cards, and the board,
estimate hero's share of wins/ties. No float money, no I/O, deterministic
(exact) or seeded-random (Monte Carlo) depending on the estimator.

This is the Task 9 grounding layer — it depends only on Core value objects,
never on Vision/Capture/State.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poker_engine.core.value_objects import Card


@dataclass(frozen=True)
class EquityResult:
    """Outcome of an equity evaluation for the hero.

    ``win``, ``tie``, ``loss`` are the fractions (0..1) of outcomes where the
    hero wins, ties, or loses against ALL opponents. ``equity`` is the hero's
    exact expected pot share: each outcome contributes 1.0 for an outright win,
    0.0 for a loss, and ``1 / (k + 1)`` for a tie split with the ``k`` opponents
    that tie with the hero (standard poker: a tie splits only among the players
    who tied, not the whole table). ``samples`` is the number of outcomes
    evaluated (exact count for enumeration, trials for Monte Carlo).
    """
    win: float
    tie: float
    loss: float
    equity: float
    samples: int

    def __post_init__(self) -> None:
        for name in ("win", "tie", "loss", "equity"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0 + 1e-9):
                raise ValueError(f"{name} must be in [0,1], got {v}")


class EquityEstimator(Protocol):
    """Computes hero equity against a set of opponents."""

    def estimate(
        self,
        hero: tuple[Card, ...],
        opponents: tuple[tuple[Card, ...], ...],
        board: tuple[Card, ...] = (),
    ) -> EquityResult:
        """Return hero equity given known hole cards and board cards."""
