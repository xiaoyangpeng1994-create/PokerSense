"""Pot odds / break-even equity helper (money-safe, Decimal only).

Pot odds compares the price of a call to the size of the pot you would win,
expressed as a required break-even equity (a fraction, not a money amount).
This is the decision-relevant number: call if your hand equity exceeds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from poker_engine.core.value_objects import ChipAmount


@dataclass(frozen=True)
class PotOdds:
    """Break-even equity required to justify a call.

    ``required_equity`` = call / (pot + call): the minimum hand equity (0..1)
    that makes calling profitable in the long run, ignoring future betting.
    """

    required_equity: float   # a ratio/probability, so float is acceptable here

    def __post_init__(self) -> None:
        if not (0.0 <= self.required_equity <= 1.0):
            raise ValueError("required_equity must be in [0,1]")


def pot_odds(pot: ChipAmount, call: ChipAmount) -> PotOdds:
    """Compute break-even equity for calling ``call`` into a ``pot``.

    Both amounts are non-negative ChipAmount (Decimal-backed). The ratio is a
    probability (float) — it is NOT a money amount, so float is safe.
    """
    if call.value == Decimal(0):  # a check (no call) has no cost
        return PotOdds(required_equity=0.0)
    num = call.value
    denom = pot.value + call.value
    if denom <= Decimal(0):
        raise ValueError("pot + call must be positive")
    return PotOdds(required_equity=float(num / denom))


def equity_call_is_profitable(equity: float, odds: PotOdds) -> bool:
    """True if hand equity strictly beats the break-even pot odds."""
    return equity > odds.required_equity


__all__ = ["PotOdds", "pot_odds", "equity_call_is_profitable"]
