"""Equity Engine — win/tie/loss estimation and pot-odds (Task 9)."""

from .calculator import EquityEstimator, EquityResult
from .enumeration import EnumerationEquity
from .evaluator import compare, evaluate
from .montecarlo import MonteCarloEquity
from .pot_odds import PotOdds, equity_call_is_profitable, pot_odds
from .range import Range, enumeration_range_equity

__all__ = [
    "EquityResult",
    "EquityEstimator",
    "EnumerationEquity",
    "MonteCarloEquity",
    "Range",
    "enumeration_range_equity",
    "evaluate",
    "compare",
    "PotOdds",
    "pot_odds",
    "equity_call_is_profitable",
]
