"""Report contracts: EquityReport, StrategyReport, ReasoningReport, Decision.

These are the reasoning-layer output contracts. They are pure data — no
computation lives here. Serialization is deferred to Task 1E.

``ReasoningReport`` intentionally does NOT store a chain-of-thought; it keeps
only an auditable structured summary (see architecture v0.2.1 §5.11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from ._freeze import _require_aware_dt, freeze_mapping, utc_now
from .enums import ActionType
from .value_objects import ChipAmount


class EquityMethod(str, Enum):
    """How equity was computed."""

    ENUMERATION = "enumeration"
    MONTECARLO = "montecarlo"


class StrategySource(str, Enum):
    """Which strategy tier produced a StrategyReport."""

    CACHE = "cache"
    PREFLOP_DB = "preflop_db"
    PRECOMPUTED = "precomputed"
    SOLVER = "solver"


class DecisionPath(str, Enum):
    """Which pipeline produced a Decision."""

    FAST = "fast"
    SLOW = "slow"


def _check_confidence(value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("confidence must be a float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"confidence must be finite, got {value}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"confidence must be in [0.0, 1.0], got {value}")


def _check_ratio(value: Any, name: str) -> None:
    """Validate a float ratio in [0, 1]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


def _check_non_negative_float(value: Any, name: str) -> None:
    """Validate a finite float that is >= 0.

    Rejects bool/non-numeric with TypeError; rejects NaN/inf/negative with
    ValueError.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class EquityReport:
    """Equity / pot-odds result for the current state."""

    win_rate: float
    tie_rate: float
    pot_odds: float
    implied_odds: float
    estimated_ev: ChipAmount
    method: EquityMethod
    timestamp: datetime

    def __post_init__(self) -> None:
        _check_ratio(self.win_rate, "win_rate")
        _check_ratio(self.tie_rate, "tie_rate")
        _check_non_negative_float(self.pot_odds, "pot_odds")
        _check_non_negative_float(self.implied_odds, "implied_odds")
        if not isinstance(self.estimated_ev, ChipAmount):
            raise TypeError("estimated_ev must be a ChipAmount")
        if not isinstance(self.method, EquityMethod):
            raise TypeError("method must be an EquityMethod enum")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)


@dataclass(frozen=True)
class StrategyReport:
    """Strategy recommendation from one of the strategy tiers."""

    action_frequencies: Mapping[ActionType, float] = field(default_factory=dict)
    bet_sizes: tuple[ChipAmount, ...] = ()
    ev: ChipAmount = field(default_factory=lambda: ChipAmount("0"))
    strategy_source: StrategySource = StrategySource.CACHE
    confidence: float = 1.0
    cache_hit: bool = False
    solver_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        af = freeze_mapping(self.action_frequencies)
        # Validate keys/values of action frequencies.
        for k, v in af.items():
            if not isinstance(k, ActionType):
                raise TypeError("action_frequencies keys must be ActionType")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise TypeError("action_frequencies values must be float")
            if not (0.0 <= v <= 1.0):
                raise ValueError("action_frequencies values must be in [0, 1]")
        object.__setattr__(self, "action_frequencies", af)

        bet_sizes = tuple(self.bet_sizes)
        if not all(isinstance(b, ChipAmount) for b in bet_sizes):
            raise TypeError("bet_sizes must be ChipAmount instances")
        object.__setattr__(self, "bet_sizes", bet_sizes)

        if not isinstance(self.ev, ChipAmount):
            raise TypeError("ev must be a ChipAmount")
        if not isinstance(self.strategy_source, StrategySource):
            raise TypeError("strategy_source must be a StrategySource enum")
        _check_confidence(self.confidence)
        if not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit must be a bool")
        object.__setattr__(
            self, "solver_metadata", freeze_mapping(self.solver_metadata)
        )


@dataclass(frozen=True)
class ReasoningReport:
    """Poker Reasoning Layer output (auditable structured summary only)."""

    analysis_summary: str
    key_factors: tuple[str, ...]
    suggested_action: ActionType
    suggested_size: ChipAmount
    confidence: float
    source: str
    hand_id: str
    request_id: str
    model_metadata: Mapping[str, Any] = field(default_factory=dict)
    state_version: int = 0
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_summary, str):
            raise TypeError("analysis_summary must be a str")
        key_factors = tuple(self.key_factors)
        if not all(isinstance(k, str) for k in key_factors):
            raise TypeError("key_factors items must be str")
        object.__setattr__(self, "key_factors", key_factors)
        if not isinstance(self.suggested_action, ActionType):
            raise TypeError("suggested_action must be an ActionType")
        if not isinstance(self.suggested_size, ChipAmount):
            raise TypeError("suggested_size must be a ChipAmount")
        _check_confidence(self.confidence)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty str")
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty str")
        object.__setattr__(self, "model_metadata", freeze_mapping(self.model_metadata))
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)


@dataclass(frozen=True)
class Decision:
    """Final decision from the Decision Engine (the single exit point)."""

    action: ActionType
    confidence: float
    evidence_chain: tuple[str, ...] = ()
    raise_size: ChipAmount | None = None
    fast_or_slow: DecisionPath = DecisionPath.FAST
    timestamp: datetime = field(default_factory=utc_now)
    state_version: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionType):
            raise TypeError("action must be an ActionType")
        _check_confidence(self.confidence)
        evidence = tuple(self.evidence_chain)
        if not all(isinstance(e, str) for e in evidence):
            raise TypeError("evidence_chain items must be str")
        object.__setattr__(self, "evidence_chain", evidence)
        if self.raise_size is not None and not isinstance(
            self.raise_size, ChipAmount
        ):
            raise TypeError("raise_size must be a ChipAmount or None")
        if not isinstance(self.fast_or_slow, DecisionPath):
            raise TypeError("fast_or_slow must be a DecisionPath enum")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")


__all__ = [
    "EquityMethod",
    "StrategySource",
    "DecisionPath",
    "EquityReport",
    "StrategyReport",
    "ReasoningReport",
    "Decision",
]
