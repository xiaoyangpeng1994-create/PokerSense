"""Deadline-aware exact/Monte Carlo multi-player equity orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from math import comb
import time
from typing import Callable

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.errors import InvalidStateError

from .contracts import DecisionContext
from .equity_cache import (
    EquityCache,
    EquityCacheQuery,
    EquityCacheState,
    EquityMethod,
)
from .multiway_equity import (
    MultiwayEquityResult,
    exact_multiway_pot_share,
    monte_carlo_multiway_pot_share,
    monte_carlo_multiway_ranges,
)
from .range_tracker import enumerate_joint_assignments


class EquityComputationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class AdaptiveEquityPolicy:
    engine_version: str = "adaptive-equity-v2-m1-pro"
    default_deadline_ms: int = 300
    exact_outcome_limit: int = 100_000
    exact_outcomes_per_ms: int = 3
    minimum_mc_trials: int = 2_000
    maximum_mc_trials: int = 50_000
    mc_trials_per_ms: int = 2
    target_half_width: Decimal = Decimal("0.02")
    joint_assignment_limit: int = 100_000
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version must be a non-empty str")
        for name in (
            "default_deadline_ms",
            "exact_outcome_limit",
            "exact_outcomes_per_ms",
            "minimum_mc_trials",
            "maximum_mc_trials",
            "mc_trials_per_ms",
            "joint_assignment_limit",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if name == "exact_outcome_limit":
                if value < 0:
                    raise ValueError("exact_outcome_limit must be >= 0")
            elif value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.minimum_mc_trials > self.maximum_mc_trials:
            raise ValueError("minimum_mc_trials cannot exceed maximum_mc_trials")
        if not isinstance(self.target_half_width, Decimal):
            raise TypeError("target_half_width must be a Decimal")
        if not self.target_half_width.is_finite() or not (
            Decimal("0") < self.target_half_width < Decimal("1")
        ):
            raise ValueError("target_half_width must be in (0, 1)")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")


@dataclass(frozen=True)
class AdaptiveEquityReport:
    status: EquityComputationStatus
    method: EquityMethod
    result: MultiwayEquityResult
    estimated_outcomes: int
    trials: int
    confidence_low: Decimal
    confidence_high: Decimal
    confidence_level: Decimal
    numerical_confidence: Decimal
    cache_state: EquityCacheState
    evidence: tuple[str, ...]
    expires_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, EquityComputationStatus):
            raise TypeError("status must be an EquityComputationStatus")
        if not isinstance(self.method, EquityMethod):
            raise TypeError("method must be an EquityMethod")
        if not isinstance(self.result, MultiwayEquityResult):
            raise TypeError("result must be a MultiwayEquityResult")
        for name in ("estimated_outcomes", "trials"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in (
            "confidence_low",
            "confidence_high",
            "confidence_level",
            "numerical_confidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TypeError(f"{name} must be a finite Decimal")
        if not Decimal("0") <= self.confidence_low <= self.confidence_high <= 1:
            raise ValueError("confidence interval must be in [0, 1]")
        if not Decimal("0") < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not Decimal("0") <= self.numerical_confidence <= 1:
            raise ValueError("numerical_confidence must be in [0, 1]")
        if not isinstance(self.cache_state, EquityCacheState):
            raise TypeError("cache_state must be an EquityCacheState")
        evidence = tuple(self.evidence)
        if not evidence or not all(isinstance(item, str) and item for item in evidence):
            raise ValueError("evidence must contain non-empty references")
        object.__setattr__(self, "evidence", evidence)
        if self.expires_at is not None:
            if not isinstance(self.expires_at, datetime):
                raise TypeError("expires_at must be a datetime or None")
            _require_aware_dt(self.expires_at)


def calculate_adaptive_equity(
    context: DecisionContext,
    *,
    now: datetime,
    policy: AdaptiveEquityPolicy | None = None,
    cache: EquityCache | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> AdaptiveEquityReport:
    """Select exact or seeded MC from context size and the request deadline."""
    if not isinstance(context, DecisionContext):
        raise TypeError("context must be a DecisionContext")
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime")
    _require_aware_dt(now)
    policy = policy or AdaptiveEquityPolicy()
    if not isinstance(policy, AdaptiveEquityPolicy):
        raise TypeError("policy must be an AdaptiveEquityPolicy")
    if cache is not None and not isinstance(cache, EquityCache):
        raise TypeError("cache must be an EquityCache or None")
    if not callable(monotonic_clock):
        raise TypeError("monotonic_clock must be callable")
    started_at = monotonic_clock()
    if context.request.expires_at is not None and now >= context.request.expires_at:
        raise InvalidStateError("equity_deadline_expired")
    if len(context.hero_cards) != 2:
        raise InvalidStateError("hero_cards_missing")
    if not context.villain_ranges:
        raise InvalidStateError("villain_ranges_missing")
    cartesian_size = 1
    for distribution in context.villain_ranges:
        cartesian_size *= len(distribution.combo_weights)
    assignments = None
    if cartesian_size <= policy.joint_assignment_limit:
        assignments = enumerate_joint_assignments(
            context.villain_ranges,
            context.hero_cards + context.board_cards,
            max_combinations=policy.joint_assignment_limit,
        )
    board_needed = 5 - len(context.board_cards)
    remaining = (
        52
        - len(context.hero_cards)
        - len(context.board_cards)
        - 2 * len(context.villain_ranges)
    )
    runouts = comb(remaining, board_needed)
    estimated_assignments = (
        len(assignments) if assignments is not None else cartesian_size
    )
    estimated_outcomes = estimated_assignments * runouts
    deadline_ms = context.request.deadline_ms or policy.default_deadline_ms
    wall_deadline = started_at + deadline_ms / 1000
    exact_budget = min(
        policy.exact_outcome_limit,
        deadline_ms * policy.exact_outcomes_per_ms,
    )
    if assignments is not None and estimated_outcomes <= exact_budget:
        method = EquityMethod.EXACT
        trials = None
        seed = None
    else:
        method = EquityMethod.MONTE_CARLO
        trials = max(
            1,
            min(
                policy.maximum_mc_trials,
                deadline_ms * policy.mc_trials_per_ms,
            ),
        )
        seed = policy.seed
    query = EquityCacheQuery(
        hero_seat=context.hero_seat,
        hero_cards=context.hero_cards,
        board_cards=context.board_cards,
        villain_ranges=context.villain_ranges,
        pots=context.pots,
        method=method,
        engine_version=policy.engine_version,
        trials=trials,
        seed=seed,
    )
    lookup_state = EquityCacheState.NOT_FOUND
    if cache is not None:
        lookup = cache.get(query, now=now)
        lookup_state = lookup.state
        if lookup.state is EquityCacheState.HIT:
            return _report_from_cache(
                lookup.entry,
                method,
                estimated_outcomes,
                trials,
                policy,
                context.request.expires_at,
            )
    if method is EquityMethod.EXACT:
        result = exact_multiway_pot_share(
            context.hero_seat,
            context.hero_cards,
            assignments,
            context.board_cards,
            context.pots,
            max_outcomes=estimated_outcomes,
        )
        low = high = result.pot_equity
        numerical_confidence = Decimal("1")
        status = EquityComputationStatus.COMPLETE
        evidence = (f"equity://exact/{policy.engine_version}",)
    else:
        if assignments is None:
            sampled = monte_carlo_multiway_ranges(
                context.hero_seat,
                context.hero_cards,
                context.villain_ranges,
                context.board_cards,
                context.pots,
                trials=trials,
                seed=seed,
                deadline_at=wall_deadline,
                monotonic_clock=monotonic_clock,
            )
        else:
            sampled = monte_carlo_multiway_pot_share(
                context.hero_seat,
                context.hero_cards,
                assignments,
                context.board_cards,
                context.pots,
                trials=trials,
                seed=seed,
                deadline_at=wall_deadline,
                monotonic_clock=monotonic_clock,
            )
        result = sampled.result
        low = sampled.confidence_low
        high = sampled.confidence_high
        numerical_confidence = _numerical_confidence(
            result.samples,
            low,
            high,
            policy,
        )
        status = (
            EquityComputationStatus.COMPLETE
            if result.samples >= policy.minimum_mc_trials
            and (high - low) / 2 <= policy.target_half_width
            else EquityComputationStatus.PARTIAL
        )
        evidence = (
            f"equity://monte-carlo/{policy.engine_version}"
            f"?trials={result.samples}&planned={trials}&seed={seed}",
        )
    if cache is not None:
        cache.put(
            query,
            result,
            created_at=now,
            expires_at=context.request.expires_at,
            evidence=evidence,
            confidence_low=low,
            confidence_high=high,
            confidence_level=Decimal("0.95"),
            numerical_confidence=numerical_confidence,
        )
    return AdaptiveEquityReport(
        status=status,
        method=method,
        result=result,
        estimated_outcomes=estimated_outcomes,
        trials=result.samples,
        confidence_low=low,
        confidence_high=high,
        confidence_level=Decimal("0.95"),
        numerical_confidence=numerical_confidence,
        cache_state=lookup_state,
        evidence=evidence,
        expires_at=context.request.expires_at,
    )


def _report_from_cache(
    entry,
    method: EquityMethod,
    estimated_outcomes: int,
    trials: int | None,
    policy: AdaptiveEquityPolicy,
    expires_at: datetime | None,
) -> AdaptiveEquityReport:
    low = entry.confidence_low
    high = entry.confidence_high
    confidence = entry.numerical_confidence
    level = entry.confidence_level
    if any(value is None for value in (low, high, confidence, level)):
        if method is not EquityMethod.EXACT:
            raise InvalidStateError("cached_monte_carlo_metadata_missing")
        low = high = entry.result.pot_equity
        confidence = Decimal("1")
        level = Decimal("0.95")
    count = entry.result.samples
    status = (
        EquityComputationStatus.COMPLETE
        if method is EquityMethod.EXACT
        or (
            count >= policy.minimum_mc_trials
            and (high - low) / 2 <= policy.target_half_width
        )
        else EquityComputationStatus.PARTIAL
    )
    return AdaptiveEquityReport(
        status=status,
        method=method,
        result=entry.result,
        estimated_outcomes=estimated_outcomes,
        trials=entry.result.samples,
        confidence_low=low,
        confidence_high=high,
        confidence_level=level,
        numerical_confidence=confidence,
        cache_state=EquityCacheState.HIT,
        evidence=entry.evidence + (f"cache://equity/{entry.key}",),
        expires_at=expires_at,
    )


def _numerical_confidence(
    trials: int,
    low: Decimal,
    high: Decimal,
    policy: AdaptiveEquityPolicy,
) -> Decimal:
    sample_factor = min(
        Decimal("1"),
        Decimal(trials) / Decimal(policy.minimum_mc_trials),
    )
    half_width = (high - low) / 2
    precision_factor = (
        Decimal("1")
        if half_width == 0
        else min(Decimal("1"), policy.target_half_width / half_width)
    )
    return min(sample_factor, precision_factor)


__all__ = [
    "AdaptiveEquityPolicy",
    "AdaptiveEquityReport",
    "EquityComputationStatus",
    "calculate_adaptive_equity",
]
