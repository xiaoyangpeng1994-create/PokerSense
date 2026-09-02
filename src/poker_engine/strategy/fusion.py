"""Baseline-preserving Decision Fusion and optional opponent adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from poker_engine.core._freeze import freeze_mapping
from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipDelta

from .advice import Advice, build_advice
from .contracts import DecisionContext
from .exploit_fusion import (
    ExploitAdjustment,
    ExploitAdjustmentPolicy,
    adjust_for_opponent,
)
from .provider import LookupState, MatchKind, StrategyCandidate
from .router import RouteResult
from .safety import GateResult


@dataclass(frozen=True)
class OpponentAdjustmentInput:
    q_values: Mapping[ActionType, ChipDelta]
    profile_quality: Decimal
    sample_size: int
    policy: ExploitAdjustmentPolicy = ExploitAdjustmentPolicy()

    def __post_init__(self) -> None:
        values = dict(self.q_values)
        if not values or not all(
            isinstance(action, ActionType) and isinstance(value, ChipDelta)
            for action, value in values.items()
        ):
            raise TypeError("q_values must map ActionType to ChipDelta")
        object.__setattr__(self, "q_values", freeze_mapping(values))
        if not isinstance(self.profile_quality, Decimal):
            raise TypeError("profile_quality must be a Decimal")
        if not self.profile_quality.is_finite() or not (
            Decimal("0") <= self.profile_quality <= Decimal("1")
        ):
            raise ValueError("profile_quality must be finite and in [0, 1]")
        if not isinstance(self.sample_size, int) or isinstance(
            self.sample_size, bool
        ):
            raise TypeError("sample_size must be an int")
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if not isinstance(self.policy, ExploitAdjustmentPolicy):
            raise TypeError("policy must be an ExploitAdjustmentPolicy")


@dataclass(frozen=True)
class FusionOutcome:
    advice: Advice
    baseline: StrategyCandidate | None
    selected: StrategyCandidate | None
    adjustment: ExploitAdjustment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.advice, Advice):
            raise TypeError("advice must be Advice")
        for name in ("baseline", "selected"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, StrategyCandidate):
                raise TypeError(f"{name} must be StrategyCandidate or None")
        if self.adjustment is not None and not isinstance(
            self.adjustment, ExploitAdjustment
        ):
            raise TypeError("adjustment must be ExploitAdjustment or None")
        if self.baseline is None and self.selected is not None:
            raise ValueError("selected candidate requires a baseline")
        if self.adjustment is not None:
            if self.baseline is None or self.selected is not self.adjustment.candidate:
                raise ValueError("adjustment candidate must be the selected candidate")


class DecisionFusion:
    """Keep one routed baseline; optionally apply one auditable adjustment."""

    def fuse(
        self,
        context: DecisionContext,
        route: RouteResult,
        *,
        opponent: OpponentAdjustmentInput | None = None,
        math_report: Mapping[str, Any] | None = None,
        hard_gates: tuple[GateResult, ...] = (),
        now: datetime | None = None,
    ) -> FusionOutcome:
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be DecisionContext")
        if not isinstance(route, RouteResult):
            raise TypeError("route must be RouteResult")
        if opponent is not None and not isinstance(
            opponent, OpponentAdjustmentInput
        ):
            raise TypeError("opponent must be OpponentAdjustmentInput or None")
        baseline = route.selected
        if baseline is not None and (
            baseline.hand_id != context.hand_id
            or baseline.state_version != context.state_version
            or baseline.request_id != context.request_id
        ):
            raise ValueError("baseline_context_mismatch")
        if baseline is None or opponent is None:
            return FusionOutcome(
                build_advice(
                    context,
                    route,
                    math_report=math_report,
                    hard_gates=hard_gates,
                    now=now,
                ),
                baseline,
                baseline,
            )
        adjustment = adjust_for_opponent(
            baseline,
            opponent.q_values,
            profile_quality=opponent.profile_quality,
            sample_size=opponent.sample_size,
            policy=opponent.policy,
        )
        selected = adjustment.candidate
        fused_route = RouteResult(
            LookupState.HIT_EXACT
            if selected.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE,
            selected,
            route.provider_results,
            route.reasons,
        )
        advice = build_advice(
            context,
            fused_route,
            math_report=math_report,
            hard_gates=hard_gates,
            now=now,
        )
        return FusionOutcome(advice, baseline, selected, adjustment)


__all__ = ["DecisionFusion", "FusionOutcome", "OpponentAdjustmentInput"]
