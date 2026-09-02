"""Fast Advice plus version-safe asynchronous strategy refinement."""

from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from poker_engine.core._freeze import _require_aware_dt, utc_now

from .advice import Advice, AdviceStatus
from .contracts import DecisionContext
from .fusion import DecisionFusion, OpponentAdjustmentInput
from .provider import (
    LookupState,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyProvider,
)
from .router import RouteResult, StrategyRouteProvider
from .safety import GateResult


class RefinementState(str, Enum):
    PENDING = "PENDING"
    APPLIED = "APPLIED"
    NO_UPDATE = "NO_UPDATE"
    DISCARDED = "DISCARDED"
    FAILED = "FAILED"


@runtime_checkable
class SlowResolver(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def source_version(self) -> str: ...

    @property
    def capability(self) -> ProviderCapability: ...

    def submit(self, context: DecisionContext) -> Future[ProviderResult]: ...


class ThreadedSlowResolver:
    """Run an ordinary Provider on a caller-owned executor."""

    def __init__(self, provider: StrategyProvider, executor: Executor) -> None:
        if not isinstance(provider, StrategyProvider):
            raise TypeError("provider must implement StrategyProvider")
        if not isinstance(executor, Executor):
            raise TypeError("executor must be an Executor")
        self._provider = provider
        self._executor = executor

    @property
    def provider_id(self) -> str:
        return self._provider.provider_id

    @property
    def source_version(self) -> str:
        return self._provider.source_version

    @property
    def capability(self) -> ProviderCapability:
        return self._provider.capability

    def submit(self, context: DecisionContext) -> Future[ProviderResult]:
        return self._executor.submit(self._provider.query, context)


@dataclass(frozen=True)
class SlowHandle:
    hand_id: str
    state_version: int
    request_id: str
    provider_id: str
    provider_version: str
    base_match_kind: MatchKind | None
    future: Future[ProviderResult]


@dataclass(frozen=True)
class StrategyCycle:
    fast_advice: Advice
    slow_handle: SlowHandle | None


@dataclass(frozen=True)
class SlowRefinement:
    state: RefinementState
    advice: Advice | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, RefinementState):
            raise TypeError("state must be a RefinementState")
        if self.advice is not None and not isinstance(self.advice, Advice):
            raise TypeError("advice must be an Advice or None")
        if (self.state is RefinementState.APPLIED) != (self.advice is not None):
            raise ValueError("only APPLIED refinement can carry Advice")
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise TypeError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)


class StrategyOrchestrator:
    """Return Fast Advice immediately and optionally schedule a Slow source."""

    def __init__(
        self,
        fast_router: StrategyRouteProvider,
        slow_resolver: SlowResolver | None = None,
        fusion: DecisionFusion | None = None,
    ) -> None:
        if not isinstance(fast_router, StrategyRouteProvider):
            raise TypeError("fast_router must implement StrategyRouteProvider")
        if slow_resolver is not None and not isinstance(
            slow_resolver, SlowResolver
        ):
            raise TypeError("slow_resolver must implement SlowResolver")
        self._fast_router = fast_router
        self._slow_resolver = slow_resolver
        self._fusion = fusion or DecisionFusion()

    def request(
        self,
        context: DecisionContext,
        *,
        math_report: Mapping[str, Any] | None = None,
        opponent_adjustment: OpponentAdjustmentInput | None = None,
        hard_gates: tuple[GateResult, ...] = (),
        now: datetime | None = None,
    ) -> StrategyCycle:
        now = now or utc_now()
        _require_aware_dt(now)
        route = self._fast_router.route(context, now=now)
        advice = self._fusion.fuse(
            context,
            route,
            opponent=opponent_adjustment,
            math_report=math_report,
            hard_gates=hard_gates,
            now=now,
        ).advice
        resolver = self._slow_resolver
        if (
            resolver is None
            or advice.status in (AdviceStatus.STALE, AdviceStatus.ABSTAIN)
            and not context.is_decision_ready
            or route.selected is not None
            and route.selected.match_kind is MatchKind.EXACT
            or not resolver.capability.match(context).applicable
            or context.request.is_expired(now)
        ):
            return StrategyCycle(advice, None)
        future = resolver.submit(context)
        if not isinstance(future, Future):
            raise TypeError("slow resolver submit must return Future")
        return StrategyCycle(advice, SlowHandle(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=resolver.provider_id,
            provider_version=resolver.source_version,
            base_match_kind=(
                route.selected.match_kind if route.selected is not None else None
            ),
            future=future,
        ))

    def collect(
        self,
        handle: SlowHandle,
        current_context: DecisionContext,
        *,
        math_report: Mapping[str, Any] | None = None,
        opponent_adjustment: OpponentAdjustmentInput | None = None,
        hard_gates: tuple[GateResult, ...] = (),
        now: datetime | None = None,
    ) -> SlowRefinement:
        if not isinstance(handle, SlowHandle):
            raise TypeError("handle must be a SlowHandle")
        if not isinstance(current_context, DecisionContext):
            raise TypeError("current_context must be a DecisionContext")
        now = now or utc_now()
        _require_aware_dt(now)
        identity = (
            current_context.hand_id,
            current_context.state_version,
            current_context.request_id,
        )
        if identity != (handle.hand_id, handle.state_version, handle.request_id):
            handle.future.cancel()
            return SlowRefinement(
                RefinementState.DISCARDED, reasons=("stale_context",)
            )
        if current_context.request.is_expired(now):
            handle.future.cancel()
            return SlowRefinement(
                RefinementState.DISCARDED, reasons=("expired_request",)
            )
        if not handle.future.done():
            return SlowRefinement(RefinementState.PENDING)
        try:
            result = handle.future.result()
        except Exception as exc:
            return SlowRefinement(
                RefinementState.FAILED,
                reasons=(f"slow_resolver_error:{type(exc).__name__}",),
            )
        rejection = _validate_slow_result(handle, current_context, result, now)
        if rejection is not None:
            return rejection
        candidate = result.candidate
        if candidate is None:
            return SlowRefinement(
                RefinementState.NO_UPDATE,
                reasons=result.reasons or ("slow_no_strategy",),
            )
        if _match_rank(candidate.match_kind) <= _match_rank(
            handle.base_match_kind
        ):
            return SlowRefinement(
                RefinementState.NO_UPDATE,
                reasons=("slow_result_not_better",),
            )
        route = RouteResult(result.state, candidate, (result,))
        advice = self._fusion.fuse(
            current_context,
            route,
            opponent=opponent_adjustment,
            math_report=math_report,
            hard_gates=hard_gates,
            now=now,
        ).advice
        if advice.status is not AdviceStatus.READY:
            return SlowRefinement(
                RefinementState.NO_UPDATE,
                reasons=advice.rejection_reasons or ("slow_advice_not_ready",),
            )
        return SlowRefinement(RefinementState.APPLIED, advice=advice)


def _validate_slow_result(
    handle: SlowHandle,
    context: DecisionContext,
    result: object,
    now: datetime,
) -> SlowRefinement | None:
    if not isinstance(result, ProviderResult):
        return SlowRefinement(
            RefinementState.FAILED, reasons=("invalid_provider_result_type",)
        )
    if result.provider_id != handle.provider_id:
        return SlowRefinement(
            RefinementState.FAILED, reasons=("provider_id_mismatch",)
        )
    candidate = result.candidate
    if candidate is None:
        return None
    if candidate.provider_id != handle.provider_id:
        return SlowRefinement(
            RefinementState.FAILED, reasons=("candidate_provider_id_mismatch",)
        )
    if candidate.provider_version != handle.provider_version:
        return SlowRefinement(
            RefinementState.FAILED,
            reasons=("candidate_provider_version_mismatch",),
        )
    if (
        candidate.hand_id != context.hand_id
        or candidate.state_version != context.state_version
        or candidate.request_id != context.request_id
    ):
        return SlowRefinement(
            RefinementState.DISCARDED, reasons=("stale_candidate_context",)
        )
    if candidate.expires_at is not None and now >= candidate.expires_at:
        return SlowRefinement(
            RefinementState.DISCARDED, reasons=("expired_candidate",)
        )
    expected = (
        LookupState.HIT_EXACT
        if candidate.match_kind is MatchKind.EXACT
        else LookupState.HIT_APPROXIMATE
    )
    if result.state is not expected:
        return SlowRefinement(
            RefinementState.FAILED,
            reasons=("lookup_state_match_kind_mismatch",),
        )
    return None


def _match_rank(value: MatchKind | None) -> int:
    return {
        None: 0,
        MatchKind.EQUITY_ONLY: 0,
        MatchKind.HEURISTIC: 1,
        MatchKind.INTERPOLATED: 2,
        MatchKind.EXACT: 3,
    }[value]


__all__ = [
    "RefinementState",
    "SlowHandle",
    "SlowRefinement",
    "SlowResolver",
    "StrategyCycle",
    "StrategyOrchestrator",
    "ThreadedSlowResolver",
]
