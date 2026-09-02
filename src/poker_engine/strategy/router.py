"""Capability-safe Strategy Provider routing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from poker_engine.core._freeze import _require_aware_dt, utc_now

from .contracts import DecisionContext
from .provider import (
    CapabilityMatch,
    LookupState,
    MatchKind,
    ProviderResult,
    StrategyCandidate,
    StrategyProvider,
)


@dataclass(frozen=True)
class RouteResult:
    state: LookupState
    selected: StrategyCandidate | None
    provider_results: tuple[ProviderResult, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, LookupState):
            raise TypeError("state must be a LookupState")
        if self.selected is not None and not isinstance(
            self.selected, StrategyCandidate
        ):
            raise TypeError("selected must be a StrategyCandidate or None")
        hit = self.state in (LookupState.HIT_EXACT, LookupState.HIT_APPROXIMATE)
        if hit != (self.selected is not None):
            raise ValueError("hit state and selected candidate must agree")
        results = tuple(self.provider_results)
        if not all(isinstance(item, ProviderResult) for item in results):
            raise TypeError("provider_results must contain ProviderResult values")
        object.__setattr__(self, "provider_results", results)
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise TypeError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)


@runtime_checkable
class StrategyRouteProvider(Protocol):
    def route(
        self,
        context: DecisionContext,
        *,
        now: datetime | None = None,
    ) -> RouteResult: ...


class FastSourceLayer(str, Enum):
    CACHE = "cache"
    PREFLOP_DB = "preflop_db"
    PRESOLVED = "presolved"
    MODEL = "model"


class StrategyRouter:
    def __init__(self, providers: tuple[StrategyProvider, ...] = ()) -> None:
        self._providers: dict[str, StrategyProvider] = {}
        for provider in providers:
            self.register(provider)

    @property
    def providers(self) -> tuple[StrategyProvider, ...]:
        return tuple(
            sorted(self._providers.values(), key=lambda item: item.provider_id)
        )

    def register(self, provider: StrategyProvider) -> None:
        if not isinstance(provider, StrategyProvider):
            raise TypeError("provider must implement StrategyProvider")
        if not provider.provider_id:
            raise ValueError("provider_id must be non-empty")
        if not provider.source_version:
            raise ValueError("source_version must be non-empty")
        if provider.provider_id in self._providers:
            existing = self._providers[provider.provider_id]
            raise ValueError(
                "duplicate provider_id "
                f"{provider.provider_id!r}: {existing.source_version!r} vs "
                f"{provider.source_version!r}"
            )
        self._providers[provider.provider_id] = provider

    def route(
        self,
        context: DecisionContext,
        *,
        now: datetime | None = None,
    ) -> RouteResult:
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be a DecisionContext")
        now = now or utc_now()
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        _require_aware_dt(now)
        if not context.is_decision_ready:
            reasons = list(context.missing_fields)
            reasons.extend(context.input_quality.hard_failures)
            if context.actor_seat != context.hero_seat:
                reasons.append("hero_not_actor")
            return RouteResult(
                LookupState.NO_STRATEGY,
                None,
                (),
                tuple(dict.fromkeys(reasons or ("context_not_ready",))),
            )

        results = []
        candidates = []
        for provider in self.providers:
            match = provider.capability.match(context)
            if not match.applicable:
                results.append(ProviderResult(
                    LookupState.NOT_APPLICABLE,
                    provider.provider_id,
                    reasons=match.reasons,
                ))
                continue
            try:
                result = provider.query(context)
            except Exception as exc:
                results.append(ProviderResult(
                    LookupState.REJECTED,
                    provider.provider_id,
                    reasons=(f"provider_error:{type(exc).__name__}",),
                ))
                continue
            rejection = self._validate_result(
                provider, context, result, match, now
            )
            if rejection is not None:
                results.append(rejection)
                continue
            normalized = _apply_capability_match(result, match)
            results.append(normalized)
            if normalized.candidate is not None:
                candidates.append((provider, normalized.candidate))

        if not candidates:
            reasons = tuple(
                dict.fromkeys(
                    reason for result in results for reason in result.reasons
                )
            )
            return RouteResult(
                LookupState.NO_STRATEGY,
                None,
                tuple(results),
                reasons or ("no_strategy",),
            )

        selected_provider, selected = max(
            candidates,
            key=lambda item: (
                _match_rank(item[1].match_kind),
                item[1].state_match_score,
                -item[0].capability.priority,
                item[0].provider_id,
            ),
        )
        del selected_provider
        state = (
            LookupState.HIT_EXACT
            if selected.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE
        )
        return RouteResult(state, selected, tuple(results))

    @staticmethod
    def _validate_result(
        provider: StrategyProvider,
        context: DecisionContext,
        result: object,
        capability_match: CapabilityMatch,
        now: datetime,
    ) -> ProviderResult | None:
        if not isinstance(result, ProviderResult):
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("invalid_provider_result_type",),
            )
        if result.provider_id != provider.provider_id:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("provider_id_mismatch",),
            )
        candidate = result.candidate
        if candidate is None:
            return None
        if candidate.provider_id != provider.provider_id:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("candidate_provider_id_mismatch",),
            )
        if candidate.provider_version != provider.source_version:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("candidate_provider_version_mismatch",),
            )
        if (
            candidate.hand_id != context.hand_id
            or candidate.state_version != context.state_version
            or candidate.request_id != context.request_id
        ):
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("stale_candidate_context",),
            )
        if candidate.expires_at is not None and now >= candidate.expires_at:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("expired_candidate",),
            )
        if capability_match.match_kind is MatchKind.INTERPOLATED and (
            candidate.match_kind is MatchKind.EXACT
        ):
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("candidate_overstates_match",),
            )
        if candidate.state_match_score > capability_match.score:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("candidate_overstates_match_score",),
            )
        expected_state = (
            LookupState.HIT_EXACT
            if candidate.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE
        )
        if result.state is not expected_state:
            return ProviderResult(
                LookupState.REJECTED,
                provider.provider_id,
                reasons=("lookup_state_match_kind_mismatch",),
            )
        return None


def _apply_capability_match(
    result: ProviderResult,
    match: CapabilityMatch,
) -> ProviderResult:
    candidate = result.candidate
    if candidate is None or not match.dimensions:
        return result
    dimensions = {item.name: item for item in candidate.match_dimensions}
    for item in match.dimensions:
        dimensions.setdefault(item.name, item)
    normalized = replace(
        candidate,
        match_dimensions=tuple(
            dimensions[name] for name in sorted(dimensions)
        ),
    )
    return replace(result, candidate=normalized)


def _match_rank(match_kind: MatchKind) -> int:
    return {
        MatchKind.EXACT: 3,
        MatchKind.INTERPOLATED: 2,
        MatchKind.HEURISTIC: 1,
        MatchKind.EQUITY_ONLY: 0,
    }[match_kind]


class TieredStrategyRouter:
    """Query Fast source layers in order and stop at the first usable tier."""

    def __init__(
        self,
        layers: Mapping[FastSourceLayer, tuple[StrategyProvider, ...]],
    ) -> None:
        if not isinstance(layers, Mapping) or not layers:
            raise ValueError("layers must be a non-empty mapping")
        routers = {}
        for layer, providers in layers.items():
            if not isinstance(layer, FastSourceLayer):
                raise TypeError("layer keys must be FastSourceLayer")
            providers = tuple(providers)
            if not providers:
                raise ValueError("configured source layer cannot be empty")
            routers[layer] = StrategyRouter(providers)
        self._routers = routers

    @property
    def layers(self) -> tuple[FastSourceLayer, ...]:
        return tuple(layer for layer in FastSourceLayer if layer in self._routers)

    def route(
        self,
        context: DecisionContext,
        *,
        now: datetime | None = None,
    ) -> RouteResult:
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be a DecisionContext")
        now = now or utc_now()
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        _require_aware_dt(now)
        results = []
        reasons = []
        for layer in self.layers:
            routed = self._routers[layer].route(context, now=now)
            results.extend(routed.provider_results)
            reasons.extend(routed.reasons)
            if routed.selected is not None:
                return RouteResult(
                    routed.state,
                    routed.selected,
                    tuple(results),
                )
        return RouteResult(
            LookupState.NO_STRATEGY,
            None,
            tuple(results),
            tuple(dict.fromkeys(reasons or ("no_strategy",))),
        )


__all__ = [
    "FastSourceLayer",
    "RouteResult",
    "StrategyRouteProvider",
    "StrategyRouter",
    "TieredStrategyRouter",
]
