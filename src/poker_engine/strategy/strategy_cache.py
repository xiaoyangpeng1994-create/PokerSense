"""Canonical, versioned LRU cache for identity-bound strategy candidates."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from threading import RLock
from typing import Callable

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from .contracts import DecisionContext
from .provider import (
    ActionOption,
    LookupState,
    MatchDimension,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
    StrategyProvider,
)
from .serialization import strategy_serialize


class StrategyCacheState(str, Enum):
    HIT = "HIT"
    STALE = "STALE"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True)
class StrategyCacheQuery:
    key: str
    context_digest: str
    provider_id: str
    provider_version: str
    provider_asset_id: str
    engine_version: str
    hand_id: str
    state_version: int
    request_id: str

    @classmethod
    def from_context(
        cls,
        context: DecisionContext,
        *,
        provider_id: str,
        provider_version: str,
        provider_asset_id: str,
        engine_version: str,
    ) -> "StrategyCacheQuery":
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be a DecisionContext")
        metadata = {
            "provider_id": _text(provider_id, "provider_id"),
            "provider_version": _text(provider_version, "provider_version"),
            "provider_asset_id": _text(provider_asset_id, "provider_asset_id"),
            "engine_version": _text(engine_version, "engine_version"),
        }
        context_digest = canonical_context_digest(context)
        key = _digest({**metadata, "context_digest": context_digest})
        return cls(
            key=key,
            context_digest=context_digest,
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            **metadata,
        )


@dataclass(frozen=True)
class StrategyTemplate:
    match_kind: MatchKind
    state_match_score: float
    match_dimensions: tuple[MatchDimension, ...]
    action_probabilities: tuple[tuple[ActionType, Decimal], ...]
    recommended_sizes: tuple[
        tuple[ActionType, tuple[ChipAmount, ...]], ...
    ]
    action_options: tuple[ActionOption, ...]
    action_ev: tuple[tuple[ActionType, ChipDelta], ...]
    confidence: float
    evidence: tuple[str, ...]
    assumptions: tuple[str, ...]
    produced_at: datetime | None

    @classmethod
    def from_candidate(cls, value: StrategyCandidate) -> "StrategyTemplate":
        return cls(
            match_kind=value.match_kind,
            state_match_score=value.state_match_score,
            match_dimensions=value.match_dimensions,
            action_probabilities=tuple(value.action_probabilities.items()),
            recommended_sizes=tuple(value.recommended_sizes.items()),
            action_options=value.action_options,
            action_ev=tuple(value.action_ev.items()),
            confidence=value.confidence,
            evidence=value.evidence,
            assumptions=value.assumptions,
            produced_at=value.produced_at,
        )

    def materialize(
        self,
        query: StrategyCacheQuery,
        context: DecisionContext,
    ) -> StrategyCandidate:
        return StrategyCandidate(
            hand_id=context.hand_id,
            state_version=context.state_version,
            request_id=context.request_id,
            provider_id=query.provider_id,
            provider_version=query.provider_version,
            match_kind=self.match_kind,
            state_match_score=self.state_match_score,
            match_dimensions=self.match_dimensions,
            action_probabilities=dict(self.action_probabilities),
            recommended_sizes=dict(self.recommended_sizes),
            action_options=self.action_options,
            action_ev=dict(self.action_ev),
            confidence=self.confidence,
            evidence=self.evidence + (f"strategy_cache_key:{query.key}",),
            assumptions=self.assumptions,
            produced_at=self.produced_at,
            expires_at=context.request.expires_at,
        )


@dataclass(frozen=True)
class StrategyCacheEntry:
    query_key: str
    template: StrategyTemplate
    inserted_at: float


@dataclass(frozen=True)
class StrategyCacheLookup:
    state: StrategyCacheState
    candidate: StrategyCandidate | None = None

    def __post_init__(self) -> None:
        hit = self.state is StrategyCacheState.HIT
        if hit != (self.candidate is not None):
            raise ValueError("only HIT lookup can carry a candidate")


class StrategyCache:
    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool):
            raise TypeError("max_entries must be an int")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if not isinstance(ttl_seconds, (int, float)) or isinstance(
            ttl_seconds, bool
        ) or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and > 0")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._max_entries = max_entries
        self._ttl_seconds = float(ttl_seconds)
        self._monotonic = monotonic
        self._entries: OrderedDict[str, StrategyCacheEntry] = OrderedDict()
        self._lock = RLock()

    def put(
        self,
        query: StrategyCacheQuery,
        candidate: StrategyCandidate,
    ) -> None:
        if not isinstance(query, StrategyCacheQuery):
            raise TypeError("query must be a StrategyCacheQuery")
        if not isinstance(candidate, StrategyCandidate):
            raise TypeError("candidate must be a StrategyCandidate")
        if (
            candidate.hand_id != query.hand_id
            or candidate.state_version != query.state_version
            or candidate.request_id != query.request_id
        ):
            raise ValueError("candidate identity does not match cache query")
        if candidate.provider_id != query.provider_id:
            raise ValueError("candidate provider_id does not match cache query")
        if candidate.provider_version != query.provider_version:
            raise ValueError("candidate provider_version does not match cache query")
        entry = StrategyCacheEntry(
            query.key,
            StrategyTemplate.from_candidate(candidate),
            self._monotonic(),
        )
        with self._lock:
            self._entries[query.key] = entry
            self._entries.move_to_end(query.key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def lookup(
        self,
        query: StrategyCacheQuery,
        context: DecisionContext,
    ) -> StrategyCacheLookup:
        if not isinstance(query, StrategyCacheQuery):
            raise TypeError("query must be a StrategyCacheQuery")
        if not isinstance(context, DecisionContext):
            raise TypeError("context must be a DecisionContext")
        if canonical_context_digest(context) != query.context_digest:
            raise ValueError("context does not match cache query digest")
        if (
            context.hand_id != query.hand_id
            or context.state_version != query.state_version
            or context.request_id != query.request_id
        ):
            raise ValueError("context identity does not match cache query")
        with self._lock:
            entry = self._entries.get(query.key)
            if entry is None:
                return StrategyCacheLookup(StrategyCacheState.NOT_FOUND)
            age = self._monotonic() - entry.inserted_at
            if age >= self._ttl_seconds:
                del self._entries[query.key]
                return StrategyCacheLookup(StrategyCacheState.STALE)
            self._entries.move_to_end(query.key)
            template = entry.template
        return StrategyCacheLookup(
            StrategyCacheState.HIT,
            template.materialize(query, context),
        )

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class CachingStrategyProvider:
    """Cache-first wrapper around one versioned StrategyProvider."""

    def __init__(
        self,
        provider: StrategyProvider,
        cache: StrategyCache,
        *,
        provider_asset_id: str,
        engine_version: str,
    ) -> None:
        if not isinstance(provider, StrategyProvider):
            raise TypeError("provider must implement StrategyProvider")
        if not isinstance(cache, StrategyCache):
            raise TypeError("cache must be a StrategyCache")
        self._provider = provider
        self._cache = cache
        self._asset_id = _text(provider_asset_id, "provider_asset_id")
        self._engine_version = _text(engine_version, "engine_version")

    @property
    def provider_id(self) -> str:
        return self._provider.provider_id

    @property
    def source_version(self) -> str:
        return self._provider.source_version

    @property
    def capability(self) -> ProviderCapability:
        return self._provider.capability

    def query(self, context: DecisionContext) -> ProviderResult:
        query = StrategyCacheQuery.from_context(
            context,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            provider_asset_id=self._asset_id,
            engine_version=self._engine_version,
        )
        cached = self._cache.lookup(query, context)
        if cached.state is StrategyCacheState.HIT:
            candidate = cached.candidate
            state = (
                LookupState.HIT_EXACT
                if candidate.match_kind is MatchKind.EXACT
                else LookupState.HIT_APPROXIMATE
            )
            return ProviderResult(state, self.provider_id, candidate)
        result = self._provider.query(context)
        if result.candidate is not None:
            try:
                self._cache.put(query, result.candidate)
            except (TypeError, ValueError):
                return ProviderResult(
                    LookupState.REJECTED,
                    self.provider_id,
                    reasons=("strategy_cache_store_rejected",),
                )
        return result


class StrategyCacheProvider:
    """Lookup-only cache source for the first TieredStrategyRouter layer."""

    def __init__(
        self,
        *,
        provider_id: str,
        source_version: str,
        capability: ProviderCapability,
        cache: StrategyCache,
        provider_asset_id: str,
        engine_version: str,
    ) -> None:
        self._provider_id = _text(provider_id, "provider_id")
        self._source_version = _text(source_version, "source_version")
        if not isinstance(capability, ProviderCapability):
            raise TypeError("capability must be a ProviderCapability")
        if not isinstance(cache, StrategyCache):
            raise TypeError("cache must be a StrategyCache")
        self._capability = capability
        self._cache = cache
        self._asset_id = _text(provider_asset_id, "provider_asset_id")
        self._engine_version = _text(engine_version, "engine_version")

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def source_version(self) -> str:
        return self._source_version

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    def query(self, context: DecisionContext) -> ProviderResult:
        query = StrategyCacheQuery.from_context(
            context,
            provider_id=self.provider_id,
            provider_version=self.source_version,
            provider_asset_id=self._asset_id,
            engine_version=self._engine_version,
        )
        cached = self._cache.lookup(query, context)
        if cached.state is not StrategyCacheState.HIT:
            reason = (
                "strategy_cache_stale"
                if cached.state is StrategyCacheState.STALE
                else "strategy_cache_miss"
            )
            return ProviderResult(
                LookupState.NOT_FOUND,
                self.provider_id,
                reasons=(reason,),
            )
        candidate = cached.candidate
        state = (
            LookupState.HIT_EXACT
            if candidate.match_kind is MatchKind.EXACT
            else LookupState.HIT_APPROXIMATE
        )
        return ProviderResult(state, self.provider_id, candidate)


def canonical_context_digest(context: DecisionContext) -> str:
    if not isinstance(context, DecisionContext):
        raise TypeError("context must be a DecisionContext")
    payload = strategy_serialize(context)
    payload.pop("request", None)
    payload.pop("input_quality", None)
    payload.pop("input_provenance", None)
    for seat in payload["seats"]:
        seat.pop("player_id", None)
    payload["action_history"] = [
        {
            "event_type": item["event_type"],
            "payload": item["payload"],
            "source": item["source"],
        }
        for item in payload["action_history"]
    ]
    return _digest(payload)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty str")
    return value


__all__ = [
    "CachingStrategyProvider",
    "StrategyCacheProvider",
    "StrategyCache",
    "StrategyCacheEntry",
    "StrategyCacheLookup",
    "StrategyCacheQuery",
    "StrategyCacheState",
    "StrategyTemplate",
    "canonical_context_digest",
]
