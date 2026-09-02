from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from datetime import timedelta

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.provider import ActionOption, MatchDimension, MatchKind
from poker_engine.strategy.provider import FakeProvider, LookupState, ProviderResult
from poker_engine.strategy.strategy_cache import (
    CachingStrategyProvider,
    StrategyCache,
    StrategyCacheProvider,
    StrategyCacheQuery,
    StrategyCacheState,
    canonical_context_digest,
)

from .helpers import candidate, capability, card, context, hit_result


def query(ctx, *, version="v1", asset="sha256:asset-1", engine="strategy-v1"):
    return StrategyCacheQuery.from_context(
        ctx,
        provider_id="provider",
        provider_version=version,
        provider_asset_id=asset,
        engine_version=engine,
    )


def value(ctx, **overrides):
    return candidate(ctx, "provider", overrides.pop("version", "v1"), **overrides)


def test_equivalent_contexts_ignore_request_and_player_identity():
    first = context()
    second_request = replace(
        first.request,
        hand_id="another-hand",
        state_version=99,
        request_id="another-request",
    )
    second = replace(
        first,
        request=second_request,
        seats=tuple(
            replace(seat, player_id=f"anonymous-{seat.seat_id}")
            for seat in first.seats
        ),
    )
    assert canonical_context_digest(first) == canonical_context_digest(second)
    assert query(first).key == query(second).key


def test_event_identity_is_ignored_but_action_payload_is_canonical_input():
    base = context(action_line="raise")
    first_event = StateEvent(
        EventType.RAISE,
        base.hand_id,
        base.state_version,
        {"seat_id": 0, "amount": ChipAmount("3")},
        base.request.requested_at,
        "state_engine",
    )
    second_event = StateEvent(
        EventType.RAISE,
        "other-hand",
        999,
        {"seat_id": 0, "amount": ChipAmount("3")},
        base.request.requested_at + timedelta(seconds=5),
        "state_engine",
    )
    changed_amount = replace(
        second_event, payload={"seat_id": 0, "amount": ChipAmount("4")}
    )
    first = replace(base, action_history=(first_event,))
    equivalent = replace(base, action_history=(second_event,))
    changed = replace(base, action_history=(changed_amount,))
    assert canonical_context_digest(first) == canonical_context_digest(equivalent)
    assert canonical_context_digest(first) != canonical_context_digest(changed)


def test_cache_hit_rebinds_candidate_to_current_request_identity():
    first = context()
    first_query = query(first)
    cache = StrategyCache()
    cache.put(first_query, value(first))

    second_request = replace(
        first.request,
        hand_id="hand-2",
        state_version=2,
        request_id="request-2",
    )
    second = replace(first, request=second_request)
    result = cache.lookup(query(second), second)

    assert result.state is StrategyCacheState.HIT
    assert result.candidate.hand_id == "hand-2"
    assert result.candidate.state_version == 2
    assert result.candidate.request_id == "request-2"
    assert result.candidate.expires_at == second.request.expires_at
    assert any(item.startswith("strategy_cache_key:")
               for item in result.candidate.evidence)


def test_cache_preserves_structured_match_dimensions():
    ctx = context()
    dimension = MatchDimension(
        "pot_bb", "6.5", "6", Decimal("0.5"), Decimal("1")
    )
    cache = StrategyCache()
    cache.put(query(ctx), value(
        ctx,
        match_kind=MatchKind.INTERPOLATED,
        score=0.5,
        match_dimensions=(dimension,),
    ))

    result = cache.lookup(query(ctx), ctx)

    assert result.candidate.match_dimensions == (dimension,)


@pytest.mark.parametrize(
    "change",
    [
        lambda ctx: (
            query(replace(ctx, action_line="raise")),
            replace(ctx, action_line="raise"),
        ),
        lambda ctx: (
            query(replace(ctx, effective_stack_bb=Decimal("80"))),
            replace(ctx, effective_stack_bb=Decimal("80")),
        ),
        lambda ctx: (
            query(replace(ctx, hero_cards=(card("As"), card("Qd")))),
            replace(ctx, hero_cards=(card("As"), card("Qd"))),
        ),
        lambda ctx: (query(ctx, version="v2"), ctx),
        lambda ctx: (query(ctx, asset="sha256:asset-2"), ctx),
        lambda ctx: (query(ctx, engine="strategy-v2"), ctx),
    ],
)
def test_context_provider_asset_or_engine_change_is_cache_miss(change):
    ctx = context()
    cache = StrategyCache()
    cache.put(query(ctx), value(ctx))
    changed_query, changed_context = change(ctx)
    assert changed_query.key != query(ctx).key
    assert cache.lookup(changed_query, changed_context).state is (
        StrategyCacheState.NOT_FOUND
    )


def test_lookup_reports_not_found_for_different_provider_version():
    ctx = context()
    cache = StrategyCache()
    cache.put(query(ctx), value(ctx))
    assert cache.lookup(query(ctx, version="v2"), ctx).state is (
        StrategyCacheState.NOT_FOUND
    )


def test_ttl_returns_stale_once_then_not_found():
    clock = [10.0]
    ctx = context()
    cache = StrategyCache(ttl_seconds=5, monotonic=lambda: clock[0])
    cache.put(query(ctx), value(ctx))
    clock[0] = 15.0
    assert cache.lookup(query(ctx), ctx).state is StrategyCacheState.STALE
    assert cache.lookup(query(ctx), ctx).state is StrategyCacheState.NOT_FOUND


def test_lru_evicts_least_recently_used_entry():
    base = context()
    first = base
    second = replace(base, action_line="raise")
    third = replace(base, action_line="three_bet")
    cache = StrategyCache(max_entries=2)
    cache.put(query(first), value(first))
    cache.put(query(second), value(second))
    assert cache.lookup(query(first), first).state is StrategyCacheState.HIT
    cache.put(query(third), value(third))
    assert cache.lookup(query(second), second).state is StrategyCacheState.NOT_FOUND
    assert cache.lookup(query(first), first).state is StrategyCacheState.HIT


def test_put_rejects_stale_or_wrong_provider_candidate():
    ctx = context()
    cache = StrategyCache()
    with pytest.raises(ValueError, match="identity"):
        cache.put(query(ctx), replace(value(ctx), request_id="stale"))
    with pytest.raises(ValueError, match="provider_id"):
        cache.put(query(ctx), replace(value(ctx), provider_id="other"))


def test_action_options_and_exact_size_frequencies_survive_cache():
    ctx = context()
    options = (
        ActionOption(ActionType.CHECK, Decimal("0.4"), source_label="check"),
        ActionOption(
            ActionType.RAISE, Decimal("0.6"), ChipAmount("2.5"), "raise_to_250"
        ),
    )
    cached = replace(value(ctx), action_options=options)
    cache = StrategyCache()
    cache.put(query(ctx), cached)
    result = cache.lookup(query(ctx), ctx)
    assert result.candidate.action_options == options
    assert result.candidate.recommended_sizes[ActionType.RAISE] == (
        ChipAmount("2.5"),
    )


def test_cache_lookup_is_thread_safe():
    ctx = context()
    cache = StrategyCache()
    cache.put(query(ctx), value(ctx))
    with ThreadPoolExecutor(max_workers=8) as executor:
        states = tuple(executor.map(
            lambda _: cache.lookup(query(ctx), ctx).state,
            range(100),
        ))
    assert set(states) == {StrategyCacheState.HIT}
    assert cache.size == 1


def test_caching_provider_queries_source_once_then_rebinds_cache_hit():
    first = context()
    calls = []

    def source(ctx):
        calls.append(ctx.request_id)
        return hit_result(value(ctx))

    provider = FakeProvider(
        "provider", "v1", capability((2,)), source
    )
    cached_provider = CachingStrategyProvider(
        provider,
        StrategyCache(),
        provider_asset_id="sha256:asset-1",
        engine_version="strategy-v1",
    )
    first_result = cached_provider.query(first)
    second_request = replace(
        first.request,
        hand_id="h2",
        state_version=2,
        request_id="r2",
    )
    second = replace(first, request=second_request)
    second_result = cached_provider.query(second)

    assert first_result.state is LookupState.HIT_EXACT
    assert second_result.state is LookupState.HIT_EXACT
    assert calls == [first.request_id]
    assert second_result.candidate.request_id == "r2"
    assert any(item.startswith("strategy_cache_key:")
               for item in second_result.candidate.evidence)


def test_caching_provider_requeries_after_ttl_expiry():
    clock = [0.0]
    ctx = context()
    calls = []

    def source(value_context):
        calls.append(1)
        return hit_result(value(value_context))

    provider = FakeProvider("provider", "v1", capability((2,)), source)
    wrapped = CachingStrategyProvider(
        provider,
        StrategyCache(ttl_seconds=1, monotonic=lambda: clock[0]),
        provider_asset_id="asset",
        engine_version="engine",
    )
    wrapped.query(ctx)
    clock[0] = 1.0
    wrapped.query(ctx)
    assert len(calls) == 2


def test_caching_provider_does_not_cache_miss_result():
    ctx = context()
    calls = []

    def source(_):
        calls.append(1)
        return ProviderResult(
            LookupState.NOT_FOUND, "provider", reasons=("miss",)
        )

    provider = FakeProvider("provider", "v1", capability((2,)), source)
    wrapped = CachingStrategyProvider(
        provider, StrategyCache(), provider_asset_id="asset", engine_version="v1"
    )
    assert wrapped.query(ctx).state is LookupState.NOT_FOUND
    assert wrapped.query(ctx).state is LookupState.NOT_FOUND
    assert len(calls) == 2


def test_lookup_only_cache_provider_reports_miss_hit_and_stale():
    clock = [0.0]
    ctx = context()
    cache = StrategyCache(ttl_seconds=1, monotonic=lambda: clock[0])
    provider = StrategyCacheProvider(
        provider_id="provider",
        source_version="v1",
        capability=capability(),
        cache=cache,
        provider_asset_id="sha256:asset-1",
        engine_version="strategy-v1",
    )

    miss = provider.query(ctx)
    cache.put(query(ctx), value(ctx))
    hit = provider.query(ctx)
    clock[0] = 1.0
    stale = provider.query(ctx)

    assert miss.state is LookupState.NOT_FOUND
    assert miss.reasons == ("strategy_cache_miss",)
    assert hit.state is LookupState.HIT_EXACT
    assert stale.state is LookupState.NOT_FOUND
    assert stale.reasons == ("strategy_cache_stale",)
