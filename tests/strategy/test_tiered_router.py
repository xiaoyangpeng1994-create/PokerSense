"""Ordered Fast-source fallback and lookup-only cache integration tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.provider import (
    FakeProvider,
    LookupState,
    MatchKind,
    ProviderResult,
)
from poker_engine.strategy.router import FastSourceLayer, TieredStrategyRouter
from poker_engine.strategy.strategy_cache import (
    CachingStrategyProvider,
    StrategyCache,
    StrategyCacheProvider,
)

from .helpers import NOW, candidate, capability, context, hit_result


FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _provider(ctx, provider_id, result, calls):
    def query(_):
        calls.append(provider_id)
        return result

    return FakeProvider(provider_id, "v1", capability(), query)


def test_first_usable_layer_stops_all_lower_sources():
    ctx = context()
    calls = []
    cache_hit = _provider(
        ctx, "cache", hit_result(candidate(ctx, "cache", "v1")), calls
    )
    db = _provider(
        ctx, "db", hit_result(candidate(ctx, "db", "v1")), calls
    )
    model = _provider(
        ctx,
        "model",
        hit_result(candidate(
            ctx, "model", "v1", match_kind=MatchKind.HEURISTIC
        )),
        calls,
    )
    router = TieredStrategyRouter({
        FastSourceLayer.MODEL: (model,),
        FastSourceLayer.PREFLOP_DB: (db,),
        FastSourceLayer.CACHE: (cache_hit,),
    })

    result = router.route(ctx, now=NOW)

    assert result.selected.provider_id == "cache"
    assert calls == ["cache"]
    assert router.layers == (
        FastSourceLayer.CACHE,
        FastSourceLayer.PREFLOP_DB,
        FastSourceLayer.MODEL,
    )


def test_miss_and_rejection_fall_through_until_model_hit():
    ctx = context()
    calls = []
    cache = _provider(
        ctx,
        "cache",
        ProviderResult(LookupState.NOT_FOUND, "cache", reasons=("cache_miss",)),
        calls,
    )
    presolved = _provider(
        ctx,
        "presolved",
        ProviderResult(
            LookupState.REJECTED, "presolved", reasons=("asset_unavailable",)
        ),
        calls,
    )
    model = _provider(
        ctx,
        "model",
        hit_result(candidate(
            ctx, "model", "v1", match_kind=MatchKind.HEURISTIC
        )),
        calls,
    )
    router = TieredStrategyRouter({
        FastSourceLayer.CACHE: (cache,),
        FastSourceLayer.PRESOLVED: (presolved,),
        FastSourceLayer.MODEL: (model,),
    })

    result = router.route(ctx, now=NOW)

    assert result.selected.provider_id == "model"
    assert calls == ["cache", "presolved", "model"]
    assert [item.state for item in result.provider_results] == [
        LookupState.NOT_FOUND,
        LookupState.REJECTED,
        LookupState.HIT_APPROXIMATE,
    ]


def test_all_layer_failures_are_aggregated_without_fake_candidate():
    ctx = context()
    calls = []
    cache = _provider(
        ctx,
        "cache",
        ProviderResult(LookupState.NOT_FOUND, "cache", reasons=("cache_miss",)),
        calls,
    )
    db = _provider(
        ctx,
        "db",
        ProviderResult(LookupState.NOT_APPLICABLE, "db", reasons=("db_miss",)),
        calls,
    )
    router = TieredStrategyRouter({
        FastSourceLayer.CACHE: (cache,),
        FastSourceLayer.PREFLOP_DB: (db,),
    })

    result = router.route(ctx, now=NOW)

    assert result.state is LookupState.NO_STRATEGY
    assert result.selected is None
    assert result.reasons == ("cache_miss", "db_miss")
    assert calls == ["cache", "db"]


def test_lookup_only_cache_layer_populates_via_write_through_db_layer():
    first = context()
    source_calls = []

    def source(ctx):
        source_calls.append(ctx.request_id)
        return hit_result(candidate(ctx, "db", "v1"))

    cache = StrategyCache()
    cap = capability()
    backend = FakeProvider("db", "v1", cap, source)
    write_through = CachingStrategyProvider(
        backend,
        cache,
        provider_asset_id="sha256:db-v1",
        engine_version="strategy-v1",
    )
    lookup_only = StrategyCacheProvider(
        provider_id="db",
        source_version="v1",
        capability=cap,
        cache=cache,
        provider_asset_id="sha256:db-v1",
        engine_version="strategy-v1",
    )
    router = TieredStrategyRouter({
        FastSourceLayer.CACHE: (lookup_only,),
        FastSourceLayer.PREFLOP_DB: (write_through,),
    })

    first_result = router.route(first, now=NOW)
    second = replace(first, request=replace(
        first.request,
        hand_id="h2",
        state_version=2,
        request_id="r2",
    ))
    second_result = router.route(second, now=NOW)

    assert first_result.selected.provider_id == "db"
    assert [item.state for item in first_result.provider_results] == [
        LookupState.NOT_FOUND, LookupState.HIT_EXACT,
    ]
    assert second_result.selected.request_id == "r2"
    assert len(second_result.provider_results) == 1
    assert "strategy_cache_key:" in second_result.selected.evidence[-1]
    assert source_calls == [first.request_id]


def test_tiered_router_is_accepted_by_production_orchestrator():
    ctx = context()
    calls = []
    db = _provider(ctx, "db", hit_result(candidate(ctx, "db", "v1")), calls)
    router = TieredStrategyRouter({FastSourceLayer.PREFLOP_DB: (db,)})

    cycle = StrategyOrchestrator(router).request(ctx, now=NOW)

    assert cycle.fast_advice.status is AdviceStatus.READY
    assert cycle.fast_advice.strategy_source == "db"


@pytest.mark.parametrize(
    "layers",
    (
        {},
        {"cache": ()},
        {FastSourceLayer.CACHE: ()},
    ),
)
def test_invalid_layer_configuration_is_rejected(layers):
    with pytest.raises((TypeError, ValueError)):
        TieredStrategyRouter(layers)


@pytest.mark.parametrize(
    "fixture",
    [
        item for item in (
            json.loads(line) for line in FIXTURES.read_text().splitlines()
        )
        if item["fixture_id"].startswith("MOCK-FAST-FALLBACK-")
    ],
)
def test_generated_fast_fallback_fixtures_execute_through_router(fixture):
    ctx = context()
    calls = []
    layers = {}
    for item in fixture["input"]["fast_source_layers"]:
        layer = FastSourceLayer(item["layer"])
        provider_id = item["provider_id"]
        lookup = LookupState(item["lookup_state"])
        if lookup in (LookupState.HIT_EXACT, LookupState.HIT_APPROXIMATE):
            kind = (
                MatchKind.HEURISTIC
                if layer is FastSourceLayer.MODEL else MatchKind.EXACT
            )
            value = candidate(
                ctx, provider_id, "v1", match_kind=kind
            )
            result = ProviderResult(lookup, provider_id, value)
        else:
            result = ProviderResult(
                lookup, provider_id, reasons=(f"{layer.value}_miss",)
            )
        layers[layer] = (_provider(ctx, provider_id, result, calls),)

    result = TieredStrategyRouter(layers).route(ctx, now=NOW)

    expected = fixture["expected"]
    assert calls == [f"mock-{name}" for name in expected["queried_layers"]]
    assert (
        None if result.selected is None else result.selected.provider_id
    ) == (
        None if expected["selected_layer"] is None
        else f"mock-{expected['selected_layer']}"
    )
    assert [item.state.value for item in result.provider_results] == [
        item["state"] for item in expected["provider_lookups"]
    ]
