"""Canonical equity cache key, expiry, and eviction tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import PotState, RangeDistribution
from poker_engine.strategy.equity_cache import (
    EquityCache,
    EquityCacheQuery,
    EquityCacheState,
    EquityMethod,
)
from poker_engine.strategy.multiway_equity import exact_multiway_pot_share
from poker_engine.strategy.range_tracker import enumerate_joint_assignments

from .helpers import card


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def _range(
    seat: int = 0,
    *,
    version: str = "range-v1",
    weights: dict[str, str] | None = None,
) -> RangeDistribution:
    weights = weights or {"KsKd": "0.75", "5s6s": "0.25"}
    return RangeDistribution(
        seat,
        {combo: Decimal(weight) for combo, weight in weights.items()},
        "test-range",
        version,
        confidence=0.8,
    )


def _query(**changes) -> EquityCacheQuery:
    values = {
        "hero_seat": 1,
        "hero_cards": (card("As"), card("Ad")),
        "board_cards": (
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        "villain_ranges": (_range(),),
        "pots": (PotState("main", ChipAmount("100"), (0, 1)),),
        "method": EquityMethod.EXACT,
        "engine_version": "equity-v1",
    }
    values.update(changes)
    return EquityCacheQuery(**values)


def _result(query: EquityCacheQuery):
    assignments = enumerate_joint_assignments(
        query.villain_ranges,
        query.hero_cards + query.board_cards,
    )
    return exact_multiway_pot_share(
        query.hero_seat,
        query.hero_cards,
        assignments,
        query.board_cards,
        query.pots,
    )


def test_cache_key_is_canonical_for_card_order_and_weight_scale():
    original = _query()
    equivalent = _query(
        hero_cards=tuple(reversed(original.hero_cards)),
        board_cards=tuple(reversed(original.board_cards)),
        villain_ranges=(_range(weights={"5s6s": "1", "KsKd": "3"}),),
    )

    assert equivalent.key == original.key


@pytest.mark.parametrize(
    "changed",
    (
        {"hero_cards": (card("Ah"), card("Ad"))},
        {"board_cards": (
            card("2c"), card("3d"), card("4h"), card("9s"), card("Jc"),
        )},
        {"villain_ranges": (_range(version="range-v2"),)},
        {"pots": (PotState("main", ChipAmount("101"), (0, 1)),)},
        {"pots": (PotState("main", ChipAmount("100"), (1,)),)},
        {"engine_version": "equity-v2"},
        {
            "method": EquityMethod.MONTE_CARLO,
            "trials": 1000,
            "seed": 7,
        },
    ),
)
def test_every_semantic_query_dimension_changes_the_key(changed):
    assert _query(**changed).key != _query().key


def test_cache_hit_preserves_result_and_evidence():
    query = _query()
    result = _result(query)
    cache = EquityCache()
    cache.put(
        query,
        result,
        created_at=NOW,
        expires_at=NOW + timedelta(seconds=10),
        evidence=("equity://exact/v1",),
    )

    lookup = cache.get(query, now=NOW + timedelta(seconds=1))

    assert lookup.state is EquityCacheState.HIT
    assert lookup.entry.result == result
    assert lookup.entry.evidence == ("equity://exact/v1",)


def test_changed_key_is_not_found_and_expired_entry_is_stale_once():
    query = _query()
    cache = EquityCache()
    cache.put(
        query,
        _result(query),
        created_at=NOW,
        expires_at=NOW + timedelta(seconds=1),
        evidence=("equity://exact/v1",),
    )

    assert cache.get(
        _query(engine_version="equity-v2"),
        now=NOW,
    ).state is EquityCacheState.NOT_FOUND
    assert cache.get(
        query,
        now=NOW + timedelta(seconds=1),
    ).state is EquityCacheState.STALE
    assert cache.get(
        query,
        now=NOW + timedelta(seconds=2),
    ).state is EquityCacheState.NOT_FOUND


def test_cache_evicts_least_recently_used_entry():
    first = _query(engine_version="v1")
    second = _query(engine_version="v2")
    third = _query(engine_version="v3")
    cache = EquityCache(max_entries=2)
    for query in (first, second):
        cache.put(
            query,
            _result(query),
            created_at=NOW,
            expires_at=None,
            evidence=("equity://exact/v1",),
        )
    assert cache.get(first, now=NOW).state is EquityCacheState.HIT
    cache.put(
        third,
        _result(third),
        created_at=NOW,
        expires_at=None,
        evidence=("equity://exact/v1",),
    )

    assert cache.get(second, now=NOW).state is EquityCacheState.NOT_FOUND
    assert cache.get(first, now=NOW).state is EquityCacheState.HIT
    assert cache.get(third, now=NOW).state is EquityCacheState.HIT


def test_cache_rejects_result_for_different_query_identity():
    query = _query()
    wrong_query = _query(
        hero_seat=2,
        pots=(PotState("main", ChipAmount("100"), (0, 2)),),
    )
    cache = EquityCache()

    with pytest.raises(ValueError, match="hero_seat"):
        cache.put(
            wrong_query,
            _result(query),
            created_at=NOW,
            expires_at=None,
            evidence=("equity://exact/v1",),
        )


def test_query_rejects_ambiguous_seat_or_pot_identity():
    with pytest.raises(ValueError, match="hero_seat"):
        _query(villain_ranges=(_range(seat=1),))
    with pytest.raises(ValueError, match="without a range"):
        _query(pots=(PotState("main", ChipAmount("100"), (0, 1, 2)),))
