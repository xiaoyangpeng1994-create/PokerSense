"""Provider capability and StrategyRouter tests."""

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import Position, Street
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import PotState
from poker_engine.strategy.provider import (
    FakeProvider,
    LookupState,
    MatchKind,
    ProviderResult,
)
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, context, hit_result


FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _abstraction_fixtures():
    return [
        item for item in (
            json.loads(line) for line in FIXTURES.read_text().splitlines()
        )
        if item["fixture_id"].startswith("MOCK-ABSTRACTION-")
    ]


def _provider(ctx, provider_id, cap, *, kind=MatchKind.EXACT, score=1.0):
    value = candidate(
        ctx, provider_id, "v1", match_kind=kind, score=score,
    )
    return FakeProvider(provider_id, "v1", cap, hit_result(value))


@pytest.mark.parametrize("player_count", range(3, 10))
def test_hu_provider_never_routes_multiplayer_preflop(player_count):
    ctx = context(player_count)
    provider = _provider(ctx, "hu", capability((2,)))
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.NO_STRATEGY
    assert result.selected is None
    assert result.provider_results[0].state is LookupState.NOT_APPLICABLE
    assert "unsupported_player_count" in result.provider_results[0].reasons


@pytest.mark.parametrize("player_count", range(2, 10))
def test_corresponding_provider_routes_each_player_count(player_count):
    ctx = context(player_count)
    provider_id = f"mock-{player_count}p"
    provider = _provider(ctx, provider_id, capability((player_count,)))
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.HIT_EXACT
    assert result.selected.provider_id == provider_id


def test_postflop_capability_uses_active_not_dealt_count():
    ctx = context(9, street=Street.FLOP, active_count=2)
    provider = _provider(
        ctx, "hu-postflop", capability((2,), streets=(Street.FLOP,)),
    )
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.HIT_EXACT


def test_exact_candidate_beats_higher_priority_heuristic():
    ctx = context(6)
    heuristic = _provider(
        ctx, "heuristic", capability(
            (6,), match_kind=MatchKind.HEURISTIC, priority=1,
        ), kind=MatchKind.HEURISTIC, score=1.0,
    )
    exact = _provider(ctx, "exact", capability((6,), priority=100))
    result = StrategyRouter((heuristic, exact)).route(ctx, now=NOW)
    assert result.selected.provider_id == "exact"
    assert result.state is LookupState.HIT_EXACT


def test_stack_interpolation_is_transparent():
    ctx = context(effective_stack_bb=Decimal("95"))
    cap = capability(
        (2,), interpolate=True, max_distance=Decimal("10"),
    )
    value = candidate(
        ctx, "interp", "v1", match_kind=MatchKind.INTERPOLATED, score=0.5,
    )
    result = StrategyRouter((
        FakeProvider("interp", "v1", cap, hit_result(value)),
    )).route(ctx, now=NOW)
    assert result.state is LookupState.HIT_APPROXIMATE
    assert result.selected.match_kind is MatchKind.INTERPOLATED
    assert result.selected.match_dimensions[0].name == "effective_stack_bb"
    assert result.selected.match_dimensions[0].requested == "95"
    assert result.selected.match_dimensions[0].matched == "100"
    assert result.selected.match_dimensions[0].distance == Decimal("5")


def test_stack_outside_interpolation_distance_is_not_applicable():
    ctx = context(effective_stack_bb=Decimal("50"))
    provider = _provider(
        ctx, "interp", capability(
            (2,), interpolate=True, max_distance=Decimal("10"),
        ), kind=MatchKind.INTERPOLATED,
    )
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].state is LookupState.NOT_APPLICABLE


def test_position_capability_is_exact_and_fail_closed():
    ctx = context()
    provider = _provider(
        ctx,
        "btn-only",
        capability((2,), hero_positions=(Position.BTN,)),
    )

    result = StrategyRouter((provider,)).route(ctx, now=NOW)

    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].reasons == ("unsupported_hero_position",)


def test_pot_interpolation_reports_structured_dimension():
    ctx = context()
    cap = capability(
        (2,),
        pot_buckets=(Decimal("1"), Decimal("2")),
        interpolate_pot=True,
        max_pot_distance=Decimal("1"),
    )
    value = candidate(
        ctx, "pot-interp", "v1",
        match_kind=MatchKind.INTERPOLATED,
        score=0.5,
    )

    result = StrategyRouter((
        FakeProvider("pot-interp", "v1", cap, hit_result(value)),
    )).route(ctx, now=NOW)

    assert result.state is LookupState.HIT_APPROXIMATE
    dimension = result.selected.match_dimensions[0]
    assert dimension.name == "pot_bb"
    assert dimension.requested == "1.5"
    assert dimension.matched == "1"
    assert dimension.distance == Decimal("0.5")
    assert dimension.maximum_distance == Decimal("1")
    assert dimension.score == 0.5


def test_pot_outside_approved_distance_does_not_query_provider():
    ctx = context()
    provider = _provider(
        ctx,
        "pot-too-far",
        capability(
            (2,),
            pot_buckets=(Decimal("10"),),
            interpolate_pot=True,
            max_pot_distance=Decimal("1"),
        ),
        kind=MatchKind.INTERPOLATED,
    )

    result = StrategyRouter((provider,)).route(ctx, now=NOW)

    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].reasons == ("unsupported_pot",)


def test_candidate_cannot_overstate_capability_match_score():
    ctx = context()
    cap = capability(
        (2,),
        pot_buckets=(Decimal("1"), Decimal("2")),
        interpolate_pot=True,
        max_pot_distance=Decimal("1"),
    )
    value = candidate(
        ctx, "overstated", "v1",
        match_kind=MatchKind.INTERPOLATED,
        score=0.51,
    )

    result = StrategyRouter((
        FakeProvider("overstated", "v1", cap, hit_result(value)),
    )).route(ctx, now=NOW)

    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].reasons == (
        "candidate_overstates_match_score",
    )


def test_aggressive_size_interpolation_is_structured_and_bounded():
    ctx = context()
    ctx = replace(ctx, action_history=(StateEvent(
        EventType.RAISE,
        ctx.hand_id,
        ctx.state_version,
        payload={"amount_total_street": "2.5"},
        timestamp=NOW,
    ),))
    cap = capability(
        (2,),
        aggressive_size_buckets=(Decimal("2"), Decimal("3")),
        interpolate_aggressive_size=True,
        max_aggressive_size_distance=Decimal("1"),
    )
    value = candidate(
        ctx, "size-interp", "v1",
        match_kind=MatchKind.INTERPOLATED,
        score=0.5,
    )

    result = StrategyRouter((
        FakeProvider("size-interp", "v1", cap, hit_result(value)),
    )).route(ctx, now=NOW)

    assert result.state is LookupState.HIT_APPROXIMATE
    dimension = result.selected.match_dimensions[0]
    assert dimension.name == "last_aggressive_total_bb"
    assert (dimension.requested, dimension.matched) == ("2.5", "2")
    assert dimension.distance == Decimal("0.5")


def test_required_aggressive_size_without_event_is_not_applicable():
    ctx = context()
    provider = _provider(
        ctx,
        "size-required",
        capability(
            (2,),
            aggressive_size_buckets=(Decimal("2"),),
        ),
    )

    result = StrategyRouter((provider,)).route(ctx, now=NOW)

    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].reasons == (
        "missing_aggressive_size",
    )


@pytest.mark.parametrize(
    "fixture", _abstraction_fixtures(), ids=lambda item: item["fixture_id"]
)
def test_abstraction_mock_cases_execute_router(fixture):
    data = fixture["input"]["abstraction_match"]
    requested = data["requested"]
    declared = data["capability"]
    ctx = context(effective_stack_bb=Decimal(requested["effective_stack_bb"]))
    hero_position = Position(requested["hero_position"])
    ctx = replace(
        ctx,
        seats=tuple(
            replace(seat, position=hero_position)
            if seat.seat_id == ctx.hero_seat else seat
            for seat in ctx.seats
        ),
        pots=(PotState(
            "main",
            ChipAmount(requested["pot_bb"]),
            ctx.active_seats,
        ),),
        action_history=(StateEvent(
            EventType.RAISE,
            ctx.hand_id,
            ctx.state_version,
            payload={
                "amount_total_street": requested["last_aggressive_total_bb"]
            },
            timestamp=NOW,
        ),),
    )
    cap = capability(
        (2,),
        hero_positions=tuple(Position(value) for value in declared["hero_positions"]),
        interpolate=True,
        max_distance=Decimal(declared["max_stack_distance_bb"]),
        pot_buckets=tuple(Decimal(value) for value in declared["pot_buckets_bb"]),
        interpolate_pot=True,
        max_pot_distance=Decimal(declared["max_pot_distance_bb"]),
        aggressive_size_buckets=tuple(
            Decimal(value) for value in declared["aggressive_size_buckets_bb"]
        ),
        interpolate_aggressive_size=True,
        max_aggressive_size_distance=Decimal(
            declared["max_aggressive_size_distance_bb"]
        ),
    )
    match = cap.match(ctx)
    kind = match.match_kind or MatchKind.EXACT
    value = candidate(
        ctx,
        "fixture-provider",
        "v1",
        match_kind=kind,
        score=match.score if match.applicable else 1.0,
        match_dimensions=match.dimensions if match.applicable else (),
    )
    result = StrategyRouter((FakeProvider(
        "fixture-provider", "v1", cap, hit_result(value)
    ),)).route(ctx, now=NOW)

    expected_lookup = fixture["expected"]["provider_lookups"][0]["state"]
    if result.selected is None:
        assert result.provider_results[0].state.value == expected_lookup
        assert fixture["expected"]["advice"]["reason_codes"][0] in (
            result.provider_results[0].reasons
        )
    else:
        assert result.state.value == expected_lookup
        assert [
            {
                "name": item.name,
                "requested": item.requested,
                "matched": item.matched,
                "distance": str(item.distance),
                "maximum_distance": str(item.maximum_distance),
            }
            for item in result.selected.match_dimensions
        ] == fixture["expected"]["match_dimensions"]


def test_stack_ante_pair_prevents_unsupported_cartesian_product():
    cap = replace(
        capability((2,)),
        stack_buckets_bb=(Decimal("20"), Decimal("100")),
        ante_values=(ChipAmount("0"), ChipAmount("0.5")),
        ante_values_are_bb=True,
        stack_ante_pairs_bb=(
            (Decimal("20"), Decimal("0")),
            (Decimal("100"), Decimal("0.5")),
        ),
    )
    ctx = context(effective_stack_bb=Decimal("100"))
    match = cap.match(ctx)
    assert not match.applicable
    assert match.reasons == ("unsupported_stack_ante_combination",)


def test_stack_ante_pair_matches_after_ante_is_normalized_by_big_blind():
    cap = replace(
        capability((2,)),
        ante_values=(ChipAmount("0.5"),),
        ante_values_are_bb=True,
        stack_ante_pairs_bb=((Decimal("100"), Decimal("0.5")),),
    )
    ctx = context(effective_stack_bb=Decimal("100"))
    ctx = replace(ctx, game_config=replace(
        ctx.game_config,
        small_blind=ChipAmount("1"),
        big_blind=ChipAmount("2"),
        ante=ChipAmount("1"),
        minimum_chip=ChipAmount("1"),
    ))
    match = cap.match(ctx)
    assert match.applicable
    assert match.score == 1.0


def test_stack_ante_pairs_must_be_sorted_unique_and_reference_stack_bucket():
    base = capability((2,))
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(base, stack_ante_pairs_bb=(
            (Decimal("100"), Decimal("0")),
            (Decimal("100"), Decimal("0")),
        ))
    with pytest.raises(ValueError, match="exist in stack buckets"):
        replace(base, stack_ante_pairs_bb=((Decimal("20"), Decimal("0")),))


def test_duplicate_provider_id_is_rejected():
    ctx = context()
    first = _provider(ctx, "same", capability((2,)))
    second = _provider(ctx, "same", capability((2,)))
    with pytest.raises(ValueError, match="duplicate provider_id"):
        StrategyRouter((first, second))


def test_stale_candidate_is_rejected():
    ctx = context()
    stale = candidate(
        ctx, "stale", "v1", expires_at=NOW - timedelta(seconds=1),
    )
    provider = FakeProvider(
        "stale", "v1", capability((2,)), hit_result(stale),
    )
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].state is LookupState.REJECTED
    assert result.provider_results[0].reasons == ("expired_candidate",)


def test_provider_exception_is_recoverable_rejection():
    ctx = context()

    def fail(_context):
        raise RuntimeError("boom")

    provider = FakeProvider("broken", "v1", capability((2,)), fail)
    result = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert result.state is LookupState.NO_STRATEGY
    assert result.provider_results[0].state is LookupState.REJECTED
    assert result.provider_results[0].reasons == ("provider_error:RuntimeError",)


def test_provider_id_mismatch_is_rejected():
    ctx = context()
    result = ProviderResult(LookupState.NOT_FOUND, "other")
    provider = FakeProvider("expected", "v1", capability((2,)), result)
    routed = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert routed.provider_results[0].state is LookupState.REJECTED
    assert routed.provider_results[0].reasons == ("provider_id_mismatch",)
