"""Deterministic builders for strategy tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.request_context import RequestContext
from poker_engine.core.value_objects import Card, ChipAmount, ChipDelta
from poker_engine.strategy.contracts import (
    ContextQuality,
    DecisionContext,
    DecisionSeat,
    EffectiveStack,
    GameConfig,
    GameType,
    InputProvenance,
    InputSource,
    LegalAction,
    PotState,
    QualityStatus,
    RangeDistribution,
)
from poker_engine.strategy.provider import (
    LookupState,
    MatchDimension,
    MatchKind,
    ProviderCapability,
    ProviderResult,
    StrategyCandidate,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 22, tzinfo=UTC)
ACTION_LINES = frozenset({
    "unopened", "limp", "multi_limp", "raise", "three_bet",
    "four_bet", "squeeze", "iso_raise", "all_in",
})
POSITIONS = {
    2: (Position.BTN, Position.BB),
    3: (Position.BTN, Position.SB, Position.BB),
    4: (Position.CO, Position.BTN, Position.SB, Position.BB),
    5: (Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB),
    6: (
        Position.UTG, Position.HJ, Position.CO, Position.BTN,
        Position.SB, Position.BB,
    ),
    7: (
        Position.UTG, Position.LJ, Position.HJ, Position.CO,
        Position.BTN, Position.SB, Position.BB,
    ),
    8: (
        Position.UTG, Position.UTG1, Position.LJ, Position.HJ,
        Position.CO, Position.BTN, Position.SB, Position.BB,
    ),
    9: (
        Position.UTG, Position.UTG1, Position.UTG2, Position.LJ,
        Position.HJ, Position.CO, Position.BTN, Position.SB, Position.BB,
    ),
}


def card(value: str) -> Card:
    return Card(Rank(value[0]), Suit(value[1]))


def context(
    player_count: int = 2,
    *,
    street: Street = Street.PREFLOP,
    active_count: int | None = None,
    missing_fields: tuple[str, ...] = (),
    hard_failures: tuple[str, ...] = (),
    actor_is_hero: bool = True,
    action_line: str = "unopened",
    effective_stack_bb: Decimal = Decimal("100"),
    expires_at=None,
) -> DecisionContext:
    active_count = active_count or player_count
    positions = POSITIONS[player_count]
    hero_seat = player_count - 1
    seats = []
    inactive = player_count - active_count
    for seat_id, position in enumerate(positions):
        folded = seat_id < inactive
        seats.append(DecisionSeat(
            seat_id=seat_id,
            player_id="hero" if seat_id == hero_seat else f"p{seat_id}",
            position=position,
            stack=ChipAmount("100"),
            street_committed=ChipAmount("0"),
            hand_committed=ChipAmount("0"),
            status=PlayerStatus.FOLDED if folded else PlayerStatus.ACTIVE,
            is_hero=seat_id == hero_seat,
            is_dealer=position is Position.BTN,
        ))
    active_seats = tuple(range(inactive, player_count))
    board = {
        Street.PREFLOP: (),
        Street.FLOP: (card("2c"), card("7d"), card("Jh")),
        Street.TURN: (card("2c"), card("7d"), card("Jh"), card("9s")),
        Street.RIVER: (
            card("2c"), card("7d"), card("Jh"), card("9s"), card("3h"),
        ),
    }[street]
    actor = hero_seat if actor_is_hero else active_seats[0]
    expiry = expires_at or NOW + timedelta(seconds=2)
    requested_at = NOW if expiry > NOW else expiry - timedelta(seconds=2)
    request = RequestContext(
        hand_id=f"h-{player_count}-{street.value}",
        state_version=1,
        request_id=f"r-{player_count}-{street.value}",
        requested_at=requested_at,
        expires_at=expiry,
        deadline_ms=300,
    )
    return DecisionContext(
        request=request,
        game_config=GameConfig(
            variant="NLHE",
            game_type=GameType.CASH,
            max_seats=player_count,
            dealt_player_count=player_count,
            small_blind=ChipAmount("0.5"),
            big_blind=ChipAmount("1"),
            ante=ChipAmount("0"),
            rake_percent=Decimal("0"),
            rake_cap=ChipAmount("0"),
            minimum_chip=ChipAmount("0.5"),
        ),
        seats=tuple(seats),
        hero_seat=hero_seat,
        actor_seat=actor,
        active_seats=active_seats,
        hero_cards=(card("As"), card("Kd")),
        board_cards=board,
        street=street,
        pots=(PotState(
            pot_id="main",
            amount=ChipAmount("1.5" if street is Street.PREFLOP else "6"),
            eligible_seats=active_seats,
        ),),
        legal_actions=(
            LegalAction(ActionType.CHECK, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("100")),
        ),
        action_history=(),
        effective_stacks=tuple(
            EffectiveStack(seat, ChipAmount("100"))
            for seat in active_seats if seat != hero_seat
        ),
        hero_range=RangeDistribution(
            hero_seat, {"AsKd": Decimal("1")}, "known", "v1",
            confidence=1.0,
        ),
        villain_ranges=tuple(
            RangeDistribution(
                seat, {"AA": Decimal("0.5"), "AKs": Decimal("0.5")},
                "mock", "v1", confidence=0.5,
            )
            for seat in active_seats if seat != hero_seat
        ),
        input_quality=ContextQuality(
            overall_confidence=0.9,
            field_confidences={"hero_cards": 0.99, "stacks": 0.9},
            hard_failures=hard_failures,
        ),
        input_provenance=(
            InputProvenance(
                "hero_cards", InputSource.VISION, QualityStatus.VALID,
                0.99, "mock://hero", NOW,
            ),
            InputProvenance(
                "stacks", InputSource.MANUAL, QualityStatus.VALID,
                1.0, "manual://stacks", NOW,
            ),
        ),
        missing_fields=missing_fields,
        action_line=action_line,
        effective_stack_bb=effective_stack_bb,
    )


def capability(
    player_counts=(2,),
    *,
    streets=(Street.PREFLOP,),
    match_kind=MatchKind.EXACT,
    priority=100,
    interpolate=False,
    max_distance=Decimal("0"),
    hero_positions=None,
    pot_buckets=(),
    interpolate_pot=False,
    max_pot_distance=Decimal("0"),
    aggressive_size_buckets=(),
    interpolate_aggressive_size=False,
    max_aggressive_size_distance=Decimal("0"),
) -> ProviderCapability:
    return ProviderCapability(
        player_counts=frozenset(player_counts),
        streets=frozenset(streets),
        game_types=frozenset({GameType.CASH}),
        stack_buckets_bb=(Decimal("100"),),
        ante_values=(ChipAmount("0"),),
        rake_percent_values=(Decimal("0"),),
        action_lines=ACTION_LINES,
        base_match_kind=match_kind,
        priority=priority,
        allow_stack_interpolation=interpolate,
        max_stack_distance_bb=max_distance,
        hero_positions=frozenset(hero_positions or (
            position for positions in POSITIONS.values() for position in positions
        )),
        pot_buckets_bb=tuple(pot_buckets),
        allow_pot_interpolation=interpolate_pot,
        max_pot_distance_bb=max_pot_distance,
        aggressive_size_buckets_bb=tuple(aggressive_size_buckets),
        allow_aggressive_size_interpolation=interpolate_aggressive_size,
        max_aggressive_size_distance_bb=max_aggressive_size_distance,
    )


def candidate(
    ctx: DecisionContext,
    provider_id="mock-2p",
    provider_version="v1",
    *,
    match_kind=MatchKind.EXACT,
    score=1.0,
    probabilities=None,
    expires_at=None,
    match_dimensions=(),
) -> StrategyCandidate:
    probabilities = probabilities or {
        ActionType.CHECK: Decimal("0.4"),
        ActionType.RAISE: Decimal("0.6"),
    }
    dimensions = tuple(match_dimensions)
    if match_kind is MatchKind.INTERPOLATED and not dimensions:
        dimensions = (MatchDimension(
            "provider_abstraction",
            "requested_state",
            "matched_state",
            Decimal(str(1 - score)),
            Decimal("1"),
        ),)
    return StrategyCandidate(
        hand_id=ctx.hand_id,
        state_version=ctx.state_version,
        request_id=ctx.request_id,
        provider_id=provider_id,
        provider_version=provider_version,
        match_kind=match_kind,
        state_match_score=score,
        match_dimensions=dimensions,
        action_probabilities=probabilities,
        recommended_sizes=(
            {ActionType.RAISE: (ChipAmount("2.5"),)}
            if ActionType.RAISE in probabilities else {}
        ),
        action_ev={
            ActionType.CHECK: ChipDelta("0"),
            ActionType.RAISE: ChipDelta("1.25"),
        } if set(probabilities) >= {ActionType.CHECK, ActionType.RAISE} else {},
        confidence=0.8,
        evidence=(f"mock://provider/{provider_id}/{provider_version}",),
        expires_at=expires_at or ctx.request.expires_at,
    )


def hit_result(value: StrategyCandidate) -> ProviderResult:
    state = (
        LookupState.HIT_EXACT
        if value.match_kind is MatchKind.EXACT
        else LookupState.HIT_APPROXIMATE
    )
    return ProviderResult(state, value.provider_id, value)
