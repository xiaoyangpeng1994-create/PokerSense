"""Shared fixtures/helpers for Task 1E serialization and contract tests."""

import json
from datetime import datetime, timezone

from poker_engine.core.serialization import deserialize, serialize

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandHistory, HandSummary
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import OpponentProfile, PlayerState
from poker_engine.core.reports import (
    Decision,
    DecisionPath,
    EquityMethod,
    EquityReport,
    ReasoningReport,
    StrategyReport,
    StrategySource,
)
from poker_engine.core.request_context import RequestContext
from poker_engine.core.state import PokerState, StateContext, ValidationResult
from poker_engine.core.value_objects import Card, ChipAmount, ChipDelta

UTC = timezone.utc

Ac = Card(Rank.ACE, Suit.CLUBS)
Ad = Card(Rank.ACE, Suit.DIAMONDS)
Kc = Card(Rank.KING, Suit.CLUBS)
Kh = Card(Rank.KING, Suit.HEARTS)
Qh = Card(Rank.QUEEN, Suit.HEARTS)
Jh = Card(Rank.JACK, Suit.HEARTS)
Th = Card(Rank.TEN, Suit.HEARTS)


def aware(**kw):
    base = dict(year=2026, month=8, day=19, hour=0, minute=30, tzinfo=UTC)
    base.update(kw)
    return datetime(**base)


def player(seat=0, pid=None, hero=False) -> PlayerState:
    return PlayerState(
        player_id=pid or f"p{seat}",
        seat=seat,
        position=Position.BTN if seat == 0 else Position.SB,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("1.25"),
        committed_this_hand=ChipAmount("1.25"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=(seat == 0),
    )


def poker_state() -> PokerState:
    return PokerState(
        state_version=3,
        hand_id="h1",
        street=Street.FLOP,
        hero_cards=(Ac, Kh),
        board_cards=(Kc, Qh, Jh),
        players=(player(0, hero=True), player(1)),
        pot=ChipAmount("30.5"),
        current_bet=ChipAmount("10"),
        to_call=ChipAmount("10"),
        actor=1,
    )


def state_event() -> StateEvent:
    return StateEvent(
        event_type=EventType.RAISE,
        hand_id="h1",
        state_version=2,
        payload={"amount": ChipAmount("12.5"), "cards": (Ac, Kh)},
        timestamp=aware(),
        source="state_engine",
    )


def observation_field(value, status=ValidationStatus.VALID):
    return ObservationField(
        value=value,
        confidence=0.95,
        source="test",
        evidence={"box": [1, 2, 3]},
        timestamp=aware(),
        validation_status=status,
    )


def raw_observation() -> RawObservation:
    return RawObservation(
        frame_seq=1,
        timestamp=aware(),
        hero_cards=observation_field((Ac, Kh)),
        board_cards=observation_field(()),
        pot=observation_field(ChipAmount("10")),
        stacks=observation_field((ChipAmount("100"),)),
        bet_size=observation_field(ChipAmount("5")),
        action=observation_field(ActionType.RAISE),
        street=observation_field(Street.PREFLOP),
        dealer_pos=observation_field(0),
        actor=observation_field(1),
        overall_confidence=0.9,
    )


def hand_summary() -> HandSummary:
    return HandSummary(
        final_pot=ChipAmount("100"),
        winners=("p0",),
        winnings={"p0": ChipAmount("100")},
        net_result={"p0": ChipDelta("50"), "p1": ChipDelta("-50")},
    )


def hand_history() -> HandHistory:
    return HandHistory(
        hand_id="h1",
        players=(player(0, hero=True), player(1)),
        events=(state_event(),),
        summary=hand_summary(),
        start_time=aware(),
        end_time=None,
    )


def equity_report() -> EquityReport:
    return EquityReport(
        win_rate=0.5,
        tie_rate=0.1,
        pot_odds=2.0,
        implied_odds=1.5,
        estimated_ev=ChipAmount("10"),
        method=EquityMethod.MONTECARLO,
        timestamp=aware(),
    )


def strategy_report() -> StrategyReport:
    return StrategyReport(
        action_frequencies={ActionType.FOLD: 0.2, ActionType.RAISE: 0.8},
        bet_sizes=(ChipAmount("10"), ChipAmount("20")),
        ev=ChipAmount("5"),
        strategy_source=StrategySource.CACHE,
        confidence=0.9,
        cache_hit=True,
        solver_metadata={"nodes": 100},
    )


def reasoning_report() -> ReasoningReport:
    return ReasoningReport(
        analysis_summary="nut flush draw",
        key_factors=("SPR = 2.8", "villain aggression"),
        suggested_action=ActionType.RAISE,
        suggested_size=ChipAmount("20"),
        confidence=0.85,
        source="poker_skill",
        hand_id="h1",
        request_id="r1",
        model_metadata={"model": "v1"},
        state_version=3,
        timestamp=aware(),
    )


def decision() -> Decision:
    return Decision(
        action=ActionType.RAISE,
        confidence=0.9,
        evidence_chain=("EquityReport", "StrategyReport"),
        raise_size=ChipAmount("15"),
        fast_or_slow=DecisionPath.FAST,
        timestamp=aware(),
        state_version=3,
    )


def opponent_profile() -> OpponentProfile:
    return OpponentProfile(
        player_id="p1",
        vpip=0.25,
        pfr=0.15,
        af=2.0,
        cbet_freq=0.6,
        threebet_freq=0.08,
        bluff_freq=0.3,
        sample_size=100,
        last_updated=aware(),
    )


def request_context() -> RequestContext:
    return RequestContext(
        hand_id="h1",
        state_version=3,
        request_id="req-abc",
        requested_at=aware(),
    )


def state_context() -> StateContext:
    return StateContext(
        previous_state=poker_state(),
        platform_rules={"blind": {"sb": "1", "bb": "2"}},
        confidence_thresholds={"hero_cards": 0.995},
        recent_events=(state_event(),),
    )


def validation_result() -> ValidationResult:
    return ValidationResult(
        is_valid=False,
        errors=("e1", "e2"),
        warnings=("w1",),
    )


def roundtrip(obj):
    """Full JSON chain: obj -> serialize -> json.dumps -> json.loads -> deserialize."""
    return deserialize(type(obj), json.loads(json.dumps(serialize(obj))))
