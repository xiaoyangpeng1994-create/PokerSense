"""Production live-stream to StrategyOrchestrator binding tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from poker_engine.core.enums import PlayerStatus, Position, Street
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.desktop.strategy_live import LiveStrategySession
from poker_engine.realtime.analysis import (
    ConfidenceSnapshot,
    EquitySnapshot,
    RealtimeAnalysis,
    StateSnapshot,
)
from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.contracts import GameConfig, GameType
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.provider import FakeProvider
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, card, hit_result


def _player(seat: int, *, hero: bool) -> PlayerState:
    return PlayerState(
        player_id="hero" if hero else "villain",
        seat=seat,
        position=Position.BB if hero else Position.BTN,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("1"),
        committed_this_hand=ChipAmount("1"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=not hero,
    )


def _state(*, actor: int | None = 1, version: int = 1) -> PokerState:
    return PokerState(
        state_version=version,
        hand_id="live-hand",
        street=Street.PREFLOP,
        hero_cards=(card("As"), card("Kd")),
        board_cards=(),
        players=(_player(0, hero=False), _player(1, hero=True)),
        pot=ChipAmount("2"),
        current_bet=ChipAmount("1"),
        to_call=ChipAmount("0"),
        actor=actor,
    )


def _analysis(
    state: PokerState,
    *,
    valid: bool = True,
    frame_seq: int = 1,
) -> RealtimeAnalysis:
    status = "valid" if valid else "unknown"
    return RealtimeAnalysis(
        frame_seq,
        StateSnapshot.from_state(state),
        EquitySnapshot(0.52, 0.02),
        ConfidenceSnapshot(
            0.95 if valid else 0.0,
            tuple(
                (field, status)
                for field in (
                    "hero_cards",
                    "board_cards",
                    "street",
                    "pot",
                    "stacks",
                    "bet_size",
                    "action",
                )
            ),
        ),
    )


def _config() -> GameConfig:
    return GameConfig(
        variant="NLHE",
        game_type=GameType.CASH,
        max_seats=2,
        dealt_player_count=2,
        small_blind=ChipAmount("0.5"),
        big_blind=ChipAmount("1"),
        minimum_chip=ChipAmount("0.5"),
    )


def test_incomplete_live_inputs_emit_atomic_abstain_without_actions():
    state = _state(actor=None)
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter()),
        _config(),
        clock=lambda: NOW,
    )

    frame = session.frame(_analysis(state, valid=False), state)

    assert frame.advice.status is AdviceStatus.ABSTAIN
    assert not frame.advice.action_probabilities
    assert "actor" in frame.advice.missing_inputs
    assert session.current_context.input_quality.hard_failures == (
        "live_input_not_valid:hero_cards",
        "live_input_not_valid:street",
        "live_input_not_valid:pot",
        "live_input_not_valid:stacks",
        "live_input_not_valid:action",
    )


def test_valid_live_state_can_reach_ready_through_injected_provider():
    state = _state()
    provider = FakeProvider(
        "live-fast",
        "v1",
        capability((2,)),
        lambda context: hit_result(candidate(
            context, provider_id="live-fast", provider_version="v1"
        )),
    )
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter((provider,))),
        _config(),
        clock=lambda: NOW,
        action_line_resolver=lambda state, history: "unopened",
    )

    frame = session.frame(_analysis(state), state)

    assert frame.advice.status is AdviceStatus.READY
    assert frame.advice.strategy_source == "live-fast"
    assert frame.advice.hand_id == state.hand_id
    assert frame.advice.state_version == state.state_version


def test_same_state_reuses_request_and_new_version_gets_new_identity():
    now = [NOW]
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter()),
        _config(),
        clock=lambda: now[0],
    )
    state = _state(actor=None)
    session.frame(_analysis(state, valid=False), state)
    first_request = session.current_context.request_id

    session.frame(_analysis(state, valid=False, frame_seq=2), state)
    assert session.current_context.request_id == first_request

    updated = replace(state, state_version=2)
    session.frame(_analysis(updated, valid=False, frame_seq=3), updated)
    assert session.current_context.request_id != first_request


def test_expired_live_request_is_replaced_even_without_state_change():
    now = [NOW]
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter()),
        _config(),
        deadline_ms=100,
        clock=lambda: now[0],
    )
    state = _state(actor=None)
    session.frame(_analysis(state, valid=False), state)
    first_request = session.current_context.request_id

    now[0] += timedelta(milliseconds=101)
    session.frame(_analysis(state, valid=False, frame_seq=2), state)

    assert session.current_context.request_id != first_request


def test_same_state_confidence_drop_cannot_reuse_ready_advice():
    state = _state()
    provider = FakeProvider(
        "live-fast",
        "v1",
        capability((2,)),
        lambda context: hit_result(candidate(
            context, provider_id="live-fast", provider_version="v1"
        )),
    )
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter((provider,))),
        _config(),
        clock=lambda: NOW,
        action_line_resolver=lambda state, history: "unopened",
    )
    ready = session.frame(_analysis(state), state).advice

    abstain = session.frame(
        _analysis(state, valid=False, frame_seq=2), state
    ).advice

    assert ready.status is AdviceStatus.READY
    assert abstain.status is AdviceStatus.ABSTAIN
    assert not abstain.action_probabilities
    assert abstain.request_id != ready.request_id


def test_analysis_and_state_identity_mismatch_is_rejected():
    state = _state()
    session = LiveStrategySession(
        StrategyOrchestrator(StrategyRouter()),
        _config(),
        clock=lambda: NOW,
    )

    mismatched = replace(state, state_version=2)
    try:
        session.frame(_analysis(state), mismatched)
    except ValueError as exc:
        assert str(exc) == "analysis and canonical state identity must match"
    else:
        raise AssertionError("identity mismatch must fail")
