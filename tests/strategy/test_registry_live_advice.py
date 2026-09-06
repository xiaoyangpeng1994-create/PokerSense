"""End-to-end proof that the wired production pipeline emits real advice.

These tests exercise the exact objects ``live.py`` builds -- registry-built
router, production action-line resolver, ``LiveStrategySession`` -- so a
regression in any layer shows up here as advice falling back to ABSTAIN.
"""

from __future__ import annotations

import pytest

from poker_engine.core.enums import ActionType, PlayerStatus, Position, Street
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
from poker_engine.strategy.action_line import build_action_line_resolver
from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.contracts import GameConfig, GameType
from poker_engine.strategy.orchestration import StrategyOrchestrator
from poker_engine.strategy.registry import (
    build_strategy_router,
    registered_provider_ids,
)

from .helpers import NOW, card

# Seat order follows upstream's get_positions(): an 8-handed table runs
# UTG, UTG+1, MP(LJ), HJ, CO, BTN, SB, BB -- there is no UTG+2 until 9-handed.
_SEAT_ORDER = {
    6: (Position.UTG, Position.HJ, Position.CO,
        Position.BTN, Position.SB, Position.BB),
    7: (Position.UTG, Position.LJ, Position.HJ, Position.CO,
        Position.BTN, Position.SB, Position.BB),
    8: (Position.UTG, Position.UTG1, Position.LJ, Position.HJ,
        Position.CO, Position.BTN, Position.SB, Position.BB),
}

# 100bb measured against the live GameConfig's 2-chip big blind.
_STACK = "200"
_BIG_BLIND = "2"


def _state(
    *,
    seats: int = 6,
    hero_cards=("As", "Ks"),
    hero_seat: int = 0,
    street: Street = Street.PREFLOP,
) -> PokerState:
    order = _SEAT_ORDER[seats]
    sb_seat = seats - 2
    bb_seat = seats - 1
    small_blind = "1"

    def committed(seat: int) -> str:
        if seat == sb_seat:
            return small_blind
        if seat == bb_seat:
            return _BIG_BLIND
        return "0"

    players = tuple(
        PlayerState(
            player_id=f"seat-{seat}",
            seat=seat,
            position=order[seat],
            stack=ChipAmount(_STACK),
            committed_this_street=ChipAmount(committed(seat)),
            committed_this_hand=ChipAmount(committed(seat)),
            status=PlayerStatus.ACTIVE,
            has_cards=True,
            is_hero=seat == hero_seat,
            is_dealer=seat == seats - 2,
        )
        for seat in range(seats)
    )
    board = ("2c", "7d", "Jh") if street is not Street.PREFLOP else ()
    return PokerState(
        state_version=1,
        hand_id="registry-hand",
        street=street,
        hero_cards=tuple(card(value) for value in hero_cards),
        board_cards=tuple(card(value) for value in board),
        players=players,
        pot=ChipAmount("3"),
        current_bet=ChipAmount(_BIG_BLIND),
        to_call=ChipAmount(_BIG_BLIND),
        actor=hero_seat,
    )


def _analysis(state: PokerState) -> RealtimeAnalysis:
    return RealtimeAnalysis(
        1,
        StateSnapshot.from_state(state),
        EquitySnapshot(0.62, 0.02),
        ConfidenceSnapshot(
            0.95,
            tuple(
                (field, "valid")
                for field in (
                    "hero_cards", "board_cards", "street", "pot",
                    "stacks", "bet_size", "action",
                )
            ),
        ),
    )


def _session(max_seats: int = 6) -> LiveStrategySession:
    return LiveStrategySession(
        StrategyOrchestrator(build_strategy_router()),
        GameConfig(
            variant="NLHE",
            game_type=GameType.CASH,
            max_seats=max_seats,
            dealt_player_count=max_seats,
            small_blind=ChipAmount("1"),
            big_blind=ChipAmount(_BIG_BLIND),
            minimum_chip=ChipAmount("1"),
        ),
        clock=lambda: NOW,
        action_line_resolver=build_action_line_resolver(),
    )


def test_registry_registers_the_bundled_rfi_provider():
    assert registered_provider_ids() == ("preflopr-explicit-rfi-heuristic",)


def test_six_max_utg_open_hand_emits_ready_advice():
    state = _state(hero_cards=("As", "Ks"))

    frame = _session().frame(_analysis(state), state)

    assert frame.advice.status is AdviceStatus.READY
    assert frame.advice.strategy_source == "preflopr-explicit-rfi-heuristic"
    assert frame.advice.hand_id == state.hand_id
    assert frame.advice.state_version == state.state_version


def test_six_max_utg_trash_hand_emits_ready_advice():
    state = _state(hero_cards=("7c", "2d"))

    frame = _session().frame(_analysis(state), state)

    assert frame.advice.status is AdviceStatus.READY
    assert frame.advice.action_probabilities


def test_open_range_hand_prefers_a_raise():
    state = _state(hero_cards=("As", "Ks"))

    advice = _session().frame(_analysis(state), state).advice

    assert advice.preferred_action is ActionType.RAISE


def test_trash_hand_prefers_a_fold():
    state = _state(hero_cards=("7c", "2d"))

    advice = _session().frame(_analysis(state), state).advice

    assert advice.preferred_action is ActionType.FOLD


@pytest.mark.parametrize("seats", [7, 8])
def test_seven_and_eight_max_emit_derived_advice(seats):
    """7/8-max are covered by same-named 9-handed keys, disclosed as derived.

    Coverage is real, but it is one approximation step removed from the
    authored chart, so the advice must arrive labelled and down-weighted
    rather than masquerading as a table-size-specific chart.
    """
    state = _state(seats=seats, hero_cards=("As", "Ks"))

    advice = _session(max_seats=seats).frame(_analysis(state), state).advice

    assert advice.status is AdviceStatus.READY
    assert advice.strategy_source == "preflopr-explicit-rfi-heuristic"
    assert advice.preferred_action is ActionType.RAISE
    assert advice.confidence == pytest.approx(0.3)
    assert any(
        item.startswith(f"derived_range:{seats}_UTG<-9_UTG:")
        for item in advice.evidence
    )


def test_derived_advice_is_down_weighted_against_authored_advice():
    """A derived range must not claim the confidence of an authored one."""
    six_max = _session().frame(_analysis(_state()), _state()).advice
    seven_max = (
        _session(max_seats=7)
        .frame(_analysis(_state(seats=7)), _state(seats=7))
        .advice
    )

    assert six_max.confidence == pytest.approx(0.4)
    assert seven_max.confidence < six_max.confidence


def test_derived_advice_discloses_the_substitution_in_assumptions():
    state = _state(seats=8)

    advice = _session(max_seats=8).frame(_analysis(state), state).advice

    assert any(
        "9_handed" in item or "tighter" in item for item in advice.assumptions
    )


def test_postflop_emits_no_strategy_because_none_is_released():
    state = _state(street=Street.FLOP, hero_cards=("As", "Ks"))

    advice = _session().frame(_analysis(state), state).advice

    assert advice.status is not AdviceStatus.READY
    assert advice.strategy_source is None
