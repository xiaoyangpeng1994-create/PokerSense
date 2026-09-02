"""Rules-derived legal actions, side pots, and context builder tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Street,
)
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.opponents import PlayerState
from poker_engine.core.request_context import RequestContext
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import (
    ActionAmountSemantics,
    DecisionSeat,
    GameConfig,
    GameType,
    LegalAction,
    PotState,
    InputProvenance,
    InputSource,
    QualityStatus,
)
from poker_engine.strategy.context_factory import ContextQualityPolicy
from poker_engine.strategy.input_provenance import collect_input_provenance
from poker_engine.strategy.state import (
    build_decision_context,
    calculate_legal_actions,
    calculate_side_pots,
)

from .helpers import card


UTC = timezone.utc
FIXTURES = (
    Path(__file__).parents[1] / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)
SIDE_POT_IDS = {
    "MOCK-SIDE-POT-EQUAL-ALLIN",
    "MOCK-SIDE-POT-FOLDED-CONTRIBUTOR",
    "MOCK-SIDE-POT-THREE-ALLIN",
    "MOCK-SIDE-POT-TWO-ALLIN",
}
EXPECTED_ELIGIBILITY = {
    "MOCK-SIDE-POT-EQUAL-ALLIN": ((0, 1, 2, 3),),
    "MOCK-SIDE-POT-FOLDED-CONTRIBUTOR": ((1, 2),),
    "MOCK-SIDE-POT-THREE-ALLIN": ((0, 1, 2, 3), (1, 2, 3), (2, 3)),
    "MOCK-SIDE-POT-TWO-ALLIN": ((0, 1, 2), (1, 2)),
}


def _config(player_count: int = 2) -> GameConfig:
    return GameConfig(
        variant="NLHE",
        game_type=GameType.CASH,
        max_seats=player_count,
        dealt_player_count=player_count,
        small_blind=ChipAmount("1"),
        big_blind=ChipAmount("2"),
        minimum_chip=ChipAmount("1"),
    )


def _player(
    seat: int,
    *,
    stack: str = "100",
    street_committed: str = "0",
    hand_committed: str | None = None,
    status: PlayerStatus = PlayerStatus.ACTIVE,
    is_hero: bool = False,
    has_cards: bool = True,
    position: Position | None = None,
) -> PlayerState:
    committed = hand_committed or street_committed
    return PlayerState(
        player_id="hero" if is_hero else f"p{seat}",
        seat=seat,
        position=position or (Position.BB if is_hero else Position.SB),
        stack=ChipAmount(stack),
        committed_this_street=ChipAmount(street_committed),
        committed_this_hand=ChipAmount(committed),
        status=status,
        has_cards=has_cards,
        is_hero=is_hero,
        is_dealer=position is Position.BTN,
    )


def _state(
    *,
    players: tuple[PlayerState, ...] | None = None,
    pot: str = "0",
    current_bet: str = "0",
    to_call: str = "0",
    actor: int | None = 1,
    street: Street = Street.PREFLOP,
    hand_id: str = "hand-1",
    state_version: int = 1,
) -> PokerState:
    players = players or (
        _player(0, position=Position.SB),
        _player(1, is_hero=True, position=Position.BB),
    )
    board = {
        Street.PREFLOP: (),
        Street.FLOP: (card("2c"), card("7d"), card("Jh")),
        Street.TURN: (card("2c"), card("7d"), card("Jh"), card("9s")),
        Street.RIVER: (
            card("2c"), card("7d"), card("Jh"), card("9s"), card("3h"),
        ),
    }[street]
    return PokerState(
        state_version=state_version,
        hand_id=hand_id,
        street=street,
        hero_cards=(card("As"), card("Kd")),
        board_cards=board,
        players=players,
        pot=ChipAmount(pot),
        current_bet=ChipAmount(current_bet),
        to_call=ChipAmount(to_call),
        actor=actor,
    )


def _request(state: PokerState) -> RequestContext:
    requested = datetime(2026, 8, 22, tzinfo=UTC)
    return RequestContext(
        state.hand_id,
        state.state_version,
        "request-1",
        requested,
        expires_at=requested + timedelta(seconds=2),
        deadline_ms=300,
    )


def _side_pot_fixtures() -> list[dict]:
    values = []
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item["fixture_id"] in SIDE_POT_IDS:
            values.append(item)
    return sorted(values, key=lambda item: item["fixture_id"])


def _fixture_seats(item: dict) -> tuple[DecisionSeat, ...]:
    return tuple(
        DecisionSeat(
            seat_id=seat["seat_id"],
            player_id=seat["player_id"],
            position=Position(seat["position"]),
            stack=ChipAmount(seat["stack"]),
            street_committed=ChipAmount(seat["street_committed"]),
            hand_committed=ChipAmount(seat["hand_committed"]),
            status=PlayerStatus(seat["status"]),
            occupied=seat["occupied"],
            is_hero=seat["is_hero"],
            is_dealer=seat["is_dealer"],
        )
        for seat in item["input"]["state"]["seats"]
    )


@pytest.mark.parametrize(
    "fixture",
    _side_pot_fixtures(),
    ids=lambda item: item["fixture_id"],
)
def test_side_pots_match_synthetic_regression_fixtures(fixture):
    seats = _fixture_seats(fixture)
    result = calculate_side_pots(seats)

    assert [str(pot.amount) for pot in result.pots] == fixture["expected"][
        "pot_amounts"
    ]
    assert sum(
        (amount.value for amount in result.uncalled_returns.values()), Decimal("0")
    ) == Decimal(fixture["expected"]["uncalled_return"])
    folded = {
        seat.seat_id
        for seat in _fixture_seats(fixture)
        if seat.status is PlayerStatus.FOLDED
    }
    assert all(
        not folded.intersection(pot.eligible_seats) for pot in result.pots
    )
    assert tuple(pot.eligible_seats for pot in result.pots) == (
        EXPECTED_ELIGIBILITY[fixture["fixture_id"]]
    )
    committed = sum((seat.hand_committed.value for seat in seats), Decimal("0"))
    accounted = sum(
        (pot.amount.value for pot in result.pots), Decimal("0")
    ) + sum(
        (amount.value for amount in result.uncalled_returns.values()), Decimal("0")
    )
    assert accounted == committed


def test_side_pots_return_a_single_unmatched_commitment():
    seats = (
        DecisionSeat(
            0,
            "hero",
            Position.BB,
            ChipAmount("90"),
            ChipAmount("10"),
            ChipAmount("10"),
            PlayerStatus.ACTIVE,
            is_hero=True,
        ),
        DecisionSeat(
            1,
            "p1",
            Position.SB,
            ChipAmount("100"),
            ChipAmount("0"),
            ChipAmount("0"),
            PlayerStatus.FOLDED,
        ),
    )

    result = calculate_side_pots(seats)

    assert result.pots == ()
    assert result.uncalled_returns == {0: ChipAmount("10")}


def test_open_betting_round_keeps_unmatched_chips_in_provisional_pot():
    seats = (
        DecisionSeat(
            0,
            "p0",
            Position.SB,
            ChipAmount("99"),
            ChipAmount("1"),
            ChipAmount("1"),
            PlayerStatus.ACTIVE,
        ),
        DecisionSeat(
            1,
            "hero",
            Position.BB,
            ChipAmount("98"),
            ChipAmount("2"),
            ChipAmount("2"),
            PlayerStatus.ACTIVE,
            is_hero=True,
        ),
        DecisionSeat(
            2,
            "p2",
            Position.BTN,
            ChipAmount("100"),
            ChipAmount("0"),
            ChipAmount("0"),
            PlayerStatus.ACTIVE,
        ),
    )

    result = calculate_side_pots(seats, settle_uncalled=False)

    assert result.uncalled_returns == {}
    assert result.pots == (PotState("main", ChipAmount("3"), (0, 1, 2)),)


def test_side_pots_reject_a_tranche_with_no_eligible_player():
    seats = tuple(
        DecisionSeat(
            seat,
            f"p{seat}",
            Position.SB if seat == 0 else Position.BB,
            ChipAmount("90"),
            ChipAmount("10"),
            ChipAmount("10"),
            PlayerStatus.FOLDED,
        )
        for seat in range(2)
    )

    with pytest.raises(InvalidStateError, match="no eligible"):
        calculate_side_pots(seats)


def test_side_pots_reject_empty_seat_collection():
    with pytest.raises(ValueError, match="empty"):
        calculate_side_pots(())


def test_legal_actions_allow_check_and_bet_when_unopened():
    actions = calculate_legal_actions(_state(), _config())

    assert actions == (
        _legal(ActionType.CHECK, "0", "0", ActionAmountSemantics.NONE),
        _legal(ActionType.BET, "2", "100", ActionAmountSemantics.TOTAL_STREET),
    )


def test_legal_actions_allow_fold_call_and_raise_when_facing_bet():
    players = (
        _player(0, street_committed="10", position=Position.SB),
        _player(1, street_committed="2", is_hero=True, position=Position.BB),
    )
    state = _state(
        players=players,
        pot="12",
        current_bet="10",
        to_call="8",
    )

    actions = calculate_legal_actions(
        state,
        _config(),
        minimum_raise_increment=ChipAmount("6"),
    )

    assert actions == (
        _legal(ActionType.FOLD, "0", "0", ActionAmountSemantics.NONE),
        _legal(ActionType.CALL, "8", "8", ActionAmountSemantics.ADDITIONAL),
        _legal(
            ActionType.RAISE,
            "16",
            "102",
            ActionAmountSemantics.TOTAL_STREET,
        ),
    )


def test_short_stack_call_is_all_in_only():
    players = (
        _player(0, street_committed="10"),
        _player(1, stack="5", street_committed="2", is_hero=True),
    )
    state = _state(
        players=players,
        pot="12",
        current_bet="10",
        to_call="8",
    )

    actions = calculate_legal_actions(state, _config())

    assert actions == (
        _legal(ActionType.FOLD, "0", "0", ActionAmountSemantics.NONE),
        _legal(ActionType.ALL_IN, "5", "5", ActionAmountSemantics.ADDITIONAL),
    )


def test_stack_above_call_but_below_minimum_raise_adds_all_in():
    players = (
        _player(0, street_committed="10"),
        _player(1, stack="9", street_committed="2", is_hero=True),
    )
    state = _state(
        players=players,
        pot="12",
        current_bet="10",
        to_call="8",
    )

    actions = calculate_legal_actions(
        state,
        _config(),
        minimum_raise_increment=ChipAmount("6"),
    )

    assert [action.action for action in actions] == [
        ActionType.FOLD,
        ActionType.CALL,
        ActionType.ALL_IN,
    ]
    assert actions[-1].amount_semantics is ActionAmountSemantics.ADDITIONAL


def test_legal_actions_reject_non_positive_raise_increment():
    state = _state(
        players=(
            _player(0, street_committed="10"),
            _player(1, street_committed="2", is_hero=True),
        ),
        pot="12",
        current_bet="10",
        to_call="8",
    )

    with pytest.raises(ValueError, match="> 0"):
        calculate_legal_actions(
            state,
            _config(),
            minimum_raise_increment=ChipAmount("0"),
        )


@pytest.mark.parametrize(
    "state",
    (
        _state(actor=None),
        _state(players=(
            _player(0),
            _player(1, status=PlayerStatus.ALL_IN, is_hero=True),
        )),
    ),
)
def test_no_legal_actions_without_an_active_actor(state):
    assert calculate_legal_actions(state, _config()) == ()


def test_context_builder_produces_ready_multi_player_context():
    players = (
        _player(0, stack="90", hand_committed="10", position=Position.BTN),
        _player(1, stack="90", hand_committed="10", position=Position.SB),
        _player(
            2,
            stack="90",
            hand_committed="10",
            is_hero=True,
            position=Position.BB,
        ),
    )
    state = _state(players=players, pot="30", actor=2)

    context = build_decision_context(
        state,
        _request(state),
        _config(3),
        action_line="three_bet",
    )

    assert context.is_decision_ready
    assert context.strategy_player_count == 3
    assert context.effective_stack_bb == Decimal("45")
    assert context.input_quality.hard_failures == ()
    assert [str(pot.amount) for pot in context.pots] == ["30"]


def test_context_builder_aggregates_provenance_with_quality_policy():
    state = _state()
    provenance = (
        InputProvenance(
            "hero_cards", InputSource.VISION, QualityStatus.VALID,
            0.96, "frame://cards", datetime(2026, 8, 22, tzinfo=UTC),
        ),
        InputProvenance(
            "actor", InputSource.DERIVED, QualityStatus.VALID,
            0.84, "state://actor", datetime(2026, 8, 22, tzinfo=UTC),
        ),
    )
    value = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="unopened",
        input_provenance=provenance,
        quality_policy=ContextQualityPolicy(("hero_cards", "actor"), 0.8),
    )
    assert value.input_quality.overall_confidence == 0.84
    assert value.input_quality.is_decision_ready


def test_context_builder_accepts_automatically_collected_inputs():
    state = _state()
    collected = collect_input_provenance(
        manual_inputs={"actor": state.actor},
        config_inputs={"blinds": (ChipAmount("1"), ChipAmount("2"))},
    )
    value = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="unopened",
        collected_inputs=collected,
        quality_policy=ContextQualityPolicy(("actor", "blinds"), 0.8),
    )
    assert value.input_provenance == collected.provenance
    assert value.input_quality.is_decision_ready
    assert {item.source for item in value.input_provenance} == {
        InputSource.MANUAL, InputSource.CONFIG,
    }


def test_context_builder_collected_conflict_blocks_decision():
    state = _state()
    collected = collect_input_provenance(
        manual_inputs={"actor": state.actor},
        inferred_inputs={"actor": (state.actor or 0) + 1},
    )
    value = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="unopened",
        collected_inputs=collected,
        quality_policy=ContextQualityPolicy(("actor",), 0.8),
    )
    assert not value.is_decision_ready
    assert "conflict:actor" in value.input_quality.hard_failures


def test_context_builder_rejects_duplicate_provenance_interfaces():
    state = _state()
    collected = collect_input_provenance(manual_inputs={"actor": state.actor})
    with pytest.raises(ValueError, match="input_provenance or collected_inputs"):
        build_decision_context(
            state,
            _request(state),
            _config(),
            action_line="unopened",
            collected_inputs=collected,
            input_provenance=collected.provenance,
        )


def test_context_builder_quality_policy_missing_provenance_blocks_decision():
    state = _state()
    value = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="unopened",
        input_provenance=(InputProvenance(
            "hero_cards", InputSource.VISION, QualityStatus.VALID,
            0.96, "frame://cards", datetime(2026, 8, 22, tzinfo=UTC),
        ),),
        quality_policy=ContextQualityPolicy(("hero_cards", "actor"), 0.8),
    )
    assert value.input_quality.overall_confidence == 0.0
    assert "missing_provenance:actor" in value.input_quality.hard_failures
    assert not value.is_decision_ready


def test_context_builder_accepts_normal_open_blind_commitments():
    players = (
        _player(
            0,
            stack="99",
            street_committed="1",
            hand_committed="1",
            position=Position.SB,
        ),
        _player(
            1,
            stack="98",
            street_committed="2",
            hand_committed="2",
            is_hero=True,
            position=Position.BB,
        ),
    )
    state = _state(
        players=players,
        pot="3",
        current_bet="2",
        to_call="0",
        actor=1,
    )

    context = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="unopened",
    )

    assert context.is_decision_ready
    assert context.input_quality.hard_failures == ()
    assert context.pots == (PotState("main", ChipAmount("3"), (0, 1)),)


def test_context_builder_computes_pairwise_effective_stacks():
    players = (
        _player(0, stack="100", hand_committed="10", position=Position.BTN),
        _player(1, stack="25", hand_committed="10", position=Position.SB),
        _player(
            2,
            stack="50",
            hand_committed="10",
            is_hero=True,
            position=Position.BB,
        ),
    )
    state = _state(players=players, pot="30", actor=2)

    context = build_decision_context(
        state,
        _request(state),
        _config(3),
        action_line="raise",
    )

    assert [
        (item.opponent_seat, item.amount) for item in context.effective_stacks
    ] == [(0, ChipAmount("50")), (1, ChipAmount("25"))]
    assert context.effective_stack_bb == Decimal("12.5")


def test_context_builder_rejects_request_for_different_state():
    state = _state()
    wrong = RequestContext(
        "another-hand",
        state.state_version,
        "request-1",
        datetime(2026, 8, 22, tzinfo=UTC),
    )

    with pytest.raises(InvalidStateError, match="reference"):
        build_decision_context(state, wrong, _config(), action_line="unopened")


def test_context_builder_marks_missing_actor_position_and_action_line():
    players = (
        _player(0),
        _player(1, is_hero=True, position=Position.UNKNOWN),
    )
    state = _state(players=players, actor=None)

    context = build_decision_context(state, _request(state), _config())

    assert context.missing_fields == ("hero_position", "actor", "action_line")
    assert not context.is_decision_ready


def test_context_builder_refuses_inconsistent_commitment_breakdown():
    players = (
        _player(0, hand_committed="10"),
        _player(1, hand_committed="10", is_hero=True),
    )
    state = _state(players=players, pot="25")

    context = build_decision_context(
        state,
        _request(state),
        _config(),
        action_line="raise",
    )

    assert "commitment_breakdown_mismatch" in context.input_quality.hard_failures
    assert not context.is_decision_ready


def test_postflop_context_uses_active_count_not_dealt_count():
    players = tuple(
        _player(
            seat,
            stack="90",
            hand_committed="10",
            status=PlayerStatus.FOLDED if seat < 4 else PlayerStatus.ACTIVE,
            has_cards=seat >= 4,
            is_hero=seat == 5,
            position=(Position.SB if seat == 4 else Position.BB),
        )
        for seat in range(6)
    )
    state = _state(
        players=players,
        pot="60",
        actor=5,
        street=Street.FLOP,
    )

    context = build_decision_context(
        state,
        _request(state),
        _config(6),
        action_line="raise",
    )

    assert context.game_config.dealt_player_count == 6
    assert context.active_seats == (4, 5)
    assert context.strategy_player_count == 2


def _legal(
    action: ActionType,
    minimum: str,
    maximum: str,
    semantics: ActionAmountSemantics,
) -> LegalAction:
    return LegalAction(
        action,
        ChipAmount(minimum),
        ChipAmount(maximum),
        semantics,
    )
