from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Street,
)
from poker_engine.core.events import EventType
from poker_engine.core.hand import HandSummary
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState, StateContext
from poker_engine.core.value_objects import ChipAmount
from poker_engine.memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.state_engine.platform_mapping import (
    CandidateMappingStatus,
    PlatformMappedStateEngine,
    PlatformSeatMapping,
    map_action_candidate,
)

from .helpers import card


NOW = datetime(2026, 8, 23, tzinfo=timezone.utc)


def amount(value):
    return ChipAmount(value)


def player(
    seat,
    *,
    stack="100",
    committed="0",
    status=PlayerStatus.ACTIVE,
    hero=False,
):
    positions = (Position.BTN, Position.SB, Position.BB)
    return PlayerState(
        player_id="hero" if hero else f"p{seat}",
        seat=seat,
        position=positions[seat],
        stack=amount(stack),
        committed_this_street=amount(committed),
        committed_this_hand=amount(committed),
        status=status,
        has_cards=status is not PlayerStatus.FOLDED,
        is_hero=hero,
        is_dealer=seat == 0,
    )


def state(
    *,
    actor_player=None,
    second=None,
    third=None,
    version=1,
    pot="0",
    current_bet="0",
    street=Street.PREFLOP,
    board=(),
):
    return PokerState(
        state_version=version,
        hand_id="mapping-hand",
        street=street,
        hero_cards=(card("As"), card("Kd")),
        board_cards=tuple(board),
        players=(
            actor_player or player(0),
            second or player(1),
            third or player(2, hero=True),
        ),
        pot=amount(pot),
        current_bet=amount(current_bet),
        to_call=amount("0"),
        actor=2,
    )


@pytest.fixture
def mapping():
    return PlatformSeatMapping(
        platform_id="synthetic",
        layout_id="three-seat-v1",
        version="mock-1",
        stack_slot_to_seat={30: 0, 31: 1, 32: 2},
        action_slot_to_seat={20: 0, 21: 1, 22: 2},
        actor_slot_to_seat={10: 0, 11: 1, 12: 2},
        dealer_slot_to_seat={40: 0, 41: 1, 42: 2},
        occupancy_slot_to_seat={50: 0, 51: 1, 52: 2},
    )


def field(value=None, status=ValidationStatus.UNKNOWN, confidence=1.0):
    return ObservationField(
        value=value,
        confidence=confidence,
        source="synthetic-replay",
        evidence={"frame": 101},
        timestamp=NOW,
        validation_status=status,
    )


def valid(value):
    return field(value, ValidationStatus.VALID)


def observation(
    *,
    actor_slot=10,
    action=None,
    action_slots=(),
    stacks=(),
    pot=None,
    dealer_slot=40,
    street=None,
    board=None,
    occupancies=(),
):
    return RawObservation(
        frame_seq=101,
        timestamp=NOW,
        hero_cards=valid((card("As"), card("Kd"))),
        board_cards=valid(tuple(board)) if board is not None else field(),
        pot=valid(amount(pot)) if pot is not None else field(),
        stacks=field(),
        bet_size=field(),
        action=valid(action) if action is not None else field(),
        street=valid(street) if street is not None else field(),
        dealer_pos=(
            valid(dealer_slot) if dealer_slot is not None else field()
        ),
        actor=valid(actor_slot) if actor_slot is not None else field(),
        overall_confidence=1.0,
        slot_stacks=tuple(
            SlotObservation(slot, valid(amount(value)))
            for slot, value in sorted(stacks)
        ),
        slot_actions=tuple(
            SlotObservation(slot, valid(value))
            for slot, value in sorted(action_slots)
        ),
        slot_occupancies=tuple(
            SlotObservation(slot, valid(value))
            for slot, value in sorted(occupancies)
        ),
    )


@pytest.mark.parametrize(
    ("before", "obs", "expected", "additional"),
    [
        (state(), observation(action=ActionType.CHECK), EventType.CHECK, "0"),
        (state(), observation(action=ActionType.FOLD), EventType.FOLD, "0"),
        (
            state(),
            observation(
                action=ActionType.BET, stacks=((30, "80"),), pot="20"
            ),
            EventType.BET,
            "20",
        ),
        (
            state(
                second=player(1, stack="90", committed="10"),
                pot="10",
                current_bet="10",
            ),
            observation(
                action=ActionType.CALL, stacks=((30, "90"),), pot="20"
            ),
            EventType.CALL,
            "10",
        ),
        (
            state(
                second=player(1, stack="90", committed="10"),
                pot="10",
                current_bet="10",
            ),
            observation(
                action=ActionType.RAISE, stacks=((30, "70"),), pot="40"
            ),
            EventType.RAISE,
            "30",
        ),
        (
            state(
                actor_player=player(0, stack="40"),
                second=player(
                    1, stack="0", committed="100",
                    status=PlayerStatus.ALL_IN,
                ),
                pot="100",
                current_bet="100",
            ),
            observation(
                action=ActionType.ALL_IN, stacks=((30, "0"),), pot="140"
            ),
            EventType.ALL_IN,
            "40",
        ),
    ],
)
def test_three_player_action_families_build_exact_candidate(
    mapping, before, obs, expected, additional,
):
    result = map_action_candidate(before, obs, mapping)
    assert result.status is CandidateMappingStatus.EXACT
    assert result.state.state_version == before.state_version + 1
    assert result.event.event_type is expected
    assert result.event.payload["amount_additional"] == additional
    assert result.event.source == (
        "platform_mapping:synthetic:three-seat-v1:mock-1"
    )


def test_slot_action_can_supply_actor_and_label(mapping):
    result = map_action_candidate(
        state(),
        observation(actor_slot=None, action_slots=((20, ActionType.CHECK),)),
        mapping,
    )
    assert result.status is CandidateMappingStatus.EXACT
    assert result.actor_seat == 0


@pytest.mark.parametrize("player_count", range(2, 10))
def test_mapping_contract_is_player_count_agnostic(player_count):
    players = tuple(
        PlayerState(
            "hero" if seat == player_count - 1 else f"p{seat}",
            seat,
            Position.BTN if seat == 0 else (
                Position.BB if seat == player_count - 1 else Position.UNKNOWN
            ),
            amount("100"),
            amount("0"),
            amount("0"),
            PlayerStatus.ACTIVE,
            True,
            seat == player_count - 1,
            seat == 0,
        )
        for seat in range(player_count)
    )
    before = PokerState(
        1,
        f"mapping-{player_count}p",
        Street.PREFLOP,
        (card("As"), card("Kd")),
        (),
        players,
        amount("0"),
        amount("0"),
        amount("0"),
        player_count - 1,
    )
    mapping = PlatformSeatMapping(
        "synthetic",
        f"{player_count}-seat",
        "mock-1",
        {100 + seat: seat for seat in range(player_count)},
        {200 + seat: seat for seat in range(player_count)},
        {300 + seat: seat for seat in range(player_count)},
        {400 + seat: seat for seat in range(player_count)},
    )
    obs = RawObservation(
        101,
        NOW,
        valid((card("As"), card("Kd"))),
        field(),
        field(),
        field(),
        field(),
        valid(ActionType.CHECK),
        field(),
        valid(400),
        valid(300),
        1.0,
    )
    result = map_action_candidate(before, obs, mapping)
    assert result.status is CandidateMappingStatus.EXACT
    assert result.event.event_type is EventType.CHECK


def test_no_action_evidence_is_a_noop_not_an_error(mapping):
    result = map_action_candidate(
        state(), observation(actor_slot=None, action=None), mapping
    )
    assert result.status is CandidateMappingStatus.NO_ACTION
    assert result.state is None and result.event is None


@pytest.mark.parametrize(
    ("obs", "reason", "status"),
    [
        (
            observation(actor_slot=None, action=ActionType.CHECK),
            "actor_missing",
            CandidateMappingStatus.AMBIGUOUS,
        ),
        (
            observation(actor_slot=99, action=ActionType.CHECK),
            "unmapped_actor_slot",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(
                actor_slot=10,
                action_slots=((21, ActionType.CHECK),),
            ),
            "conflicting_actor_slots",
            CandidateMappingStatus.AMBIGUOUS,
        ),
        (
            observation(
                action=ActionType.CHECK,
                action_slots=((20, ActionType.FOLD),),
            ),
            "conflicting_action_labels",
            CandidateMappingStatus.AMBIGUOUS,
        ),
        (
            observation(action_slots=((99, ActionType.CHECK),)),
            "unmapped_action_slot",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.CHECK, dealer_slot=99),
            "unmapped_dealer_slot",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.CHECK, dealer_slot=41),
            "dealer_mapping_conflicts_with_state",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.CHECK, stacks=((99, "100"),)),
            "unmapped_stack_slot",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.BET, stacks=((31, "90"),), pot="10"),
            "multiple_players_changed",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.BET, stacks=(), pot="10"),
            "actor_stack_missing_for_chip_action",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.BET, stacks=((30, "90"),)),
            "pot_missing_for_chip_action",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.BET, stacks=((30, "90"),), pot="9"),
            "chip_delta_mismatch",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.CHECK, stacks=((30, "90"),)),
            "non_chip_action_changed_stack",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.CHECK, street=Street.FLOP),
            "cards_or_street_changed_during_action",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(
                action=ActionType.CHECK,
                board=(card("2c"), card("7d"), card("Jh")),
            ),
            "cards_or_street_changed_during_action",
            CandidateMappingStatus.INVALID,
        ),
        (
            observation(action=ActionType.POST_BB),
            "forced_action_not_supported",
            CandidateMappingStatus.INVALID,
        ),
    ],
)
def test_mapping_faults_fail_closed(mapping, obs, reason, status):
    result = map_action_candidate(state(), obs, mapping)
    assert result.status is status
    assert result.reasons == (reason,)
    assert result.state is None
    assert result.event is None


def test_mapping_contract_is_immutable_and_one_to_one():
    source = {0: 0}
    mapping = PlatformSeatMapping(
        "p", "l", "v", source, {}, {}, {}
    )
    source[0] = 9
    assert mapping.stack_slot_to_seat[0] == 0
    with pytest.raises(TypeError):
        mapping.stack_slot_to_seat[1] = 1
    with pytest.raises(ValueError, match="one-to-one"):
        PlatformSeatMapping("p", "l", "v", {0: 0, 1: 0}, {}, {}, {})
    with pytest.raises(TypeError, match="slot keys"):
        PlatformSeatMapping("p", "l", "v", {True: 0}, {}, {}, {})


def test_mapped_engine_preserves_base_engine_for_non_action_frame(mapping):
    engine = PlatformMappedStateEngine(mapping)
    before = state()
    obs = observation(
        actor_slot=None,
        dealer_slot=None,
        street=Street.FLOP,
        board=(card("2c"), card("7d"), card("Jh")),
    )
    result = engine.transition(before, obs, StateContext(before))
    assert result.changed and result.validation.is_valid
    assert result.state.street is Street.FLOP
    assert [event.event_type for event in result.events] == [
        EventType.STREET_CHANGE,
        EventType.DEAL,
    ]


def test_snapshot_maps_occupancy_stack_dealer_and_positions(mapping):
    before = state(second=replace(
        player(1),
        stack=amount("0"),
        position=Position.UNKNOWN,
        status=PlayerStatus.SITTING_OUT,
        has_cards=False,
    ))
    obs = observation(
        actor_slot=None,
        action=None,
        stacks=((31, "75"),),
        occupancies=((50, True), (51, True), (52, True)),
    )
    result = PlatformMappedStateEngine(mapping).transition(
        before, obs, StateContext(before)
    )
    assert result.changed and result.validation.is_valid
    seats = {item.seat: item for item in result.state.players}
    assert seats[1].status is PlayerStatus.ACTIVE
    assert seats[1].stack == amount("75")
    assert [seats[index].position for index in range(3)] == [
        Position.BTN, Position.SB, Position.BB,
    ]


def test_snapshot_marks_empty_seat_sitting_out(mapping):
    before = state()
    obs = observation(
        actor_slot=None,
        action=None,
        dealer_slot=42,
        occupancies=((50, True), (51, False), (52, True)),
    )
    result = PlatformMappedStateEngine(mapping).transition(
        before, obs, StateContext(before)
    )
    seats = {item.seat: item for item in result.state.players}
    assert seats[1].status is PlayerStatus.SITTING_OUT
    assert not seats[1].has_cards
    assert seats[2].position is Position.BTN
    assert seats[0].position is Position.BB


def test_persistent_action_glyph_is_emitted_once_after_live_baseline(mapping):
    engine = PlatformMappedStateEngine(mapping)
    current = replace(state(), state_version=0)
    shown = observation(
        actor_slot=None,
        action=None,
        action_slots=((20, ActionType.CHECK),),
    )
    first = engine.transition(current, shown, StateContext(current))
    assert not first.changed and first.events == ()
    repeated = engine.transition(current, shown, StateContext(current))
    assert not repeated.changed and repeated.events == ()
    hidden = observation(actor_slot=None, action=None, action_slots=())
    engine.transition(current, hidden, StateContext(current))
    appeared = engine.transition(current, shown, StateContext(current))
    assert [event.event_type for event in appeared.events] == [EventType.CHECK]


def test_current_actor_semantics_promote_hero_without_stealing_action_actor(
    mapping,
):
    current_mapping = replace(mapping, actor_observation_is_current=True)
    before = state()
    hero_turn = observation(actor_slot=12, action=None)
    snapshot = PlatformMappedStateEngine(current_mapping).transition(
        before, hero_turn, StateContext(before)
    )
    assert snapshot.state.actor == 2

    opponent_action = observation(
        actor_slot=12,
        action=None,
        action_slots=((20, ActionType.CHECK),),
    )
    result = map_action_candidate(before, opponent_action, current_mapping)
    assert result.status is CandidateMappingStatus.EXACT
    assert result.actor_seat == 0
    assert result.state.actor == 2

    no_current_actor = observation(
        actor_slot=None,
        action=None,
        action_slots=((20, ActionType.CHECK),),
    )
    cleared = map_action_candidate(
        replace(before, actor=2), no_current_actor, current_mapping
    )
    assert cleared.status is CandidateMappingStatus.EXACT
    assert cleared.actor_seat == 0
    assert cleared.state.actor is None


def test_orchestrator_persists_mapped_state_and_event_atomically(mapping):
    memory = InMemoryHandMemory()
    engine = PlatformMappedStateEngine(mapping)
    app = ApplicationOrchestrator(engine, memory)
    before = state()
    app.start_hand(before, started_at=NOW)
    result = app.process_observation(
        observation(
            action=ActionType.BET, stacks=((30, "80"),), pot="20"
        )
    )
    assert result.persisted
    assert memory.latest_state(before.hand_id) == result.transition.state
    history = app.complete_hand(
        before.hand_id,
        HandSummary(final_pot=amount("20"), winners=()),
        ended_at=NOW,
    )
    assert len(history.events) == 1
    assert history.events[0].event_type is EventType.BET


def test_invalid_mapping_never_advances_memory(mapping):
    memory = InMemoryHandMemory()
    app = ApplicationOrchestrator(PlatformMappedStateEngine(mapping), memory)
    before = state()
    app.start_hand(before, started_at=NOW)
    result = app.process_observation(
        observation(
            action=ActionType.BET, stacks=((30, "90"),), pot="9"
        )
    )
    assert not result.persisted
    assert not result.transition.validation.is_valid
    assert memory.latest_state(before.hand_id) == before


def test_sequential_replay_advances_versions_without_guessing(mapping):
    engine = PlatformMappedStateEngine(mapping)
    current = state()
    replay = [
        observation(action=ActionType.CHECK),
        observation(action=ActionType.BET, stacks=((30, "90"),), pot="10"),
    ]
    versions = []
    for obs in replay:
        transition = engine.transition(current, obs, StateContext(current))
        assert transition.validation.is_valid
        assert transition.changed
        current = transition.state
        versions.append(current.state_version)
    assert versions == [2, 3]
    assert current.players[0].stack == amount("90")
    assert current.players[0].committed_this_street == amount("10")
    assert current.pot == amount("10")


def test_low_confidence_slot_values_do_not_enter_candidate(mapping):
    obs = observation(action=ActionType.BET, stacks=((30, "90"),), pot="10")
    obs = replace(
        obs,
        slot_stacks=(SlotObservation(
            30,
            field(amount("90"), ValidationStatus.LOW_CONFIDENCE),
        ),),
    )
    result = map_action_candidate(state(), obs, mapping)
    assert result.status is CandidateMappingStatus.INVALID
    assert result.reasons == ("actor_stack_missing_for_chip_action",)
