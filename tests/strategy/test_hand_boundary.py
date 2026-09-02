from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from poker_engine.confidence import ConfidenceGate
from poker_engine.core.enums import (
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.events import EventType
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.realtime.analysis import EquitySnapshot
from poker_engine.realtime.hand_boundary import (
    HandBoundaryPolicy,
    HandBoundaryStatus,
    detect_hand_boundary,
)
from poker_engine.realtime.pipeline import RealtimePipeline
from poker_engine.state_engine import StateEngine


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)
FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "fixtures/strategy/v1/fixtures.jsonl"
)
HERO = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
BOARD = (
    Card(Rank.TWO, Suit.CLUBS),
    Card(Rank.SEVEN, Suit.DIAMONDS),
    Card(Rank.JACK, Suit.HEARTS),
    Card(Rank.NINE, Suit.SPADES),
    Card(Rank.THREE, Suit.HEARTS),
)


def card(value: str) -> Card:
    return Card(Rank(value[0]), Suit(value[1]))


def value_field(value, status=ValidationStatus.VALID):
    return ObservationField(
        value=value,
        confidence=0.99,
        source="vision:test",
        evidence={"frame": 2},
        timestamp=NOW,
        validation_status=status,
    )


def players(*, dealer_seat=0, stack0="100", stack1="100"):
    return (
        PlayerState(
            "villain", 0, Position.BTN, ChipAmount(stack0),
            ChipAmount("10"), ChipAmount("10"), PlayerStatus.ACTIVE,
            True, False, dealer_seat == 0,
        ),
        PlayerState(
            "hero", 1, Position.BB, ChipAmount(stack1),
            ChipAmount("10"), ChipAmount("10"), PlayerStatus.ACTIVE,
            True, True, dealer_seat == 1,
        ),
    )


def state(*, hand_id="h1", street=Street.RIVER, board=BOARD, pot="20"):
    return PokerState(
        state_version=5,
        hand_id=hand_id,
        street=street,
        hero_cards=HERO,
        board_cards=board,
        players=players(),
        pot=ChipAmount(pot),
        current_bet=ChipAmount("10"),
        to_call=ChipAmount("0"),
        actor=1,
    )


def observation(
    *,
    hero=HERO,
    board=(),
    street=Street.PREFLOP,
    pot="1.5",
    dealer_slot=0,
    stacks=("100", "100"),
    conflict_field=None,
):
    values = {
        "hero_cards": value_field(hero),
        "board_cards": value_field(board),
        "pot": value_field(ChipAmount(pot)),
        "stacks": value_field(tuple(ChipAmount(item) for item in stacks)),
        "bet_size": value_field(ChipAmount("0")),
        "action": value_field(None, ValidationStatus.UNKNOWN),
        "street": value_field(street),
        "dealer_pos": value_field(dealer_slot),
        "actor": value_field(1),
    }
    if conflict_field:
        values[conflict_field] = replace(
            values[conflict_field], validation_status=ValidationStatus.CONFLICT
        )
    return RawObservation(
        frame_seq=2,
        timestamp=NOW,
        overall_confidence=0.99,
        **values,
    )


def test_different_confirmed_hero_pair_is_sufficient_boundary():
    new_hero = (
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
    )
    result = detect_hand_boundary(state(), observation(hero=new_hero))
    assert result.status is HandBoundaryStatus.CONFIRMED
    assert result.evidence == ("hero_cards_changed",)


def test_postflop_reset_requires_street_board_and_supporting_signal():
    result = detect_hand_boundary(state(), observation())
    assert result.status is HandBoundaryStatus.CONFIRMED
    assert result.evidence == (
        "street_reset_to_preflop", "board_cleared", "pot_reset",
    )


def test_two_reset_hints_are_ambiguous_and_do_not_start_hand():
    result = detect_hand_boundary(state(), observation(pot="20"))
    assert result.status is HandBoundaryStatus.AMBIGUOUS
    assert result.reasons == ("insufficient_boundary_evidence",)


def test_single_reset_hint_is_same_hand():
    value = observation(board=BOARD, pot="20")
    result = detect_hand_boundary(state(), value)
    assert result.status is HandBoundaryStatus.SAME_HAND
    assert result.evidence == ("street_reset_to_preflop",)


def test_conflict_in_boundary_evidence_is_ambiguous_even_if_cards_change():
    new_hero = (
        Card(Rank.QUEEN, Suit.CLUBS),
        Card(Rank.QUEEN, Suit.DIAMONDS),
    )
    result = detect_hand_boundary(
        state(), observation(hero=new_hero, conflict_field="pot")
    )
    assert result.status is HandBoundaryStatus.AMBIGUOUS
    assert result.reasons == ("pot_conflict",)


def test_preflop_reset_needs_explicit_dealer_and_stack_mappings():
    previous = state(street=Street.PREFLOP, board=(), pot="20")
    current = observation(
        street=Street.PREFLOP,
        board=(),
        pot="1.5",
        dealer_slot=7,
        stacks=("150", "100"),
    )
    policy = HandBoundaryPolicy(
        dealer_slot_to_seat={7: 1},
        stack_index_to_seat=(0, 1),
    )
    result = detect_hand_boundary(previous, current, policy)
    assert result.status is HandBoundaryStatus.CONFIRMED
    assert set(result.evidence) == {
        "pot_reset", "dealer_changed", "stack_reset_or_payout",
    }


def test_visual_indices_are_not_inferred_without_platform_mapping():
    previous = state(street=Street.PREFLOP, board=(), pot="20")
    current = observation(
        street=Street.PREFLOP,
        board=(),
        pot="1.5",
        dealer_slot=7,
        stacks=("150", "100"),
    )
    result = detect_hand_boundary(previous, current)
    assert result.status is HandBoundaryStatus.SAME_HAND
    assert result.evidence == ("pot_reset",)


def test_invalid_or_non_unique_platform_mappings_are_rejected():
    with pytest.raises(ValueError, match="one-to-one"):
        HandBoundaryPolicy(dealer_slot_to_seat={0: 1, 2: 1})
    with pytest.raises(ValueError, match="unique"):
        HandBoundaryPolicy(stack_index_to_seat=(0, 0))
    with pytest.raises(TypeError):
        HandBoundaryPolicy(dealer_slot_to_seat={True: 1})


def test_orchestrator_records_unsettled_hand_end_before_successor():
    memory = InMemoryHandMemory()
    orchestrator = ApplicationOrchestrator(StateEngine(), memory)
    previous = state()
    successor = PokerState(
        state_version=0,
        hand_id="h2",
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=players(dealer_seat=1),
        pot=ChipAmount("0"),
        current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"),
        actor=0,
    )
    orchestrator.start_hand(previous, started_at=NOW)
    history = orchestrator.start_next_hand(
        successor,
        ended_at=NOW + timedelta(seconds=1),
        started_at=NOW + timedelta(seconds=1),
    )

    assert history.events[-1].event_type is EventType.HAND_END
    assert history.events[-1].payload == {
        "reason": "confirmed_hand_boundary",
        "successor_hand_id": "h2",
        "settled": False,
    }
    assert memory.active_hand_id == "h2"


def test_non_boundary_types_are_rejected():
    with pytest.raises(TypeError, match="previous"):
        detect_hand_boundary(object(), observation())
    with pytest.raises(TypeError, match="observation"):
        detect_hand_boundary(state(), object())
    with pytest.raises(TypeError, match="policy"):
        detect_hand_boundary(state(), observation(), object())


def test_realtime_pipeline_closes_hand_on_confirmed_reset_signature():
    class Frame:
        frame_seq = 2

    class Source:
        def __init__(self):
            self.done = False

        def next_frame(self):
            if self.done:
                return None
            self.done = True
            return Frame()

    class Vision:
        def process(self, frame, table_map):
            return observation()

    class Equity:
        def compute(self, current):
            return EquitySnapshot(win_rate=0.5, tie_rate=0.0)

    memory = InMemoryHandMemory()
    orchestrator = ApplicationOrchestrator(
        StateEngine(),
        memory,
        ConfidenceGate(thresholds={name: 0.9 for name in (
            "hero_cards", "board_cards", "pot", "stacks", "bet_size",
            "action", "street",
        )}),
    )
    previous = state()
    orchestrator.start_hand(previous, started_at=NOW - timedelta(seconds=1))
    clean_players = tuple(
        replace(
            player,
            committed_this_street=ChipAmount("0"),
            committed_this_hand=ChipAmount("0"),
        )
        for player in players(dealer_seat=1)
    )

    def successor():
        return PokerState(
            state_version=0,
            hand_id="h2",
            street=Street.PREFLOP,
            hero_cards=(),
            board_cards=(),
            players=clean_players,
            pot=ChipAmount("0"),
            current_bet=ChipAmount("0"),
            to_call=ChipAmount("0"),
            actor=0,
        )

    pipeline = RealtimePipeline(
        Source(),
        Vision(),
        object(),
        orchestrator,
        equity_strategy=Equity(),
        new_hand_state_factory=successor,
    )
    step = pipeline.step()

    assert step is not None
    assert step.hand_boundary.status is HandBoundaryStatus.CONFIRMED
    assert step.change.changed_fields == ("hand_boundary",)
    assert step.analysis.state.hand_id == "h2"
    assert memory.active_hand_id == "h2"
    assert memory.get_hand_history("h1").events[-1].event_type is EventType.HAND_END


def test_hand_boundary_mock_cases_execute_against_production_detector():
    fixtures = [
        json.loads(line) for line in FIXTURES.read_text().splitlines()
        if '"fixture_id": "MOCK-HAND-BOUNDARY-' in line
    ]
    assert len(fixtures) == 6

    for fixture in fixtures:
        spec = fixture["input"]["hand_boundary"]
        before = spec["previous"]
        current = spec["current"]
        previous = replace(
            state(
                street=Street(before["street"]),
                board=tuple(card(item) for item in before["board_cards"]),
                pot=before["pot"],
            ),
            hero_cards=tuple(card(item) for item in before["hero_cards"]),
        )
        policy_data = spec["policy"]
        current_obs = observation(
            hero=tuple(card(item) for item in current["hero_cards"]),
            board=tuple(card(item) for item in current["board_cards"]),
            street=Street(current["street"]),
            pot=current["pot"],
            dealer_slot=policy_data.get("dealer_slot", 0),
            stacks=tuple(policy_data.get("stacks", ("100", "100"))),
            conflict_field=current["conflict_field"],
        )
        policy = HandBoundaryPolicy(
            dealer_slot_to_seat={
                int(slot): seat for slot, seat in
                policy_data.get("dealer_slot_to_seat", {}).items()
            },
            stack_index_to_seat=tuple(
                policy_data.get("stack_index_to_seat", ())
            ),
        )
        result = detect_hand_boundary(previous, current_obs, policy)
        expected = fixture["expected"]["hand_boundary"]
        assert result.status.value == expected["status"], fixture["fixture_id"]
        assert list(result.evidence) == expected["evidence"]
        assert list(result.reasons) == expected["reasons"]
