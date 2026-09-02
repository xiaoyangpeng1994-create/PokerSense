from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from poker_engine.core.enums import ActionType, Rank, Street, Suit
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.confidence import ConfidenceGate
from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.memory.hand_memory import InMemoryHandMemory
from poker_engine.orchestrator import ApplicationOrchestrator
from poker_engine.realtime.analysis import EquitySnapshot
from poker_engine.realtime.pipeline import RealtimePipeline
from poker_engine.realtime.temporal_consensus import TemporalConsensus
from poker_engine.state_engine import StateEngine


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)
FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "fixtures/strategy/v1/fixtures.jsonl"
)


def field(value, *, status=ValidationStatus.VALID, confidence=0.95):
    return ObservationField(
        value=value,
        confidence=confidence,
        source="vision:test",
        evidence={"fixture": "temporal"},
        timestamp=NOW,
        validation_status=status,
    )


def observation(
    frame_seq: int,
    *,
    pot="10",
    hero=None,
    action=ActionType.CALL,
    action_status=ValidationStatus.VALID,
    slot_stack="100",
) -> RawObservation:
    hero = hero or (
        Card(Rank.ACE, Suit.SPADES),
        Card(Rank.KING, Suit.HEARTS),
    )
    return RawObservation(
        frame_seq=frame_seq,
        timestamp=NOW + timedelta(milliseconds=frame_seq),
        hero_cards=field(hero),
        board_cards=field(()),
        pot=field(ChipAmount(pot)),
        stacks=field((ChipAmount("100"), ChipAmount("99"))),
        bet_size=field(ChipAmount("1")),
        action=field(action, status=action_status),
        street=field(Street.PREFLOP),
        dealer_pos=field(0),
        actor=field(1),
        slot_stacks=(
            SlotObservation(4, field(ChipAmount(slot_stack))),
        ),
        slot_actions=(
            SlotObservation(4, field(action, status=action_status)),
        ),
    )


def test_all_fields_require_two_identical_consecutive_frames():
    consensus = TemporalConsensus(default_frames=2)
    first = consensus.apply(observation(1))
    second = consensus.apply(observation(2))

    assert first.confirmed_fields == ()
    assert first.observation.hero_cards.value is None
    assert first.observation.pot.validation_status is ValidationStatus.UNKNOWN
    assert first.observation.slot_stacks[0].field.value is None
    assert second.pending_fields == ()
    assert "hero_cards" in second.confirmed_fields
    assert "pot" in second.confirmed_fields
    assert "slot_stacks[slot_id=4]" in second.confirmed_fields
    assert second.observation.pot.value == ChipAmount("10")
    assert second.observation.slot_stacks[0].field.value == ChipAmount("100")


def test_changed_candidate_restarts_confirmation_without_leaking_value():
    consensus = TemporalConsensus({"pot": 2})
    first = consensus.apply(observation(1, pot="10"))
    changed = consensus.apply(observation(2, pot="11"))
    confirmed = consensus.apply(observation(3, pot="11"))

    assert first.observation.pot.value is None
    assert changed.observation.pot.value is None
    assert changed.pending_fields == ("pot",)
    assert confirmed.observation.pot.value == ChipAmount("11")


def test_unknown_frame_breaks_consecutive_run():
    consensus = TemporalConsensus({"action": 2})
    consensus.apply(observation(1))
    unknown = replace(
        observation(2),
        action=field(None, status=ValidationStatus.UNKNOWN),
    )
    assert consensus.apply(unknown).observation.action.value is None
    assert consensus.apply(observation(3)).observation.action.value is None
    assert consensus.apply(observation(4)).observation.action.value is ActionType.CALL


def test_dropped_frame_sequence_does_not_count_as_consecutive():
    consensus = TemporalConsensus({"pot": 2})
    consensus.apply(observation(1))
    assert consensus.apply(observation(3)).observation.pot.value is None
    assert consensus.apply(observation(4)).observation.pot.value == ChipAmount("10")


def test_conflict_is_preserved_and_reported_not_rewritten_as_unknown():
    consensus = TemporalConsensus({"action": 2})
    value = consensus.apply(observation(
        1, action=None, action_status=ValidationStatus.CONFLICT
    ))
    assert value.conflict_fields == (
        "action", "slot_actions[slot_id=4]",
    )
    assert value.observation.action.validation_status is ValidationStatus.CONFLICT


def test_missing_slot_breaks_only_that_slot_pending_run():
    consensus = TemporalConsensus({"slot_stacks": 2})
    consensus.apply(observation(1))
    without_slot = replace(observation(2), slot_stacks=())
    consensus.apply(without_slot)
    assert (
        consensus.apply(observation(3)).observation.slot_stacks[0].field.value
        is None
    )
    assert consensus.apply(
        observation(4)
    ).observation.slot_stacks[0].field.value == ChipAmount("100")


def test_slot_identity_uses_slot_id_not_tuple_index():
    consensus = TemporalConsensus({"slot_stacks": 2})
    first = replace(
        observation(1),
        slot_stacks=(SlotObservation(7, field(ChipAmount("50"))),),
    )
    second = replace(
        observation(2),
        slot_stacks=(SlotObservation(7, field(ChipAmount("50"))),),
    )
    consensus.apply(first)
    result = consensus.apply(second)
    assert "slot_stacks[slot_id=7]" in result.confirmed_fields
    assert result.observation.slot_stacks[0].field.value == ChipAmount("50")


def test_threshold_one_is_immediate_and_preserves_original_field():
    raw = observation(1)
    result = TemporalConsensus(default_frames=1).apply(raw)
    assert result.observation == raw
    assert result.pending_fields == ()


@pytest.mark.parametrize("frames", [0, -1, True, 1.5])
def test_invalid_confirmation_frames_are_rejected(frames):
    error = TypeError if isinstance(frames, (bool, float)) else ValueError
    with pytest.raises(error):
        TemporalConsensus({"pot": frames})


def test_unknown_field_and_non_increasing_frame_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        TemporalConsensus({"not_a_field": 2})
    consensus = TemporalConsensus()
    consensus.apply(observation(2))
    with pytest.raises(ValueError, match="strictly"):
        consensus.apply(observation(2))


def test_realtime_pipeline_applies_consensus_before_canonical_state():
    class Frame:
        def __init__(self, frame_seq):
            self.frame_seq = frame_seq

    class Source:
        def __init__(self):
            self.frames = [Frame(1), Frame(2)]

        def next_frame(self):
            return self.frames.pop(0) if self.frames else None

    class Vision:
        def process(self, frame, table_map):
            assert table_map is marker
            return observation(frame.frame_seq, pot="10")

    class Equity:
        def compute(self, state):
            return EquitySnapshot(win_rate=0.5, tie_rate=0.0)

    marker = object()
    players = (
        PlayerState(
            player_id="villain",
            seat=0,
            position=Position.BTN,
            stack=ChipAmount("100"),
            committed_this_street=ChipAmount("0"),
            committed_this_hand=ChipAmount("0"),
            status=PlayerStatus.ACTIVE,
            has_cards=True,
            is_hero=False,
            is_dealer=True,
        ),
        PlayerState(
            player_id="hero",
            seat=1,
            position=Position.BB,
            stack=ChipAmount("100"),
            committed_this_street=ChipAmount("0"),
            committed_this_hand=ChipAmount("0"),
            status=PlayerStatus.ACTIVE,
            has_cards=True,
            is_hero=True,
            is_dealer=False,
        ),
    )
    initial = PokerState(
        state_version=0,
        hand_id="temporal-live",
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=players,
        pot=ChipAmount("0"),
        current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"),
        actor=1,
    )
    memory = InMemoryHandMemory()
    orchestrator = ApplicationOrchestrator(
        StateEngine(),
        memory,
        ConfidenceGate(thresholds={name: 0.9 for name in (
            "hero_cards", "board_cards", "pot", "stacks", "bet_size",
            "action", "street",
        )}),
    )
    orchestrator.start_hand(initial)
    pipeline = RealtimePipeline(
        Source(),
        Vision(),
        marker,
        orchestrator,
        equity_strategy=Equity(),
        confirmation_frames={"pot": 2},
    )

    first = pipeline.step()
    second = pipeline.step()

    assert first is not None and second is not None
    assert first.analysis.state.pot == ChipAmount("0")
    assert dict(first.analysis.confidence.field_status)["pot"] == "unknown"
    assert second.analysis.state.pot == ChipAmount("10")
    assert dict(second.analysis.confidence.field_status)["pot"] == "valid"


def test_temporal_mock_sequences_execute_against_production_consensus():
    fixtures = [
        json.loads(line) for line in FIXTURES.read_text().splitlines()
        if '"fixture_id": "MOCK-TEMPORAL-' in line
    ]
    assert len(fixtures) == 8

    for fixture in fixtures:
        spec = fixture["input"]["temporal_sequence"]
        expected = fixture["expected"]["temporal_consensus"]
        field_name = spec["field"]
        consensus = TemporalConsensus({
            field_name: spec["confirmation_frames"]
        })
        emitted = []
        confirmed_frames = []
        conflict_frames = []
        for item in spec["frames"]:
            status_name = item["status"]
            raw = observation(item["frame_seq"])
            if field_name == "pot":
                value = (
                    ChipAmount(item["value"])
                    if item["value"] is not None else None
                )
                raw = replace(raw, pot=field(
                    value, status=ValidationStatus[status_name]
                ))
                path = "pot"
            elif field_name == "action":
                value = (
                    ActionType(item["value"])
                    if item["value"] is not None else None
                )
                raw = replace(raw, action=field(
                    value, status=ValidationStatus[status_name]
                ))
                path = "action"
            elif field_name == "street":
                value = Street(item["value"])
                raw = replace(raw, street=field(value))
                path = "street"
            else:
                slot_id = item["slot_id"]
                path = f"slot_stacks[slot_id={slot_id}]"
                slots = () if status_name == "MISSING" else (
                    SlotObservation(
                        slot_id, field(ChipAmount(item["value"]))
                    ),
                )
                raw = replace(raw, slot_stacks=slots)

            result = consensus.apply(raw)
            if field_name == "slot_stacks":
                current = (
                    result.observation.slot_stacks[0].field.value
                    if result.observation.slot_stacks else None
                )
            else:
                current = getattr(result.observation, field_name).value
            emitted.append(
                current.value if isinstance(current, ActionType) else
                current.value if isinstance(current, Street) else
                str(current) if isinstance(current, ChipAmount) else current
            )
            if path in result.confirmed_fields:
                confirmed_frames.append(item["frame_seq"])
            if path in result.conflict_fields:
                conflict_frames.append(item["frame_seq"])

        assert emitted == expected["emitted_values"], fixture["fixture_id"]
        assert confirmed_frames == expected["confirmed_frames"]
        assert conflict_frames == expected["conflict_frames"]
