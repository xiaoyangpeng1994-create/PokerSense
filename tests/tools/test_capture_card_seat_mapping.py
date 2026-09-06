"""Tests for stage J seat mapping (capture-card platform).

Hardware-free and private-data-free: every test builds synthetic per-slot
``FieldValue`` evidence. The acceptance list mirrors guide section 13:

- 2-8 player occupancy combinations map one-to-one;
- the dealer badge is accepted in every visual slot;
- the hero is fixed at canonical seat 0;
- seats joining / leaving only change that seat's own entry;
- zero or multiple dealer reads fail closed (UNKNOWN / CONFLICT);
- an EMPTY slot carrying a stack is a cross-field conflict;
- the serialized mapping round-trips and rejects malformed documents;
- the production mapping plugs into ``map_snapshot_candidate`` and rejects
  unmapped slots.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Street
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.desktop.live import load_platform_seat_mapping
from poker_engine.state_engine.platform_mapping import (
    PlatformSeatMapping,
    map_snapshot_candidate,
)

from tools.capture_card_calibration import SLOT_COUNT
from tools.capture_card_calibration.schema import FieldValue, Occupancy
from tools.capture_card_calibration.seat_mapping import (
    CAPTURE_CARD_LAYOUT_ID,
    CAPTURE_CARD_PLATFORM_ID,
    SeatConsistencyError,
    build_capture_card_seat_mapping,
    check_occupancy_stack_consistency,
    derive_canonical_dealer,
    identity_slot_map,
    load_seat_mapping_file,
    seat_mapping_document,
    summarize_occupancy,
    write_seat_mapping_file,
)


NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


# --- mapping shape ---------------------------------------------------------


def test_identity_map_covers_the_full_ring_with_hero_at_zero():
    slots = identity_slot_map()
    assert slots == {i: i for i in range(SLOT_COUNT)}
    assert slots[0] == 0  # hero is always canonical seat 0


def test_build_mapping_uses_independent_maps_per_field():
    mapping = build_capture_card_seat_mapping()
    assert isinstance(mapping, PlatformSeatMapping)
    assert mapping.platform_id == CAPTURE_CARD_PLATFORM_ID
    assert mapping.layout_id == CAPTURE_CARD_LAYOUT_ID
    for field_map in (
        mapping.stack_slot_to_seat,
        mapping.action_slot_to_seat,
        mapping.actor_slot_to_seat,
        mapping.dealer_slot_to_seat,
        mapping.occupancy_slot_to_seat,
    ):
        assert dict(field_map) == identity_slot_map()
    assert mapping.actor_observation_is_current is True


def test_document_round_trip(tmp_path):
    path = write_seat_mapping_file(tmp_path / "seat_mapping.draft.json")
    mapping = load_seat_mapping_file(path)
    assert mapping.platform_id == CAPTURE_CARD_PLATFORM_ID
    assert dict(mapping.stack_slot_to_seat) == identity_slot_map()
    assert mapping.actor_observation_is_current is True


def test_document_matches_ldplayer_schema_shape():
    doc = seat_mapping_document()
    assert doc["schema_version"] == 1
    for name in (
        "stack_slot_to_seat",
        "action_slot_to_seat",
        "actor_slot_to_seat",
        "dealer_slot_to_seat",
        "occupancy_slot_to_seat",
    ):
        assert doc[name] == {str(i): i for i in range(SLOT_COUNT)}


@pytest.mark.parametrize(
    "patch",
    [
        {"schema_version": 2},
        {"platform_id": ""},
        {"stack_slot_to_seat": {}},
        {"stack_slot_to_seat": {"0": 0, "1": 0}},  # not one-to-one
        {"stack_slot_to_seat": {"0": 8}},  # outside the ring
        {"occupancy_slot_to_seat": {"1": 1, "2": 2}},  # hero slot missing
    ],
)
def test_load_rejects_malformed_documents(tmp_path, patch):
    doc = seat_mapping_document()
    doc.update(patch)
    path = tmp_path / "bad.json"
    import json

    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_seat_mapping_file(path)


# --- dealer reduction -------------------------------------------------------


def _dealers(*positive_slots: int) -> dict[int, FieldValue]:
    return {
        slot: FieldValue.valid(slot in positive_slots)
        for slot in range(SLOT_COUNT)
    }


@pytest.mark.parametrize("slot", range(SLOT_COUNT))
def test_dealer_accepted_in_every_visual_slot(slot):
    result = derive_canonical_dealer(_dealers(slot))
    assert result.status.name == "VALID"
    assert result.value == slot


def test_dealer_none_readable_is_unknown_not_guessed():
    unknowns = {slot: FieldValue.unknown() for slot in range(SLOT_COUNT)}
    assert derive_canonical_dealer(unknowns).status.name == "UNKNOWN"


def test_dealer_all_false_is_unknown():
    assert derive_canonical_dealer(_dealers()).status.name == "UNKNOWN"


def test_dealer_multiple_positives_conflict():
    assert derive_canonical_dealer(_dealers(2, 5)).status.name == "CONFLICT"


# --- occupancy / stack cross-field consistency -------------------------------


def test_empty_slot_with_stack_conflicts():
    occ = {0: FieldValue.valid(Occupancy.EMPTY)}
    stacks = {0: FieldValue.valid(178)}
    with pytest.raises(SeatConsistencyError):
        check_occupancy_stack_consistency(occ, stacks)


def test_empty_slot_with_stack_value_string_conflicts():
    occ = {3: FieldValue.valid(Occupancy.EMPTY.value)}
    stacks = {3: FieldValue.valid(50)}
    with pytest.raises(SeatConsistencyError):
        check_occupancy_stack_consistency(occ, stacks)


def test_empty_slot_without_stack_is_consistent():
    occ = {0: FieldValue.valid(Occupancy.EMPTY)}
    stacks = {0: FieldValue.unknown()}
    check_occupancy_stack_consistency(occ, stacks)


def test_occupied_slot_with_stack_is_consistent():
    occ = {1: FieldValue.valid(Occupancy.OCCUPIED)}
    stacks = {1: FieldValue.valid(250)}
    check_occupancy_stack_consistency(occ, stacks)


def test_unknown_occupancy_never_conflicts():
    occ = {2: FieldValue.unknown()}
    stacks = {2: FieldValue.valid(88)}
    check_occupancy_stack_consistency(occ, stacks)


# --- occupancy combinations (2-8 players, joins/leaves) ---------------------


def _occupancies(occupied: set[int]) -> dict[int, FieldValue]:
    return {
        slot: FieldValue.valid(
            Occupancy.OCCUPIED if slot in occupied else Occupancy.EMPTY
        )
        for slot in range(SLOT_COUNT)
    }


@pytest.mark.parametrize(
    "occupied",
    [
        {0, 4},                     # heads-up
        {0, 1, 4},                  # 3-handed
        {0, 1, 3, 5},               # 4-handed, gaps
        {0, 1, 2, 4, 6},            # 5-handed
        {0, 1, 2, 3, 5, 7},         # 6-handed
        {0, 1, 2, 3, 4, 5, 6},      # 7-handed
        set(range(SLOT_COUNT)),     # full ring
    ],
)
def test_occupancy_combinations_map_exactly(occupied):
    assert summarize_occupancy(_occupancies(occupied)) == tuple(sorted(occupied))


def test_occupancy_join_and_leave_change_only_that_seat():
    before = summarize_occupancy(_occupancies({0, 1, 4}))
    after_join = summarize_occupancy(_occupancies({0, 1, 4, 6}))
    after_leave = summarize_occupancy(_occupancies({0, 1}))
    assert set(after_join) - set(before) == {6}
    assert set(before) - set(after_leave) == {4}


def test_occupancy_all_unknown_yields_none_not_empty_table():
    unknowns = {slot: FieldValue.unknown() for slot in range(SLOT_COUNT)}
    assert summarize_occupancy(unknowns) is None


# --- integration with the production mapper ---------------------------------


def _observation(slot_occupancies: dict[int, bool]) -> RawObservation:
    def blank():
        return ObservationField(
            value=None, confidence=0.0, source="test",
            validation_status=ValidationStatus.UNKNOWN,
        )

    return RawObservation(
        frame_seq=1,
        timestamp=NOW,
        hero_cards=blank(),
        board_cards=blank(),
        pot=blank(),
        stacks=blank(),
        bet_size=blank(),
        action=blank(),
        street=ObservationField(
            value=Street.PREFLOP, confidence=1.0, source="test",
            validation_status=ValidationStatus.VALID,
        ),
        dealer_pos=blank(),
        actor=blank(),
        slot_occupancies=tuple(
            SlotObservation(
                slot_id=slot,
                field=ObservationField(
                    value=value, confidence=1.0, source="test",
                    validation_status=ValidationStatus.VALID,
                ),
            )
            for slot, value in slot_occupancies.items()
        ),
    )


def _state(seats) -> PokerState:
    players = tuple(
        PlayerState(
            player_id="hero" if seat == 0 else f"p{seat}",
            seat=seat,
            position=Position.UNKNOWN,
            stack=ChipAmount(100),
            committed_this_street=ChipAmount(0),
            committed_this_hand=ChipAmount(0),
            status=PlayerStatus.ACTIVE,
            has_cards=True,
            is_dealer=False,
            is_hero=seat == 0,
        )
        for seat in seats
    )
    return PokerState(
        state_version=0,
        hand_id="h1",
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=players,
        pot=ChipAmount(0),
        current_bet=ChipAmount(0),
        to_call=ChipAmount(0),
    )


def test_snapshot_mapping_sits_out_departed_seat():
    mapping = build_capture_card_seat_mapping()
    state = _state({0, 1, 4})
    obs = _observation({4: False})  # EMPTY
    snapshot, errors = map_snapshot_candidate(state, obs, mapping)
    assert errors == ()
    by_seat = {p.seat: p for p in snapshot.players}
    assert by_seat[4].status is PlayerStatus.SITTING_OUT
    assert by_seat[4].has_cards is False
    assert by_seat[0].status is PlayerStatus.ACTIVE


def test_snapshot_mapping_rejects_unmapped_slot():
    mapping = build_capture_card_seat_mapping()
    state = _state({0, 1})
    obs = _observation({9: False})  # outside the ring
    snapshot, errors = map_snapshot_candidate(state, obs, mapping)
    assert snapshot is None
    assert errors == ("unmapped_occupancy_slot",)


def test_snapshot_mapping_rejects_seat_outside_state():
    mapping = build_capture_card_seat_mapping()
    state = _state({0, 1})
    obs = _observation({5: True})  # OCCUPIED but not in state players
    snapshot, errors = map_snapshot_candidate(state, obs, mapping)
    assert snapshot is None
    assert errors == ("mapped_seat_not_in_state",)


# --- stage L: landed production resource ------------------------------------


def test_landed_config_loads_through_the_production_loader():
    """The committed seat-mapping file loads via live.load_platform_seat_mapping."""
    mapping = load_platform_seat_mapping(
        CAPTURE_CARD_PLATFORM_ID, CAPTURE_CARD_LAYOUT_ID
    )
    assert mapping.platform_id == CAPTURE_CARD_PLATFORM_ID
    assert mapping.layout_id == CAPTURE_CARD_LAYOUT_ID
    expected = identity_slot_map()
    assert dict(mapping.stack_slot_to_seat) == expected
    assert dict(mapping.action_slot_to_seat) == expected
    assert dict(mapping.actor_slot_to_seat) == expected
    assert dict(mapping.dealer_slot_to_seat) == expected
    assert dict(mapping.occupancy_slot_to_seat) == expected
    assert mapping.actor_observation_is_current is True


def test_landed_config_matches_the_stage_j_builder():
    """Land file and in-code builder never diverge (serialization contract)."""
    from pathlib import Path

    landed = load_seat_mapping_file(
        Path("configs") / "platform"
        / f"{CAPTURE_CARD_PLATFORM_ID}__{CAPTURE_CARD_LAYOUT_ID}"
        "_seat_mapping.json"
    )
    built = build_capture_card_seat_mapping()
    assert landed == built


def test_landed_config_loads_through_both_loaders():
    landed = load_platform_seat_mapping(
        CAPTURE_CARD_PLATFORM_ID, CAPTURE_CARD_LAYOUT_ID
    )
    assert landed == build_capture_card_seat_mapping()
