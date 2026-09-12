"""A full, currently calibrated seat census is required for live equity."""

from dataclasses import replace
from datetime import datetime, timezone

from poker_engine.confidence import ConfidenceGate
from poker_engine.core.enums import ActionType, PlayerStatus
from poker_engine.core.observation import (
    ObservationField, RawObservation, SlotObservation, ValidationStatus,
)
from poker_engine.core.value_objects import ChipAmount
from poker_engine.desktop.equity_guard import build_equity_input_guard
from poker_engine.desktop.live import _seed_state
from poker_engine.equity._deck import full_deck
from poker_engine.realtime.change_detector import detect_change
from poker_engine.state_engine.platform_mapping import PlatformSeatMapping


def _field(value, confidence=1.0, status=ValidationStatus.VALID):
    return ObservationField(value=value, confidence=confidence, source="test",
                            evidence={}, timestamp=datetime.now(timezone.utc),
                            validation_status=status)


def _scene():
    seed = _seed_state()
    state = replace(seed, hero_cards=full_deck()[:2], players=tuple(
        replace(p, status=PlayerStatus.ACTIVE, has_cards=True) for p in seed.players))
    obs = RawObservation(
        frame_seq=1, timestamp=datetime.now(timezone.utc),
        hero_cards=_field(state.hero_cards), board_cards=_field(()),
        pot=_field(state.pot), stacks=_field(()), bet_size=_field(ChipAmount("0")),
        action=_field(ActionType.CHECK), street=_field(state.street),
        dealer_pos=_field(0), actor=_field(0),
        slot_occupancies=tuple(SlotObservation(i, _field(True)) for i in range(8)),
    )
    identity = dict(enumerate(range(8)))
    mapping = PlatformSeatMapping(
        "wpk", "test", "1", identity, identity, identity, identity, identity)
    return state, obs, build_equity_input_guard(ConfidenceGate(), mapping)


def test_all_seats_must_be_present_and_pass_calibration():
    state, obs, guard = _scene()
    assert guard(state, obs) is None
    assert guard(state, replace(obs, slot_occupancies=obs.slot_occupancies[:-1])) == (
        "live_equity_occupancy_incomplete")
    weak = obs.slot_occupancies[:-1] + (SlotObservation(7, _field(True, 0.1)),)
    assert guard(state, replace(obs, slot_occupancies=weak)) == (
        "live_equity_occupancy_incomplete")


def test_stale_board_cannot_be_used_after_live_board_quality_drops():
    state, obs, guard = _scene()
    lost = replace(obs, board_cards=_field(None, status=ValidationStatus.UNKNOWN))
    assert guard(state, lost) == "live_equity_input_unavailable:board_cards"
    mismatch = replace(obs, hero_cards=_field(full_deck()[2:4]))
    assert guard(state, mismatch) == "live_equity_state_mismatch:hero_cards"


def test_actor_dealer_and_slot_only_changes_trigger_state_processing():
    _, obs, _ = _scene()
    changed = replace(obs, actor=_field(1), dealer_pos=_field(1),
                      slot_stacks=(SlotObservation(0, _field(ChipAmount("198"))),))
    assert set(detect_change(obs, changed).changed_fields) == {
        "actor", "dealer_pos", "slot_stacks"}
    changed = replace(obs, slot_occupancies=obs.slot_occupancies[:-1] + (
        SlotObservation(7, _field(False)),))
    assert detect_change(obs, changed).changed_fields == ("slot_occupancies",)
    lost = replace(obs, slot_occupancies=obs.slot_occupancies[:-1])
    assert not detect_change(obs, lost).changed
