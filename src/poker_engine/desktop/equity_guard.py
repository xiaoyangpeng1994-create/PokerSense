"""Prevent incomplete live recognition from becoming a plausible equity."""

from poker_engine.confidence import ConfidenceGate
from poker_engine.core.enums import PlayerStatus
from poker_engine.core.observation import RawObservation, ValidationStatus
from poker_engine.core.state import PokerState
from poker_engine.state_engine.platform_mapping import PlatformSeatMapping


def build_equity_input_guard(gate: ConfidenceGate, mapping: PlatformSeatMapping):
    """Require current, calibrated cards and the complete mapped seat census."""
    def guard(state: PokerState, observation: RawObservation) -> str | None:
        obs = gate.apply(observation).observation
        for name in ("hero_cards", "street", "board_cards"):
            field = getattr(obs, name)
            if field.validation_status is not ValidationStatus.VALID:
                return f"live_equity_input_unavailable:{name}"
            if field.value != getattr(state, name):
                return f"live_equity_state_mismatch:{name}"
        slots = {slot.slot_id: slot.field for slot in obs.slot_occupancies}
        players = {player.seat: player for player in state.players}
        if set(mapping.occupancy_slot_to_seat.values()) != set(players):
            return "live_equity_seat_mapping_incomplete"
        for slot_id, seat in mapping.occupancy_slot_to_seat.items():
            field = slots.get(slot_id)
            if field is None or field.validation_status is not ValidationStatus.VALID:
                return "live_equity_occupancy_incomplete"
            player = players[seat]
            if player.status is PlayerStatus.UNKNOWN:
                return "live_equity_player_status_unavailable"
            if field.value != (player.status is not PlayerStatus.SITTING_OUT):
                return "live_equity_occupancy_state_mismatch"
        return None
    return guard
