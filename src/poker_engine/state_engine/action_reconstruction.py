"""Reconstruct one canonical player-action event from adjacent states."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.enums import ActionType, PlayerStatus
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.state import PokerState


class ReconstructionStatus(str, Enum):
    EXACT = "EXACT"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


_EVENT_TYPE = {
    ActionType.FOLD: EventType.FOLD,
    ActionType.CHECK: EventType.CHECK,
    ActionType.CALL: EventType.CALL,
    ActionType.BET: EventType.BET,
    ActionType.RAISE: EventType.RAISE,
    ActionType.ALL_IN: EventType.ALL_IN,
}


@dataclass(frozen=True)
class ActionReconstruction:
    status: ReconstructionStatus
    candidates: tuple[ActionType, ...]
    event: StateEvent | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        reasons = tuple(self.reasons)
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidates must be unique")
        if not all(isinstance(item, ActionType) for item in candidates):
            raise TypeError("candidates must contain ActionType values")
        if not all(isinstance(item, str) and item for item in reasons):
            raise TypeError("reasons must contain non-empty strings")
        if self.status is ReconstructionStatus.EXACT:
            if len(candidates) != 1 or not isinstance(self.event, StateEvent):
                raise ValueError("EXACT requires one candidate and one event")
        elif self.event is not None:
            raise ValueError("non-EXACT reconstruction cannot expose an event")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "reasons", reasons)

    @property
    def blocks_strategy(self) -> bool:
        return self.status is not ReconstructionStatus.EXACT


def _invalid(reason: str) -> ActionReconstruction:
    return ActionReconstruction(
        ReconstructionStatus.INVALID, (), reasons=(reason,)
    )


def reconstruct_action_event(
    previous: PokerState,
    current: PokerState,
    *,
    actor_seat: int,
    observed_action: ActionType | None,
    timestamp: datetime,
    source: str = "state_engine.action_reconstruction",
) -> ActionReconstruction:
    """Reconcile action label with stack/commitment/pot deltas.

    The function is pure and conservative.  It emits no event when evidence
    admits multiple action meanings or violates chip/state invariants.
    """
    if not isinstance(previous, PokerState) or not isinstance(current, PokerState):
        raise TypeError("previous and current must be PokerState values")
    if not isinstance(actor_seat, int) or isinstance(actor_seat, bool):
        raise TypeError("actor_seat must be an int")
    if observed_action is not None and not isinstance(observed_action, ActionType):
        raise TypeError("observed_action must be an ActionType or None")
    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    _require_aware_dt(timestamp)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty str")
    if observed_action in {
        ActionType.POST_SB, ActionType.POST_BB, ActionType.POST_ANTE,
    }:
        return _invalid("forced_action_not_supported")
    if previous.hand_id != current.hand_id:
        return _invalid("hand_id_mismatch")
    if current.state_version <= previous.state_version:
        return _invalid("state_version_not_advanced")
    if previous.street is not current.street:
        return _invalid("street_changed_during_action")
    if previous.hero_cards != current.hero_cards or (
        previous.board_cards != current.board_cards
    ):
        return _invalid("cards_changed_during_action")
    before_by_seat = {player.seat: player for player in previous.players}
    after_by_seat = {player.seat: player for player in current.players}
    if set(before_by_seat) != set(after_by_seat):
        return _invalid("seat_set_changed")
    if actor_seat not in before_by_seat:
        return _invalid("actor_not_found")
    for seat in before_by_seat:
        before = before_by_seat[seat]
        after = after_by_seat[seat]
        if (
            before.player_id != after.player_id
            or before.position is not after.position
            or before.is_hero != after.is_hero
            or before.is_dealer != after.is_dealer
        ):
            return _invalid("player_identity_changed")
        if seat == actor_seat:
            continue
        if (
            before.stack != after.stack
            or before.committed_this_street != after.committed_this_street
            or before.committed_this_hand != after.committed_this_hand
            or before.status is not after.status
            or before.has_cards != after.has_cards
        ):
            return _invalid("multiple_players_changed")

    before = before_by_seat[actor_seat]
    after = after_by_seat[actor_seat]
    if before.status is not PlayerStatus.ACTIVE or not before.has_cards:
        return _invalid("actor_not_active")
    if after.status is PlayerStatus.FOLDED and after.has_cards:
        return _invalid("folded_actor_still_has_cards")
    if after.status is PlayerStatus.ALL_IN and after.stack.value != 0:
        return _invalid("all_in_actor_has_stack")
    if after.stack.value == 0 and after.status is not PlayerStatus.ALL_IN:
        return _invalid("zero_stack_actor_not_all_in")
    if after.status not in {
        PlayerStatus.ACTIVE, PlayerStatus.FOLDED, PlayerStatus.ALL_IN,
    }:
        return _invalid("actor_status_invalid")
    stack_spent = before.stack.value - after.stack.value
    street_delta = (
        after.committed_this_street.value
        - before.committed_this_street.value
    )
    hand_delta = (
        after.committed_this_hand.value - before.committed_this_hand.value
    )
    pot_delta = current.pot.value - previous.pot.value
    if min(stack_spent, street_delta, hand_delta, pot_delta) < 0:
        return _invalid("money_regression")
    if not stack_spent == street_delta == hand_delta == pot_delta:
        return _invalid("chip_delta_mismatch")
    expected_current_bet = max(
        previous.current_bet.value,
        after.committed_this_street.value,
    )
    if current.current_bet.value != expected_current_bet:
        return _invalid("current_bet_mismatch")
    to_call_before = max(
        Decimal("0"),
        previous.current_bet.value - before.committed_this_street.value,
    )

    candidates: list[ActionType] = []
    folded = (
        before.status is not PlayerStatus.FOLDED
        and after.status is PlayerStatus.FOLDED
    )
    all_in = after.stack.value == 0 and stack_spent > 0
    if folded:
        if stack_spent != 0:
            return _invalid("fold_spent_chips")
        candidates.append(ActionType.FOLD)
    elif after.status is PlayerStatus.FOLDED:
        return _invalid("actor_already_folded")
    elif stack_spent == 0:
        if to_call_before != 0:
            return _invalid("check_facing_bet")
        candidates.append(ActionType.CHECK)
    else:
        new_total = after.committed_this_street.value
        if all_in:
            candidates.append(ActionType.ALL_IN)
        if to_call_before > 0 and stack_spent == min(
            to_call_before, before.stack.value
        ):
            candidates.append(ActionType.CALL)
        elif previous.current_bet.value == 0 and new_total > 0:
            candidates.append(ActionType.BET)
        elif new_total > previous.current_bet.value:
            candidates.append(ActionType.RAISE)
        else:
            return _invalid("action_amount_not_legal")

    candidates = list(dict.fromkeys(candidates))
    if observed_action is not None:
        if observed_action not in candidates:
            return ActionReconstruction(
                ReconstructionStatus.INVALID,
                tuple(candidates),
                reasons=("observed_action_conflicts_with_deltas",),
            )
        candidates = [observed_action]
    if len(candidates) != 1:
        return ActionReconstruction(
            ReconstructionStatus.AMBIGUOUS,
            tuple(candidates),
            reasons=("multiple_legal_action_interpretations",),
        )

    action = candidates[0]
    event = StateEvent(
        event_type=_EVENT_TYPE[action],
        hand_id=current.hand_id,
        state_version=current.state_version,
        payload={
            "seat_id": actor_seat,
            "action": action.value,
            "amount_additional": format(stack_spent, "f"),
            "amount_total_street": format(
                after.committed_this_street.value, "f"
            ),
            "pot_delta": format(pot_delta, "f"),
            "stack_before": format(before.stack.value, "f"),
            "stack_after": format(after.stack.value, "f"),
            "all_in": all_in,
        },
        timestamp=timestamp,
        source=source,
    )
    return ActionReconstruction(
        ReconstructionStatus.EXACT, (action,), event=event
    )


__all__ = [
    "ActionReconstruction",
    "ReconstructionStatus",
    "reconstruct_action_event",
]
