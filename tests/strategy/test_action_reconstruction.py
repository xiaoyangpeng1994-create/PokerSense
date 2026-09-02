from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import ActionType, PlayerStatus, Position, Street
from poker_engine.core.events import EventType
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.state_engine.action_reconstruction import (
    ReconstructionStatus,
    reconstruct_action_event,
)

from .helpers import card


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def player(
    seat,
    *,
    stack="100",
    committed="0",
    status=PlayerStatus.ACTIVE,
    hero=False,
):
    return PlayerState(
        f"p{seat}", seat,
        Position.BTN if seat == 0 else Position.BB,
        ChipAmount(stack), ChipAmount(committed), ChipAmount(committed),
        status, status is not PlayerStatus.FOLDED, hero, seat == 0,
    )


def state(
    *,
    actor=player(0),
    villain=player(1, hero=True),
    version=1,
    hand_id="h1",
    pot="0",
    current_bet="0",
    street=Street.PREFLOP,
    board=(),
):
    return PokerState(
        version, hand_id, street, (card("As"), card("Kd")), tuple(board),
        (actor, villain), ChipAmount(pot), ChipAmount(current_bet),
        ChipAmount("0"), 0,
    )


def result(before, after, action=None):
    return reconstruct_action_event(
        before, after, actor_seat=0, observed_action=action, timestamp=NOW
    )


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        (
            state(),
            state(actor=player(0, status=PlayerStatus.FOLDED), version=2),
            ActionType.FOLD,
        ),
        (state(), state(version=2), ActionType.CHECK),
        (
            state(villain=player(1, committed="10", hero=True),
                  pot="10", current_bet="10"),
            state(actor=player(0, stack="90", committed="10"),
                  villain=player(1, committed="10", hero=True),
                  version=2, pot="20", current_bet="10"),
            ActionType.CALL,
        ),
        (
            state(),
            state(actor=player(0, stack="80", committed="20"),
                  version=2, pot="20", current_bet="20"),
            ActionType.BET,
        ),
        (
            state(villain=player(1, committed="10", hero=True),
                  pot="10", current_bet="10"),
            state(actor=player(0, stack="70", committed="30"),
                  villain=player(1, committed="10", hero=True),
                  version=2, pot="40", current_bet="30"),
            ActionType.RAISE,
        ),
    ],
)
def test_unambiguous_actions_are_reconstructed(before, after, expected):
    value = result(before, after)
    assert value.status is ReconstructionStatus.EXACT
    assert value.candidates == (expected,)
    assert value.event.event_type is EventType(expected.value)
    assert not value.blocks_strategy


def test_event_payload_has_both_amount_semantics_and_chip_evidence():
    before = state()
    after = state(
        actor=player(0, stack="80", committed="20"),
        version=2, pot="20", current_bet="20",
    )
    event = result(before, after).event
    assert dict(event.payload) == {
        "seat_id": 0,
        "action": "bet",
        "amount_additional": "20",
        "amount_total_street": "20",
        "pot_delta": "20",
        "stack_before": "100",
        "stack_after": "80",
        "all_in": False,
    }
    assert event.timestamp == NOW
    assert event.state_version == 2


def test_short_all_in_call_without_label_is_ambiguous_and_blocks():
    before = state(
        actor=player(0, stack="40"),
        villain=player(1, stack="0", committed="100", hero=True),
        pot="100", current_bet="100",
    )
    after = state(
        actor=player(0, stack="0", committed="40",
                     status=PlayerStatus.ALL_IN),
        villain=player(1, stack="0", committed="100", hero=True),
        version=2, pot="140", current_bet="100",
    )
    value = result(before, after)
    assert value.status is ReconstructionStatus.AMBIGUOUS
    assert value.candidates == (ActionType.ALL_IN, ActionType.CALL)
    assert value.event is None
    assert value.blocks_strategy


@pytest.mark.parametrize("label", (ActionType.ALL_IN, ActionType.CALL))
def test_observed_label_resolves_short_all_in_call(label):
    before = state(
        actor=player(0, stack="40"),
        villain=player(1, stack="0", committed="100", hero=True),
        pot="100", current_bet="100",
    )
    after = state(
        actor=player(0, stack="0", committed="40",
                     status=PlayerStatus.ALL_IN),
        villain=player(1, stack="0", committed="100", hero=True),
        version=2, pot="140", current_bet="100",
    )
    value = result(before, after, label)
    assert value.status is ReconstructionStatus.EXACT
    assert value.candidates == (label,)
    assert value.event.payload["all_in"] is True


def test_all_in_raise_without_label_is_ambiguous():
    before = state(
        actor=player(0, stack="100"),
        villain=player(1, committed="50", hero=True),
        pot="50", current_bet="50",
    )
    after = state(
        actor=player(0, stack="0", committed="100",
                     status=PlayerStatus.ALL_IN),
        villain=player(1, committed="50", hero=True),
        version=2, pot="150", current_bet="100",
    )
    value = result(before, after)
    assert value.status is ReconstructionStatus.AMBIGUOUS
    assert value.candidates == (ActionType.ALL_IN, ActionType.RAISE)


def test_conflicting_observed_label_is_invalid_and_exposes_no_event():
    value = result(state(), state(version=2), ActionType.FOLD)
    assert value.status is ReconstructionStatus.INVALID
    assert value.candidates == (ActionType.CHECK,)
    assert value.reasons == ("observed_action_conflicts_with_deltas",)
    assert value.event is None


@pytest.mark.parametrize(
    ("before", "after", "reason"),
    [
        (state(), state(hand_id="h2", version=2), "hand_id_mismatch"),
        (state(), state(version=1), "state_version_not_advanced"),
        (
            state(),
            state(version=2, street=Street.FLOP,
                  board=(card("2c"), card("7d"), card("Jh"))),
            "street_changed_during_action",
        ),
        (
            state(),
            state(actor=player(0, stack="90", committed="10"),
                  version=2, pot="9", current_bet="10"),
            "chip_delta_mismatch",
        ),
        (
            state(),
            state(actor=player(0, stack="90", committed="10"),
                  villain=player(1, stack="99", committed="1", hero=True),
                  version=2, pot="11", current_bet="10"),
            "multiple_players_changed",
        ),
        (
            state(),
            state(actor=player(0, stack="90", committed="10"),
                  version=2, pot="10", current_bet="9"),
            "current_bet_mismatch",
        ),
    ],
)
def test_invalid_state_deltas_fail_closed(before, after, reason):
    value = result(before, after)
    assert value.status is ReconstructionStatus.INVALID
    assert value.reasons == (reason,)
    assert value.event is None
    assert value.blocks_strategy


def test_check_facing_bet_is_invalid():
    before = state(
        villain=player(1, committed="10", hero=True),
        pot="10", current_bet="10",
    )
    after = replace(before, state_version=2)
    assert result(before, after).reasons == ("check_facing_bet",)


def test_partial_non_all_in_call_is_invalid():
    before = state(
        villain=player(1, committed="10", hero=True),
        pot="10", current_bet="10",
    )
    after = state(
        actor=player(0, stack="95", committed="5"),
        villain=player(1, committed="10", hero=True),
        version=2, pot="15", current_bet="10",
    )
    assert result(before, after).reasons == ("action_amount_not_legal",)


@pytest.mark.parametrize(
    "forced", (ActionType.POST_SB, ActionType.POST_BB, ActionType.POST_ANTE)
)
def test_forced_actions_require_a_separate_reconstruction_path(forced):
    value = result(state(), state(version=2), forced)
    assert value.status is ReconstructionStatus.INVALID
    assert value.reasons == ("forced_action_not_supported",)


def test_contract_type_errors_are_not_converted_to_domain_results():
    with pytest.raises(TypeError, match="actor_seat"):
        reconstruct_action_event(
            state(), state(version=2), actor_seat=True,
            observed_action=None, timestamp=NOW,
        )
    with pytest.raises(TypeError, match="observed_action"):
        reconstruct_action_event(
            state(), state(version=2), actor_seat=0,
            observed_action="check", timestamp=NOW,
        )
    with pytest.raises(TypeError, match="timezone-aware"):
        reconstruct_action_event(
            state(), state(version=2), actor_seat=0,
            observed_action=None, timestamp=datetime(2026, 8, 22),
        )


@pytest.mark.parametrize(
    ("before_actor", "after_actor", "reason"),
    [
        (
            player(0, status=PlayerStatus.FOLDED),
            player(0, status=PlayerStatus.FOLDED),
            "actor_not_active",
        ),
        (
            player(0),
            replace(player(0, status=PlayerStatus.FOLDED), has_cards=True),
            "folded_actor_still_has_cards",
        ),
        (
            player(0),
            player(0, stack="1", status=PlayerStatus.ALL_IN),
            "all_in_actor_has_stack",
        ),
        (
            player(0),
            player(0, stack="0", committed="100"),
            "zero_stack_actor_not_all_in",
        ),
        (
            player(0),
            player(0, status=PlayerStatus.UNKNOWN),
            "actor_status_invalid",
        ),
    ],
)
def test_invalid_actor_status_transitions_fail_closed(
    before_actor, after_actor, reason
):
    before = state(actor=before_actor)
    spent = before_actor.stack.value - after_actor.stack.value
    after = state(
        actor=after_actor,
        version=2,
        pot=str(spent),
        current_bet=str(max(
            before.current_bet.value,
            after_actor.committed_this_street.value,
        )),
    )
    assert result(before, after).reasons == (reason,)


def test_player_identity_change_is_invalid():
    after_actor = replace(player(0), player_id="replacement")
    value = result(state(), state(actor=after_actor, version=2))
    assert value.reasons == ("player_identity_changed",)
