"""Fail-closed hand-boundary detection from stable observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from poker_engine.core.observation import RawObservation, ValidationStatus
from poker_engine.core.state import PokerState


class HandBoundaryStatus(str, Enum):
    SAME_HAND = "SAME_HAND"
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class HandBoundaryPolicy:
    """Platform semantics needed to interpret global visual slot numbers."""

    dealer_slot_to_seat: Mapping[int, int] = field(default_factory=dict)
    stack_index_to_seat: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        dealer = dict(self.dealer_slot_to_seat)
        if not all(
            isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0
            and isinstance(seat, int) and not isinstance(seat, bool) and seat >= 0
            for slot, seat in dealer.items()
        ):
            raise TypeError("dealer slot and seat IDs must be non-negative ints")
        if len(dealer.values()) != len(set(dealer.values())):
            raise ValueError("dealer_slot_to_seat must be one-to-one")
        stack_order = tuple(self.stack_index_to_seat)
        if not all(
            isinstance(seat, int) and not isinstance(seat, bool) and seat >= 0
            for seat in stack_order
        ):
            raise TypeError("stack_index_to_seat must contain non-negative ints")
        if len(stack_order) != len(set(stack_order)):
            raise ValueError("stack_index_to_seat seats must be unique")
        object.__setattr__(
            self, "dealer_slot_to_seat", MappingProxyType(dealer)
        )
        object.__setattr__(self, "stack_index_to_seat", stack_order)


@dataclass(frozen=True)
class HandBoundaryDetection:
    status: HandBoundaryStatus
    evidence: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        reasons = tuple(self.reasons)
        if not all(isinstance(item, str) and item for item in evidence + reasons):
            raise TypeError("evidence and reasons must contain non-empty strings")
        if self.status is HandBoundaryStatus.CONFIRMED and not evidence:
            raise ValueError("CONFIRMED boundary requires evidence")
        if self.status is HandBoundaryStatus.AMBIGUOUS and not reasons:
            raise ValueError("AMBIGUOUS boundary requires reasons")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "reasons", reasons)


def detect_hand_boundary(
    previous: PokerState,
    observation: RawObservation,
    policy: HandBoundaryPolicy | None = None,
) -> HandBoundaryDetection:
    """Detect a new deal without treating weak reset hints as authoritative."""
    if not isinstance(previous, PokerState):
        raise TypeError("previous must be a PokerState")
    if not isinstance(observation, RawObservation):
        raise TypeError("observation must be a RawObservation")
    if policy is None:
        policy = HandBoundaryPolicy()
    elif not isinstance(policy, HandBoundaryPolicy):
        raise TypeError("policy must be a HandBoundaryPolicy or None")

    relevant = {
        "hero_cards": observation.hero_cards,
        "board_cards": observation.board_cards,
        "street": observation.street,
        "pot": observation.pot,
        "dealer_pos": observation.dealer_pos,
        "stacks": observation.stacks,
    }
    conflicts = tuple(
        f"{name}_conflict" for name, value in relevant.items()
        if value.validation_status is ValidationStatus.CONFLICT
    )
    if conflicts:
        return HandBoundaryDetection(
            HandBoundaryStatus.AMBIGUOUS,
            reasons=conflicts,
        )

    evidence: list[str] = []
    hero = observation.hero_cards
    if (
        hero.validation_status is ValidationStatus.VALID
        and hero.value is not None
        and len(hero.value) == 2
        and len(previous.hero_cards) == 2
        and tuple(hero.value) != previous.hero_cards
    ):
        evidence.append("hero_cards_changed")
        return HandBoundaryDetection(
            HandBoundaryStatus.CONFIRMED, tuple(evidence)
        )

    street = observation.street
    if (
        street.validation_status is ValidationStatus.VALID
        and street.value is not None
        and street.value.value == "preflop"
        and previous.street.value != "preflop"
    ):
        evidence.append("street_reset_to_preflop")
    board = observation.board_cards
    if (
        board.validation_status is ValidationStatus.VALID
        and board.value is not None
        and previous.board_cards
        and tuple(board.value) == ()
    ):
        evidence.append("board_cleared")
    pot = observation.pot
    if (
        pot.validation_status is ValidationStatus.VALID
        and pot.value is not None
        and pot.value < previous.pot
    ):
        evidence.append("pot_reset")

    dealer = observation.dealer_pos
    previous_dealer = next(
        (player.seat for player in previous.players if player.is_dealer), None
    )
    if (
        dealer.validation_status is ValidationStatus.VALID
        and dealer.value is not None
        and dealer.value in policy.dealer_slot_to_seat
        and previous_dealer is not None
        and policy.dealer_slot_to_seat[dealer.value] != previous_dealer
    ):
        evidence.append("dealer_changed")

    stacks = observation.stacks
    if (
        stacks.validation_status is ValidationStatus.VALID
        and stacks.value is not None
        and policy.stack_index_to_seat
        and len(stacks.value) == len(policy.stack_index_to_seat)
    ):
        by_seat = {player.seat: player for player in previous.players}
        if any(
            seat in by_seat and amount > by_seat[seat].stack
            for seat, amount in zip(policy.stack_index_to_seat, stacks.value)
        ):
            evidence.append("stack_reset_or_payout")

    evidence_set = set(evidence)
    postflop_reset = {
        "street_reset_to_preflop", "board_cleared",
    } <= evidence_set and bool(evidence_set & {
        "pot_reset", "dealer_changed", "stack_reset_or_payout",
    })
    preflop_reset = {
        "pot_reset", "dealer_changed", "stack_reset_or_payout",
    } <= evidence_set
    if postflop_reset or preflop_reset:
        return HandBoundaryDetection(
            HandBoundaryStatus.CONFIRMED, tuple(evidence)
        )
    if len(evidence) >= 2:
        return HandBoundaryDetection(
            HandBoundaryStatus.AMBIGUOUS,
            tuple(evidence),
            ("insufficient_boundary_evidence",),
        )
    return HandBoundaryDetection(
        HandBoundaryStatus.SAME_HAND, tuple(evidence)
    )


__all__ = [
    "HandBoundaryDetection",
    "HandBoundaryPolicy",
    "HandBoundaryStatus",
    "detect_hand_boundary",
]
