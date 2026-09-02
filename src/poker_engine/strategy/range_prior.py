"""Versioned preflop concrete-combo prior lookup without random fallback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from poker_engine.core.enums import Position, Rank, Suit
from poker_engine.core.value_objects import Card

from .contracts import RangeDistribution
from .heuristic_provider import PreflopRfiHeuristicProvider


class RangePriorState(str, Enum):
    HIT = "HIT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RangePriorQuery:
    seat_id: int
    player_count: int
    position: Position
    effective_stack_bb: Decimal
    action_line: str
    known_cards: tuple[Card, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.seat_id, int) or isinstance(self.seat_id, bool):
            raise TypeError("seat_id must be an int")
        if self.seat_id < 0:
            raise ValueError("seat_id must be >= 0")
        if not isinstance(self.player_count, int) or isinstance(
            self.player_count, bool
        ):
            raise TypeError("player_count must be an int")
        if not 2 <= self.player_count <= 9:
            raise ValueError("player_count must be in [2, 9]")
        if not isinstance(self.position, Position):
            raise TypeError("position must be a Position")
        if not isinstance(self.effective_stack_bb, Decimal):
            raise TypeError("effective_stack_bb must be a Decimal")
        if (
            not self.effective_stack_bb.is_finite()
            or self.effective_stack_bb < 0
        ):
            raise ValueError("effective_stack_bb must be finite and >= 0")
        if not isinstance(self.action_line, str) or not self.action_line:
            raise ValueError("action_line must be a non-empty str")
        cards = tuple(self.known_cards)
        if not all(isinstance(card, Card) for card in cards):
            raise TypeError("known_cards must contain Card values")
        if len(cards) != len(set(cards)):
            raise ValueError("known_cards must be distinct")
        object.__setattr__(self, "known_cards", cards)


@dataclass(frozen=True)
class RangePriorResult:
    state: RangePriorState
    distribution: RangeDistribution | None = None
    evidence: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, RangePriorState):
            raise TypeError("state must be a RangePriorState")
        hit = self.state is RangePriorState.HIT
        if hit != (self.distribution is not None):
            raise ValueError("HIT state and distribution presence must agree")
        if self.distribution is not None and not isinstance(
            self.distribution, RangeDistribution
        ):
            raise TypeError("distribution must be RangeDistribution or None")
        for name in ("evidence", "reasons"):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if hit and (not self.evidence or self.reasons):
            raise ValueError("HIT requires evidence and cannot carry reasons")
        if not hit and not self.reasons:
            raise ValueError("non-HIT result requires reasons")


class PreflopRfiRangePrior:
    """Concrete uniform-combo prior for an observed first-in raise."""

    def __init__(self, provider: PreflopRfiHeuristicProvider) -> None:
        if not isinstance(provider, PreflopRfiHeuristicProvider):
            raise TypeError("provider must be PreflopRfiHeuristicProvider")
        self._provider = provider

    @classmethod
    def from_builtin(cls) -> "PreflopRfiRangePrior":
        return cls(PreflopRfiHeuristicProvider.from_builtin())

    @property
    def source_version(self) -> str:
        return f"{self._provider.source_version}:uniform-combo-prior/v1"

    def lookup(self, query: RangePriorQuery) -> RangePriorResult:
        if not isinstance(query, RangePriorQuery):
            raise TypeError("query must be a RangePriorQuery")
        reasons = []
        if query.player_count not in self._provider.capability.player_counts:
            reasons.append("unsupported_player_count")
        if query.effective_stack_bb != Decimal("100"):
            reasons.append("unsupported_stack")
        if query.action_line != "open_raise":
            reasons.append("unsupported_action_line")
        hand_classes = self._provider.explicit_range(
            query.player_count, query.position
        )
        if hand_classes is None:
            reasons.append("unsupported_position")
        if reasons:
            return RangePriorResult(
                RangePriorState.NOT_APPLICABLE,
                reasons=tuple(reasons),
            )

        blocked = set(query.known_cards)
        combos = sorted({
            combo
            for hand_class_value in hand_classes
            for combo, cards in expand_hand_class(hand_class_value)
            if not blocked.intersection(cards)
        })
        if not combos:
            return RangePriorResult(
                RangePriorState.UNKNOWN,
                reasons=("range_card_collision",),
            )
        unit = Decimal("1") / Decimal(len(combos))
        weights = {combo: unit for combo in combos}
        weights[combos[-1]] = Decimal("1") - sum(
            (weights[combo] for combo in combos[:-1]), Decimal("0")
        )
        source = "preflopr-explicit-rfi-heuristic"
        distribution = RangeDistribution(
            seat_id=query.seat_id,
            combo_weights=weights,
            source=source,
            source_version=self.source_version,
            entropy=Decimal(str(math.log(len(combos)))),
            effective_sample_size=0,
            confidence=0.4,
        )
        return RangePriorResult(
            RangePriorState.HIT,
            distribution,
            evidence=(
                f"{self._provider.source_url}/tree/{self._provider.source_revision}",
                f"asset_sha256:{self._provider.asset_sha256}",
                f"range_prior:{query.player_count}:{query.position.value}:open_raise",
                f"combo_expansion:uniform-combo-v1:{len(combos)}",
            ),
        )


def expand_hand_class(value: str) -> tuple[tuple[str, tuple[Card, Card]], ...]:
    """Expand AA/AKs/AKo to 6/4/12 canonical concrete combinations."""
    if not isinstance(value, str):
        raise TypeError("hand class must be a str")
    if len(value) not in (2, 3):
        raise ValueError("hand class must be pair, suited, or offsuit")
    try:
        first = Rank(value[0])
        second = Rank(value[1])
    except ValueError as exc:
        raise ValueError(f"invalid hand class {value!r}") from exc
    suits = tuple(Suit)
    holdings = []
    if len(value) == 2:
        if first is not second:
            raise ValueError("two-character hand class must be a pair")
        for first_index, first_suit in enumerate(suits):
            for second_suit in suits[first_index + 1:]:
                cards = Card(first, first_suit), Card(second, second_suit)
                holdings.append(("".join(str(card) for card in cards), cards))
    else:
        if first is second or value[2] not in "so":
            raise ValueError("three-character hand class must be suited/offsuit")
        for first_suit in suits:
            for second_suit in suits:
                if (value[2] == "s") != (first_suit is second_suit):
                    continue
                cards = Card(first, first_suit), Card(second, second_suit)
                holdings.append(("".join(str(card) for card in cards), cards))
    return tuple(holdings)


__all__ = [
    "PreflopRfiRangePrior",
    "RangePriorQuery",
    "RangePriorResult",
    "RangePriorState",
    "expand_hand_class",
]
