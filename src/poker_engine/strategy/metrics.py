"""Exact Decimal-derived strategy metrics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from poker_engine.core.value_objects import ChipAmount

from .contracts import EffectiveStack, PotState


class MetricStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PairwiseSpr:
    opponent_seat: int
    value: Decimal | None
    status: MetricStatus

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_seat, int) or isinstance(
            self.opponent_seat, bool
        ):
            raise TypeError("opponent_seat must be an int")
        if not isinstance(self.status, MetricStatus):
            raise TypeError("status must be a MetricStatus")
        if self.status is MetricStatus.UNKNOWN:
            if self.value is not None:
                raise ValueError("UNKNOWN SPR must not contain a value")
        elif not isinstance(self.value, Decimal) or not self.value.is_finite():
            raise TypeError("KNOWN SPR must contain a finite Decimal")


@dataclass(frozen=True)
class PotOddsMetric:
    required_equity: Decimal
    no_call_cost: bool

    def __post_init__(self) -> None:
        _require_ratio(self.required_equity, "required_equity")
        if not isinstance(self.no_call_cost, bool):
            raise TypeError("no_call_cost must be a bool")
        if self.no_call_cost and self.required_equity != 0:
            raise ValueError("a no-cost action must require zero equity")


@dataclass(frozen=True)
class NormalizedActionSize:
    additional_amount: ChipAmount
    total_street_amount: ChipAmount
    size_bb: Decimal
    pot_fraction: Decimal | None
    raise_multiplier: Decimal | None

    def __post_init__(self) -> None:
        for name in ("additional_amount", "total_street_amount"):
            if not isinstance(getattr(self, name), ChipAmount):
                raise TypeError(f"{name} must be a ChipAmount")
        if self.total_street_amount < self.additional_amount:
            raise ValueError("total_street_amount cannot be below additional_amount")
        for name in ("size_bb", "pot_fraction", "raise_multiplier"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite Decimal >= 0 or None")


def calculate_pairwise_spr(
    effective_stacks: tuple[EffectiveStack, ...],
    pots: tuple[PotState, ...],
) -> tuple[PairwiseSpr, ...]:
    """Calculate effective-stack / total-pot SPR for each active opponent."""
    stacks = tuple(effective_stacks)
    pots = tuple(pots)
    if not all(isinstance(item, EffectiveStack) for item in stacks):
        raise TypeError("effective_stacks must contain EffectiveStack values")
    if len({item.opponent_seat for item in stacks}) != len(stacks):
        raise ValueError("effective_stacks must have unique opponent seats")
    if not all(isinstance(item, PotState) for item in pots):
        raise TypeError("pots must contain PotState values")
    total_pot = sum((item.amount.value for item in pots), Decimal("0"))
    if total_pot == 0:
        return tuple(
            PairwiseSpr(item.opponent_seat, None, MetricStatus.UNKNOWN)
            for item in stacks
        )
    return tuple(
        PairwiseSpr(
            item.opponent_seat,
            item.amount.value / total_pot,
            MetricStatus.KNOWN,
        )
        for item in stacks
    )


def calculate_pot_odds(
    pot: ChipAmount,
    to_call: ChipAmount,
) -> PotOddsMetric:
    """Return exact immediate break-even equity: call / (pot + call)."""
    _require_amount(pot, "pot")
    _require_amount(to_call, "to_call")
    if to_call.value == 0:
        return PotOddsMetric(Decimal("0"), True)
    return PotOddsMetric(
        to_call.value / (pot.value + to_call.value),
        False,
    )


def normalize_action_size(
    *,
    additional_amount: ChipAmount,
    total_street_amount: ChipAmount,
    pot_before_action: ChipAmount,
    big_blind: ChipAmount,
    current_bet: ChipAmount,
) -> NormalizedActionSize:
    """Normalize explicit additional/total amounts without guessing semantics."""
    for name, value in (
        ("additional_amount", additional_amount),
        ("total_street_amount", total_street_amount),
        ("pot_before_action", pot_before_action),
        ("big_blind", big_blind),
        ("current_bet", current_bet),
    ):
        _require_amount(value, name)
    if big_blind.value <= 0:
        raise ValueError("big_blind must be > 0")
    if total_street_amount < additional_amount:
        raise ValueError("total_street_amount cannot be below additional_amount")
    return NormalizedActionSize(
        additional_amount=additional_amount,
        total_street_amount=total_street_amount,
        size_bb=additional_amount.value / big_blind.value,
        pot_fraction=(
            additional_amount.value / pot_before_action.value
            if pot_before_action.value > 0 else None
        ),
        raise_multiplier=(
            total_street_amount.value / current_bet.value
            if current_bet.value > 0 else None
        ),
    )


def _require_amount(value: ChipAmount, name: str) -> None:
    if not isinstance(value, ChipAmount):
        raise TypeError(f"{name} must be a ChipAmount")


def _require_ratio(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise TypeError(f"{name} must be a finite Decimal")
    if not Decimal("0") <= value <= Decimal("1"):
        raise ValueError(f"{name} must be in [0, 1]")


__all__ = [
    "MetricStatus",
    "NormalizedActionSize",
    "PairwiseSpr",
    "PotOddsMetric",
    "calculate_pairwise_spr",
    "calculate_pot_odds",
    "normalize_action_size",
]
