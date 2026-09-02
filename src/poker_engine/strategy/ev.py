"""Exact Decimal EV primitives with explicit completeness semantics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping

from poker_engine.core._freeze import freeze_mapping
from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta


class EvStatus(str, Enum):
    COMPLETE = "COMPLETE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ActionEvEstimate:
    status: EvStatus
    ev: ChipDelta | None
    components: Mapping[str, Decimal]
    missing_inputs: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvStatus):
            raise TypeError("status must be an EvStatus")
        if self.ev is not None and not isinstance(self.ev, ChipDelta):
            raise TypeError("ev must be a ChipDelta or None")
        if (self.status is EvStatus.COMPLETE) != (self.ev is not None):
            raise ValueError("COMPLETE status and EV presence must agree")
        components = dict(self.components)
        if not all(isinstance(name, str) and name for name in components):
            raise TypeError("component names must be non-empty strings")
        if not all(
            isinstance(value, Decimal) and value.is_finite()
            for value in components.values()
        ):
            raise ValueError("components must be finite Decimal values")
        object.__setattr__(self, "components", freeze_mapping(components))
        for name in ("missing_inputs", "assumptions"):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class EvGapResult:
    complete: bool
    best_action: ActionType | None
    second_action: ActionType | None
    gap: ChipDelta | None
    missing_actions: tuple[ActionType, ...] = ()


def calculate_call_ev(
    *,
    expected_pot_share: Decimal,
    pot_before_call: ChipAmount,
    to_call: ChipAmount,
) -> ActionEvEstimate:
    """Immediate call EV relative to folding.

    ``expected_pot_share`` includes wins and split-pot shares. The pot input is
    the amount before Hero calls, so ``EV = share * (pot + call) - call``.
    Future-street strategic effects are intentionally not modeled.
    """
    _probability(expected_pot_share, "expected_pot_share")
    _amount(pot_before_call, "pot_before_call")
    _amount(to_call, "to_call")
    final_pot = pot_before_call.value + to_call.value
    gross_share = expected_pot_share * final_pot
    ev = gross_share - to_call.value
    return ActionEvEstimate(
        EvStatus.COMPLETE,
        ChipDelta(ev),
        {
            "expected_pot_share": expected_pot_share,
            "pot_before_call": pot_before_call.value,
            "to_call": to_call.value,
            "gross_expected_return": gross_share,
        },
        assumptions=("immediate_call_ev_no_future_actions",),
    )


def calculate_aggressive_ev(
    *,
    pot_before_action: ChipAmount,
    amount: ChipAmount,
    fold_probability: Decimal,
    call_probability: Decimal,
    raise_probability: Decimal,
    call_continuation_ev: ChipDelta | None,
    raise_continuation_ev: ChipDelta | None,
) -> ActionEvEstimate:
    """Aggregate a bet/raise EV tree from explicit net continuation values.

    Continuation EVs are net values relative to the decision point and must
    already include the action cost. A fold wins the existing pot; the wager is
    returned and therefore is not subtracted in that branch. ``amount`` is kept
    as provenance and validated even though the supplied continuation values
    own its downstream economics.
    """
    _amount(pot_before_action, "pot_before_action")
    _amount(amount, "amount")
    probabilities = {
        "fold_probability": fold_probability,
        "call_probability": call_probability,
        "raise_probability": raise_probability,
    }
    for name, value in probabilities.items():
        _probability(value, name)
    if sum(probabilities.values(), Decimal("0")) != Decimal("1"):
        raise ValueError("branch probabilities must sum exactly to 1")
    missing = []
    if call_probability > 0 and call_continuation_ev is None:
        missing.append("call_continuation_ev")
    if raise_probability > 0 and raise_continuation_ev is None:
        missing.append("raise_continuation_ev")
    for name, value in (
        ("call_continuation_ev", call_continuation_ev),
        ("raise_continuation_ev", raise_continuation_ev),
    ):
        if value is not None and not isinstance(value, ChipDelta):
            raise TypeError(f"{name} must be a ChipDelta or None")
    components = {
        **probabilities,
        "pot_before_action": pot_before_action.value,
        "amount": amount.value,
        "fold_branch_ev": pot_before_action.value,
    }
    if call_continuation_ev is not None:
        components["call_continuation_ev"] = call_continuation_ev.value
    if raise_continuation_ev is not None:
        components["raise_continuation_ev"] = raise_continuation_ev.value
    if missing:
        return ActionEvEstimate(
            EvStatus.UNKNOWN,
            None,
            components,
            missing_inputs=tuple(missing),
            assumptions=("continuation_values_are_net_decision_point_ev",),
        )
    call_ev = call_continuation_ev.value if call_continuation_ev else Decimal("0")
    raise_ev = (
        raise_continuation_ev.value if raise_continuation_ev else Decimal("0")
    )
    ev = (
        fold_probability * pot_before_action.value
        + call_probability * call_ev
        + raise_probability * raise_ev
    )
    return ActionEvEstimate(
        EvStatus.COMPLETE,
        ChipDelta(ev),
        components,
        assumptions=("continuation_values_are_net_decision_point_ev",),
    )


def calculate_ev_gap(
    legal_actions: tuple[ActionType, ...],
    action_ev: Mapping[ActionType, ChipDelta | None],
) -> EvGapResult:
    """Return best-vs-second gap only when every legal action has an EV."""
    legal = tuple(dict.fromkeys(legal_actions))
    if not legal or not all(isinstance(action, ActionType) for action in legal):
        raise ValueError("legal_actions must contain at least one ActionType")
    values = dict(action_ev)
    if not all(isinstance(action, ActionType) for action in values):
        raise TypeError("action_ev keys must be ActionType")
    if not all(
        value is None or isinstance(value, ChipDelta) for value in values.values()
    ):
        raise TypeError("action_ev values must be ChipDelta or None")
    missing = tuple(
        action for action in legal
        if action not in values or values[action] is None
    )
    if missing:
        return EvGapResult(False, None, None, None, missing)
    ordered = sorted(
        legal,
        key=lambda action: (values[action].value, action.value),
        reverse=True,
    )
    best = ordered[0]
    if len(ordered) == 1:
        return EvGapResult(True, best, None, None)
    second = ordered[1]
    return EvGapResult(
        True,
        best,
        second,
        ChipDelta(values[best].value - values[second].value),
    )


def _probability(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or not Decimal("0") <= value <= Decimal("1"):
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _amount(value: ChipAmount, name: str) -> None:
    if not isinstance(value, ChipAmount):
        raise TypeError(f"{name} must be a ChipAmount")


__all__ = [
    "ActionEvEstimate",
    "EvGapResult",
    "EvStatus",
    "calculate_aggressive_ev",
    "calculate_call_ev",
    "calculate_ev_gap",
]
