"""Auditable hard-gate contracts for fail-closed strategy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class GateResult:
    """One named decision gate and its stable reason codes."""

    name: str
    status: GateStatus
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("gate name must be a non-empty str")
        if not isinstance(self.status, GateStatus):
            raise TypeError("gate status must be GateStatus")
        reasons = tuple(self.reasons)
        if not all(isinstance(value, str) and value for value in reasons):
            raise TypeError("gate reasons must contain non-empty strings")
        if self.status is GateStatus.FAIL and not reasons:
            raise ValueError("failed gate requires reasons")
        if self.status is not GateStatus.FAIL and reasons:
            raise ValueError("only failed gate can carry reasons")
        object.__setattr__(self, "reasons", reasons)


def validate_gate_set(
    values: tuple[GateResult, ...],
    *,
    reserved_names: frozenset[str] = frozenset(),
) -> tuple[GateResult, ...]:
    values = tuple(values)
    if not all(isinstance(value, GateResult) for value in values):
        raise TypeError("hard_gates must contain GateResult values")
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise ValueError("hard gate names must be unique")
    overlap = reserved_names.intersection(names)
    if overlap:
        raise ValueError(f"reserved hard gate name: {min(overlap)}")
    return values


__all__ = ["GateResult", "GateStatus", "validate_gate_set"]
