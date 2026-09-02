"""Conservative named confidence aggregation for Advice."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from poker_engine.core._freeze import freeze_mapping


@dataclass(frozen=True)
class ConfidenceAggregate:
    overall: float
    components: Mapping[str, float]
    limiting_components: tuple[str, ...]
    missing_components: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_components


def aggregate_confidence(
    components: Mapping[str, float | None],
    *,
    required_components: tuple[str, ...] | None = None,
) -> ConfidenceAggregate:
    """Use the minimum supplied quality; a missing required value yields 0."""
    values = dict(components)
    if not values:
        raise ValueError("components cannot be empty")
    if not all(isinstance(name, str) and name for name in values):
        raise TypeError("component names must be non-empty strings")
    present = {}
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"confidence component {name!r} must be a float")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f"confidence component {name!r} must be finite and in [0, 1]"
            )
        present[name] = float(value)
    required = tuple(required_components or values.keys())
    if len(required) != len(set(required)) or not all(
        isinstance(name, str) and name for name in required
    ):
        raise ValueError("required_components must be unique non-empty strings")
    unknown_required = tuple(name for name in required if name not in values)
    if unknown_required:
        raise ValueError(
            f"required components not declared: {','.join(unknown_required)}"
        )
    missing = tuple(name for name in required if values[name] is None)
    if missing:
        return ConfidenceAggregate(
            0.0, freeze_mapping(present), (), missing
        )
    overall = min(present.values(), default=0.0)
    limiting = tuple(
        sorted(name for name, value in present.items() if value == overall)
    )
    return ConfidenceAggregate(
        overall, freeze_mapping(present), limiting, ()
    )


__all__ = ["ConfidenceAggregate", "aggregate_confidence"]
