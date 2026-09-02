"""Confidence Gate — field-level sanitization before State Engine.

Low-confidence key fields become UNKNOWN (value=None) so they never enter the
State Engine as reliable canonical data. This is a pure, deterministic,
field-level gate; it never drops the whole observation and never modifies the
input (Frozen Core).

Frozen thresholds (architecture v0.2.1):
    hero_cards >= 0.995, board_cards >= 0.995, street >= 0.999,
    pot >= 0.99, stacks >= 0.99, bet_size >= 0.99, action >= 0.99.

Equality passes: confidence == threshold -> PASS; below -> BLOCK.

Fields without a frozen threshold (actor, dealer_pos, overall_confidence) are
NOT gated here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)

from .errors import ConfidenceGateError

# Fixed field name -> threshold. Order is the deterministic blocked_fields order.
_FROZEN_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("hero_cards", 0.995),
    ("board_cards", 0.995),
    ("street", 0.999),
    ("pot", 0.99),
    ("stacks", 0.99),
    ("bet_size", 0.99),
    ("action", 0.99),
)


@dataclass(frozen=True)
class ConfidenceGateResult:
    """Outcome of gating one observation.

    ``observation`` is the sanitized copy (never the input object).
    ``blocked_fields`` lists field names demoted to UNKNOWN, deterministic order.
    """

    observation: RawObservation
    blocked_fields: tuple[str, ...]


def _validate_threshold_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"threshold {name!r} must be a finite float")
    # int is allowed here but treated as float; reject non-finite floats.
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfidenceGateError(f"threshold {name!r} must be finite (not NaN/inf)")
    if not (0.0 <= value <= 1.0):
        raise ConfidenceGateError(f"threshold {name!r} must be in [0.0, 1.0]")
    return float(value)


class ConfidenceGate:
    """Field-level confidence gate."""

    def __init__(
        self,
        thresholds: Mapping[str, float] | None = None,
        *,
        slot_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        if thresholds is None:
            resolved: dict[str, float] = dict(_FROZEN_THRESHOLDS)
        else:
            resolved = self._validate_thresholds(thresholds)
        self._thresholds = resolved
        supplied_slot = dict(slot_thresholds or {})
        unknown_slot = set(supplied_slot) - {"occupancy"}
        if unknown_slot:
            raise ConfidenceGateError(
                f"unknown slot threshold field(s): {sorted(unknown_slot)}"
            )
        self._slot_thresholds = {
            "occupancy": _validate_threshold_value(
                supplied_slot.get("occupancy", 0.99), "occupancy"
            )
        }

    @staticmethod
    def _validate_thresholds(thresholds: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(thresholds, Mapping):
            raise TypeError("thresholds must be a Mapping")
        expected = {name for name, _ in _FROZEN_THRESHOLDS}
        provided = set(thresholds.keys())
        missing = expected - provided
        if missing:
            raise ConfidenceGateError(
                f"missing threshold(s): {sorted(missing)}"
            )
        unknown = provided - expected
        if unknown:
            raise ConfidenceGateError(
                f"unknown threshold field(s): {sorted(unknown)}"
            )
        resolved: dict[str, float] = {}
        for name, _ in _FROZEN_THRESHOLDS:
            resolved[name] = _validate_threshold_value(thresholds[name], name)
        return resolved

    @property
    def thresholds(self) -> Mapping[str, float]:
        return MappingProxyType(self._thresholds)

    @property
    def slot_thresholds(self) -> Mapping[str, float]:
        return MappingProxyType(self._slot_thresholds)

    def apply(self, observation: RawObservation) -> ConfidenceGateResult:
        if not isinstance(observation, RawObservation):
            raise TypeError("observation must be a RawObservation")

        blocked: list[str] = []
        replacements: dict[str, ObservationField] = {}
        slot_replacements: dict[str, tuple] = {}

        for name, threshold in self._thresholds.items():
            field = getattr(observation, name)
            new_field = self._gate_field(field, threshold)
            if new_field is not field:
                # demoted -> record blocked (only demotions, not no-change)
                blocked.append(name)
                replacements[name] = new_field

        # ADR-002 additive per-slot fields: gate every slot independently,
        # reusing the field-level threshold. blocked path encodes the visual
        # slot_id (NOT tuple position).
        slot_stacks = self._gate_slots(
            observation.slot_stacks,
            self._thresholds["stacks"],
            "slot_stacks",
            blocked,
        )
        slot_actions = self._gate_slots(
            observation.slot_actions,
            self._thresholds["action"],
            "slot_actions",
            blocked,
        )
        slot_occupancies = self._gate_slots(
            observation.slot_occupancies,
            self._slot_thresholds["occupancy"],
            "slot_occupancies",
            blocked,
        )
        if slot_stacks is not observation.slot_stacks:
            slot_replacements["slot_stacks"] = slot_stacks
        if slot_actions is not observation.slot_actions:
            slot_replacements["slot_actions"] = slot_actions
        if slot_occupancies is not observation.slot_occupancies:
            slot_replacements["slot_occupancies"] = slot_occupancies

        # Build the sanitized observation via dataclasses.replace (returns a new
        # object, never mutates input).
        sanitized = dataclasses.replace(
            observation, **replacements, **slot_replacements
        )
        return ConfidenceGateResult(
            observation=sanitized,
            blocked_fields=tuple(blocked),
        )

    @staticmethod
    def _gate_slots(
        slots: tuple,
        threshold: float,
        path_prefix: str,
        blocked: list[str],
    ) -> tuple:
        """Gate each slot independently; only demoted slots are blocked.

        Returns the original tuple object when nothing changed (so identity
        comparison in ``apply`` can skip replacement).
        """
        out: list = []
        changed = False
        # slots are strictly-ascending by slot_id (already validated by Core);
        # iterate in order to preserve deterministic ascending blocked paths.
        for slot in slots:
            new_field = ConfidenceGate._gate_field(slot.field, threshold)
            if new_field is not slot.field:
                blocked.append(f"{path_prefix}[slot_id={slot.slot_id}]")
                out.append(SlotObservation(slot_id=slot.slot_id, field=new_field))
                changed = True
            else:
                out.append(slot)
        return tuple(out) if changed else slots

    @staticmethod
    def _gate_field(
        field: ObservationField, threshold: float
    ) -> ObservationField:
        status = field.validation_status

        # CONFLICT and UNKNOWN: never change.
        if status is ValidationStatus.CONFLICT:
            return field
        if status is ValidationStatus.UNKNOWN:
            return field

        # LOW_CONFIDENCE: always demote to UNKNOWN (never upgrade).
        if status is ValidationStatus.LOW_CONFIDENCE:
            return ObservationField(
                value=None,
                confidence=field.confidence,
                source=field.source,
                evidence=field.evidence,
                timestamp=field.timestamp,
                validation_status=ValidationStatus.UNKNOWN,
            )

        # VALID: keep if confidence >= threshold, else demote to UNKNOWN.
        if status is ValidationStatus.VALID:
            if field.confidence >= threshold:
                return field
            return ObservationField(
                value=None,
                confidence=field.confidence,
                source=field.source,
                evidence=field.evidence,
                timestamp=field.timestamp,
                validation_status=ValidationStatus.UNKNOWN,
            )

        # Unreachable given the enum's members, but be explicit & safe.
        return field


__all__ = ["ConfidenceGate", "ConfidenceGateResult"]
