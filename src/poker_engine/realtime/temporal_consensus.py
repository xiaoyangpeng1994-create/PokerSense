"""Deterministic consecutive-frame confirmation for recognition fields."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)


FIELD_NAMES = (
    "hero_cards",
    "board_cards",
    "pot",
    "stacks",
    "bet_size",
    "action",
    "street",
    "dealer_pos",
    "actor",
)
SLOT_FIELD_NAMES = ("slot_stacks", "slot_actions", "slot_occupancies")
ALL_FIELD_NAMES = FIELD_NAMES + SLOT_FIELD_NAMES


@dataclass(frozen=True)
class TemporalConsensusResult:
    observation: RawObservation
    confirmed_fields: tuple[str, ...]
    pending_fields: tuple[str, ...]
    conflict_fields: tuple[str, ...]


@dataclass(frozen=True)
class _PendingValue:
    value: object
    count: int
    frame_seq: int


class TemporalConsensus:
    """Confirm each VALID value after N consecutive matching frames.

    Pending candidates are emitted as UNKNOWN instead of reusing a prior
    value. CONFLICT and LOW_CONFIDENCE remain visible for the downstream
    confidence gate. Slot observations are tracked by stable visual slot ID,
    never by tuple index.
    """

    def __init__(
        self,
        confirmation_frames: Mapping[str, int] | None = None,
        *,
        default_frames: int = 1,
    ) -> None:
        self._validate_frames(default_frames, "default_frames")
        values = dict(confirmation_frames or {})
        unknown = sorted(set(values) - set(ALL_FIELD_NAMES))
        if unknown:
            raise ValueError(
                f"unknown temporal consensus fields: {', '.join(unknown)}"
            )
        for name, frames in values.items():
            self._validate_frames(frames, name)
        self._thresholds = {
            name: values.get(name, default_frames) for name in ALL_FIELD_NAMES
        }
        self._pending: dict[str, _PendingValue] = {}
        self._last_frame_seq: int | None = None

    @staticmethod
    def _validate_frames(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} confirmation frames must be an int")
        if value < 1:
            raise ValueError(f"{name} confirmation frames must be >= 1")

    def apply(self, observation: RawObservation) -> TemporalConsensusResult:
        if not isinstance(observation, RawObservation):
            raise TypeError("observation must be a RawObservation")
        if (
            self._last_frame_seq is not None
            and observation.frame_seq <= self._last_frame_seq
        ):
            raise ValueError("frame_seq must increase strictly")
        self._last_frame_seq = observation.frame_seq

        confirmed: list[str] = []
        pending: list[str] = []
        conflicts: list[str] = []
        replacements = {}
        for name in FIELD_NAMES:
            replacements[name] = self._apply_field(
                name,
                getattr(observation, name),
                observation.frame_seq,
                confirmed,
                pending,
                conflicts,
            )

        for collection_name in SLOT_FIELD_NAMES:
            slots = getattr(observation, collection_name)
            seen_paths = set()
            output_slots = []
            for slot in slots:
                path = f"{collection_name}[slot_id={slot.slot_id}]"
                seen_paths.add(path)
                field = self._apply_field(
                    path,
                    slot.field,
                    observation.frame_seq,
                    confirmed,
                    pending,
                    conflicts,
                    threshold_name=collection_name,
                )
                output_slots.append(SlotObservation(slot.slot_id, field))
            prefix = f"{collection_name}["
            for path in tuple(self._pending):
                if path.startswith(prefix) and path not in seen_paths:
                    self._pending.pop(path, None)
            replacements[collection_name] = tuple(output_slots)

        return TemporalConsensusResult(
            observation=replace(observation, **replacements),
            confirmed_fields=tuple(confirmed),
            pending_fields=tuple(pending),
            conflict_fields=tuple(conflicts),
        )

    def _apply_field(
        self,
        path: str,
        field: ObservationField,
        frame_seq: int,
        confirmed: list[str],
        pending: list[str],
        conflicts: list[str],
        *,
        threshold_name: str | None = None,
    ) -> ObservationField:
        if field.validation_status is not ValidationStatus.VALID or (
            field.value is None
        ):
            self._pending.pop(path, None)
            if field.validation_status is ValidationStatus.CONFLICT:
                conflicts.append(path)
            return field

        threshold = self._thresholds[threshold_name or path]
        prior = self._pending.get(path)
        if (
            prior is not None
            and prior.value == field.value
            and prior.frame_seq + 1 == frame_seq
        ):
            count = min(threshold, prior.count + 1)
        else:
            count = 1
        self._pending[path] = _PendingValue(field.value, count, frame_seq)
        if count >= threshold:
            confirmed.append(path)
            return field
        pending.append(path)
        return replace(
            field,
            value=None,
            confidence=0.0,
            validation_status=ValidationStatus.UNKNOWN,
        )


__all__ = [
    "ALL_FIELD_NAMES",
    "FIELD_NAMES",
    "SLOT_FIELD_NAMES",
    "TemporalConsensus",
    "TemporalConsensusResult",
]
