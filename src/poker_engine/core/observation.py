"""Observation contracts: ValidationStatus, ObservationField[T], RawObservation.

RawObservation is what the Vision Engine will produce in the future — a
raw, per-frame snapshot of the table with confidence-scored evidence.
It deliberately contains NO poker judgement, NO OCR correction, and NO
state inference. It is pure evidence.

Task 1C scope only. Confidence Gate business logic lives in a later task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar

from ._freeze import _require_aware_dt, deep_freeze, freeze_mapping, utc_now
from .enums import ActionType, Street
from .value_objects import Card, ChipAmount


class ValidationStatus(str, Enum):
    """Confidence/validation status of a single observation field.

    Distinct from the State Engine's ``ValidationResult``: this enum tags an
    individual recognized value, whereas ValidationResult describes whether a
    whole PokerState is valid.
    """

    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


T = TypeVar("T")


@dataclass(frozen=True)
class ObservationField(Generic[T]):
    """A single recognized value with confidence and evidence.

    Immutability:
    - frozen dataclass (attribute reassignment raises FrozenInstanceError)
    - ``value`` AND ``evidence`` are deep-frozen via recursive defensive copy;
      external mutation of the original dict/list cannot affect this object,
      and the stored value cannot be mutated in place.
    """

    value: T | None
    confidence: float
    source: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    validation_status: ValidationStatus = ValidationStatus.VALID

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("confidence must be a float")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not isinstance(self.source, str):
            raise TypeError("source must be a str")
        if not self.source:
            raise ValueError("source must be a non-empty str")
        if not isinstance(self.validation_status, ValidationStatus):
            raise TypeError(
                "validation_status must be a ValidationStatus enum"
            )
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)
        # Deep-freeze both value and evidence defensively.
        object.__setattr__(self, "value", deep_freeze(self.value))
        object.__setattr__(self, "evidence", freeze_mapping(self.evidence))


@dataclass(frozen=True)
class SlotObservation(Generic[T]):
    """A per-visual-slot observation (ADR-002, additive to Frozen Core).

    ``slot_id`` is a *visual geometry slot index* — it is NOT a player identity
    and does NOT reference ``PlayerState.seat`` / ``player_id``. Mapping a
    visual slot to a player is a future TableMap / Platform Contract concern.

    Immutability: frozen; ``field`` is an already-deep-immutable
    ``ObservationField``.
    """

    slot_id: int
    field: ObservationField[T]

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, int) or isinstance(self.slot_id, bool):
            raise TypeError("slot_id must be an int")
        if self.slot_id < 0:
            raise ValueError("slot_id must be >= 0")
        if not isinstance(self.field, ObservationField):
            raise TypeError("field must be an ObservationField")


@dataclass(frozen=True)
class RawObservation:
    """A single frame's raw observation of the table.

    Fields carry ``ObservationField`` wrappers so confidence/evidence are
    attached to every recognized value. When a field's ``value`` is not None,
    its runtime type is also validated (Contract Validation only — this is NOT
    State Engine business logic). No poker logic lives here.
    """

    frame_seq: int
    timestamp: datetime

    hero_cards: ObservationField[tuple[Card, ...]]
    board_cards: ObservationField[tuple[Card, ...]]
    pot: ObservationField[ChipAmount]
    stacks: ObservationField[tuple[ChipAmount, ...]]
    bet_size: ObservationField[ChipAmount]
    action: ObservationField[ActionType]
    street: ObservationField[Street]
    dealer_pos: ObservationField[int]
    actor: ObservationField[int]

    # Historical positional field: a positional trailing argument after ``actor``
    # has always meant ``overall_confidence``. Keep it in its historical slot so
    # the additive ADR-002 fields do NOT shift positional semantics.
    overall_confidence: float = field(default=0.0)

    # ADR-002 additive per-visual-slot observations (default empty tuple).
    # Placed AFTER overall_confidence (both have defaults) to preserve the
    # historical positional constructor contract.
    # slot_id is visual geometry only, NOT player identity. Each tuple's
    # slot_id values are unique and strictly ascending.
    slot_stacks: tuple[SlotObservation[ChipAmount], ...] = ()
    slot_actions: tuple[SlotObservation[ActionType], ...] = ()
    slot_occupancies: tuple[SlotObservation[bool], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_seq, int) or isinstance(
            self.frame_seq, bool
        ):
            raise TypeError("frame_seq must be an int")
        if self.frame_seq < 0:
            raise ValueError("frame_seq must be >= 0")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)
        if not isinstance(self.overall_confidence, (int, float)) or isinstance(
            self.overall_confidence, bool
        ):
            raise TypeError("overall_confidence must be a float")
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(
                "overall_confidence must be in [0.0, 1.0]"
            )

        # Ensure all observation fields are ObservationField instances, and
        # validate the runtime type of each non-None value.
        _validate_field_type(self.hero_cards, "hero_cards", tuple)
        _validate_field_type(self.board_cards, "board_cards", tuple)
        _validate_field_type(self.pot, "pot", ChipAmount)
        _validate_field_type(self.stacks, "stacks", tuple)
        _validate_field_type(self.bet_size, "bet_size", ChipAmount)
        _validate_field_type(self.action, "action", ActionType)
        _validate_field_type(self.street, "street", Street)
        _validate_field_type(self.dealer_pos, "dealer_pos", int)
        _validate_field_type(self.actor, "actor", int)

        # ADR-002 additive per-slot fields: normalize to tuple, validate slot_id
        # uniqueness + strict ascending order, deep-freeze the tuples.
        for name, expected in (
            ("slot_stacks", ChipAmount),
            ("slot_actions", ActionType),
            ("slot_occupancies", bool),
        ):
            slots = tuple(getattr(self, name))
            if not all(isinstance(s, SlotObservation) for s in slots):
                raise TypeError(f"{name} must be SlotObservation instances")
            _validate_slot_order(slots, name)
            object.__setattr__(self, name, slots)
            # Runtime type invariant for each slot's field.value.
            for s in slots:
                v = s.field.value
                if v is None:
                    continue
                if not isinstance(v, expected) or (
                    expected is bool and type(v) is not bool
                ):
                    raise TypeError(
                        f"{name} slot field.value must be "
                        f"{expected.__name__} or None, got {type(v).__name__}"
                    )

        # Element-level type checks for tuple fields.
        _validate_tuple_elements(self.hero_cards, "hero_cards", Card)
        _validate_tuple_elements(self.board_cards, "board_cards", Card)
        _validate_tuple_elements(self.stacks, "stacks", ChipAmount)


def _validate_slot_order(slots: tuple, name: str) -> None:
    """Require slot_id uniqueness and strictly ascending order within a tuple."""
    seen: set[int] = set()
    prev: int | None = None
    for s in slots:
        if s.slot_id in seen:
            raise ValueError(f"{name} must not contain duplicate slot_id")
        if prev is not None and s.slot_id <= prev:
            raise ValueError(f"{name} slot_id must be strictly ascending")
        seen.add(s.slot_id)
        prev = s.slot_id


def _validate_field_type(
    field_obj: object, name: str, expected_type: type
) -> None:
    """Require field_obj to be an ObservationField and its value typed."""
    if not isinstance(field_obj, ObservationField):
        raise TypeError(f"{name} must be an ObservationField")
    value = field_obj.value  # type: ignore[attr-defined]
    if value is None:
        # UNKNOWN fields may carry None value.
        return
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name}.value must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def _validate_tuple_elements(
    field_obj: object, name: str, element_type: type
) -> None:
    """Require every element of a tuple-typed field to be element_type."""
    if not isinstance(field_obj, ObservationField):
        raise TypeError(f"{name} must be an ObservationField")
    value = field_obj.value  # type: ignore[attr-defined]
    if value is None:
        return
    for elem in value:  # type: ignore[union-attr]
        if not isinstance(elem, element_type):
            raise TypeError(
                f"{name}.value elements must be {element_type.__name__}, "
                f"got {type(elem).__name__}"
            )


__all__ = [
    "ValidationStatus",
    "ObservationField",
    "SlotObservation",
    "RawObservation",
]
