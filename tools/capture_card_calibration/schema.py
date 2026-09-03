"""Label, manifest and metrics schemas for capture-card calibration.

Mirrors ``docs/capture-card-calibration-guide.zh-CN.md`` sections 4 (stage A
device manifest), 9 (stage F per-frame ground truth) and 12 (stage I field
metrics).

The guide's failure-closed rules are enforced here rather than left to
convention:

- ``UNKNOWN`` and ``CONFLICT`` may never carry a value, so they can never be
  serialized as a plausible-looking default (section 12).
- ``REPLACE_ME`` placeholders block the pipeline instead of silently passing
  as data (stage A must freeze the hardware before anything downstream runs).
- Card codes, streets, actions and occupancy are validated against the
  enumerations in section 9, and Hero actor evidence is restricted to
  ``HERO`` so opponent countdowns cannot be smuggled in as actor recognition
  (section 8).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from . import SCHEMA_VERSION, SLOT_COUNT

PLACEHOLDER = "REPLACE_ME"
RANKS = "23456789TJQKA"
SUITS = "CDHS"

# Only human-confirmed pixel reading is an accepted labelling method.
REVIEW_METHODS = frozenset({"manual_source_pixels"})
# Section 8: opponent countdowns may be collected for the future but must not
# masquerade as supported actor recognition.
ACTOR_VALUES = frozenset({"HERO"})

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


class SchemaError(ValueError):
    """Raised when a label or manifest violates the guide's contract."""


class LabelStatus(str, Enum):
    VALID = "VALID"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class Scene(str, Enum):
    TABLE = "table"
    DEAL_TRANSITION = "deal_transition"
    ACTION_TRANSITION = "action_transition"
    RESULT = "result"
    MENU = "menu"
    OVERLAY = "overlay"
    SIGNAL_LOSS = "signal_loss"
    RECONNECT = "reconnect"


class Street(str, Enum):
    PRE_FLOP = "PRE_FLOP"
    FLOP = "FLOP"
    TURN = "TURN"
    RIVER = "RIVER"


class CompletedAction(str, Enum):
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"
    ALL_IN = "ALL_IN"


class Occupancy(str, Enum):
    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"


# --- shared validation helpers -------------------------------------------


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _coerce_enum(name: str, value: Any, enum_cls: type[Enum]) -> Enum:
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise SchemaError(
            f"{name} value {value!r} is not one of: {allowed}"
        ) from exc


def is_card_code(value: Any) -> bool:
    """Return True for a rank+suit code such as ``TS`` or ``QD``."""
    return (
        isinstance(value, str)
        and len(value) == 2
        and value[0] in RANKS
        and value[1] in SUITS
    )


def validate_card_list(
    name: str, value: Any, *, min_cards: int, max_cards: int
) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{name} must be a list of card codes")
    if not min_cards <= len(value) <= max_cards:
        raise SchemaError(
            f"{name} must hold {min_cards}-{max_cards} cards, got {len(value)}"
        )
    for card in value:
        if not is_card_code(card):
            raise SchemaError(f"{name} contains an invalid card code: {card!r}")
    return list(value)


# --- fields ---------------------------------------------------------------


@dataclass(frozen=True)
class FieldValue:
    """A labelled field: a status plus, only when VALID, a value."""

    status: LabelStatus
    value: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, LabelStatus):
            object.__setattr__(self, "status", _coerce_enum(
                "status", self.status, LabelStatus
            ))
        if self.status is LabelStatus.VALID and self.value is None:
            raise SchemaError("a VALID field must carry a non-None value")
        if self.status is not LabelStatus.VALID and self.value is not None:
            raise SchemaError(
                f"{self.status.value} must not carry a value; UNKNOWN and "
                "CONFLICT must never be serialized as a default"
            )

    # -- constructors ----------------------------------------------------

    @classmethod
    def valid(cls, value: Any) -> "FieldValue":
        return cls(LabelStatus.VALID, value)

    @classmethod
    def unknown(cls) -> "FieldValue":
        return cls(LabelStatus.UNKNOWN, None)

    @classmethod
    def conflict(cls) -> "FieldValue":
        return cls(LabelStatus.CONFLICT, None)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldValue":
        if not isinstance(data, Mapping):
            raise SchemaError("field must be an object with status/value")
        if "status" not in data:
            raise SchemaError("field is missing 'status'")
        return cls(LabelStatus(data["status"]), data.get("value"))

    def validate_enum(self, name: str, enum_cls: type[Enum]) -> None:
        if self.status is LabelStatus.VALID:
            _coerce_enum(name, self.value, enum_cls)


@dataclass(frozen=True)
class Review:
    """Who confirmed the label and how (section 9)."""

    reviewer: str
    method: str = "manual_source_pixels"
    notes: str = ""

    def __post_init__(self) -> None:
        _require_str("review.reviewer", self.reviewer)
        if self.reviewer == PLACEHOLDER:
            raise SchemaError(
                "review.reviewer must name a real reviewer, not REPLACE_ME"
            )
        if self.method not in REVIEW_METHODS:
            allowed = ", ".join(sorted(REVIEW_METHODS))
            raise SchemaError(
                f"review.method must be one of: {allowed}; got {self.method!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "method": self.method,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Review":
        return cls(
            reviewer=data.get("reviewer", ""),
            method=data.get("method", "manual_source_pixels"),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class SlotLabel:
    """Per-visual-slot ground truth for one frame (section 9)."""

    slot_id: int
    occupancy: FieldValue
    stack: FieldValue
    dealer: FieldValue
    completed_action: FieldValue
    current_actor: FieldValue

    def __post_init__(self) -> None:
        if isinstance(self.slot_id, bool) or not isinstance(self.slot_id, int):
            raise SchemaError("slot_id must be an int")
        if not 0 <= self.slot_id < SLOT_COUNT:
            raise SchemaError(
                f"slot_id must be in [0, {SLOT_COUNT}), got {self.slot_id}"
            )
        self.occupancy.validate_enum("occupancy", Occupancy)
        self.completed_action.validate_enum("completed_action", CompletedAction)
        if self.stack.status is LabelStatus.VALID:
            _require_non_negative_int("stack", self.stack.value)
        if (
            self.dealer.status is LabelStatus.VALID
            and not isinstance(self.dealer.value, bool)
        ):
            raise SchemaError("dealer value must be a bool when VALID")
        if (
            self.current_actor.status is LabelStatus.VALID
            and self.current_actor.value not in ACTOR_VALUES
        ):
            raise SchemaError(
                "current_actor may only be HERO; opponent actor evidence is "
                "not supported and must stay UNKNOWN"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "occupancy": self.occupancy.to_dict(),
            "stack": self.stack.to_dict(),
            "dealer": self.dealer.to_dict(),
            "completed_action": self.completed_action.to_dict(),
            "current_actor": self.current_actor.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SlotLabel":
        return cls(
            slot_id=data["slot_id"],
            occupancy=FieldValue.from_dict(data["occupancy"]),
            stack=FieldValue.from_dict(data["stack"]),
            dealer=FieldValue.from_dict(data["dealer"]),
            completed_action=FieldValue.from_dict(data["completed_action"]),
            current_actor=FieldValue.from_dict(data["current_actor"]),
        )


@dataclass(frozen=True)
class FrameLabel:
    """One line of ``labels/frames.jsonl`` (section 9)."""

    frame: str
    sha256: str
    session_id: str
    hand_id: str
    timestamp_ms: int
    stable: bool
    scene: Scene
    hero_cards: FieldValue
    board_cards: FieldValue
    street: FieldValue
    pot: FieldValue
    slots: tuple[SlotLabel, ...]
    review: Review
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_str("frame", self.frame)
        if not isinstance(self.sha256, str) or not _HEX64.match(self.sha256):
            raise SchemaError("sha256 must be 64 lowercase hex characters")
        _require_str("session_id", self.session_id)
        _require_str("hand_id", self.hand_id)
        _require_non_negative_int("timestamp_ms", self.timestamp_ms)
        if not isinstance(self.stable, bool):
            raise SchemaError("stable must be a bool")
        if not isinstance(self.scene, Scene):
            object.__setattr__(self, "scene", _coerce_enum(
                "scene", self.scene, Scene
            ))
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        self.street.validate_enum("street", Street)
        if self.hero_cards.status is LabelStatus.VALID:
            validate_card_list(
                "hero_cards", self.hero_cards.value, min_cards=2, max_cards=2
            )
        if self.board_cards.status is LabelStatus.VALID:
            validate_card_list(
                "board_cards", self.board_cards.value, min_cards=1, max_cards=5
            )
        if self.pot.status is LabelStatus.VALID:
            _require_non_negative_int("pot", self.pot.value)
        if not isinstance(self.review, Review):
            raise SchemaError("review must be a Review instance")
        slots = tuple(self.slots)
        if len(slots) != SLOT_COUNT:
            raise SchemaError(
                f"slots must list all {SLOT_COUNT} visual slots, "
                f"got {len(slots)}"
            )
        seen = [slot.slot_id for slot in slots]
        if sorted(seen) != list(range(SLOT_COUNT)):
            raise SchemaError(
                f"slots must cover slot_id 0..{SLOT_COUNT - 1} exactly once"
            )
        object.__setattr__(self, "slots", slots)

    # -- derived helpers -------------------------------------------------

    @property
    def group_id(self) -> str:
        """Split key: a hand never straddles two splits (section 11)."""
        return f"{self.session_id}::{self.hand_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame": self.frame,
            "sha256": self.sha256,
            "session_id": self.session_id,
            "hand_id": self.hand_id,
            "timestamp_ms": self.timestamp_ms,
            "stable": self.stable,
            "scene": self.scene.value,
            "hero_cards": self.hero_cards.to_dict(),
            "board_cards": self.board_cards.to_dict(),
            "street": self.street.to_dict(),
            "pot": self.pot.to_dict(),
            "slots": [slot.to_dict() for slot in self.slots],
            "review": self.review.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameLabel":
        return cls(
            frame=data["frame"],
            sha256=data["sha256"],
            session_id=data["session_id"],
            hand_id=data["hand_id"],
            timestamp_ms=data["timestamp_ms"],
            stable=bool(data["stable"]),
            scene=Scene(data["scene"]),
            hero_cards=FieldValue.from_dict(data["hero_cards"]),
            board_cards=FieldValue.from_dict(data["board_cards"]),
            street=FieldValue.from_dict(data["street"]),
            pot=FieldValue.from_dict(data["pot"]),
            slots=tuple(
                SlotLabel.from_dict(slot) for slot in data.get("slots", [])
            ),
            review=Review.from_dict(data.get("review", {})),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FrameLabel":
        return cls.from_dict(json.loads(text))


# --- stage A: device and capture manifest ---------------------------------

DEVICE_SECTIONS = (
    "phone",
    "app",
    "video_adapter",
    "capture_card",
    "uvc",
    "recording",
)

# Section 5 nominally asks for three independent capture sessions, but the
# project owner explicitly waived the third session on 2026-09-03: the two
# existing sessions already cover both fixed 8-max (session_001) and dynamic
# 6-8 player MTT (session_002) table setups, and the remaining coverage gaps
# are field-level (action / temporal / anomaly / metrics), not session-level.
# This is a recorded owner decision, deliberately NOT a silent relaxation of
# the guide, so the acceptance floor is 2 until the owner says otherwise.
MIN_SESSIONS = 2


def _collect_placeholders(node: Any, prefix: str, found: list[str]) -> None:
    if isinstance(node, str):
        if node == PLACEHOLDER:
            found.append(prefix)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            _collect_placeholders(value, f"{prefix}.{key}", found)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _collect_placeholders(value, f"{prefix}[{index}]", found)


@dataclass(frozen=True)
class DeviceAndCapture:
    """``source/device_and_capture.json`` (section 4)."""

    phone: Mapping[str, Any]
    app: Mapping[str, Any]
    video_adapter: Mapping[str, Any]
    capture_card: Mapping[str, Any]
    uvc: Mapping[str, Any]
    recording: Mapping[str, Any]
    sessions: tuple[Mapping[str, Any], ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"unsupported schema_version {self.schema_version!r}"
            )
        for name in DEVICE_SECTIONS:
            section = getattr(self, name)
            if not isinstance(section, Mapping) or not section:
                raise SchemaError(
                    f"device_and_capture.{name} must be a non-empty object"
                )

    # -- stage A gate ----------------------------------------------------

    def placeholders(self) -> list[str]:
        """Return dotted paths still holding a REPLACE_ME placeholder."""
        found: list[str] = []
        for name in DEVICE_SECTIONS:
            _collect_placeholders(getattr(self, name), name, found)
        return found

    @property
    def is_ready(self) -> bool:
        return not self.placeholders()

    def require_ready(self) -> None:
        """Fail closed until the hardware manifest is actually filled in."""
        remaining = self.placeholders()
        if remaining:
            joined = ", ".join(remaining)
            raise SchemaError(
                "stage A hardware manifest is not frozen; still REPLACE_ME at: "
                + joined
            )

    def require_min_sessions(self, minimum: int = MIN_SESSIONS) -> None:
        if len(self.sessions) < minimum:
            raise SchemaError(
                f"section 5 requires at least {minimum} independent capture "
                f"sessions, found {len(self.sessions)}"
            )

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
        }
        for name in DEVICE_SECTIONS:
            data[name] = dict(getattr(self, name))
        data["sessions"] = [dict(session) for session in self.sessions]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceAndCapture":
        missing = [name for name in DEVICE_SECTIONS if name not in data]
        if missing:
            raise SchemaError(
                "device_and_capture is missing sections: " + ", ".join(missing)
            )
        return cls(
            **{name: data[name] for name in DEVICE_SECTIONS},
            sessions=tuple(data.get("sessions", ())),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> "DeviceAndCapture":
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )


# --- stage I: per-field metrics -------------------------------------------


@dataclass(frozen=True)
class FieldMetrics:
    """``evidence/field_metrics.json`` entry for one field (section 12)."""

    field: str
    algorithm_version: str
    threshold: float
    train_samples: int
    calibration_positive_samples: int
    calibration_negative_samples: int
    validation_positive_samples: int
    validation_negative_samples: int
    correct_valid: int
    false_valid: int
    unknown_on_positive: int
    conflict: int
    lowest_accepted_positive: float
    highest_rejected_negative: float
    source_sessions: tuple[str, ...]
    code_sha256: str
    config_sha256: str
    template_sha256: str

    _COUNT_FIELDS = (
        "train_samples",
        "calibration_positive_samples",
        "calibration_negative_samples",
        "validation_positive_samples",
        "validation_negative_samples",
        "correct_valid",
        "false_valid",
        "unknown_on_positive",
        "conflict",
    )

    def __post_init__(self) -> None:
        _require_str("field", self.field)
        _require_str("algorithm_version", self.algorithm_version)
        if self.algorithm_version == PLACEHOLDER:
            raise SchemaError("algorithm_version must not be REPLACE_ME")
        for name in self._COUNT_FIELDS:
            _require_non_negative_int(name, getattr(self, name))
        if not isinstance(self.threshold, (int, float)) or isinstance(
            self.threshold, bool
        ):
            raise SchemaError("threshold must be a number")
        for name in ("lowest_accepted_positive", "highest_rejected_negative"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SchemaError(f"{name} must be a number")
        sessions = tuple(self.source_sessions)
        if not sessions:
            raise SchemaError("source_sessions must name at least one session")
        for session in sessions:
            _require_str("source_sessions entry", session)
        object.__setattr__(self, "source_sessions", sessions)
        for name in ("code_sha256", "config_sha256", "template_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX64.match(value):
                raise SchemaError(f"{name} must be 64 lowercase hex characters")

    @property
    def zero_false_valid(self) -> bool:
        """Section 12: the locked validation set must have zero false VALID."""
        return self.false_valid == 0

    @property
    def recall_on_validation(self) -> float:
        """Recall over validation positives; UNKNOWN counts as missed."""
        positives = self.validation_positive_samples
        if positives <= 0:
            return 0.0
        return (positives - self.unknown_on_positive) / positives

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "algorithm_version": self.algorithm_version,
            "threshold": self.threshold,
            "train_samples": self.train_samples,
            "calibration_positive_samples": self.calibration_positive_samples,
            "calibration_negative_samples": self.calibration_negative_samples,
            "validation_positive_samples": self.validation_positive_samples,
            "validation_negative_samples": self.validation_negative_samples,
            "correct_valid": self.correct_valid,
            "false_valid": self.false_valid,
            "unknown_on_positive": self.unknown_on_positive,
            "conflict": self.conflict,
            "lowest_accepted_positive": self.lowest_accepted_positive,
            "highest_rejected_negative": self.highest_rejected_negative,
            "source_sessions": list(self.source_sessions),
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "template_sha256": self.template_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldMetrics":
        return cls(
            field=data["field"],
            algorithm_version=data["algorithm_version"],
            threshold=data["threshold"],
            train_samples=data["train_samples"],
            calibration_positive_samples=data["calibration_positive_samples"],
            calibration_negative_samples=data["calibration_negative_samples"],
            validation_positive_samples=data["validation_positive_samples"],
            validation_negative_samples=data["validation_negative_samples"],
            correct_valid=data["correct_valid"],
            false_valid=data["false_valid"],
            unknown_on_positive=data["unknown_on_positive"],
            conflict=data["conflict"],
            lowest_accepted_positive=data["lowest_accepted_positive"],
            highest_rejected_negative=data["highest_rejected_negative"],
            source_sessions=tuple(data.get("source_sessions", ())),
            code_sha256=data["code_sha256"],
            config_sha256=data["config_sha256"],
            template_sha256=data["template_sha256"],
        )


__all__ = [
    "ACTOR_VALUES",
    "CompletedAction",
    "DEVICE_SECTIONS",
    "DeviceAndCapture",
    "FieldMetrics",
    "FieldValue",
    "FrameLabel",
    "LabelStatus",
    "MIN_SESSIONS",
    "Occupancy",
    "PLACEHOLDER",
    "RANKS",
    "REVIEW_METHODS",
    "Review",
    "Scene",
    "SchemaError",
    "SlotLabel",
    "Street",
    "SUITS",
    "is_card_code",
    "validate_card_list",
]
