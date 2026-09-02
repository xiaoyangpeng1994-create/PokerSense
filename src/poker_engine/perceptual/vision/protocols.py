"""Vision detector/adapter protocols and immutable recognition result objects.

These are PURE contracts: inputs/outputs are plain data (numpy arrays on the
way in, immutable dataclasses on the way out). No OpenCV/Paddle object types
leak into these contracts or into the Frozen Core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, TypeVar

import numpy as np

from poker_engine.core.enums import ActionType, Rank, Street, Suit
from poker_engine.core.observation import ValidationStatus
from poker_engine.core.value_objects import Card, ChipAmount


def freeze_templates(templates: Mapping, version_hint: str = "") -> Mapping:
    """Deep-freeze a template mapping for immutable exposure.

    - The mapping is wrapped in ``MappingProxyType`` (cannot add/remove keys).
    - Every value MUST be a numpy ndarray (anything else is rejected — a
      template set is, by contract, a set of image arrays).
    - Every ndarray value is COPIED into a **bytes-backed** read-only array:
      the underlying buffer is immutable ``bytes``, so the array's WRITEABLE
      flag cannot be re-enabled (``setflags(write=True)`` raises) and callers
      cannot mutate template pixels via a shared reference.

    Returns a read-only mapping whose ndarray values are write-protected at the
    buffer level.
    """
    frozen: dict = {}
    for key, value in templates.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"template {key!r} must be a numpy.ndarray, "
                f"got {type(value).__name__}"
            )
        arr = np.array(value, copy=True)
        # Rebuild on an immutable bytes buffer: this cannot be made writable.
        buf = arr.tobytes()
        read_only = np.frombuffer(buf, dtype=arr.dtype).reshape(arr.shape)
        frozen[key] = read_only
    return MappingProxyType(frozen)


class BoardSlotOccupancy(str, Enum):
    CARD = "card"
    EMPTY = "empty"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Recognition result objects (frozen, deterministic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardSlotResult:
    """A single card-position recognition (one rank or suit match slot)."""

    rank_score: float
    suit_score: float
    rank: Rank | None = None
    suit: Suit | None = None

    def __post_init__(self) -> None:
        _check_raw_score(self.rank_score, "rank_score")
        _check_raw_score(self.suit_score, "suit_score")


@dataclass(frozen=True)
class CardRecognition:
    value: tuple[Card, ...] | None
    raw_score: float
    slots: tuple[CardSlotResult, ...]

    def __post_init__(self) -> None:
        _check_raw_score(self.raw_score, "raw_score")


@dataclass(frozen=True)
class AmountRecognition:
    value: ChipAmount | None
    raw_score: float

    def __post_init__(self) -> None:
        _check_raw_score(self.raw_score, "raw_score")


@dataclass(frozen=True)
class ActionRecognition:
    value: ActionType | None
    raw_score: float
    runner_up_score: float = 0.0

    def __post_init__(self) -> None:
        _check_raw_score(self.raw_score, "raw_score")
        _check_raw_score(self.runner_up_score, "runner_up_score")


@dataclass(frozen=True)
class BoardSlotResult:
    # raw_score semantics: "evidence strength for the SELECTED occupancy".
    # higher == stronger evidence (CARD: card-presence evidence; EMPTY: empty/
    # background evidence; UNKNOWN: weak evidence). Normalized to a finite
    # value in a sane range (nominally [0,1]).
    #
    # occupancy is the INDEPENDENT presence/empty signal. ``card`` is the
    # SEPARATE rank/suit identity result. They may disagree: a slot can be
    # occupancy=CARD with card=None (presence says "card" but identity failed
    # to recognize it) — this is a valid Vision-layer intermediate state that
    # the engine promotes to CONFLICT. Only EMPTY/UNKNOWN require card=None
    # (a non-empty identity on an empty/unknown slot is contradictory by
    # construction and rejected early).
    slot_index: int                 # 0..4
    occupancy: BoardSlotOccupancy
    card: Card | None               # None allowed for CARD (identity unconfirmed)
    raw_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.slot_index, int) or isinstance(self.slot_index, bool):
            raise TypeError("slot_index must be an int")
        if self.slot_index < 0 or self.slot_index > 4:
            raise ValueError("slot_index must be in 0..4")
        if not isinstance(self.occupancy, BoardSlotOccupancy):
            raise TypeError("occupancy must be a BoardSlotOccupancy")
        _check_raw_score(self.raw_score, "raw_score")
        # EMPTY / UNKNOWN must not carry a card identity.
        if self.occupancy is not BoardSlotOccupancy.CARD and self.card is not None:
            raise ValueError("EMPTY/UNKNOWN slot must have card=None")


@dataclass(frozen=True)
class BoardSlotsRecognition:
    slots: tuple[BoardSlotResult, ...]   # strictly 5, slot_index 0..4 ordered

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        if len(slots) != 5:
            raise ValueError("BoardSlotsRecognition must have exactly 5 slots")
        if not all(isinstance(s, BoardSlotResult) for s in slots):
            raise TypeError("slots must be BoardSlotResult instances")
        for i, s in enumerate(slots):
            if s.slot_index != i:
                raise ValueError(
                    "board slot_index must be strictly 0,1,2,3,4 in order; "
                    f"got slot_index {s.slot_index} at position {i}"
                )
        object.__setattr__(self, "slots", slots)


def _check_raw_score(v: float, name: str) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be a float")
    import math

    if not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite")
    if not (0.0 <= float(v) <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]")


@dataclass(frozen=True)
class StreetRecognition:
    street: Street | None
    status: ValidationStatus
    raw_score: float                    # street raw feature (must be calibrated)
    evidence: tuple[BoardSlotResult, ...]

    def __post_init__(self) -> None:
        _check_raw_score(self.raw_score, "raw_score")


@dataclass(frozen=True)
class CalibratedConfidence:
    confidence: float                   # [0,1]; no ">= threshold -> VALID" semantics

    def __post_init__(self) -> None:
        _check_raw_score(self.confidence, "confidence")


# ---------------------------------------------------------------------------
# Detector / adapter protocols
# ---------------------------------------------------------------------------

T = TypeVar("T")


class CardRecognizer(Protocol):
    def recognize(self, roi_image: np.ndarray, card_model) -> CardRecognition:
        ...


class AmountRecognizer(Protocol):
    def recognize(self, roi_image: np.ndarray) -> AmountRecognition:
        ...


class ActionRecognizer(Protocol):
    def recognize(self, roi_image: np.ndarray, slot_id: int) -> ActionRecognition:
        ...


class BoardSlotDetector(Protocol):
    def detect(self, board_roi_image: np.ndarray) -> BoardSlotsRecognition:
        ...


class StreetDetector(Protocol):
    def derive(self, board_slots: BoardSlotsRecognition) -> StreetRecognition:
        ...


class ConfidenceCalibrator(Protocol[T]):
    def calibrate(self, raw_score: float) -> CalibratedConfidence:
        ...


__all__ = [
    "BoardSlotOccupancy",
    "CardSlotResult",
    "CardRecognition",
    "AmountRecognition",
    "ActionRecognition",
    "BoardSlotResult",
    "BoardSlotsRecognition",
    "StreetRecognition",
    "CalibratedConfidence",
    "CardRecognizer",
    "AmountRecognizer",
    "ActionRecognizer",
    "BoardSlotDetector",
    "StreetDetector",
    "ConfidenceCalibrator",
]
