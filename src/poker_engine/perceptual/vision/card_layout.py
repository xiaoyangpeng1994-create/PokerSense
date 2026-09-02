"""Card slot geometry contracts (Vision detector configuration, NOT Frozen Core).

``CardSubROI`` coordinates are normalized (0..1) RELATIVE to their parent ROI
(the BOARD_CARDS or HERO_CARDS ROI from TableMap). No absolute screen
coordinates are permitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from .errors import TableMapError


@dataclass(frozen=True)
class CardSubROI:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {v}")
            object.__setattr__(self, name, float(v))
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("CardSubROI width/height must be > 0")


@dataclass(frozen=True)
class BoardSlotLayout:
    """5 card sub-ROIs (relative to BOARD_CARDS)."""

    layout_id: str
    version: int
    slots: tuple[CardSubROI, ...]

    def __post_init__(self) -> None:
        _validate_layout(self, 5, "BoardSlotLayout")


@dataclass(frozen=True)
class HeroSlotLayout:
    """2 card sub-ROIs (relative to HERO_CARDS)."""

    layout_id: str
    version: int
    slots: tuple[CardSubROI, ...]

    def __post_init__(self) -> None:
        _validate_layout(self, 2, "HeroSlotLayout")


def _validate_layout(obj, expected_len: int, name: str) -> None:
    if not isinstance(obj.layout_id, str) or not obj.layout_id:
        raise ValueError(f"{name}.layout_id must be a non-empty str")
    if isinstance(obj.version, bool) or not isinstance(obj.version, int):
        raise TypeError(f"{name}.version must be an int")
    slots = tuple(obj.slots)
    if len(slots) != expected_len:
        raise ValueError(
            f"{name} must have exactly {expected_len} slots, got {len(slots)}"
        )
    if not all(isinstance(s, CardSubROI) for s in slots):
        raise TypeError(f"{name}.slots must be CardSubROI instances")
    object.__setattr__(obj, "slots", slots)


def _subroi_to_dict(r: CardSubROI) -> dict:
    return {"x": r.x, "y": r.y, "width": r.width, "height": r.height}


def _subroi_from_dict(d: Mapping) -> CardSubROI:
    return CardSubROI(
        x=d["x"], y=d["y"], width=d["width"], height=d["height"],
    )


def board_layout_to_dict(layout: BoardSlotLayout) -> dict:
    return {
        "kind": "board",
        "layout_id": layout.layout_id,
        "version": layout.version,
        "slots": [_subroi_to_dict(s) for s in layout.slots],
    }


def hero_layout_to_dict(layout: HeroSlotLayout) -> dict:
    return {
        "kind": "hero",
        "layout_id": layout.layout_id,
        "version": layout.version,
        "slots": [_subroi_to_dict(s) for s in layout.slots],
    }


def board_layout_from_dict(data: Mapping) -> BoardSlotLayout:
    if data.get("kind") != "board":
        raise TableMapError("expected board card layout")
    return BoardSlotLayout(
        layout_id=data["layout_id"],
        version=data["version"],
        slots=tuple(_subroi_from_dict(s) for s in data["slots"]),
    )


def hero_layout_from_dict(data: Mapping) -> HeroSlotLayout:
    if data.get("kind") != "hero":
        raise TableMapError("expected hero card layout")
    return HeroSlotLayout(
        layout_id=data["layout_id"],
        version=data["version"],
        slots=tuple(_subroi_from_dict(s) for s in data["slots"]),
    )


def board_layout_to_json(layout: BoardSlotLayout) -> str:
    return json.dumps(
        board_layout_to_dict(layout), sort_keys=True, separators=(",", ":")
    )


def hero_layout_to_json(layout: HeroSlotLayout) -> str:
    return json.dumps(
        hero_layout_to_dict(layout), sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "CardSubROI",
    "BoardSlotLayout",
    "HeroSlotLayout",
    "board_layout_to_dict",
    "hero_layout_to_dict",
    "board_layout_from_dict",
    "hero_layout_from_dict",
    "board_layout_to_json",
    "hero_layout_to_json",
]
