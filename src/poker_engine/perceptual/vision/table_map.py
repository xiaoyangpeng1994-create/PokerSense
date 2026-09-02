"""TableMap / ROI contracts (perceptual layer, NOT Frozen Core).

A TableMap describes *where* each visual field lives on the table, expressed as
normalized (0~1) ROI coordinates relative to a reference client-area size. It
carries NO poker semantics and does NOT interpret player identity — ``slot_id``
is purely a visual seat-slot index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .errors import TableMapError


class ROIKind(str, Enum):
    HERO_CARDS = "hero_cards"
    BOARD_CARDS = "board_cards"
    POT = "pot"
    BET_SIZE = "bet_size"
    STACK = "stack"      # per-seat: slot_id required
    ACTION = "action"    # per-seat: slot_id required
    DEALER = "dealer"
    ACTOR = "actor"


# Kinds that require a slot_id (per-seat geometry).
_PER_SEAT_KINDS = frozenset({ROIKind.STACK, ROIKind.ACTION})


@dataclass(frozen=True)
class ROI:
    kind: ROIKind
    x: float
    y: float
    width: float
    height: float
    slot_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ROIKind):
            raise TypeError("kind must be a ROIKind")
        for name in ("x", "y", "width", "height"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {v}")
            object.__setattr__(self, name, float(v))
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("ROI width/height must be > 0")
        # slot_id semantics: per-seat kinds require an int slot; global kinds
        # must be None.
        if self.kind in _PER_SEAT_KINDS:
            if isinstance(self.slot_id, bool) or not isinstance(self.slot_id, int):
                raise TypeError(f"{self.kind.value} ROI requires int slot_id")
            if self.slot_id < 0:
                raise ValueError("slot_id must be >= 0")
        else:
            if self.slot_id is not None:
                raise ValueError(
                    f"{self.kind.value} is a global field; slot_id must be None"
                )


@dataclass(frozen=True)
class TableMap:
    platform_id: str
    layout_id: str
    reference_size: tuple[int, int]
    aspect_tolerance: float = 0.02
    rois: tuple[ROI, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.platform_id, str) or not self.platform_id:
            raise ValueError("platform_id must be a non-empty str")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a non-empty str")
        rs = tuple(self.reference_size)
        if len(rs) != 2:
            raise ValueError("reference_size must be (width, height)")
        w, h = rs
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError("reference_size width must be a positive int")
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ValueError("reference_size height must be a positive int")
        object.__setattr__(self, "reference_size", rs)
        if isinstance(self.aspect_tolerance, bool) or not isinstance(
            self.aspect_tolerance, (int, float)
        ):
            raise TypeError("aspect_tolerance must be a float")
        if not (0.0 <= float(self.aspect_tolerance) <= 1.0):
            raise ValueError("aspect_tolerance must be in [0.0, 1.0]")
        object.__setattr__(self, "aspect_tolerance", float(self.aspect_tolerance))
        rois = tuple(self.rois)
        if not all(isinstance(r, ROI) for r in rois):
            raise TypeError("rois must be ROI instances")
        object.__setattr__(self, "rois", rois)
        # Reject duplicate visual keys (kind, slot_id) — prevents extract_all
        # from silently overwriting an earlier ROI.
        seen: set[tuple[ROIKind, int | None]] = set()
        for r in rois:
            key = (r.kind, r.slot_id)
            if key in seen:
                raise TableMapError(
                    f"duplicate ROI key {(r.kind.value, r.slot_id)!r}"
                )
            seen.add(key)
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError("schema_version must be an int")

    @property
    def reference_aspect_ratio(self) -> float:
        """Derived at runtime from reference_size (NOT persisted)."""
        w, h = self.reference_size
        return w / h

    # --- serialization (JSON, no pickle) ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "platform_id": self.platform_id,
            "layout_id": self.layout_id,
            "reference_size": [self.reference_size[0], self.reference_size[1]],
            "aspect_tolerance": self.aspect_tolerance,
            "rois": [
                {
                    "kind": r.kind.value,
                    "slot_id": r.slot_id,
                    "x": r.x,
                    "y": r.y,
                    "width": r.width,
                    "height": r.height,
                }
                for r in self.rois
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableMap":
        if not isinstance(data, Mapping):
            raise TypeError("data must be a Mapping")
        version = data.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise TableMapError("schema_version must be an int")
        if version != 1:
            raise TableMapError(f"unsupported schema_version {version!r}")
        rs = data.get("reference_size")
        rois_raw = data.get("rois", [])
        rois = tuple(
            ROI(
                kind=ROIKind(r["kind"]),
                slot_id=r.get("slot_id"),
                x=r["x"],
                y=r["y"],
                width=r["width"],
                height=r["height"],
            )
            for r in rois_raw
        )
        return cls(
            platform_id=data["platform_id"],
            layout_id=data["layout_id"],
            reference_size=(rs[0], rs[1]),
            aspect_tolerance=data.get("aspect_tolerance", 0.02),
            rois=rois,
            schema_version=version,
        )

    @classmethod
    def from_json(cls, text: str) -> "TableMap":
        return cls.from_dict(json.loads(text))


__all__ = ["ROIKind", "ROI", "TableMap"]
