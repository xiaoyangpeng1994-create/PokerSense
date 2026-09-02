"""Stage E geometry drafts (guide section 8).

ROI measurements are taken as integer pixel half-open intervals
``[x0, y0, x1, y1)`` on the **normalized canvas** and converted to the
normalized coordinates the production loader consumes:

    x      = x0 / canvas_width
    y      = y0 / canvas_height
    width  = (x1 - x0) / canvas_width
    height = (y1 - y0) / canvas_height

Which file a measurement lands in follows section 8's instruction that
"board card slots, Dealer and empty-seat evidence use separate layout
files, do not force them into one ROI":

| ``field`` in the CSV | Destination |
|---|---|
| ``hero_cards``, ``board_cards``, ``pot`` | ``table_map.draft.json`` (global) |
| ``stack``, ``action`` | ``table_map.draft.json`` (all 8 slots) |
| ``hero_actor`` | ``table_map.draft.json`` (optional, global) |
| ``board_card`` (x5, ordered) | ``board_slot_layout.draft.json`` |
| ``dealer_search`` (slot 0-7) | ``dealer_slot_layout.draft.json`` |
| ``empty_slot`` (slot 1-7) | ``empty_slot_layout.draft.json`` |

The TableMap draft is built through the real :class:`TableMap` class, so a
draft that fails production validation (duplicate ROI keys, out-of-range
coordinates) fails here instead of at load time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from poker_engine.perceptual.vision.table_map import ROIKind, ROI, TableMap

from . import SLOT_COUNT
from .dataset import RoiMeasurement

# --- field routing ---------------------------------------------------------

TABLE_MAP_FIELDS: Mapping[str, ROIKind] = {
    "hero_cards": ROIKind.HERO_CARDS,
    "board_cards": ROIKind.BOARD_CARDS,
    "pot": ROIKind.POT,
    "stack": ROIKind.STACK,
    "action": ROIKind.ACTION,
    "hero_actor": ROIKind.ACTOR,
}

REQUIRED_GLOBAL_FIELDS: tuple[str, ...] = ("hero_cards", "board_cards", "pot")
PER_SEAT_FIELDS: tuple[str, ...] = ("stack", "action")
OPTIONAL_GLOBAL_FIELDS: tuple[str, ...] = ("hero_actor",)

BOARD_CARD_FIELD = "board_card"
BOARD_CARD_SLOTS = 5
DEALER_SEARCH_FIELD = "dealer_search"
EMPTY_SLOT_FIELD = "empty_slot"
# Hero (slot 0) is never an empty seat, matching the existing empty layout.
EMPTY_SLOT_IDS: tuple[int, ...] = tuple(range(1, SLOT_COUNT))
DEALER_SLOT_IDS: tuple[int, ...] = tuple(range(SLOT_COUNT))


class GeometryError(ValueError):
    """Raised when ROI measurements cannot form a coherent geometry draft."""


# --- conversion ------------------------------------------------------------


def _validate_canvas(canvas: tuple[int, int]) -> tuple[int, int]:
    if len(canvas) != 2:
        raise GeometryError("canvas must be (width, height)")
    width, height = canvas
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GeometryError(f"canvas {name} must be a positive int")
    return (width, height)


def normalized_roi(
    measurement: RoiMeasurement, canvas: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Convert a pixel measurement to normalized ``(x, y, width, height)``."""
    width, height = _validate_canvas(canvas)
    if measurement.x1 > width or measurement.y1 > height:
        raise GeometryError(
            f"{measurement.field} extends to "
            f"({measurement.x1}, {measurement.y1}) which is outside the "
            f"{width}x{height} canvas"
        )
    return (
        measurement.x0 / width,
        measurement.y0 / height,
        measurement.width / width,
        measurement.height / height,
    )


def _group(
    measurements: Iterable[RoiMeasurement],
) -> dict[str, list[RoiMeasurement]]:
    grouped: dict[str, list[RoiMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault(measurement.field, []).append(measurement)
    return grouped


def _reject_unknown_fields(grouped: Mapping[str, list[RoiMeasurement]]) -> None:
    known = set(TABLE_MAP_FIELDS) | {
        BOARD_CARD_FIELD,
        DEALER_SEARCH_FIELD,
        EMPTY_SLOT_FIELD,
    }
    unknown = sorted(set(grouped) - known)
    if unknown:
        raise GeometryError(
            "unrecognized ROI measurement fields: "
            + ", ".join(unknown)
            + f"; expected one of: {', '.join(sorted(known))}"
        )


def _pick_global(
    grouped: Mapping[str, list[RoiMeasurement]], field: str
) -> RoiMeasurement:
    entries = grouped.get(field, [])
    if len(entries) != 1:
        raise GeometryError(
            f"{field} must be measured exactly once, found {len(entries)}"
        )
    entry = entries[0]
    if entry.slot_id is not None:
        raise GeometryError(f"{field} is a global field; slot_id must be empty")
    return entry


def _pick_per_seat(
    grouped: Mapping[str, list[RoiMeasurement]], field: str
) -> list[RoiMeasurement]:
    entries = grouped.get(field, [])
    if not entries:
        raise GeometryError(f"{field} has no measurements")
    by_slot = {entry.slot_id: entry for entry in entries}
    if len(by_slot) != len(entries):
        raise GeometryError(f"{field} has duplicate slot measurements")
    missing = [slot for slot in range(SLOT_COUNT) if slot not in by_slot]
    if missing:
        raise GeometryError(
            f"{field} must cover all {SLOT_COUNT} slots; missing "
            + ", ".join(str(slot) for slot in missing)
        )
    return [by_slot[slot] for slot in range(SLOT_COUNT)]


# --- TableMap draft --------------------------------------------------------


def build_table_map_draft(
    measurements: Sequence[RoiMeasurement],
    *,
    platform_id: str,
    layout_id: str,
    canvas: tuple[int, int],
    aspect_tolerance: float = 0.01,
) -> TableMap:
    """Build a production-loadable TableMap draft from pixel measurements."""
    if not platform_id:
        raise GeometryError("platform_id must be non-empty")
    if not layout_id:
        raise GeometryError("layout_id must be non-empty")
    _validate_canvas(canvas)
    grouped = _group(measurements)
    _reject_unknown_fields(grouped)

    rois: list[ROI] = []
    for field in REQUIRED_GLOBAL_FIELDS:
        entry = _pick_global(grouped, field)
        x, y, width, height = normalized_roi(entry, canvas)
        rois.append(
            ROI(kind=TABLE_MAP_FIELDS[field], x=x, y=y,
                width=width, height=height, slot_id=None)
        )
    for field in PER_SEAT_FIELDS:
        for entry in _pick_per_seat(grouped, field):
            x, y, width, height = normalized_roi(entry, canvas)
            rois.append(
                ROI(
                    kind=TABLE_MAP_FIELDS[field],
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    slot_id=entry.slot_id,
                )
            )
    for field in OPTIONAL_GLOBAL_FIELDS:
        entries = grouped.get(field, [])
        if not entries:
            continue
        entry = _pick_global(grouped, field)
        x, y, width, height = normalized_roi(entry, canvas)
        rois.append(
            ROI(kind=TABLE_MAP_FIELDS[field], x=x, y=y,
                width=width, height=height, slot_id=None)
        )
    return TableMap(
        platform_id=platform_id,
        layout_id=layout_id,
        reference_size=canvas,
        aspect_tolerance=aspect_tolerance,
        rois=tuple(rois),
    )


# --- separate layout drafts ------------------------------------------------


def build_relative_slot_layout(
    measurements: Sequence[RoiMeasurement],
    *,
    kind: str,
    layout_id: str,
    parent: RoiMeasurement,
    expected_count: int,
    version: int = 1,
) -> dict[str, Any]:
    """Slot layout in fractions of a parent ROI (board/hero card slots).

    Matches ``configs/vision/wepoker_android/board_slot_layout.json``, whose
    slot rectangles are fractions of the board-card ROI rather than of the
    whole canvas.
    """
    if len(measurements) != expected_count:
        raise GeometryError(
            f"{kind} slot layout requires exactly {expected_count} "
            f"measurements, got {len(measurements)}"
        )
    parent_width = parent.width
    parent_height = parent.height
    slots = []
    for entry in measurements:
        if entry.slot_id is not None:
            raise GeometryError(
                f"{kind} slot layout entries must not carry a slot_id"
            )
        if not (
            parent.x0 <= entry.x0
            and entry.x1 <= parent.x1
            and parent.y0 <= entry.y0
            and entry.y1 <= parent.y1
        ):
            raise GeometryError(
                f"{kind} slot {entry.source_frame or '?'} is not inside the "
                f"parent {parent.field} ROI"
            )
        slots.append(
            {
                "x": (entry.x0 - parent.x0) / parent_width,
                "y": (entry.y0 - parent.y0) / parent_height,
                "width": entry.width / parent_width,
                "height": entry.height / parent_height,
            }
        )
    return {
        "kind": kind,
        "layout_id": layout_id,
        "status": "draft",
        "version": version,
        "slots": slots,
    }


def build_indexed_slot_layout(
    measurements: Sequence[RoiMeasurement],
    *,
    layout_id: str,
    canvas: tuple[int, int],
    expected_slots: Sequence[int],
    version: int = 1,
) -> dict[str, Any]:
    """Slot layout in canvas coordinates, keyed by ``slot_id``.

    Matches ``configs/vision/wepoker_android/dealer_slot_layout.json``.
    """
    by_slot = {entry.slot_id: entry for entry in measurements}
    if len(by_slot) != len(measurements):
        raise GeometryError("duplicate slot_id in slot layout measurements")
    if any(entry.slot_id is None for entry in measurements):
        raise GeometryError("indexed slot layout entries require a slot_id")
    missing = [slot for slot in expected_slots if slot not in by_slot]
    if missing:
        raise GeometryError(
            "slot layout is missing slot ids: "
            + ", ".join(str(slot) for slot in missing)
        )
    slots = []
    for slot_id in expected_slots:
        entry = by_slot[slot_id]
        x, y, width, height = normalized_roi(entry, canvas)
        slots.append(
            {
                "slot_id": slot_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
    return {
        "layout_id": layout_id,
        "status": "draft",
        "version": version,
        "slots": slots,
    }


def write_geometry_drafts(
    measurements: Sequence[RoiMeasurement],
    geometry_dir: Path | str,
    *,
    platform_id: str,
    layout_id: str,
    canvas: tuple[int, int],
) -> dict[str, Path]:
    """Write every stage E draft file and return their paths."""
    target = Path(geometry_dir)
    target.mkdir(parents=True, exist_ok=True)
    grouped = _group(measurements)
    _reject_unknown_fields(grouped)

    written: dict[str, Path] = {}
    table_map = build_table_map_draft(
        measurements,
        platform_id=platform_id,
        layout_id=layout_id,
        canvas=canvas,
    )
    written["table_map"] = _write_json(
        target / "table_map.draft.json", table_map.to_dict()
    )

    board_cards = grouped.get(BOARD_CARD_FIELD, [])
    if board_cards:
        parent = _pick_global(grouped, "board_cards")
        written["board_slot_layout"] = _write_json(
            target / "board_slot_layout.draft.json",
            build_relative_slot_layout(
                sorted(board_cards, key=lambda m: m.x0),
                kind="board",
                layout_id=layout_id,
                parent=parent,
                expected_count=BOARD_CARD_SLOTS,
            ),
        )
    dealer = grouped.get(DEALER_SEARCH_FIELD, [])
    if dealer:
        written["dealer_slot_layout"] = _write_json(
            target / "dealer_slot_layout.draft.json",
            build_indexed_slot_layout(
                dealer,
                layout_id=layout_id,
                canvas=canvas,
                expected_slots=DEALER_SLOT_IDS,
            ),
        )
    empty = grouped.get(EMPTY_SLOT_FIELD, [])
    if empty:
        written["empty_slot_layout"] = _write_json(
            target / "empty_slot_layout.draft.json",
            build_indexed_slot_layout(
                empty,
                layout_id=layout_id,
                canvas=canvas,
                expected_slots=EMPTY_SLOT_IDS,
            ),
        )
    return written


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "BOARD_CARD_FIELD",
    "BOARD_CARD_SLOTS",
    "DEALER_SEARCH_FIELD",
    "DEALER_SLOT_IDS",
    "EMPTY_SLOT_FIELD",
    "EMPTY_SLOT_IDS",
    "GeometryError",
    "OPTIONAL_GLOBAL_FIELDS",
    "PER_SEAT_FIELDS",
    "REQUIRED_GLOBAL_FIELDS",
    "TABLE_MAP_FIELDS",
    "build_indexed_slot_layout",
    "build_relative_slot_layout",
    "build_table_map_draft",
    "normalized_roi",
    "write_geometry_drafts",
]
