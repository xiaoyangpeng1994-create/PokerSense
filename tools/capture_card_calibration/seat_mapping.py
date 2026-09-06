# -*- coding: utf-8 -*-
"""Stage J: visual-slot to canonical-seat mapping for the capture-card
platform (``wepoker_android_capture_card``).

The platform fixes the hero at the bottom-centre of the canvas. Visual
slots are numbered 0..7 starting at the hero and ascending in the
**direction of play** (up the left column, across the top, down the right
column):

- 0 = hero (bottom), 1 = lower-left, 2 = mid-left, 3 = upper-left,
  4 = top, 5 = upper-right, 6 = mid-right, 7 = lower-right.

Direction-of-play evidence (private corpus ``labels/frames.jsonl``): the
dealer badge advances through ascending slot numbers across consecutive
hands — session_002 shows 2 -> 4 -> 7 -> 1 -> 2 -> 4 (skipping empty
slots, wrapping past the hero) — matching poker's clockwise button
movement. Ascending slot therefore equals ascending canonical seat, and
the mapping is the identity. Hero is canonical seat 0 by platform
convention, which matches ``_position_players`` (ascending seat order =
BTN, SB, BB, ...).

Consistency rules (failure-closed, mirroring the guide section 13):

- exactly one dealer badge across all slots, otherwise the dealer seat is
  ``UNKNOWN`` (none readable) or a conflict (more than one);
- a slot read EMPTY must not carry a stack value (``EMPTY`` + stack is a
  cross-field conflict);
- the hero is always canonical seat 0; a seat mapping that does not map
  every field's slot set explicitly is rejected.

This module never modifies the LDPlayer mapping; the capture-card mapping
is an independent artifact with independent geometry evidence (guide
rules 1 and 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from poker_engine.state_engine.platform_mapping import PlatformSeatMapping

from . import SLOT_COUNT
from .seat_reader import HERO_SLOT
from .schema import FieldValue, LabelStatus, Occupancy

__all__ = [
    "CAPTURE_CARD_LAYOUT_ID",
    "CAPTURE_CARD_PLATFORM_ID",
    "CAPTURE_CARD_SEAT_MAPPING_VERSION",
    "SeatConsistencyError",
    "build_capture_card_seat_mapping",
    "check_occupancy_stack_consistency",
    "derive_canonical_dealer",
    "identity_slot_map",
    "load_seat_mapping_file",
    "seat_mapping_document",
    "summarize_occupancy",
    "write_seat_mapping_file",
]

CAPTURE_CARD_PLATFORM_ID = "wepoker_android_capture_card"
CAPTURE_CARD_LAYOUT_ID = (
    "phone_samsung_galaxy_s25_ultra__card_ugreen__uvc_1920x1080_30"
    "__canvas_498x1080__v1"
)
CAPTURE_CARD_SEAT_MAPPING_VERSION = "wepoker-android-capture-card-seat-map-v1"


class SeatConsistencyError(ValueError):
    """Raised when per-slot seat evidence is internally inconsistent."""


def identity_slot_map() -> dict[int, int]:
    """Identity slot->seat map for the full 8-max ring (hero = seat 0)."""
    return {slot: slot for slot in range(SLOT_COUNT)}


def build_capture_card_seat_mapping() -> PlatformSeatMapping:
    """Production mapping object for the capture-card platform.

    Every field uses the identity map with its own independent geometry
    evidence (guide section 13: stack/action/dealer/occupancy each map
    independently). ``actor_observation_is_current`` is True: the actor
    ROI reports the player *currently facing a decision* (the blue
    "N 跟注" decision pill), matching the Android profile semantics.
    """
    slots = identity_slot_map()
    return PlatformSeatMapping(
        platform_id=CAPTURE_CARD_PLATFORM_ID,
        layout_id=CAPTURE_CARD_LAYOUT_ID,
        version=CAPTURE_CARD_SEAT_MAPPING_VERSION,
        stack_slot_to_seat=slots,
        action_slot_to_seat=slots,
        actor_slot_to_seat=dict(slots),
        dealer_slot_to_seat=slots,
        occupancy_slot_to_seat=slots,
        actor_observation_is_current=True,
    )


def derive_canonical_dealer(
    slot_dealers: Mapping[int, FieldValue],
) -> FieldValue:
    """Reduce per-slot dealer reads to one canonical dealer seat.

    - exactly one ``VALID True`` -> that seat (VALID);
    - no positive read -> UNKNOWN (never guessed);
    - more than one positive read -> CONFLICT (contradictory evidence).
    """
    positives = sorted(
        slot for slot, fv in slot_dealers.items()
        if fv.status is LabelStatus.VALID and fv.value is True
    )
    if len(positives) == 1:
        return FieldValue.valid(positives[0])
    if not positives:
        return FieldValue.unknown()
    return FieldValue.conflict()


def check_occupancy_stack_consistency(
    slot_occupancies: Mapping[int, FieldValue],
    slot_stacks: Mapping[int, FieldValue],
) -> None:
    """Cross-field check: an EMPTY slot must never carry a stack value.

    Raises :class:`SeatConsistencyError` on the first conflict; callers
    that cannot resolve the conflict must treat the frame's seat evidence
    as unusable (fail closed).
    """
    for slot, occ in slot_occupancies.items():
        if occ.status is not LabelStatus.VALID:
            continue
        if occ.value is not Occupancy.EMPTY and occ.value != Occupancy.EMPTY.value:
            continue
        stack = slot_stacks.get(slot)
        if (
            stack is not None
            and stack.status is LabelStatus.VALID
            and stack.value is not None
        ):
            raise SeatConsistencyError(
                f"slot {slot}: EMPTY occupancy conflicts with stack "
                f"{stack.value!r}"
            )


def seat_mapping_document() -> dict:
    """Serializable seat-mapping document (schema_version 1)."""
    slots = identity_slot_map()
    return {
        "schema_version": 1,
        "platform_id": CAPTURE_CARD_PLATFORM_ID,
        "layout_id": CAPTURE_CARD_LAYOUT_ID,
        "version": CAPTURE_CARD_SEAT_MAPPING_VERSION,
        "stack_slot_to_seat": {str(k): v for k, v in slots.items()},
        "action_slot_to_seat": {str(k): v for k, v in slots.items()},
        "actor_slot_to_seat": {str(k): v for k, v in slots.items()},
        "dealer_slot_to_seat": {str(k): v for k, v in slots.items()},
        "occupancy_slot_to_seat": {str(k): v for k, v in slots.items()},
        "actor_observation_is_current": True,
    }


def _validate_slot_map(value: object, name: str) -> dict[int, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    out: dict[int, int] = {}
    for key, seat in value.items():
        try:
            slot = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} slot key {key!r} is not an int") from exc
        if not isinstance(seat, int) or isinstance(seat, bool):
            raise ValueError(f"{name} seat for slot {key!r} must be an int")
        if slot < 0 or seat < 0 or slot >= SLOT_COUNT or seat >= SLOT_COUNT:
            raise ValueError(
                f"{name} maps outside the 0..{SLOT_COUNT - 1} ring: "
                f"{key!r} -> {seat!r}"
            )
        out[slot] = seat
    if len(set(out.values())) != len(out):
        raise ValueError(f"{name} must map one-to-one")
    return out


def load_seat_mapping_file(path: Path | str) -> PlatformSeatMapping:
    """Load and validate a seat-mapping JSON document (fail closed)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("seat mapping document must be a JSON object")
    if data.get("schema_version") != 1:
        raise ValueError("unsupported seat mapping schema_version")
    for name in ("platform_id", "layout_id", "version"):
        if not isinstance(data.get(name), str) or not data[name]:
            raise ValueError(f"{name} must be a non-empty string")
    maps = {
        name: _validate_slot_map(data.get(name), name)
        for name in (
            "stack_slot_to_seat",
            "action_slot_to_seat",
            "actor_slot_to_seat",
            "dealer_slot_to_seat",
            "occupancy_slot_to_seat",
        )
    }
    if HERO_SLOT not in maps["occupancy_slot_to_seat"]:
        raise ValueError("occupancy map must cover the hero slot 0")
    return PlatformSeatMapping(
        platform_id=data["platform_id"],
        layout_id=data["layout_id"],
        version=data["version"],
        stack_slot_to_seat=maps["stack_slot_to_seat"],
        action_slot_to_seat=maps["action_slot_to_seat"],
        actor_slot_to_seat=maps["actor_slot_to_seat"],
        dealer_slot_to_seat=maps["dealer_slot_to_seat"],
        occupancy_slot_to_seat=maps["occupancy_slot_to_seat"],
        actor_observation_is_current=bool(
            data.get("actor_observation_is_current", False)
        ),
    )


def write_seat_mapping_file(path: Path | str) -> Path:
    """Write the canonical seat-mapping document to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(seat_mapping_document(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def summarize_occupancy(
    slot_occupancies: Mapping[int, FieldValue],
) -> tuple[int, ...] | None:
    """Canonical occupied-seat tuple, or ``None`` when evidence is absent.

    Seats whose occupancy is not VALID simply do not appear; an empty
    result (no VALID reads at all) yields ``None`` so callers fail closed
    instead of asserting an empty table.
    """
    occupied = sorted(
        slot for slot, fv in slot_occupancies.items()
        if fv.status is LabelStatus.VALID
        and (fv.value is Occupancy.OCCUPIED or fv.value == Occupancy.OCCUPIED.value)
    )
    known = any(fv.status is LabelStatus.VALID for fv in slot_occupancies.values())
    if not known:
        return None
    return tuple(occupied)
