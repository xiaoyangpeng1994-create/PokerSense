"""Fixed-layout visual-slot marker recognition for Android table badges."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np

from .protocols import freeze_templates


@dataclass(frozen=True)
class SlotSearchROI:
    slot_id: int
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, int) or isinstance(self.slot_id, bool):
            raise TypeError("slot_id must be an int")
        if self.slot_id < 0:
            raise ValueError("slot_id must be >= 0")
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
            object.__setattr__(self, name, float(value))
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("search ROI dimensions must be positive")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("search ROI must fit inside the frame")


@dataclass(frozen=True)
class SlotMarkerLayout:
    layout_id: str
    version: int
    slots: tuple[SlotSearchROI, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a non-empty str")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError("version must be an int")
        slots = tuple(self.slots)
        if not slots:
            raise ValueError("slots must be non-empty")
        ids = [slot.slot_id for slot in slots]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("slot ids must be unique and ascending")
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True)
class SlotMarkerRecognition:
    slot_id: int | None
    raw_score: float
    runner_up_score: float


class TemplateSlotMarkerRecognizer:
    """Find one marker across explicit per-slot search windows."""

    def __init__(
        self, template: np.ndarray, layout: SlotMarkerLayout, version: str
    ) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty str")
        self._template = freeze_templates({"marker": template})["marker"]
        self._layout = layout
        self.version = version

    def recognize(self, image: np.ndarray) -> SlotMarkerRecognition:
        if image is None or image.size == 0:
            return SlotMarkerRecognition(None, 0.0, 0.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(self._template, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scores: list[tuple[float, int]] = []
        for slot in self._layout.slots:
            x0 = round(slot.x * width)
            y0 = round(slot.y * height)
            x1 = round((slot.x + slot.width) * width)
            y1 = round((slot.y + slot.height) * height)
            search = gray[y0:y1, x0:x1]
            if (
                search.shape[0] < template.shape[0]
                or search.shape[1] < template.shape[1]
            ):
                scores.append((0.0, slot.slot_id))
                continue
            result = cv2.matchTemplate(
                search, template, cv2.TM_CCOEFF_NORMED
            )
            _, maximum, _, _ = cv2.minMaxLoc(result)
            scores.append((float(max(0.0, maximum)), slot.slot_id))
        scores.sort(reverse=True)
        best_score, best_slot = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        return SlotMarkerRecognition(best_slot, best_score, runner_up)


class TemplatePerSlotMarkerRecognizer:
    """Score the same binary marker independently in every visual slot."""

    def __init__(
        self,
        template: np.ndarray,
        layout: SlotMarkerLayout,
        version: str,
        *,
        binary_threshold: int = 140,
    ) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("version must be a non-empty str")
        if not isinstance(binary_threshold, int) or isinstance(
            binary_threshold, bool
        ):
            raise TypeError("binary_threshold must be an int")
        if not 0 <= binary_threshold <= 255:
            raise ValueError("binary_threshold must be in [0,255]")
        frozen = freeze_templates({"marker": template})["marker"]
        gray = cv2.cvtColor(frozen, cv2.COLOR_BGR2GRAY)
        self._template = cv2.threshold(
            gray, binary_threshold, 255, cv2.THRESH_BINARY
        )[1]
        self._layout = layout
        self._binary_threshold = binary_threshold
        self.version = version

    def recognize(self, image: np.ndarray) -> Mapping[int, float]:
        if image is None or image.size == 0:
            return {slot.slot_id: 0.0 for slot in self._layout.slots}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(
            gray, self._binary_threshold, 255, cv2.THRESH_BINARY
        )[1]
        height, width = binary.shape[:2]
        scores = {}
        for slot in self._layout.slots:
            x0 = round(slot.x * width)
            y0 = round(slot.y * height)
            x1 = round((slot.x + slot.width) * width)
            y1 = round((slot.y + slot.height) * height)
            search = binary[y0:y1, x0:x1]
            if (
                search.shape[0] < self._template.shape[0]
                or search.shape[1] < self._template.shape[1]
            ):
                scores[slot.slot_id] = 0.0
                continue
            result = cv2.matchTemplate(
                search, self._template, cv2.TM_CCOEFF_NORMED
            )
            _, maximum, _, _ = cv2.minMaxLoc(result)
            scores[slot.slot_id] = float(max(0.0, maximum))
        return MappingProxyType(scores)


def slot_marker_layout_from_dict(data: Mapping) -> SlotMarkerLayout:
    return SlotMarkerLayout(
        layout_id=data["layout_id"],
        version=data["version"],
        slots=tuple(SlotSearchROI(**slot) for slot in data["slots"]),
    )


__all__ = [
    "SlotSearchROI",
    "SlotMarkerLayout",
    "SlotMarkerRecognition",
    "TemplateSlotMarkerRecognizer",
    "TemplatePerSlotMarkerRecognizer",
    "slot_marker_layout_from_dict",
]
