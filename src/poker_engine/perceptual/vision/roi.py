"""Pure, deterministic ROI -> pixel transformation + layout compatibility check.

``Frame -> ROI crops`` is a pure function: same Frame + same TableMap yields
identical crops. Uses explicit floor (``int()``) for normalized -> pixel
rounding; layout compatibility fails fast when the actual frame aspect ratio
deviates beyond the TableMap's tolerance.
"""

from __future__ import annotations

import numpy as np

from ..capture.base import Frame
from .errors import TableMapMismatchError
from .table_map import ROI, TableMap


def _pixel(value: float, size: int) -> int:
    """Convert a normalized coordinate to a pixel index (explicit floor)."""
    return int(value * size)


def check_layout_compatibility(table_map: TableMap, frame: Frame) -> None:
    """Fail fast if the frame's aspect ratio is incompatible with the map."""
    if frame.width <= 0 or frame.height <= 0:
        raise TableMapMismatchError("frame has non-positive dimensions")
    actual_aspect = frame.width / frame.height
    ref_aspect = table_map.reference_aspect_ratio
    if abs(actual_aspect - ref_aspect) > table_map.aspect_tolerance:
        raise TableMapMismatchError(
            f"frame aspect {actual_aspect:.4f} deviates from reference "
            f"{ref_aspect:.4f} beyond tolerance {table_map.aspect_tolerance}"
        )


def roi_pixel_bounds(roi: ROI, frame: Frame) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) pixel bounds for an ROI on the given frame.

    Uses the frame's actual width/height for normalized -> pixel conversion.
    """
    x0 = _pixel(roi.x, frame.width)
    y0 = _pixel(roi.y, frame.height)
    x1 = _pixel(roi.x + roi.width, frame.width)
    y1 = _pixel(roi.y + roi.height, frame.height)
    return x0, y0, x1, y1


def extract_roi(frame: Frame, roi: ROI) -> np.ndarray:
    """Return the deterministic pixel crop for a single ROI."""
    x0, y0, x1, y1 = roi_pixel_bounds(roi, frame)
    if x1 <= x0 or y1 <= y0:
        raise TableMapMismatchError(
            f"ROI {roi.kind.value} collapses to zero size at frame "
            f"{frame.width}x{frame.height}"
        )
    if x1 > frame.width or y1 > frame.height:
        raise TableMapMismatchError(f"ROI {roi.kind.value} exceeds frame bounds")
    return frame.image[y0:y1, x0:x1]


def extract_all(table_map: TableMap, frame: Frame) -> dict[str, np.ndarray]:
    """Validate layout once, then deterministically crop every ROI.

    Keys are ``kind`` for global ROIs and ``f"{kind}:{slot_id}"`` for per-seat.
    """
    check_layout_compatibility(table_map, frame)
    result: dict[str, np.ndarray] = {}
    for roi in table_map.rois:
        key = (
            roi.kind.value
            if roi.slot_id is None
            else f"{roi.kind.value}:{roi.slot_id}"
        )
        result[key] = extract_roi(frame, roi)
    return result


__all__ = [
    "check_layout_compatibility",
    "roi_pixel_bounds",
    "extract_roi",
    "extract_all",
]
