"""Perceptual layer: capture + table mapping (Task 6).

Provides ``Frame`` (immutable capture) and ``TableMap``/``ROI`` + deterministic
ROI extraction for Task 7 Vision consumption. No poker recognition here.
"""

from .capture.base import (
    CaptureError,
    CaptureService,
    CaptureTarget,
    Frame,
    WindowRect,
)
from .capture.fake_backend import FakeBackend
from .capture.adb_backend import AdbBackend
from .capture.capture_card_backend import CaptureCardBackend
from .capture.normalization import NormalizationConfig, normalize
from .vision.errors import TableMapError, TableMapMismatchError
from .vision.roi import (
    check_layout_compatibility,
    extract_all,
    extract_roi,
    roi_pixel_bounds,
)
from .vision.table_map import ROIKind, ROI, TableMap

__all__ = [
    "CaptureError",
    "CaptureService",
    "CaptureTarget",
    "Frame",
    "WindowRect",
    "FakeBackend",
    "AdbBackend",
    "CaptureCardBackend",
    "NormalizationConfig",
    "normalize",
    "ROIKind",
    "ROI",
    "TableMap",
    "TableMapError",
    "TableMapMismatchError",
    "check_layout_compatibility",
    "extract_all",
    "extract_roi",
    "roi_pixel_bounds",
]
