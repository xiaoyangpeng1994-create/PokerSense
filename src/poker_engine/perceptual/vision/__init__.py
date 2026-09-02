"""Vision subpackage: table mapping (ROI contracts) + recognition (Task 7B)."""

from .errors import TableMapError, TableMapMismatchError
from .roi import (
    check_layout_compatibility,
    extract_all,
    extract_roi,
    roi_pixel_bounds,
)
from .table_map import ROIKind, ROI, TableMap
from .card_layout import (
    BoardSlotLayout,
    CardSubROI,
    HeroSlotLayout,
)
from .asset_manifest import VisionAssetManifest
from .calibration import CalibrationBins, ConfidenceCalibrator
from .protocols import (
    ActionRecognition,
    AmountRecognition,
    BoardSlotOccupancy,
    BoardSlotResult,
    BoardSlotsRecognition,
    CalibratedConfidence,
    CardRecognition,
    CardSlotResult,
    StreetRecognition,
)
from .engine import VisionEngine

__all__ = [
    "ROIKind",
    "ROI",
    "TableMap",
    "TableMapError",
    "TableMapMismatchError",
    "check_layout_compatibility",
    "extract_all",
    "extract_roi",
    "roi_pixel_bounds",
    "BoardSlotLayout",
    "CardSubROI",
    "HeroSlotLayout",
    "VisionAssetManifest",
    "CalibrationBins",
    "ConfidenceCalibrator",
    "ActionRecognition",
    "AmountRecognition",
    "BoardSlotOccupancy",
    "BoardSlotResult",
    "BoardSlotsRecognition",
    "CalibratedConfidence",
    "CardRecognition",
    "CardSlotResult",
    "StreetRecognition",
    "VisionEngine",
]
