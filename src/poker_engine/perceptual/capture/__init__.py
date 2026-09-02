"""Capture subpackage: frame contracts and capture backends."""

from .base import (
    CaptureError,
    CaptureService,
    CaptureTarget,
    Frame,
    WindowRect,
)
from .fake_backend import FakeBackend
from .adb_backend import AdbBackend
from .capture_card_backend import CaptureCardBackend
from .mss_backend import MssBackend
from .normalization import NormalizationConfig, normalize
from .quartz_backend import (
    QuartzBackend,
    WindowCandidate,
    has_screen_capture_permission,
    list_window_candidates,
    request_screen_capture_permission,
)

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
    "MssBackend",
    "QuartzBackend",
    "WindowCandidate",
    "has_screen_capture_permission",
    "list_window_candidates",
    "request_screen_capture_permission",
]
