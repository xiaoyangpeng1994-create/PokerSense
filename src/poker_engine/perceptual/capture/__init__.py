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
from .mss_backend import MssBackend
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
    "MssBackend",
    "QuartzBackend",
    "WindowCandidate",
    "has_screen_capture_permission",
    "list_window_candidates",
    "request_screen_capture_permission",
]
