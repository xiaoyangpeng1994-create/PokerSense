"""Capture contracts: Frame, WindowRect, CaptureTarget, CaptureError.

Perceptual-layer objects (NOT Frozen Core). ``Frame`` carries an independent,
immutable, bytes-backed pixel buffer exposing a numpy read-only view.

Pixel immutability guarantee:
- The pixel buffer is an independent ``bytes`` copy (not a view onto the
  caller's ndarray) — mutating the source ndarray never affects ``Frame``.
- The buffer is exposed as a numpy read-only view via ``numpy.frombuffer``.
  Because the backing store is immutable ``bytes``, even ``setflags(write=True)``
  cannot be used to regain write access.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WindowRect:
    """A rectangle in physical pixels (virtual-desktop coordinates)."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Frame:
    """A single captured frame with an immutable pixel buffer.

    ``image`` is a numpy read-only view backed by an independent ``bytes``
    buffer. Callers cannot mutably write through it, and cannot re-enable
    writability via ``setflags(write=True)``.
    """

    frame_seq: int
    timestamp: datetime
    window_id: str
    window_rect: WindowRect
    image: np.ndarray
    width: int
    height: int

    def __init__(
        self,
        frame_seq: int,
        timestamp: datetime,
        window_id: str,
        window_rect: WindowRect,
        image: Any,
        width: int,
        height: int,
    ) -> None:
        # --- basic contract validation (mirrors Frozen Core style) ---
        if not isinstance(frame_seq, int) or isinstance(frame_seq, bool):
            raise TypeError("frame_seq must be an int")
        if frame_seq < 0:
            raise ValueError("frame_seq must be >= 0")
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
            raise TypeError("timestamp must be timezone-aware")
        if not isinstance(window_id, str) or not window_id:
            raise ValueError("window_id must be a non-empty str")
        if not isinstance(window_rect, WindowRect):
            raise TypeError("window_rect must be a WindowRect")
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("width must be an int")
        if not isinstance(height, int) or isinstance(height, bool):
            raise TypeError("height must be an int")

        # --- defensive, independent bytes-backed copy ---
        arr = np.asarray(image)
        if arr.ndim < 2:
            raise ValueError("image must be at least 2-D (height, width, ...)")
        if arr.shape[1] != width or arr.shape[0] != height:
            raise ValueError(
                f"image shape {arr.shape[:2]} does not match "
                f"(height={height}, width={width})"
            )
        # tobytes() returns a brand-new immutable bytes object: an independent
        # copy of the pixel data, decoupled from the caller's ndarray buffer.
        buf = arr.tobytes(order="C")
        # A read-only view over immutable bytes: cannot be made writable.
        view = np.frombuffer(buf, dtype=arr.dtype).reshape(arr.shape)

        object.__setattr__(self, "frame_seq", frame_seq)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "window_rect", window_rect)
        object.__setattr__(self, "image", view)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


class CaptureError(Exception):
    """Raised when a capture target cannot be captured (invalid/occluded/etc)."""


@dataclass(frozen=True)
class CaptureTarget:
    """What visual source to capture.

    ``window_id`` is the backend-specific stable identifier: a window title for
    desktop backends or an ADB serial for Android capture.  The historical name
    is retained to avoid changing the frozen capture contract.
    """

    window_id: str
    window_index: int | None = None
    allow_fullscreen_fallback: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValueError("window_id must be a non-empty str")
        if self.window_index is not None:
            if isinstance(self.window_index, bool) or not isinstance(
                self.window_index, int
            ):
                raise TypeError("window_index must be an int or None")
            if self.window_index < 0:
                raise ValueError("window_index must be >= 0")
        if not isinstance(self.allow_fullscreen_fallback, bool):
            raise TypeError("allow_fullscreen_fallback must be a bool")


class CaptureService:
    """Abstract visual-source capture backend.

    ``frame_seq`` is assigned internally and increases strictly monotonically
    within a session (never derived from wall-clock time, never caller-supplied).
    """

    def __init__(self) -> None:
        self._seq = 0

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def capture(self, target: CaptureTarget) -> Frame:
        raise NotImplementedError


__all__ = [
    "Frame",
    "WindowRect",
    "CaptureTarget",
    "CaptureError",
    "CaptureService",
]
