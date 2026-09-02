"""Fake capture backend for unit/integration tests (deterministic, no UI)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .base import CaptureTarget, CaptureService, Frame, WindowRect


class FakeBackend(CaptureService):
    """Returns a fixed, deterministic frame (or a caller-provided image)."""

    def __init__(
        self,
        image: Any | None = None,
        window_id: str = "fake-table",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        super().__init__()
        if image is None:
            h = height if height is not None else 60
            w = width if width is not None else 100
            self._image = np.zeros((h, w, 3), dtype=np.uint8)
            self._width = w
            self._height = h
        else:
            arr = np.asarray(image)
            # Derive dimensions from the image shape.
            self._height, self._width = arr.shape[0], arr.shape[1]
            self._image = arr
        self._window_id = window_id

    def capture(self, target: CaptureTarget) -> Frame:
        ts = datetime(2026, 8, 19, 1, 0, 0, tzinfo=timezone.utc)
        rect = WindowRect(left=0, top=0, width=self._width, height=self._height)
        return Frame(
            frame_seq=self._next_seq(),
            timestamp=ts,
            window_id=target.window_id or self._window_id,
            window_rect=rect,
            image=self._image,
            width=self._width,
            height=self._height,
        )


__all__ = ["FakeBackend"]
