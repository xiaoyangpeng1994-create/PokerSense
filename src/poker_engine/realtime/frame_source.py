"""FrameSource abstraction for the realtime pipeline.

The realtime layer consumes a *stream* of frames without knowing where they
come from. :class:`SyntheticFrameSource` replays a pre-built sequence of
frames deterministically, for tests and CI.

A live-capture source is just another implementation of the
:class:`FrameSource` protocol, pulling from a ``CaptureService`` backend
(``MssBackend`` on Windows, ``QuartzBackend`` on macOS).

Only the frame *lifecycle* is owned here: the frame's pixel data remains the
immutable ``Frame`` from the capture layer.
"""

from __future__ import annotations

from typing import Protocol

from poker_engine.perceptual.capture.base import Frame


class FrameSource(Protocol):
    """An ordered, pull-based source of frames."""

    def next_frame(self) -> Frame | None:
        """Return the next frame, or None when the stream is exhausted."""
        ...


class SyntheticFrameSource:
    """Replay a fixed sequence of frames (deterministic, test-friendly)."""

    def __init__(self, frames: tuple[Frame, ...]) -> None:
        if not isinstance(frames, tuple):
            raise TypeError("frames must be a tuple of Frame")
        for f in frames:
            if not isinstance(f, Frame):
                raise TypeError("each frame must be a Frame")
        self._frames = frames
        self._idx = 0

    def next_frame(self) -> Frame | None:
        if self._idx >= len(self._frames):
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame


__all__ = ["FrameSource", "SyntheticFrameSource"]
