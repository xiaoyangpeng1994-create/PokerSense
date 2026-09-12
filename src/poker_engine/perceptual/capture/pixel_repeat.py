"""Pixel repetition evidence, deliberately NOT a device-freeze detector.

An unchanged live table and a stale device buffer can produce identical pixels
with new host timestamps. Without an independent source signal, freshness is
unknown in both cases. This observer must never invent that missing evidence.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class PixelRepeatEvidence:
    frame_seq: int
    consecutive_identical_frames: int
    unchanged_host_seconds: float
    pixels_changed: bool | None
    source_freshness: str = "UNKNOWN"


class PixelRepeatObserver:
    def __init__(self):
        self.reset()

    def reset(self):
        self._digest = None
        self._first_seen = None
        self._last_seen = None
        self._count = 0
        self.evidence = None

    def observe(self, image: np.ndarray, frame_seq: int, observed_at: float):
        if not math.isfinite(observed_at):
            raise ValueError("host observation time must be finite")
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str((image.shape, image.dtype.str)).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(image)).cast("B"))
        current = digest.digest()
        changed = None if self._digest is None else current != self._digest
        if (self._digest is None or changed or observed_at < self._last_seen):
            self._count = 1
            self._first_seen = observed_at
        else:
            self._count += 1
        self._digest, self._last_seen = current, observed_at
        self.evidence = PixelRepeatEvidence(
            frame_seq, self._count, observed_at - self._first_seen, changed)
        return self.evidence
