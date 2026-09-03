"""Stage C content-boundary drift measurement (guide section 6).

Section 6 requires sampling frames from the beginning, middle and end of each
session (and after any reconnect), measuring the game content boundary, and
rejecting the capture chain when the drift under one ``layout_id`` exceeds two
output pixels.

Two distinct things are easy to confuse here, so this module measures both and
keeps them apart:

- **Canvas geometry** — whether the UVC letterbox/pillarbox frame was actually
  cropped away. A residual capture border is *physically fixed*: it darkens the
  same side on *every* frame. :func:`edge_content_flags` tests each edge
  directly, so a border that is present on all frames is detectable no matter
  what the game happens to be drawing.
- **Content luminance** — the game UI itself may legitimately render a dark
  band (a menu backdrop, a full-screen effect). :func:`content_bounds` will
  report that as an inset boundary, which is a fact about the *picture*, not
  about the *geometry*, and must not be scored as normalisation drift.

Reporting the raw inset of every frame as "drift" would fail a perfectly good
capture chain on the basis of a menu screen. The two measurements are therefore
reported separately and only the table frames decide pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

#: Luminance below which a pixel counts as background/black.
LUM_THRESHOLD = 16
#: Share of a row/column that must be lit for it to count as content.
CONTENT_FRACTION = 0.5
#: Section 6 tolerance for boundary drift, in output pixels.
MAX_BOUNDARY_DRIFT_PX = 2
#: Width of the edge band probed for residual capture border, in pixels.
EDGE_BAND_PX = 2

EDGES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class ContentBounds:
    """Inclusive content bounds of one frame, in output pixels."""

    left: int
    right: int
    top: int
    bottom: int


@dataclass(frozen=True)
class EdgeFlags:
    """Per-edge evidence that the canvas reaches the frame border.

    A ``True`` flag means the band on that edge contained lit pixels, so no
    residual capture border is covering it.
    """

    left: bool
    right: bool
    top: bool
    bottom: bool

    def as_dict(self) -> dict[str, bool]:
        return {edge: getattr(self, edge) for edge in EDGES}


@dataclass(frozen=True)
class DriftSummary:
    """Drift of one measured group of frames."""

    scene: str
    stable: bool
    frame_count: int
    left: tuple[int, int]
    right: tuple[int, int]
    top: tuple[int, int]
    bottom: tuple[int, int]

    @property
    def worst_drift(self) -> int:
        edges = (self.left, self.right, self.top, self.bottom)
        return max(high - low for low, high in edges)

    @property
    def within_tolerance(self) -> bool:
        return self.worst_drift <= MAX_BOUNDARY_DRIFT_PX

    def drift_by_edge(self) -> dict[str, int]:
        return {
            "left": self.left[1] - self.left[0],
            "right": self.right[1] - self.right[0],
            "top": self.top[1] - self.top[0],
            "bottom": self.bottom[1] - self.bottom[0],
        }


def load_gray(path: str | Path) -> np.ndarray:
    """Decode an image file to grayscale.

    Decoding through a byte buffer avoids OpenCV's inability to open non-ASCII
    paths on Windows, which matters for private datasets kept under a
    non-English project directory.
    """
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _as_gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError(
            f"frame must be 2-D grayscale or 3-D colour, got {array.ndim}-D"
        )
    return array


def content_bounds(
    frame: np.ndarray,
    *,
    lum_threshold: int = LUM_THRESHOLD,
    fraction: float = CONTENT_FRACTION,
) -> ContentBounds | None:
    """Measure the lit content bounds of ``frame``.

    A row or column counts as content only when at least ``fraction`` of it is
    above ``lum_threshold``; testing bare ``any()`` lets a single compression
    artefact pin the boundary to the frame edge and hide real drift.

    Returns ``None`` when no row/column reaches the threshold (a blank frame).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    gray = _as_gray(frame)
    mask = gray > lum_threshold
    rows = np.where(mask.mean(axis=1) >= fraction)[0]
    cols = np.where(mask.mean(axis=0) >= fraction)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return ContentBounds(
        left=int(cols[0]),
        right=int(cols[-1]),
        top=int(rows[0]),
        bottom=int(rows[-1]),
    )


def edge_content_flags(
    frame: np.ndarray,
    *,
    band: int = EDGE_BAND_PX,
    lum_threshold: int = LUM_THRESHOLD,
    fraction: float = CONTENT_FRACTION,
) -> EdgeFlags:
    """Report which edges contain lit pixels within ``band`` pixels.

    This is the direct test for a residual capture border: such a border is
    fixed hardware framing, so it darkens an edge on *every* frame. If any
    frame shows content at an edge, that edge is canvas, not border.
    """
    if band < 1:
        raise ValueError(f"band must be >= 1, got {band!r}")
    gray = _as_gray(frame)
    if min(gray.shape) <= band:
        raise ValueError("frame is smaller than the requested edge band")
    mask = gray > lum_threshold

    def lit(region: np.ndarray) -> bool:
        return bool(region.mean() >= fraction)

    return EdgeFlags(
        left=lit(mask[:, :band]),
        right=lit(mask[:, -band:]),
        top=lit(mask[:band, :]),
        bottom=lit(mask[-band:, :]),
    )


def summarize_drift(
    bounds: Sequence[ContentBounds],
    *,
    scene: str = "",
    stable: bool = True,
) -> DriftSummary | None:
    """Collapse a group of bounds into min/max per edge.

    Returns ``None`` for an empty group rather than inventing a zero-drift
    result for frames that were never measured.
    """
    if not bounds:
        return None
    values = {
        edge: [int(getattr(item, edge)) for item in bounds] for edge in EDGES
    }
    return DriftSummary(
        scene=scene,
        stable=stable,
        frame_count=len(bounds),
        left=(min(values["left"]), max(values["left"])),
        right=(min(values["right"]), max(values["right"])),
        top=(min(values["top"]), max(values["top"])),
        bottom=(min(values["bottom"]), max(values["bottom"])),
    )


def merge_edge_flags(flags: Iterable[EdgeFlags]) -> EdgeFlags:
    """OR the per-frame edge flags into one verdict per edge."""
    merged = {edge: False for edge in EDGES}
    for item in flags:
        for edge in EDGES:
            merged[edge] = merged[edge] or bool(getattr(item, edge))
    return EdgeFlags(**merged)


__all__ = [
    "CONTENT_FRACTION",
    "EDGE_BAND_PX",
    "EDGES",
    "LUM_THRESHOLD",
    "MAX_BOUNDARY_DRIFT_PX",
    "ContentBounds",
    "DriftSummary",
    "EdgeFlags",
    "content_bounds",
    "edge_content_flags",
    "load_gray",
    "merge_edge_flags",
    "summarize_drift",
]
