"""Tests for stage C content-boundary drift measurement (guide section 6)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.capture_card_calibration.boundary import (
    EDGE_BAND_PX,
    MAX_BOUNDARY_DRIFT_PX,
    ContentBounds,
    content_bounds,
    edge_content_flags,
    merge_edge_flags,
    summarize_drift,
)


def _frame(height: int = 40, width: int = 30, fill: int = 0) -> np.ndarray:
    return np.full((height, width), fill, dtype=np.uint8)


def test_content_bounds_of_blank_frame_is_none() -> None:
    assert content_bounds(_frame(fill=0)) is None


def test_content_bounds_of_solid_block() -> None:
    frame = _frame(fill=0)
    # 20 of 40 rows and 20 of 30 columns lit: both orientations clear 50%.
    frame[10:30, 5:25] = 200
    bounds = content_bounds(frame)
    assert bounds == ContentBounds(left=5, right=24, top=10, bottom=29)


def test_content_bounds_ignores_single_pixel_noise() -> None:
    """A lone bright pixel must not pin the boundary to the frame edge.

    A naive ``any()`` implementation passes this frame as full-bleed, which is
    exactly how real drift hides.
    """
    frame = _frame(fill=0)
    frame[10:30, 5:25] = 200
    frame[0, 0] = 255
    frame[-1, -1] = 255
    bounds = content_bounds(frame)
    assert bounds == ContentBounds(left=5, right=24, top=10, bottom=29)


def test_content_bounds_requires_majority_in_both_orientations() -> None:
    """A band that is wide but short is not a canvas edge.

    Content is only credited when a row *and* a column clear ``fraction``, so
    an isolated bright band cannot masquerade as a full-bleed canvas.
    """
    frame = _frame(fill=0)
    frame[10:12, :] = 200  # full width, but only 2 of 40 rows
    assert content_bounds(frame, fraction=0.5) is None


def test_content_bounds_respects_fraction_argument() -> None:
    frame = _frame(fill=0)
    frame[10:30, 5:25] = 200  # rows 66.7% lit, columns 50% lit
    assert content_bounds(frame, fraction=0.4) is not None
    assert content_bounds(frame, fraction=0.7) is None


def test_content_bounds_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        content_bounds(_frame(), fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        content_bounds(_frame(), fraction=1.5)


def test_content_bounds_accepts_colour_frames() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[5:15, 5:15] = (200, 200, 200)
    assert content_bounds(frame) == ContentBounds(5, 14, 5, 14)


def test_content_bounds_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="2-D grayscale"):
        content_bounds(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_edge_flags_detect_residual_border() -> None:
    """A fixed black band on the left is reported as an edge without content."""
    frame = _frame(fill=200)
    frame[:, :EDGE_BAND_PX] = 0
    flags = edge_content_flags(frame)
    assert flags.left is False
    assert flags.right is True
    assert flags.top is True
    assert flags.bottom is True


def test_edge_flags_all_lit_on_full_bleed_frame() -> None:
    flags = edge_content_flags(_frame(fill=200))
    assert flags.as_dict() == {
        "left": True,
        "right": True,
        "top": True,
        "bottom": True,
    }


def test_edge_flags_blank_frame_has_no_edge_content() -> None:
    flags = edge_content_flags(_frame(fill=0))
    assert flags.as_dict() == {
        "left": False,
        "right": False,
        "top": False,
        "bottom": False,
    }


def test_edge_flags_reject_bad_band() -> None:
    with pytest.raises(ValueError, match="band"):
        edge_content_flags(_frame(), band=0)
    with pytest.raises(ValueError, match="smaller than"):
        edge_content_flags(_frame(height=4, width=4), band=8)


def test_merge_edge_flags_ors_across_frames() -> None:
    """One frame with content at an edge proves that edge is canvas."""
    blank = edge_content_flags(_frame(fill=0))
    lit = edge_content_flags(_frame(fill=200))
    merged = merge_edge_flags([blank, blank, lit])
    assert merged.as_dict() == {
        "left": True,
        "right": True,
        "top": True,
        "bottom": True,
    }


def test_summarize_drift_reports_zero_for_identical_bounds() -> None:
    items = [ContentBounds(0, 497, 0, 1079) for _ in range(5)]
    summary = summarize_drift(items, scene="table", stable=True)
    assert summary is not None
    assert summary.frame_count == 5
    assert summary.worst_drift == 0
    assert summary.within_tolerance is True
    assert summary.drift_by_edge() == {
        "left": 0,
        "right": 0,
        "top": 0,
        "bottom": 0,
    }


def test_summarize_drift_detects_moving_edge() -> None:
    items = [
        ContentBounds(0, 497, 0, 1079),
        ContentBounds(0, 497, 0, 1074),
        ContentBounds(2, 497, 0, 1079),
    ]
    summary = summarize_drift(items)
    assert summary is not None
    assert summary.worst_drift == 5
    assert summary.left == (0, 2)
    assert summary.bottom == (1074, 1079)
    assert summary.within_tolerance is False


def test_summarize_drift_of_empty_group_is_none() -> None:
    """No frames measured is not the same as zero drift."""
    assert summarize_drift([]) is None


def test_boundary_at_tolerance_boundary_is_accepted() -> None:
    items = [
        ContentBounds(0, 497, 0, 1079),
        ContentBounds(0, 497, 0, 1079 - MAX_BOUNDARY_DRIFT_PX),
    ]
    summary = summarize_drift(items)
    assert summary is not None
    assert summary.worst_drift == MAX_BOUNDARY_DRIFT_PX
    assert summary.within_tolerance is True
