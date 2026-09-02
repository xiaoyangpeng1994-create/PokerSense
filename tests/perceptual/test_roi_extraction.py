"""Tests for deterministic ROI extraction and layout compatibility."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from poker_engine.perceptual import (
    Frame,
    ROIKind,
    ROI,
    TableMap,
    TableMapMismatchError,
    WindowRect,
    check_layout_compatibility,
    extract_all,
    extract_roi,
    roi_pixel_bounds,
)

UTC = timezone.utc


def _frame(w=1920, h=1080):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rect = WindowRect(left=0, top=0, width=w, height=h)
    return Frame(
        frame_seq=0,
        timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        window_id="t",
        window_rect=rect,
        image=img,
        width=w,
        height=h,
    )


def _map(aspect_tolerance=0.02):
    return TableMap(
        platform_id="p",
        layout_id="l",
        reference_size=(1920, 1080),
        aspect_tolerance=aspect_tolerance,
        rois=(ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.25, height=0.10),),
    )


def test_roi_pixel_bounds_floor():
    frame = _frame()
    roi = ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.25, height=0.10)
    x0, y0, x1, y1 = roi_pixel_bounds(roi, frame)
    assert x0 == int(0.5 * 1920)
    assert y0 == int(0.5 * 1080)
    assert x1 == int(0.75 * 1920)
    assert y1 == int(0.60 * 1080)


def test_extract_deterministic():
    frame = _frame()
    roi = ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.25, height=0.10)
    c1 = extract_roi(frame, roi)
    c2 = extract_roi(frame, roi)
    assert np.array_equal(c1, c2)
    assert c1.shape == (int(0.10 * 1080), int(0.25 * 1920), 3)


def test_extract_uses_actual_size():
    # a frame smaller than reference still crops by actual dims
    frame = _frame(w=960, h=540)
    roi = ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.25, height=0.10)
    crop = extract_roi(frame, roi)
    assert crop.shape == (int(0.10 * 540), int(0.25 * 960), 3)


def test_layout_compatible_ok():
    frame = _frame()
    m = _map()
    check_layout_compatibility(m, frame)  # no raise


def test_layout_mismatch_fails():
    frame = _frame(w=1000, h=1080)  # aspect 1000/1080 vs 1920/1080
    m = _map(aspect_tolerance=0.001)
    with pytest.raises(TableMapMismatchError):
        check_layout_compatibility(m, frame)


def test_extract_all_per_seat_keys():
    frame = _frame()
    m = TableMap(
        platform_id="p",
        layout_id="l",
        reference_size=(1920, 1080),
        rois=(
            ROI(kind=ROIKind.STACK, x=0.1, y=0.8, width=0.08, height=0.04, slot_id=0),
            ROI(kind=ROIKind.STACK, x=0.8, y=0.8, width=0.08, height=0.04, slot_id=1),
            ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.25, height=0.10),
        ),
    )
    crops = extract_all(m, frame)
    assert set(crops.keys()) == {"stack:0", "stack:1", "pot"}


def test_extract_roi_exceeds_bounds():
    frame = _frame(w=100, h=100)
    roi = ROI(kind=ROIKind.POT, x=0.9, y=0.9, width=0.5, height=0.5)
    with pytest.raises(TableMapMismatchError):
        extract_roi(frame, roi)
