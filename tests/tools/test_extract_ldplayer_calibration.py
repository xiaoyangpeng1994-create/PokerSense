from __future__ import annotations

import numpy as np
import pytest

from tools.extract_ldplayer_calibration import (
    ExtractionSettings,
    difference_hash,
    hamming_distance,
    normalize_frame,
    should_keep,
)


def test_normalize_frame_removes_only_host_toolbar():
    settings = ExtractionSettings(
        host_toolbar_pixels=2, expected_width=4, expected_height=3
    )
    frame = np.zeros((5, 4, 3), dtype=np.uint8)
    frame[:2] = 255
    frame[2:] = 17

    canvas = normalize_frame(frame, settings)

    assert canvas.shape == (3, 4, 3)
    assert np.all(canvas == 17)


def test_normalize_frame_rejects_unexpected_canvas_size():
    with pytest.raises(ValueError, match="expected"):
        normalize_frame(
            np.zeros((4, 4, 3), dtype=np.uint8),
            ExtractionSettings(host_toolbar_pixels=0),
        )


def test_difference_hash_and_dedup_selection_are_deterministic():
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = first.copy()
    second[:, 16:] = 255
    first_hash = difference_hash(first)
    second_hash = difference_hash(second)

    assert difference_hash(first) == first_hash
    assert hamming_distance(first_hash, first_hash) == 0
    assert hamming_distance(first_hash, second_hash) > 0
    assert not should_keep(first_hash, [first_hash], 0)
    assert should_keep(second_hash, [first_hash], 0)


def test_should_keep_rejects_negative_distance():
    with pytest.raises(ValueError, match="non-negative"):
        should_keep(0, [], -1)
