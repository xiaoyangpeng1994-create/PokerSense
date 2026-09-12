import numpy as np
import pytest

from tools.aa8_special_modes import REGIONS
from tools.aa8_special_spatial import AA8SpatialModeReader, BUYIN_TITLE_BOX


def witness():
    return np.random.default_rng(42).integers(
        0, 256, (1080, 498, 3), dtype=np.uint8)


def test_unobserved_seat_is_not_claimed():
    image = witness()
    with pytest.raises(ValueError, match="reviewed"):
        AA8SpatialModeReader({k: image for k in REGIONS}, countdown_images={0: image})


def test_optional_extension_keeps_default_api():
    image = witness()
    reader = AA8SpatialModeReader({k: image for k in REGIONS})
    result = reader.recognize(image)
    assert result["spatial_validated_slots"] == [3]
    assert result["countdown_seconds"] is None
    assert result["blocking_overlay"] == "UNKNOWN"


def test_overlay_move_blocks_even_matching_table():
    image = witness()
    reader = AA8SpatialModeReader({k: image for k in REGIONS},
                                  countdown_images={5: image, 7: image},
                                  buyin_overlay_image=image)
    moved = image.copy()
    x1, y1, x2, y2 = BUYIN_TITLE_BOX
    moved[y1 - 40:y2 - 40, x1:x2] = image[y1:y2, x1:x2]
    moved[y1:y2, x1:x2] = 0
    result = reader.recognize(moved)
    assert result["block_state_updates"]
    assert result["insurance"] == "UNKNOWN"
    assert result["insurance_countdown_slots"] == []


def test_seconds_change_cannot_change_prefix_result():
    image = witness()
    reader = AA8SpatialModeReader({k: image for k in REGIONS},
                                  countdown_images={5: image, 7: image})
    changed = image.copy()
    changed[572:590, 44:80] = 0
    result = reader.recognize(changed)
    assert 5 in result["insurance_countdown_slots"]
    assert result["countdown_seconds"] is None
