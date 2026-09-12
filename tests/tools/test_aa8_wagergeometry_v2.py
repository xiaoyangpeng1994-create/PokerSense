import numpy as np
import pytest

from tools.aa8_wagergeometry_v2 import AA8WagerReaderV2, WAGERS_V2, digit_bounds


def test_far_below_pill_edge_does_not_inflate_digit_height():
    patch = np.zeros((23, 57), np.uint8)
    patch[4:16, 44:52] = 255
    patch[19, 50:54] = 255
    assert digit_bounds(patch) == (44, 4, 52, 16)


def test_decimal_punctuation_near_baseline_is_not_silently_erased():
    patch = np.zeros((23, 57), np.uint8)
    patch[4:16, 30:38] = 255
    patch[14:16, 42:44] = 255
    assert digit_bounds(patch) is None


def test_hero_roi_contains_whole_source_coin_not_gray_button():
    x, y, w, h = WAGERS_V2[4]
    assert x <= 298 and x + w >= 315
    assert y <= 814 and y + h > 830
    assert y + h <= 833


@pytest.mark.parametrize("patch,slot", [
    (None, 0), (np.zeros((3, 3)), 0), (np.zeros((24, 78, 3), np.uint8), 8)])
def test_invalid_roi_rejected_before_matching(patch, slot):
    reader = AA8WagerReaderV2.__new__(AA8WagerReaderV2)
    assert reader.read_label(patch, slot)["reason"] == "unsupported_wager_patch"


def test_blank_mask_does_not_invent_digit():
    assert digit_bounds(np.zeros((23, 57), np.uint8)) is None


def test_fixed_smoothing_does_not_make_flat_or_digit_only_patch_a_coin():
    import cv2
    reader = AA8WagerReaderV2.__new__(AA8WagerReaderV2)
    reader.coin = np.random.default_rng(3).integers(0, 255, (17, 17), dtype=np.uint8)
    reader.coin_smoothing = True
    for text in ("", "2"):
        patch = np.full((33, 85, 3), (20, 100, 20), np.uint8)
        cv2.putText(patch, text, (40, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255))
        result = reader.read_label(patch, 6)
        assert result["value"] is None
        assert result["coin_unique"] is False
