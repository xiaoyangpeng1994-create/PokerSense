"""Photometric experiment integrity, not model accuracy tests."""

import numpy as np
import pytest

from tools.aa8_card_preflight import VARIANTS, transform_card


@pytest.mark.parametrize("variant", VARIANTS)
def test_transform_keeps_input_and_geometry(variant):
    crop = np.random.default_rng(1).integers(0, 256, (78, 53, 3), dtype=np.uint8)
    original = crop.copy()
    transformed = transform_card(crop, variant)
    assert transformed.shape == crop.shape
    assert transformed.dtype == np.uint8
    np.testing.assert_array_equal(crop, original)
    assert not np.shares_memory(crop, transformed)


@pytest.mark.parametrize("crop", [
    None, np.zeros((78, 53)), np.zeros((78, 53, 3), np.float32)])
def test_invalid_crop_rejected(crop):
    with pytest.raises(ValueError, match="valid card"):
        transform_card(crop, "gaussian_050")


def test_blank_stays_blank_no_generated_evidence():
    crop = np.zeros((78, 53, 3), np.uint8)
    np.testing.assert_array_equal(transform_card(crop, "gaussian_050"), crop)


def test_no_per_image_score_selected_variant():
    with pytest.raises(ValueError, match="unregistered"):
        transform_card(np.zeros((78, 53, 3), np.uint8), "best_score")
