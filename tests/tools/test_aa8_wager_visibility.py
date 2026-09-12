import cv2
import numpy as np
import pytest

from tools.aa8_wager_visibility import patch_visibility


def coin():
    return np.random.default_rng(17).integers(0, 255, (17, 17), dtype=np.uint8)


def felt():
    hsv = np.full((33, 85, 3), (70, 220, 120), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_positive_empty_never_means_zero():
    result = patch_visibility(felt(), coin(), scene_supported=True, unobstructed=True)
    assert result["status"] == "VISIBLE_EMPTY_CANDIDATE"
    assert result["value"] is None
    assert not result["street_wager_zero_verified"]
    assert not result["strategy_eligible"]


@pytest.mark.parametrize("pixel", [(255, 255, 255), (0, 0, 0), (80, 80, 80)])
def test_even_one_ambiguous_pixel_abstains(pixel):
    patch = felt()
    patch[10, 20] = pixel
    result = patch_visibility(patch, coin(), scene_supported=True, unobstructed=True)
    assert result["status"] == "UNKNOWN"


@pytest.mark.parametrize("scene,unobstructed", [
    (False, True), (True, False), (None, True), (True, None)])
def test_scene_and_occlusion_need_explicit_evidence(scene, unobstructed):
    result = patch_visibility(felt(), coin(), scene_supported=scene,
                              unobstructed=unobstructed)
    assert result["status"] == "UNKNOWN"


def test_coin_without_amount_never_becomes_empty():
    patch = felt()
    template = coin()
    patch[5:22, 5:22] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    result = patch_visibility(patch, template, scene_supported=True, unobstructed=True)
    assert result["status"] == "VISIBLE_COIN_CANDIDATE"
    assert result["value"] is None


def test_invalid_patch_unknown():
    assert patch_visibility(None, coin())["status"] == "UNKNOWN"
