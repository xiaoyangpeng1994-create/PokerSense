"""Scene support uses positive geometry, not absence of cards or buttons."""

import cv2
import numpy as np
import pytest

from poker_engine.perceptual.vision.supported_scene import GreenTableSupport


def reference():
    image = np.zeros((1080, 498, 3), np.uint8)
    image[:, :, 1] = 100
    cv2.rectangle(image, (190, 285), (308, 306), (0, 30, 0), 2)
    return image


def test_matching_green_geometry_supports_only_candidate():
    image = reference()
    assert GreenTableSupport(image).recognize(image).supported


@pytest.mark.parametrize("change", [
    "blank", "wrong_canvas", "different_theme", "occluded",
])
def test_missing_positive_evidence_rejects(change):
    image = reference()
    guard = GreenTableSupport(image)
    if change == "blank":
        image[:] = 0
    elif change == "wrong_canvas":
        image = image[:, :400]
    elif change == "different_theme":
        image[:, :, 2] = 150
        image[:, :, 0] = 150
    else:
        image[275:315, 180:320] = 0
    assert not guard.recognize(image).supported
