import numpy as np
import pytest

from tools.aa_card_visibility import face_card_support


def canvas():
    return np.full((1080, 498, 3), (50, 140, 20), dtype=np.uint8)


@pytest.mark.parametrize("brightness", [95, 240])
def test_normal_and_dim_faces_have_positive_support(brightness):
    image = canvas()
    image[473:551, 110:163] = brightness
    assert face_card_support(image, (110, 473, 53, 78))[0]


@pytest.mark.parametrize("colour", [(40, 40, 140), (180, 65, 25)])
def test_coloured_backs_are_not_faces(colour):
    image = canvas()
    image[473:551, 110:163] = colour
    assert not face_card_support(image, (110, 473, 53, 78))[0]


def test_raised_face_rejects_clipped_rank():
    image = canvas()
    image[459:537, 110:163] = 240
    assert face_card_support(image, (110, 473, 53, 78))[1] == (
        "face_extends_above_fixed_crop")


def test_invalid_input_fails_closed():
    assert not face_card_support(None, (110, 473, 53, 78))[0]
    assert not face_card_support(canvas(), (490, 473, 53, 78))[0]
