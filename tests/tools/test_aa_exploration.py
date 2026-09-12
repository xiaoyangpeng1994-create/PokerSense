import numpy as np

from tools.aa_exploration import bright_bounds


def test_bounds_half_open():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, 3:15] = 100
    assert bright_bounds(image) == [3, 15]


def test_dark_frame_has_no_candidate():
    assert bright_bounds(np.zeros((10, 20, 3), dtype=np.uint8)) is None
