import numpy as np

from tools.aa8_actor_ring_v2 import actor_ring_v2, union_suffix_components
from tools.aa8_continuous_state import suffix_components


def test_eight_connected_fragment_candidate_added_without_changing_source_reader():
    mask = np.zeros((20, 20), np.uint8)
    mask[3:7, 3:7] = 255
    mask[7:11, 7:11] = 255
    assert suffix_components(mask) == []
    assert len(union_suffix_components(mask)) == 1


def test_four_connected_avatar_bridge_solution_is_preserved():
    mask = np.zeros((40, 45), np.uint8)
    mask[5:16, 5:13] = 255
    mask[16, 13] = 255
    mask[17:35, 14:40] = 255
    old = suffix_components(mask)
    union = union_suffix_components(mask)
    assert len(old) == len(union) == 1
    np.testing.assert_array_equal(old[0], union[0])


def test_identical_four_eight_glyphs_are_deduplicated():
    mask = np.zeros((25, 25), np.uint8)
    mask[5:16, 5:13] = 255
    assert len(union_suffix_components(mask)) == 1


def test_large_avatar_blob_is_not_a_suffix_candidate():
    assert union_suffix_components(np.full((30, 30), 255, np.uint8)) == []


def test_missing_original_template_or_canvas_abstains():
    assert actor_ring_v2(None, {}, None)["actor"] is None
