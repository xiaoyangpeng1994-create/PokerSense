from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.aa_visual_candidate import (
    AASceneCandidate, render_geometry, table_map, validate_layout,
)


@pytest.fixture
def profile():
    path = Path(__file__).resolve().parents[2] / (
        "configs/vision/aa_android_capture_card/layout.candidate.json")
    return json.loads(path.read_text())


def test_nine_distinct_stacks_and_hero(profile):
    mapping = table_map(profile)
    stacks = [r.slot_id for r in mapping.rois if r.kind.value == "stack"]
    assert stacks == list(range(9))
    assert profile["hero_slot"] == 5
    assert mapping.platform_id == "aa_android_capture_card"


@pytest.mark.parametrize("bad", [8, 10])
def test_wrong_slot_count(profile, bad):
    profile["slots"] = (profile["slots"] + [deepcopy(profile["slots"][0])])[:bad]
    with pytest.raises(ValueError, match="nine"):
        validate_layout(profile)


def test_no_geometry_clipping(profile):
    profile["slots"][0]["stack"][0] = 497
    with pytest.raises(ValueError, match="outside"):
        validate_layout(profile)


def test_wrong_platform(profile):
    profile["platform_id"] = "wepoker_android_capture_card"
    with pytest.raises(ValueError, match="AA"):
        validate_layout(profile)


def patterned_canvas():
    image = np.full((1080, 498, 3), (50, 140, 20), dtype=np.uint8)
    image[345:413:2, 211:287:2] = 200
    return image


def test_scene_never_infers_participation_or_special_modes(profile):
    image = patterned_canvas()
    candidate = AASceneCandidate(image, profile)
    result = candidate.recognize(image)
    assert result["scene"] == "AA_TABLE_CANDIDATE"
    assert result["participation"] == "UNKNOWN"
    assert result["critical_hit_triggered"] == "UNKNOWN"
    assert result["squid_enabled"] == "UNKNOWN"
    assert not result["strategy_eligible"]


@pytest.mark.parametrize("image", [None, np.zeros((10, 20, 3), np.uint8),
                                   np.zeros((1080, 498, 3), np.float32)])
def test_wrong_canvas_rejects(profile, image):
    candidate = AASceneCandidate(patterned_canvas(), profile)
    assert candidate.recognize(image)["reason"] == "unsupported_canvas"


def test_no_state_carry_after_valid_frame(profile):
    image = patterned_canvas()
    candidate = AASceneCandidate(image, profile)
    assert candidate.recognize(image)["scene"] == "AA_TABLE_CANDIDATE"
    assert candidate.recognize(np.zeros_like(image))["scene"] == "UNKNOWN"
    assert candidate.recognize(image)["scene"] == "AA_TABLE_CANDIDATE"


def test_flat_reference_and_invalid_floor_reject(profile):
    with pytest.raises(ValueError, match="non-flat"):
        AASceneCandidate(np.zeros((1080, 498, 3), np.uint8), profile)
    with pytest.raises(ValueError, match="floor"):
        AASceneCandidate(patterned_canvas(), profile, floor=float("nan"))


def test_render_does_not_mutate(profile):
    image = patterned_canvas()
    before = image.copy()
    assert not np.array_equal(render_geometry(image, profile), before)
    assert np.array_equal(image, before)
