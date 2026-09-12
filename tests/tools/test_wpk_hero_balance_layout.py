"""Location is not participation, and absent evidence never carries old cash."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.wpk_hero_balance_layout import (
    HeroBalanceLayout, LOCATIONS, choose_layout, pill_edges, read_hero_balance,
)
from tools.probe_wpk_hero_layout import review_points


def pill():
    image = np.zeros((28, 88, 3), np.uint8)
    cv2.ellipse(image, (44, 14), (39, 11), 0, 0, 360, (180, 180, 180), 1)
    return image


def frame_with(*locations):
    image = np.zeros((1080, 498, 3), np.uint8)
    for name in locations:
        x, y, w, h = LOCATIONS[name]
        image[y:y + h, x:x + w] = pill()
    return image


@pytest.mark.parametrize("locations,expected,status", [
    (("lower",), "lower", "CANDIDATE"), (("raised",), "raised", "CANDIDATE"),
    (("lower", "raised"), None, "CONFLICT"), ((), None, "UNKNOWN"),
])
def test_only_positive_unambiguous_location_is_selected(locations, expected, status):
    reader = HeroBalanceLayout({key: pill() for key in LOCATIONS})
    result = reader.recognize(frame_with(*locations))
    assert (result.location, result.status) == (expected, status)
    assert result.participation == "UNKNOWN"


def test_central_digits_do_not_choose_the_location():
    patch = pill()
    before = pill_edges(patch)
    patch[5:23, 20:68] = 255
    assert np.array_equal(before, pill_edges(patch))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.1])
def test_invalid_score_rejected(value):
    with pytest.raises(ValueError):
        choose_layout(value, .1)


def test_no_stale_value_after_loss_or_conflict():
    selector = HeroBalanceLayout({key: pill() for key in LOCATIONS})
    calls = []

    def recognize(patch):
        calls.append(patch.shape)
        return SimpleNamespace(value=SimpleNamespace(value=200), raw_score=.95)

    ocr = SimpleNamespace(recognize=recognize)
    cal = SimpleNamespace(should_abstain=lambda score: score < .8)
    result = read_hero_balance(frame_with("raised"), selector, ocr, cal)
    assert result["value"] == "200" and not result["usable_stack_verified"]
    for names in ((), ("lower", "raised")):
        result = read_hero_balance(frame_with(*names), selector, ocr, cal)
        assert result["value"] is None
    assert len(calls) == 1


def test_location_does_not_override_amount_gate():
    selector = HeroBalanceLayout({key: pill() for key in LOCATIONS})
    ocr = SimpleNamespace(recognize=lambda patch: SimpleNamespace(
        value=SimpleNamespace(value=200), raw_score=.4))
    result = read_hero_balance(frame_with("raised"), selector, ocr,
                               SimpleNamespace(should_abstain=lambda score: score < .8))
    assert result["location"] == "raised" and result["value"] is None


def test_wrong_canvas_and_degenerate_template_abstain():
    selector = HeroBalanceLayout({key: pill() for key in LOCATIONS})
    assert selector.recognize(np.zeros((1080, 500, 3), np.uint8)).location is None
    with pytest.raises(ValueError, match="positive"):
        HeroBalanceLayout({key: np.zeros((28, 88, 3), np.uint8) for key in LOCATIONS})


def test_interval_truth_keeps_real_frame_numbers_and_zero():
    result = review_points({"intervals": [
        {"start": 10530, "end": 10532, "location": "lower", "visible_digits": "0"}]})
    assert [p["source_frame"] for p in result] == [10530, 10531, 10532]
    assert all(p["visible_digits"] == "0" for p in result)


@pytest.mark.parametrize("review", [
    {"intervals": [{"start": 5, "end": 4}]},
    {"checkpoints": [{"source_frame": 1}, {"source_frame": 1}]},
])
def test_bad_or_overlapping_truth_is_not_double_counted(review):
    with pytest.raises(ValueError):
        review_points(review)
