import numpy as np
import pytest

from tools.aa8_action_reader import (
    GlyphTransitions, badge_text, compare, mask, normalize_glyph, similarity,
)


def test_persistent_badge_emits_once_and_clear_rearms():
    tracker = GlyphTransitions(clear_frames=2)
    assert tracker.observe(1, {"4": "check"}) == []
    assert tracker.observe(2, {"4": "check"})[0]["glyph"] == "check"
    assert tracker.observe(3, {"4": "check"}) == []
    assert tracker.observe(4, {"4": None}) == []
    assert tracker.observe(5, {"4": "check"}) == []
    assert tracker.observe(6, {"4": "check"}) == []
    tracker.observe(7, {"4": None})
    tracker.observe(8, {"4": None})
    tracker.observe(9, {"4": "check"})
    assert len(tracker.observe(10, {"4": "check"})) == 1


def test_gap_does_not_confirm_previous_frame():
    tracker = GlyphTransitions()
    tracker.observe(1, {"4": "fold"})
    assert tracker.observe(3, {"4": "fold"}) == []
    assert len(tracker.observe(4, {"4": "fold"})) == 1
    with pytest.raises(ValueError):
        tracker.observe(4, {"4": "fold"})


def test_three_frame_animation_dropout_does_not_duplicate_all_in():
    tracker = GlyphTransitions()
    values = ["all_in", "all_in", None, None, None, "all_in", "all_in"]
    events = []
    for frame, value in enumerate(values):
        events.extend(tracker.observe(frame, {"1": value}))
    assert len(events) == 1


def test_sustained_absence_rearms_with_default_threshold():
    tracker = GlyphTransitions()
    values = ["check"] * 2 + [None] * 5 + ["check"] * 2
    events = []
    for frame, value in enumerate(values):
        events.extend(tracker.observe(frame, {"1": value}))
    assert len(events) == 2


def test_unknown_never_becomes_fold():
    tracker = GlyphTransitions()
    for frame in range(10):
        assert tracker.observe(frame, {"4": None}) == []


def test_hero_control_outside_completed_badge_roi():
    image = np.zeros((1080, 498, 3), np.uint8)
    image[839:909, 213:285] = (255, 150, 0)
    assert not mask(image, [213, 839, 72, 70], "call").any()
    assert not mask(image, [213, 839, 72, 70], "aggressive").any()


def test_flat_masks_do_not_match():
    empty = np.zeros((21, 68), np.uint8)
    assert similarity(empty, empty) == 0


def test_orange_pill_alone_is_not_text_evidence():
    image = np.zeros((1080, 498, 3), np.uint8)
    image[572:593, 430:490] = (0, 140, 255)
    avatar = [423, 611, 74, 68]
    assert not badge_text(image, avatar).any()
    image[578:585, 449:458] = (255, 255, 255)
    assert badge_text(image, avatar).any()


def test_badge_alignment_ignores_position_and_small_background_island():
    first = np.zeros((30, 80), np.uint8)
    first[3:23, 4:68] = 255
    first[7:18, 25:30] = 0
    second = np.zeros_like(first)
    second[6:26, 8:72] = first[3:23, 4:68]
    second[1, 1] = 255
    assert np.array_equal(normalize_glyph(first, "call"),
                          normalize_glyph(second, "call"))


def test_all_in_border_animation_is_excluded():
    value = np.zeros((28, 64), np.uint8)
    value[0:28, :3] = 255
    value[8:19, 20:25] = 255
    value[8:19, 35:40] = 255
    clean = value.copy()
    clean[:, :3] = 0
    assert np.array_equal(normalize_glyph(value, "all_in"),
                          normalize_glyph(clean, "all_in"))


def test_border_only_animation_is_unknown():
    value = np.zeros((28, 64), np.uint8)
    value[:, :4] = 255
    assert not normalize_glyph(value, "all_in").any()


def test_comparison_does_not_hide_duplicates_or_force_match():
    reference = {"streets": [{"street": "river", "actions": [
        {"kind": "call", "all_in": True, "slot": 4, "window": [10, 12]}]}]}
    events = [{"frame": 13, "slot": 4, "glyph": "all_in"},
              {"frame": 14, "slot": 4, "glyph": "all_in"},
              {"frame": 15, "slot": 6, "glyph": "fold"}]
    result = compare(events, reference)
    assert len(result["matched"]) == 1
    assert len(result["unmatched_proposals"]) == 2
    assert len(compare(events, reference, latency=0)["missed"]) == 1
