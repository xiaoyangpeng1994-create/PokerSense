import pytest

from tools.aa8_glyph_transitions_v2 import GlyphTransitionsV2


def observe(tracker, values, start=0):
    events = []
    for frame, value in enumerate(values, start=start):
        events.extend(tracker.observe(frame, {"4": value}))
    return events


def test_persistent_badge_emits_once_and_stable_clear_rearms():
    tracker = GlyphTransitionsV2()
    events = observe(
        tracker, ["check"] * 2 + [None] * 5 + ["check"] * 2)
    assert [event["glyph"] for event in events] == ["check", "check"]


def test_unsupported_scene_preserves_confirmed_identity_without_duplicate():
    tracker = GlyphTransitionsV2()
    assert observe(tracker, ["check", "check"])
    for frame in range(2, 20):
        assert tracker.observe(frame, {"4": None}, suspended=True) == []
    assert observe(tracker, ["check", "check"], start=20) == []


def test_unconfirmed_streak_cannot_bridge_an_unsupported_scene():
    tracker = GlyphTransitionsV2()
    assert tracker.observe(0, {"4": "check"}) == []
    for frame in range(1, 20):
        assert tracker.observe(frame, {"4": None}, suspended=True) == []
    assert tracker.observe(20, {"4": "check"}) == []
    assert tracker.observe(21, {"4": "check"})[0]["glyph"] == "check"


def test_suppressed_action_does_not_reappear_after_short_clear():
    tracker = GlyphTransitionsV2()
    events = observe(
        tracker, ["call", "call", "fold", "fold", None, None,
                  "fold", "fold"])
    assert [event["glyph"] for event in events] == ["call"]


def test_stable_clear_rearms_a_suppressed_action():
    tracker = GlyphTransitionsV2()
    events = observe(
        tracker, ["call", "call", "fold", "fold"] + [None] * 5
        + ["fold", "fold"])
    assert [event["glyph"] for event in events] == ["call", "fold"]


def test_later_distinct_action_can_emit_without_clear():
    tracker = GlyphTransitionsV2(rapid_change_guard_frames=3)
    events = observe(
        tracker, ["check", "check"] + ["check"] * 6 + ["call", "call"])
    assert [event["glyph"] for event in events] == ["check", "call"]


def test_explicit_hand_epoch_allows_same_glyph_after_overlay():
    tracker = GlyphTransitionsV2()
    assert observe(tracker, ["check", "check"])
    tracker.observe(2, {"4": None}, suspended=True)
    tracker.begin_epoch()
    assert tracker.observe(3, {"4": "check"}) == []
    assert tracker.observe(4, {"4": "check"})[0]["glyph"] == "check"


def test_gap_and_arguments_fail_closed():
    tracker = GlyphTransitionsV2()
    tracker.observe(1, {"4": "fold"})
    assert tracker.observe(3, {"4": "fold"}) == []
    assert tracker.observe(4, {"4": "fold"})
    with pytest.raises(ValueError, match="backwards"):
        tracker.observe(4, {"4": "fold"})
    with pytest.raises(ValueError, match="suspended boolean"):
        GlyphTransitionsV2().observe(1, {"4": None}, suspended=1)
    with pytest.raises(ValueError, match="rapid change guard"):
        GlyphTransitionsV2(rapid_change_guard_frames=1)
