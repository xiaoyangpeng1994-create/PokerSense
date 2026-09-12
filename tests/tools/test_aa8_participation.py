import pytest
import numpy as np

from tools.aa8_participation import RosterCandidates, waiting_yellow


def cues(slot, cue):
    return {str(slot): {"cue": cue}}


def test_missing_buttons_or_cards_never_mean_observer_or_fold():
    tracker = RosterCandidates()
    tracker.begin_hand("h")
    for frame in (0, 1, 2):
        result = tracker.observe(frame, {})
    assert all(row["current"] == "UNKNOWN" for row in result["slots"].values())
    assert all(row["in_this_hand"] is None for row in result["slots"].values())


def test_waiting_newcomer_is_not_dealt_in():
    tracker = RosterCandidates()
    tracker.begin_hand("h")
    tracker.observe(0, cues(6, "WAITING_POST_OR_PASS"))
    result = tracker.observe(1, cues(6, "WAITING_POST_OR_PASS"))
    assert result["slots"]["6"]["current"] == "WAITING_CANDIDATE"
    assert result["slots"]["6"]["in_this_hand"] is None


def test_waiting_does_not_erase_dealt_roster():
    tracker = RosterCandidates(1)
    tracker.begin_hand("h")
    tracker.observe(0, cues(6, "BACK_CARDS"))
    result = tracker.observe(1, cues(6, "WAITING_POST_OR_PASS"))
    assert result["slots"]["6"]["conflict"]
    assert result["slots"]["6"]["history"] == "DEALT_IN_CANDIDATE"


def test_next_hand_can_include_previous_waiting_newcomer():
    tracker = RosterCandidates(1)
    tracker.begin_hand("first")
    tracker.observe(0, cues(6, "WAITING_POST_OR_PASS"))
    tracker.begin_hand("second")
    result = tracker.observe(1, cues(6, "BACK_CARDS"))
    assert result["slots"]["6"]["current"] == "DEALT_IN_CANDIDATE"


def test_fold_history_preserved_across_unknowns():
    tracker = RosterCandidates(1)
    tracker.begin_hand("h")
    tracker.observe(0, {}, {"4": "fold"})
    result = tracker.observe(1, {})
    assert result["slots"]["4"]["current"] == "UNKNOWN"
    assert result["slots"]["4"]["history"] == "FOLDED_CANDIDATE"


def test_frame_gap_resets_confirmation():
    tracker = RosterCandidates()
    tracker.observe(0, cues(0, "EMPTY"))
    assert tracker.observe(2, cues(0, "EMPTY"))["slots"]["0"]["current"] == "UNKNOWN"


def test_boundary_not_automatically_invented():
    tracker = RosterCandidates(1)
    result = tracker.observe(0, cues(0, "BACK_CARDS"))
    assert result["hand"] is None
    assert result["slots"]["0"]["history"] == "UNKNOWN"
    assert result["boundary_authority"] == "external_not_inferred"


@pytest.mark.parametrize("frame", [-1, True, "2"])
def test_invalid_frame_rejected(frame):
    with pytest.raises(ValueError):
        RosterCandidates().observe(frame, {})


def test_nonmonotonic_rejected():
    tracker = RosterCandidates()
    tracker.observe(10, {})
    with pytest.raises(ValueError):
        tracker.observe(10, {})


@pytest.mark.parametrize("prior", ["EMPTY", "WAITING_POST_OR_PASS", "EXPLICIT_FOLD"])
def test_back_cards_do_not_reopen_same_hand_roster(prior):
    tracker = RosterCandidates(1)
    tracker.begin_hand("h")
    tracker.observe(0, cues(6, prior))
    result = tracker.observe(1, cues(6, "BACK_CARDS"))
    assert result["slots"]["6"]["conflict"]
    assert result["slots"]["6"]["current"] == "UNKNOWN"


def test_empty_to_explicit_wait_next_is_not_dealt_conflict():
    tracker = RosterCandidates(1)
    tracker.begin_hand("h")
    tracker.observe(0, cues(6, "EMPTY"))
    value = tracker.observe(1, cues(6, "WAITING_NEXT_HAND"))["slots"]["6"]
    assert value["current"] == "WAITING_CANDIDATE"
    assert not value["conflict"]


def test_wait_next_history_never_fills_current_unknown():
    tracker = RosterCandidates(1)
    tracker.begin_hand("h")
    tracker.observe(0, cues(6, "WAITING_NEXT_HAND"))
    value = tracker.observe(1, {})["slots"]["6"]
    assert value["current"] == "UNKNOWN"
    assert value["history"] == "WAITING_CANDIDATE"


def test_yellow_waiting_text_not_green_status_or_red_back():
    image = np.array([[[0, 255, 255], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert waiting_yellow(image).tolist() == [[True, False, False]]
