import copy

import pytest

from tools.aa8_hand_transition import HandTransitionCandidates


def row(frame, debit=False):
    stacks = dict.fromkeys(map(str, range(8)), "100")
    if debit:
        stacks.update({"1": "280", "2": "98", "3": "96", "4": "94", "5": "98"})
    return {"frame": frame, "scene_supported": True, "board_count": 0,
            "pot": {"value": "0"}, "stacks": stacks}


def test_post_and_deal_requires_two_confirmations_and_preserves_credit():
    tracker = HandTransitionCandidates()
    tracker.observe(row(20), {"visible": None})
    tracker.observe(row(21), {"visible": None})
    assert tracker.observe(row(22, True), {"visible": True})["candidate"] is None
    event = tracker.observe(row(23, True), {"visible": True})["candidate"]
    assert event["first_candidate_frame"] == 22
    assert event["before_frame"] == 21
    assert event["confirmed_at_frame"] == 23
    assert event["unallocated_credits"] == {"1": "180"}
    assert event["credit_semantics"] == "POSSIBLE_REFILL_NOT_PROFIT"
    assert not event["authoritative_boundary"]
    assert tracker.observe(row(24, True), {"visible": True})["candidate"] is None


def test_pot_clearing_alone_never_new_hand():
    tracker = HandTransitionCandidates()
    before = row(0)
    before["pot"] = {"value": "578"}
    before["board_count"] = 5
    tracker.observe(before, {"visible": None})
    for frame in range(1, 20):
        assert tracker.observe(row(frame), {"visible": None})["candidate"] is None


@pytest.mark.parametrize("change", [
    {"pot": None}, {"board_count": None}, {"board_count": 3},
    {"scene_supported": False}, {"stacks": {"0": "100"}}])
def test_incomplete_or_nonzero_context_abstains(change):
    tracker = HandTransitionCandidates()
    for frame in (0, 1):
        tracker.observe(row(frame), {"visible": None})
    for frame in (2, 3):
        value = row(frame, True)
        value.update(copy.deepcopy(change))
        assert tracker.observe(value, {"visible": True})["candidate"] is None


def test_no_center_card_no_boundary():
    tracker = HandTransitionCandidates()
    for frame in range(4):
        result = tracker.observe(row(frame, frame >= 2), {"visible": None})
        assert result["candidate"] is None


def test_nonconsecutive_preceding_state_cannot_be_used():
    tracker = HandTransitionCandidates()
    tracker.observe(row(0), {"visible": None})
    tracker.observe(row(1), {"visible": None})
    tracker.observe(row(3, True), {"visible": True})
    assert tracker.observe(row(4, True), {"visible": True})["candidate"] is None


def test_changed_next_frame_cancels_candidate():
    tracker = HandTransitionCandidates()
    tracker.observe(row(0), {"visible": None})
    tracker.observe(row(1), {"visible": None})
    tracker.observe(row(2, True), {"visible": True})
    assert tracker.observe(row(3), {"visible": True})["candidate"] is None


def test_single_bet_not_multiple_posts():
    tracker = HandTransitionCandidates()
    tracker.observe(row(0), {"visible": None})
    tracker.observe(row(1), {"visible": None})
    for frame in (2, 3):
        value = row(frame)
        value["stacks"]["4"] = "98"
        assert tracker.observe(value, {"visible": True})["candidate"] is None
