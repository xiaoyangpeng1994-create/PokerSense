"""Safety contracts for development continuous visual candidates."""

import numpy as np
import pytest

from tools.aa8_continuous_state import (
    ContinuousEvidence, actor_ring, board_count, short_all_in_candidate,
    suffix_components,
)


def row(actor=0, board=4, own="63", other="221"):
    return {"actor": actor, "board_count": board,
            "wagers": {"0": own, "3": other}}


def test_requires_adjacent_stable_frames():
    state = ContinuousEvidence()
    assert state.observe(1, row())["status"] == "UNKNOWN"
    result = state.observe(2, row())
    assert result["context"] == {
        "actor": 0, "own": "63", "price_lower_bound": "221", "frame": 2}
    assert result["action"] is None
    assert not result["strategy_eligible"]
    assert state.observe(4, row())["status"] == "UNKNOWN"


@pytest.mark.parametrize("change", [
    row(None), row(3), row(board=5), row(board=None), row(own=None), row(other="222")])
def test_context_resets_without_carrying_stale_values(change):
    state = ContinuousEvidence()
    state.observe(1, row())
    state.observe(2, row())
    assert state.observe(3, change).get("context") is None


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "bad", 221])
def test_invalid_money_abstains(amount):
    state = ContinuousEvidence()
    state.observe(1, row(other=amount))
    assert state.observe(2, row(other=amount))["status"] == "UNKNOWN"


def test_missing_wager_not_zero_or_check():
    state = ContinuousEvidence()
    for frame in (1, 2, 3):
        result = state.observe(frame, row(own=None))
        assert result["action"] is None
        assert result.get("context") is None


def test_ring_missing_and_multiple_abstain():
    image = np.zeros((1080, 498, 3), np.uint8)
    profile = {"slots": [{"avatar": [s * 50, 0, 40, 40]} for s in range(8)]}
    assert actor_ring(image, profile)["actor"] is None
    image[:40, :40] = (0, 255, 0)
    assert actor_ring(image, profile)["actor"] == 0
    image[:40, 50:90] = (0, 255, 0)
    assert actor_ring(image, profile)["actor"] is None


def test_board_count_invalid_partial_abstains():
    image = np.zeros((1080, 498, 3), np.uint8)
    assert board_count(image) == 0
    image[475:545, 110:159] = 255
    assert board_count(image) is None
    image[475:545, 166:215] = 255
    image[475:545, 223:272] = 255
    assert board_count(image) == 3


def cash_sequence():
    rows = []
    for i in range(4):
        rows.append({"frame": 100 + i, "board_count": 4,
                     "actor": 0 if i < 2 else None,
                     "actor_evidence": {"timer_suffix_verified": i < 2},
                     "wagers": {"0": "63", "3": "221", "1": None},
                     "stacks": {str(s): ("104" if i < 2 else "0")
                                if s == 0 else "200" for s in range(8)},
                     "pot": "474" if i < 2 else "578"})
    return rows


def test_unknown_other_wagers_allow_only_short_call_lower_bound_candidate():
    result = short_all_in_candidate(cash_sequence())
    assert result["action"] == "call"
    assert result["debit"] == "104"
    assert result["price_lower_bound"] == "221"
    assert not result["exact_street_price_known"]
    assert not result["legal_action_verified"]
    assert not result["strategy_eligible"]


@pytest.mark.parametrize("change", [
    "gap", "board", "actor", "timer", "pot", "unstable", "two_debits",
    "not_exhausted", "not_short", "unknown", "same_actor_after"])
def test_short_call_refuses_incomplete_or_conflicting_evidence(change):
    rows = cash_sequence()
    if change == "gap":
        rows[2]["frame"] = 150
    elif change == "board":
        rows[2]["board_count"] = 5
    elif change == "actor":
        rows[1]["actor"] = 3
    elif change == "timer":
        rows[1]["actor_evidence"]["timer_suffix_verified"] = False
    elif change == "pot":
        rows[2]["pot"] = rows[3]["pot"] = "579"
    elif change == "unstable":
        rows[0]["stacks"]["0"] = "105"
    elif change == "two_debits":
        rows[2]["stacks"]["3"] = rows[3]["stacks"]["3"] = "199"
    elif change == "not_exhausted":
        rows[2]["stacks"]["0"] = rows[3]["stacks"]["0"] = "1"
    elif change == "not_short":
        rows[0]["wagers"]["3"] = rows[1]["wagers"]["3"] = "160"
    elif change == "unknown":
        rows[0]["stacks"]["7"] = None
    else:
        rows[2]["actor"] = rows[3]["actor"] = 0
    assert short_all_in_candidate(rows)["action"] is None


def test_continuous_emits_short_call_once_on_second_after_frame():
    state = ContinuousEvidence()
    values = [state.observe(r["frame"], r)["unmarked_action_candidate"]
              for r in cash_sequence()]
    assert [v["action"] for v in values] == [None, None, None, "call"]


def test_suffix_components_ignore_background_translation():
    patch = np.zeros((30, 40), np.uint8)
    patch[5:16, 6:13] = 255
    first = suffix_components(patch)
    second = suffix_components(np.roll(patch, 4, axis=1))
    assert len(first) == len(second) == 1
    np.testing.assert_array_equal(first[0], second[0])


def test_diagonal_avatar_bridge_does_not_swallow_timer_component():
    patch = np.zeros((40, 45), np.uint8)
    patch[5:16, 5:13] = 255
    patch[16, 13] = 255
    patch[17:35, 14:40] = 255
    assert len(suffix_components(patch)) == 1
