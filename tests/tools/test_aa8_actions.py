import json
from pathlib import Path

import pytest

from tools.verify_aa8_actions import verify


def inputs():
    root = Path(__file__).resolve().parents[1] / "fixtures/aa_reference_hands"
    return tuple(json.loads((root / f"eight_session_56dd_{name}_v1.json").read_text(
        encoding="utf-8")) for name in ("actions", "settlement"))


def test_complete_voluntary_line_and_unmatched_return():
    result = verify(*inputs())
    assert result["action_count"] == 21
    assert result["surviving_slots"] == [1, 4]
    assert result["unmatched_return_under_reviewed_action_line"] == "94"
    assert result["contested_pot_before_unallocated_deduction"] == "529"
    assert not result["full_visual_acceptance"]
    assert not result["automated_visual_extraction"]
    assert not result["exact_event_frames_verified"]
    assert not result["rake_policy_verified"]


@pytest.mark.parametrize("street,event,field,value", [
    (0, 1, "debit", "20"),
    (0, 2, "slot", 6),
    (1, 3, "kind", "raise"),
    (1, 5, "kind", "check"),
    (2, 1, "kind", "bet"),
    (3, 1, "kind", "raise"),
    (3, 1, "all_in", False),
    (3, 1, "debit", "214"),
    (3, 2, "pot_after", "624"),
    (3, 2, "window", [3165, 3172]),
])
def test_bad_actions_do_not_pass(street, event, field, value):
    review, cash = inputs()
    review["streets"][street]["actions"][event][field] = value
    with pytest.raises(ValueError):
        verify(review, cash)


def test_missing_turn_check_and_future_board_rejected():
    review, cash = inputs()
    review["streets"][2]["actions"].pop()
    with pytest.raises(ValueError, match="incomplete street"):
        verify(review, cash)
    review, cash = inputs()
    review["streets"][1]["board"] = cash["board"]
    with pytest.raises(ValueError, match="board"):
        verify(review, cash)


def test_replay_preserves_information_time():
    result = verify(*inputs())
    assert result["timeline"][0]["board"] == []
    assert result["timeline"][7]["board"] == ["5h", "6c", "6s"]
    assert all("opponent" not in row for row in result["timeline"])
