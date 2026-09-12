"""Matching cannot hide late confirmation, wrong money or duplicate actions."""

from tools.replay_wpk_action_timeline import compare_actions


def fixture():
    truth = {"comparison": {"max_additional_confirmation_frames": 2}, "actions": [
        {"order": 1, "street": "preflop", "seat": 7, "action": "call",
         "visible_interval": [3, 5], "money_visible_by": 6, "amount_additional": "4"}]}
    event = {"kind": "action_evidence", "street": "preflop", "seat": 7,
             "action": "call", "first_frame": 4, "confirmed_frame": 6, "amount": "4.0"}
    return truth, event


def test_decimal_amount_matches_without_float_coercion():
    truth, event = fixture()
    assert compare_actions(truth, [event])["matched_count"] == 1


def test_late_confirmation_does_not_pass_a_good_first_frame():
    truth, event = fixture()
    event["confirmed_frame"] = 20
    result = compare_actions(truth, [event])
    assert result["matched_count"] == 0
    assert not result["matches"][0]["timing_matches"]


def test_extra_duplicate_action_is_reported():
    truth, event = fixture()
    result = compare_actions(truth, [event, event.copy()])
    assert len(result["unexpected_actions"]) == 1
