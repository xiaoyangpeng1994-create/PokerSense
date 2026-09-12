from tools.aa8_checkpoint_compare import fields


def row():
    return {"board_count": None, "cards": {"hero": None, "board_slots": [None] * 5},
            "stacks": {"6": {"value": None}}, "current_actor": None,
            "pot": {"value": None},
            "participation": {"slots": {"6": {"current": "UNKNOWN"}}}}


def test_unknown_is_not_empty_board_zero_stack_or_na():
    result = fields(row())
    assert result["board_cards"] is None
    assert result["stacks"]["6"] is None
    assert result["pot"] is None


def test_explicit_current_waiting_allows_na_not_numeric_zero():
    observation = row()
    observation["participation"]["slots"]["6"]["current"] = "WAITING_CANDIDATE"
    assert fields(observation)["stacks"]["6"] == {"status": "NOT_APPLICABLE"}


def test_partial_card_identity_does_not_pass_board():
    observation = row()
    observation["board_count"] = 3
    observation["cards"]["board_slots"] = ["5h", None, "6s", None, None]
    assert fields(observation)["board_cards"] is None
