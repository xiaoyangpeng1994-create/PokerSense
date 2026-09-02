"""Tests for core enums (Task 1B)."""

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)


def test_rank_members_complete():
    ranks = list(Rank)
    # 2..A = 13 distinct ranks (2,3,4,5,6,7,8,9,T,J,Q,K,A)
    assert len(ranks) == 13
    assert Rank.ACE.value == "A"
    assert Rank.TEN.value == "T"


def test_rank_value_int():
    assert Rank.TWO.value_int == 2
    assert Rank.TEN.value_int == 10
    assert Rank.ACE.value_int == 14


def test_suit_members():
    assert {s.value for s in Suit} == {"c", "d", "h", "s"}


def test_street_members():
    assert [s.value for s in Street] == [
        "preflop", "flop", "turn", "river", "showdown",
    ]


def test_action_type_core():
    assert ActionType.FOLD.value == "fold"
    assert ActionType.RAISE.value == "raise"
    assert ActionType.ALL_IN.value == "all_in"


def test_player_status_members():
    assert {s.value for s in PlayerStatus} == {
        "active", "folded", "all_in", "sitting_out", "unknown",
    }


def test_position_covers_9max():
    vals = {p.value for p in Position}
    # 9-max coverage
    expected_positions = [
        "SB", "BB", "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN",
    ]
    for expected in expected_positions:
        assert expected in vals
    assert "UNKNOWN" in vals


def test_str_enum_serializes_like_string():
    import json

    assert json.dumps(Rank.ACE) == '"A"'
    assert json.dumps(Suit.SPADES) == '"s"'
