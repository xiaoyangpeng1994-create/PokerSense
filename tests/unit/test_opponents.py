"""Tests for PlayerState and OpponentProfile (opponents.py)."""

from dataclasses import FrozenInstanceError

import pytest

from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.opponents import OpponentProfile, PlayerState
from poker_engine.core.value_objects import ChipAmount


def _make_player(**overrides):
    args = dict(
        player_id="p1",
        seat=0,
        position=Position.BTN,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=True,
        is_dealer=True,
    )
    args.update(overrides)
    return PlayerState(**args)


def test_valid_player():
    p = _make_player()
    assert p.seat == 0
    assert p.is_hero is True


def test_committed_street_exceeds_hand_rejected():
    with pytest.raises(ValueError):
        _make_player(
            committed_this_street=ChipAmount("30"),
            committed_this_hand=ChipAmount("20"),
        )


def test_negative_seat_rejected():
    with pytest.raises(ValueError):
        _make_player(seat=-1)


def test_non_int_seat_rejected():
    with pytest.raises(TypeError):
        _make_player(seat="0")  # type: ignore[arg-type]


def test_empty_player_id_rejected():
    with pytest.raises(ValueError):
        _make_player(player_id="")


def test_player_frozen():
    p = _make_player()
    with pytest.raises(FrozenInstanceError):
        p.stack = ChipAmount("50")  # type: ignore[misc]


def test_player_stack_uses_chipamount():
    p = _make_player()
    assert isinstance(p.stack, ChipAmount)
    assert isinstance(p.committed_this_street, ChipAmount)


def test_committed_equal_hand_ok():
    p = _make_player(
        committed_this_street=ChipAmount("20"),
        committed_this_hand=ChipAmount("20"),
    )
    assert p.committed_this_street == ChipAmount("20")


# ---------- OpponentProfile ----------

def _make_profile(**overrides):
    from datetime import datetime, timezone
    args = dict(
        player_id="p1",
        vpip=0.25,
        pfr=0.15,
        af=2.0,
        cbet_freq=0.60,
        threebet_freq=0.08,
        bluff_freq=0.30,
        sample_size=100,
        last_updated=datetime(2026, 8, 18, tzinfo=timezone.utc),
    )
    args.update(overrides)
    return OpponentProfile(**args)


def test_profile_valid():
    p = _make_profile()
    assert p.player_id == "p1"
    assert p.vpip == 0.25
    assert p.sample_size == 100


def test_profile_empty_player_id_rejected():
    with pytest.raises(ValueError):
        _make_profile(player_id="")


def test_profile_freq_out_of_range_rejected():
    with pytest.raises(ValueError):
        _make_profile(vpip=1.5)


def test_profile_af_can_exceed_one():
    # aggression factor has no strict upper bound
    p = _make_profile(af=5.0)
    assert p.af == 5.0


def test_profile_negative_sample_size_rejected():
    with pytest.raises(ValueError):
        _make_profile(sample_size=-1)


def test_profile_freq_must_be_float_not_bool():
    with pytest.raises(TypeError):
        _make_profile(vpip=True)


def test_profile_naive_last_updated_rejected():
    from datetime import datetime
    with pytest.raises(TypeError):
        _make_profile(last_updated=datetime(2026, 8, 18))  # naive


def test_profile_frozen():
    p = _make_profile()
    with pytest.raises(FrozenInstanceError):
        p.vpip = 0.99  # type: ignore[misc]


# ---------- 返修 v2 新增测试 ----------

def test_profile_af_nan_rejected():
    with pytest.raises(ValueError):
        _make_profile(af=float("nan"))


def test_profile_af_infinity_rejected():
    with pytest.raises(ValueError):
        _make_profile(af=float("inf"))


def test_profile_af_bool_rejected():
    with pytest.raises(TypeError):
        _make_profile(af=True)


def test_profile_af_negative_rejected():
    with pytest.raises(ValueError):
        _make_profile(af=-1.0)
