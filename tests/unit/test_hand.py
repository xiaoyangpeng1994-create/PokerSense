"""Tests for HandSummary and HandHistory."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandHistory, HandSummary
from poker_engine.core.opponents import PlayerState
from poker_engine.core.value_objects import ChipAmount, ChipDelta

UTC = timezone.utc


def _aware(**kw):
    base = dict(year=2026, month=8, day=18, hour=14, tzinfo=UTC)
    base.update(kw)
    return datetime(**base)


def _player(seat, pid=None) -> PlayerState:
    return PlayerState(
        player_id=pid or f"p{seat}",
        seat=seat,
        position=Position.BTN,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=(seat == 0),
        is_dealer=(seat == 0),
    )


def _event() -> StateEvent:
    return StateEvent(
        event_type=EventType.HAND_START, hand_id="h1", state_version=0,
        timestamp=_aware(),
    )


def _summary() -> HandSummary:
    return HandSummary(
        final_pot=ChipAmount("100"),
        winners=("p0",),
        winnings={"p0": ChipAmount("100")},
        net_result={"p0": ChipDelta("50"), "p1": ChipDelta("-50")},
    )


# ---------- HandSummary ----------

def test_summary_valid():
    s = _summary()
    assert s.final_pot == ChipAmount("100")
    assert s.winners == ("p0",)
    assert s.winnings["p0"] == ChipAmount("100")
    assert s.net_result["p1"] == ChipDelta("-50")


def test_summary_final_pot_must_be_chipamount():
    with pytest.raises(TypeError):
        HandSummary(final_pot="100", winners=("p0",))


def test_summary_winnings_must_be_chipamount():
    with pytest.raises(TypeError):
        HandSummary(
            final_pot=ChipAmount("100"), winners=("p0",),
            winnings={"p0": "100"},
        )


def test_summary_net_result_must_be_chipdelta():
    with pytest.raises(TypeError):
        HandSummary(
            final_pot=ChipAmount("100"), winners=("p0",),
            net_result={"p0": ChipAmount("50")},  # ChipAmount not ChipDelta
        )


def test_summary_winnings_net_result_deep_immutable():
    winnings = {"p0": ChipAmount("100")}
    net = {"p0": ChipDelta("50")}
    s = HandSummary(final_pot=ChipAmount("100"), winners=("p0",),
                    winnings=winnings, net_result=net)
    winnings["pX"] = ChipAmount("999")
    net["pY"] = ChipDelta("999")
    assert "pX" not in s.winnings
    assert "pY" not in s.net_result
    with pytest.raises(TypeError):
        s.winnings["pZ"] = ChipAmount("1")  # type: ignore[index]


# ---------- HandHistory ----------

def test_hand_history_valid():
    h = HandHistory(
        hand_id="h1",
        players=(_player(0), _player(1)),
        events=(_event(),),
        summary=_summary(),
        start_time=_aware(),
    )
    assert h.hand_id == "h1"
    assert len(h.players) == 2
    assert len(h.events) == 1


def test_hand_history_empty_hand_id_rejected():
    with pytest.raises(ValueError):
        HandHistory(
            hand_id="", players=(), events=(), summary=_summary(),
            start_time=_aware(),
        )


def test_hand_history_players_must_be_playerstate():
    with pytest.raises(TypeError):
        HandHistory(
            hand_id="h1", players=("x",), events=(), summary=_summary(),
            start_time=_aware(),
        )


def test_hand_history_events_must_be_stateevent():
    with pytest.raises(TypeError):
        HandHistory(
            hand_id="h1", players=(), events=("x",), summary=_summary(),
            start_time=_aware(),
        )


def test_hand_history_naive_start_time_rejected():
    with pytest.raises(TypeError):
        HandHistory(
            hand_id="h1", players=(), events=(), summary=_summary(),
            start_time=datetime(2026, 8, 18),  # naive
        )


def test_hand_history_end_before_start_rejected():
    with pytest.raises(ValueError):
        HandHistory(
            hand_id="h1", players=(), events=(), summary=_summary(),
            start_time=_aware(hour=14),
            end_time=_aware(hour=13),
        )


def test_hand_history_frozen():
    h = HandHistory(
        hand_id="h1", players=(), events=(), summary=_summary(),
        start_time=_aware(),
    )
    with pytest.raises(FrozenInstanceError):
        h.hand_id = "h2"  # type: ignore[misc]


# ---------- 返修 v2 新增测试 ----------

def test_summary_winnings_key_must_be_nonempty_str():
    with pytest.raises(TypeError):
        HandSummary(
            final_pot=ChipAmount("100"), winners=("p0",),
            winnings={"": ChipAmount("10")},
        )


def test_summary_net_result_key_must_be_nonempty_str():
    with pytest.raises(TypeError):
        HandSummary(
            final_pot=ChipAmount("100"), winners=("p0",),
            net_result={"": ChipDelta("5")},
        )


def test_summary_winners_no_duplicates():
    with pytest.raises(ValueError):
        HandSummary(
            final_pot=ChipAmount("100"),
            winners=("p0", "p0"),
        )


def test_hand_history_event_hand_id_must_match():
    mismatch_event = StateEvent(
        event_type=EventType.HAND_START, hand_id="OTHER", state_version=0,
        timestamp=_aware(),
    )
    with pytest.raises(ValueError):
        HandHistory(
            hand_id="h1",
            players=(),
            events=(mismatch_event,),
            summary=_summary(),
            start_time=_aware(),
        )


def test_hand_history_players_no_duplicate_player_id():
    with pytest.raises(ValueError):
        HandHistory(
            hand_id="h1",
            players=(_player(0, "same"), _player(1, "same")),
            events=(),
            summary=_summary(),
            start_time=_aware(),
        )


def test_hand_history_players_no_duplicate_seat():
    with pytest.raises(ValueError):
        HandHistory(
            hand_id="h1",
            players=(_player(0, "a"), _player(0, "b")),
            events=(),
            summary=_summary(),
            start_time=_aware(),
        )
