"""Tests for Hand Memory (InMemoryHandMemory)."""

from datetime import datetime, timedelta, timezone

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Street
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandHistory, HandSummary
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount, ChipDelta
from poker_engine.memory import (
    HandConflictError,
    HandLifecycleError,
    HandNotFoundError,
    InMemoryHandMemory,
)

UTC = timezone.utc


def _aw(**kw):
    base = dict(year=2026, month=8, day=19, hour=1, minute=0, tzinfo=UTC)
    base.update(kw)
    return datetime(**base)


def _player(seat=0, pid=None, hero=False):
    return PlayerState(
        player_id=pid or f"p{seat}",
        seat=seat,
        position=Position.BTN if seat == 0 else Position.SB,
        stack=ChipAmount("100"),
        committed_this_street=ChipAmount("0"),
        committed_this_hand=ChipAmount("0"),
        status=PlayerStatus.ACTIVE,
        has_cards=True,
        is_hero=hero,
        is_dealer=(seat == 0),
    )


def _state(hand_id="h1", version=0, **overrides):
    args = dict(
        state_version=version,
        hand_id=hand_id,
        street=Street.PREFLOP,
        hero_cards=(),
        board_cards=(),
        players=(_player(0, hero=True), _player(1)),
        pot=ChipAmount("1.5"),
        current_bet=ChipAmount("1"),
        to_call=ChipAmount("1"),
    )
    args.update(overrides)
    return PokerState(**args)


def _event(hand_id="h1", version=0, etype=EventType.HAND_START):
    return StateEvent(
        event_type=etype,
        hand_id=hand_id,
        state_version=version,
        timestamp=_aw(),
    )


def _summary():
    return HandSummary(
        final_pot=ChipAmount("100"),
        winners=("p0",),
        winnings={"p0": ChipAmount("100")},
        net_result={"p0": ChipDelta("50"), "p1": ChipDelta("-50")},
    )


def _started():
    m = InMemoryHandMemory()
    s0 = _state("h1", 0)
    m.start_hand("h1", s0, started_at=_aw())
    return m, s0


# ---------- A. start_hand ----------

def test_start_hand_ok():
    m, s0 = _started()
    assert m.hand_exists("h1")
    assert m.is_active("h1")
    assert m.active_hand_id == "h1"
    # initial_state saved as first snapshot
    assert m.latest_state("h1") == s0


def test_start_hand_empty_hand_id_rejected():
    m = InMemoryHandMemory()
    with pytest.raises(ValueError):
        m.start_hand("", _state("h1", 0), started_at=_aw())


def test_start_hand_mismatched_initial_state_hand_id():
    m = InMemoryHandMemory()
    with pytest.raises(HandConflictError):
        m.start_hand("h1", _state("OTHER", 0), started_at=_aw())


def test_start_hand_idempotent_identical():
    m, s0 = _started()
    # identical hand_id + initial_state + started_at -> no-op
    m.start_hand("h1", s0, started_at=_aw())
    assert m.states("h1") == (s0,)  # not duplicated


def test_start_hand_conflict_different_initial_state():
    m, s0 = _started()
    with pytest.raises(HandConflictError):
        m.start_hand("h1", _state("h1", 1), started_at=_aw())


def test_start_hand_conflict_different_started_at():
    m, s0 = _started()
    with pytest.raises(HandConflictError):
        m.start_hand("h1", s0, started_at=_aw(minute=2))


def test_start_second_active_hand_rejected():
    m, s0 = _started()
    with pytest.raises(HandLifecycleError):
        m.start_hand("h2", _state("h2", 0), started_at=_aw())


# ---------- B. record_state ----------

def test_record_state_ok_increasing():
    m, s0 = _started()
    s1 = _state("h1", 1)
    m.record_state(s1)
    assert m.latest_state("h1") == s1
    assert m.states("h1") == (s0, s1)


def test_record_state_gap_allowed():
    m, s0 = _started()
    s2 = _state("h1", 2)  # skip version 1 (gap allowed)
    m.record_state(s2)
    assert m.latest_state("h1") == s2


def test_record_state_nonexistent_hand():
    m = InMemoryHandMemory()
    with pytest.raises(HandNotFoundError):
        m.record_state(_state("hX", 0))


def test_record_state_unknown_hand_id_rejected():
    # record_state looks up by state.hand_id; unknown hand_id -> HandNotFound
    m, s0 = _started()
    with pytest.raises(HandNotFoundError):
        m.record_state(_state("UNKNOWN", 1))


def test_record_state_version_regression_rejected():
    m, s0 = _started()
    m.record_state(_state("h1", 2))
    with pytest.raises(HandConflictError):
        m.record_state(_state("h1", 1))  # regression


def test_record_state_duplicate_idempotent():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    m.record_state(_state("h1", 1))  # identical -> no-op
    assert len(m.states("h1")) == 2


def test_record_state_same_version_different_conflict():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    with pytest.raises(HandConflictError):
        m.record_state(
            _state("h1", 1, pot=ChipAmount("999"))  # same version, diff content
        )


def test_record_state_completed_hand_rejected():
    m, s0 = _started()
    m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    with pytest.raises(HandLifecycleError):
        m.record_state(_state("h1", 1))


# ---------- C. record_event ----------

def test_record_event_ok():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    e = _event("h1", 1, EventType.RAISE)
    m.record_event(e)
    assert m.events("h1") == (e,)


def test_record_event_nonexistent_hand():
    m = InMemoryHandMemory()
    with pytest.raises(HandNotFoundError):
        m.record_event(_event("hX", 0))


def test_record_event_unknown_hand_id_rejected():
    m, s0 = _started()
    with pytest.raises(HandNotFoundError):
        m.record_event(_event("UNKNOWN", 0, EventType.RAISE))


def test_record_event_unknown_state_version_rejected():
    m, s0 = _started()
    # version 5 state doesn't exist (gap beyond latest)
    with pytest.raises(HandConflictError):
        m.record_event(_event("h1", 5, EventType.RAISE))


def test_record_event_version_must_map_to_existing_state():
    m, s0 = _started()
    m.record_state(_state("h1", 2))  # gap, latest is version 2
    # version 1 was never recorded -> reject even though < latest
    with pytest.raises(HandConflictError):
        m.record_event(_event("h1", 1, EventType.RAISE))
    # version 2 exists -> ok
    m.record_event(_event("h1", 2, EventType.RAISE))
    assert len(m.events("h1")) == 1


def test_record_event_duplicate_idempotent():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    e = _event("h1", 1, EventType.RAISE)
    m.record_event(e)
    m.record_event(e)  # identical -> no-op
    assert len(m.events("h1")) == 1


def test_record_event_completed_hand_rejected():
    m, s0 = _started()
    m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    with pytest.raises(HandLifecycleError):
        m.record_event(_event("h1", 0))


# ---------- C2. atomic transition ----------

def test_record_transition_commits_state_and_events_together():
    m, s0 = _started()
    s1 = _state("h1", 1)
    events = (
        _event("h1", 1, EventType.DEAL),
        _event("h1", 1, EventType.RAISE),
    )

    m.record_transition(s1, events)
    m.record_transition(s1, events)

    assert m.states("h1") == (s0, s1)
    assert m.events("h1") == events


def test_record_transition_invalid_event_leaves_memory_unchanged():
    m, s0 = _started()
    before_states = m.states("h1")
    before_events = m.events("h1")

    with pytest.raises(HandConflictError):
        m.record_transition(
            _state("h1", 1),
            (_event("another-hand", 1, EventType.RAISE),),
        )

    assert m.states("h1") == before_states == (s0,)
    assert m.events("h1") == before_events == ()


def test_record_transition_wrong_event_version_leaves_memory_unchanged():
    m, s0 = _started()

    with pytest.raises(HandConflictError):
        m.record_transition(
            _state("h1", 1),
            (_event("h1", 0, EventType.RAISE),),
        )

    assert m.states("h1") == (s0,)
    assert m.events("h1") == ()


# ---------- D. read API ----------

def test_read_api_returns_tuples_not_internal_lists():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    m.record_event(_event("h1", 1))
    assert isinstance(m.states("h1"), tuple)
    assert isinstance(m.events("h1"), tuple)


def test_get_state_unknown_version_returns_none():
    m, s0 = _started()
    assert m.get_state("h1", 99) is None


def test_latest_state_nonexistent_hand():
    m = InMemoryHandMemory()
    assert m.latest_state("hX") is None


def test_states_ordered_by_version():
    m, s0 = _started()
    m.record_state(_state("h1", 2))
    m.record_state(_state("h1", 3))
    versions = [s.state_version for s in m.states("h1")]
    assert versions == [0, 2, 3]


def test_read_api_does_not_leak_mutable_internal():
    m, s0 = _started()
    states = m.states("h1")  # returns tuple
    # tuple is immutable; can't mutate internal list through it
    assert isinstance(states, tuple)
    # get_state returns a frozen PokerState; reassigning is forbidden
    s = m.get_state("h1", 0)
    assert s.state_version == 0


# ---------- E. complete_hand ----------

def test_complete_hand_ok():
    m, s0 = _started()
    m.record_state(_state("h1", 1))
    m.record_event(_event("h1", 1, EventType.RAISE))
    h = m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    assert isinstance(h, HandHistory)
    assert h.hand_id == "h1"
    assert h.summary.final_pot == ChipAmount("100")
    assert h.start_time == _aw()
    assert h.end_time == _aw(minute=5)
    assert len(h.events) == 1
    # players from latest_state
    assert h.players == m.states("h1")[-1].players


def test_complete_hand_sets_inactive():
    m, s0 = _started()
    m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    assert not m.is_active("h1")
    assert m.active_hand_id is None
    assert m.get_hand_history("h1") is not None


def test_complete_hand_nonexistent():
    m = InMemoryHandMemory()
    with pytest.raises(HandNotFoundError):
        m.complete_hand("hX", _summary(), ended_at=_aw())


def test_complete_hand_double_complete_rejected():
    m, s0 = _started()
    m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    with pytest.raises(HandLifecycleError):
        m.complete_hand("h1", _summary(), ended_at=_aw(minute=6))


def test_complete_hand_end_before_start_rejected():
    m, s0 = _started()
    # started_at = hour=1 minute=0; end at hour=0 minute=59 is earlier
    with pytest.raises(HandLifecycleError):
        m.complete_hand("h1", _summary(), ended_at=_aw(hour=0, minute=59))


def test_complete_hand_naive_datetime_rejected():
    m, s0 = _started()
    with pytest.raises(TypeError):
        m.complete_hand("h1", _summary(), ended_at=datetime(2026, 8, 19))


def test_complete_hand_players_from_latest_state():
    m, s0 = _started()
    m.record_state(_state("h1", 1, players=(_player(0, hero=True),)))
    h = m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    assert len(h.players) == 1


# ---------- E2. atomic active-hand replacement ----------

def test_replace_active_hand_commits_boundary_and_successor_together():
    m, s0 = _started()
    boundary = _event("h1", 0, EventType.HAND_END)

    history = m.replace_active_hand(
        _state("h2", 0),
        _summary(),
        boundary,
        ended_at=_aw(minute=5),
        started_at=_aw(minute=5),
    )

    assert history.events == (boundary,)
    assert m.get_hand_history("h1") == history
    assert m.active_hand_id == "h2"
    assert m.states("h1") == (s0,)
    assert m.states("h2") == (_state("h2", 0),)


def test_replace_active_hand_invalid_boundary_is_fully_rolled_back():
    m, s0 = _started()

    with pytest.raises(HandConflictError):
        m.replace_active_hand(
            _state("h2", 0),
            _summary(),
            _event("wrong", 0, EventType.HAND_END),
            ended_at=_aw(minute=5),
            started_at=_aw(minute=5),
        )

    assert m.active_hand_id == "h1"
    assert m.is_active("h1")
    assert m.get_hand_history("h1") is None
    assert m.states("h1") == (s0,)
    assert m.events("h1") == ()
    assert not m.hand_exists("h2")


def test_replace_active_hand_non_boundary_event_is_fully_rolled_back():
    m, s0 = _started()

    with pytest.raises(HandConflictError):
        m.replace_active_hand(
            _state("h2", 0),
            _summary(),
            _event("h1", 0, EventType.RAISE),
            ended_at=_aw(minute=5),
            started_at=_aw(minute=5),
        )

    assert m.active_hand_id == "h1"
    assert m.states("h1") == (s0,)
    assert m.events("h1") == ()
    assert not m.hand_exists("h2")


def test_replace_active_hand_existing_successor_is_fully_rolled_back():
    m = InMemoryHandMemory()
    m.start_hand("h2", _state("h2", 0), started_at=_aw())
    m.complete_hand("h2", _summary(), ended_at=_aw(minute=1))
    s1 = _state("h1", 0)
    m.start_hand("h1", s1, started_at=_aw(minute=2))

    with pytest.raises(HandConflictError):
        m.replace_active_hand(
            _state("h2", 0),
            _summary(),
            _event("h1", 0, EventType.HAND_END),
            ended_at=_aw(minute=5),
            started_at=_aw(minute=5),
        )

    assert m.active_hand_id == "h1"
    assert m.is_active("h1")
    assert m.get_hand_history("h1") is None
    assert m.states("h1") == (s1,)
    assert m.events("h1") == ()


# ---------- F. multi-hand isolation ----------

def test_multi_hand_isolation():
    m = InMemoryHandMemory()
    # hand A
    m.start_hand("A", _state("A", 0), started_at=_aw())
    m.record_state(_state("A", 1, pot=ChipAmount("10")))
    m.record_event(_event("A", 1))
    m.complete_hand("A", _summary(), ended_at=_aw(minute=5))
    # hand B
    m.start_hand("B", _state("B", 0), started_at=_aw(minute=10))
    m.record_state(_state("B", 1, pot=ChipAmount("20")))
    m.complete_hand("B", _summary(), ended_at=_aw(minute=15))

    # A and B fully isolated
    assert [s.hand_id for s in m.states("A")] == ["A", "A"]
    assert [s.hand_id for s in m.states("B")] == ["B", "B"]
    assert [e.hand_id for e in m.events("A")] == ["A"]
    assert m.events("B") == ()
    a_hist = m.get_hand_history("A")
    b_hist = m.get_hand_history("B")
    assert a_hist.hand_id == "A"
    assert b_hist.hand_id == "B"
    assert len(m.completed_hands()) == 2


def test_completed_hands_returns_histories():
    m, s0 = _started()
    m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))
    assert len(m.completed_hands()) == 1
    assert m.completed_hands()[0].hand_id == "h1"


# ---------- G. serialization compatibility ----------

def test_completed_hand_history_json_roundtrip():
    import json

    from poker_engine.core.serialization import deserialize, serialize

    m, s0 = _started()
    m.record_state(_state("h1", 1))
    m.record_event(_event("h1", 1, EventType.RAISE))
    h = m.complete_hand("h1", _summary(), ended_at=_aw(minute=5))

    s = serialize(h)
    assert s["__type__"] == "HandHistory"
    rt = deserialize(HandHistory, json.loads(json.dumps(s)))
    assert rt.hand_id == h.hand_id
    assert rt.summary.final_pot == h.summary.final_pot
    assert rt.start_time == h.start_time
    assert rt.end_time == h.end_time


# ---------- v2 返修：strict None semantics for started_at / ended_at ----------

@pytest.mark.parametrize("bad", [0, False, ""])
def test_start_hand_falsy_started_at_rejected(bad):
    m = InMemoryHandMemory()
    with pytest.raises(TypeError):
        m.start_hand("h1", _state("h1", 0), started_at=bad)


@pytest.mark.parametrize("bad", [0, False, ""])
def test_complete_hand_falsy_ended_at_rejected(bad):
    m, s0 = _started()
    with pytest.raises(TypeError):
        m.complete_hand("h1", _summary(), ended_at=bad)


def test_start_hand_none_started_at_auto_utc():
    m = InMemoryHandMemory()
    m.start_hand("h1", _state("h1", 0), started_at=None)
    assert m.is_active("h1")
    # started_at auto-assigned as aware UTC (relative to real now, not a
    # fixed date — a hardcoded ended_at here would eventually be in the
    # past relative to the auto-assigned start_time and fail spuriously).
    h = m.complete_hand(
        "h1", _summary(), ended_at=datetime.now(UTC) + timedelta(minutes=5)
    )
    assert h.start_time.tzinfo is not None
    assert h.start_time.utcoffset() is not None


def test_complete_hand_none_ended_at_auto_utc():
    # start with started_at=None so start_time == utc_now, then complete with
    # ended_at=None -> both auto (end >= start holds trivially).
    m = InMemoryHandMemory()
    m.start_hand("h1", _state("h1", 0), started_at=None)
    h = m.complete_hand("h1", _summary(), ended_at=None)
    assert h.end_time is not None
    assert h.end_time.tzinfo is not None
    assert h.end_time.utcoffset() is not None
    assert h.end_time >= h.start_time
