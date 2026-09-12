"""Visual evidence pairing is causal and cannot silently invent betting events."""

from copy import deepcopy

import pytest

from poker_engine.state_engine.visual_timeline import VisualTimeline


def row(frame, *, seat=7, action=None, balance="100", street="preflop",
        present=None, supported=True):
    seats = {0, 1, 2, 4, 6, 7}
    presence = {i: i in seats for i in range(8)}
    presence.update(present or {})
    return {"source_frame": frame,
            "scene_reason": "supported_layout_candidate" if supported else "unknown",
            "hero": ["Ac", "Qh"],
            "street_observation": {"status": "valid", "value": street},
            "seat_presence": {str(i): {"status": "valid", "value": presence[i]}
                              for i in range(8)},
            "stacks": {str(i): {"status": "unknown" if i == seat and balance is None
                                else "valid", "value": balance if i == seat else "100"}
                       for i in range(8)},
            "actions": {str(seat): {"accepted_candidate": action is not None,
                                    "value": action}}}


def started():
    engine = VisualTimeline()
    engine.consume(row(1))
    engine.consume(row(2))
    return engine


@pytest.mark.parametrize("money_first", [False, True])
def test_badge_and_money_can_arrive_in_either_order(money_first):
    engine = started()
    inputs = []
    for i in range(3, 18):
        action = "call" if (i >= 5 if money_first else True) else None
        balance = "96" if (True if money_first else i >= 5) else "100"
        inputs.append(row(i, action=action, balance=balance))
    before = deepcopy(inputs)
    for observation in inputs:
        engine.consume(observation)
    actions = [e for e in engine.events if e.kind == "action_evidence"]
    assert len(actions) == 1 and actions[0].amount == "4"
    assert actions[0].confirmed_frame >= 6
    assert inputs == before


def test_future_money_does_not_change_an_earlier_prefix():
    engine = started()
    engine.consume(row(3, action="call"))
    engine.consume(row(4, action="call"))
    prefix = tuple(engine.events)
    assert not any(e.kind == "action_evidence" for e in prefix)
    engine.consume(row(5, action="call", balance="96"))
    engine.consume(row(6, action="call", balance="96"))
    assert tuple(engine.events[:len(prefix)]) == prefix


def test_waiting_joiner_is_not_added_and_departure_is_not_a_fold():
    engine = started()
    for i in range(3, 8):
        observed = row(i, seat=5, action="bet", balance="80",
                       present={5: True, 6: False})
        engine.consume(observed)
    assert engine.roster == (0, 1, 2, 4, 6, 7)
    assert engine.departed == {6}
    actions = [e for e in engine.events if e.kind == "action_evidence"]
    assert not any(e.seat in (5, 6) for e in actions)


def test_persistent_fold_is_not_repeated_on_each_street():
    engine = started()
    for i in range(3, 14):
        stage = "preflop" if i < 7 else "flop"
        engine.consume(row(i, seat=0, action="fold", street=stage))
    assert len([e for e in engine.events if e.action == "fold"]) == 1
    assert engine.summary()["hero_action_eligible"] is False


def test_missing_amount_stays_unresolved():
    engine = started()
    for i in range(3, 19):
        engine.consume(row(i, action="call", balance=None))
    assert not any(e.kind == "action_evidence" for e in engine.events)
    assert any(e.kind == "unresolved_action" for e in engine.events)


def test_two_nearby_debits_are_not_collapsed_into_one_call():
    engine = started()
    for i in range(3, 10):
        engine.consume(row(i, balance="96" if i < 5 else "92",
                           action="call" if i >= 7 else None))
    assert not any(e.kind == "action_evidence" for e in engine.events)


def test_reappearing_badge_and_new_debit_need_more_context():
    engine = started()
    for i in range(3, 7):
        engine.consume(row(i, action="call", balance="96"))
    engine.consume(row(7, balance="96"))
    for i in range(8, 11):
        engine.consume(row(i, action="call", balance="92"))
    assert len([e for e in engine.events if e.kind == "action_evidence"]) == 1
    assert any("betting_context" in e.note for e in engine.events)


@pytest.mark.parametrize("unsupported", [False, True])
def test_gaps_do_not_pair_old_balance_with_new_frame(unsupported):
    engine = started()
    engine.consume(row(3, action="call"))
    engine.consume(row(4, action="call"))
    if unsupported:
        engine.consume(row(5, supported=False))
    for i in range(6, 10):
        engine.consume(row(i, action="call", balance="96"))
    assert not any(e.kind == "action_evidence" for e in engine.events)


@pytest.mark.parametrize("frame", [2, 1, True, -1])
def test_bad_frame_identity_cannot_confirm_again(frame):
    with pytest.raises(ValueError):
        started().consume(row(frame))
