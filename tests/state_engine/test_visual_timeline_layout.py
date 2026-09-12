import pytest

from poker_engine.state_engine.visual_timeline import VisualTimeline


def observation(frame, *, folded=None, ninth=True):
    return {"source_frame": frame, "scene_reason": "supported_layout_candidate",
            "hero": ["Ac", "Qh"],
            "street_observation": {"status": "valid", "value": "preflop"},
            "seat_presence": {str(i): {"status": "valid", "value": True}
                              for i in range(9 if ninth else 8)},
            "actions": {} if folded is None else {
                str(folded): {"accepted_candidate": True, "value": "fold"}}}


def test_ninth_slot_and_nonzero_hero_are_not_lost():
    engine = VisualTimeline(slot_count=9, hero_slot=5)
    for i in (1, 2):
        engine.consume(observation(i))
    assert engine.roster == tuple(range(9))
    for i in (3, 4):
        engine.consume(observation(i, folded=8))
    assert 8 in engine.folded
    assert engine.summary()["hero_action_eligible"] is None
    for i in (5, 6):
        engine.consume(observation(i, folded=5))
    assert engine.summary()["hero_action_eligible"] is False


def test_missing_ninth_census_cannot_open_roster():
    engine = VisualTimeline(slot_count=9, hero_slot=5)
    for i in range(1, 5):
        engine.consume(observation(i, ninth=False))
    assert engine.roster is None


def test_eight_slot_default_rejects_ninth_before_mutating():
    engine = VisualTimeline()
    with pytest.raises(ValueError, match="outside"):
        engine.consume(observation(1))
    assert engine.last_frame is None
    assert engine.events == []


@pytest.mark.parametrize("kwargs", [
    dict(slot_count=True), dict(slot_count=1), dict(slot_count=11),
    dict(hero_slot=True), dict(slot_count=9, hero_slot=9)])
def test_invalid_layout(kwargs):
    with pytest.raises(ValueError):
        VisualTimeline(**kwargs)
