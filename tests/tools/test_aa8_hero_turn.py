import pytest

from tools import aa8_hero_turn as module


@pytest.mark.parametrize("fractions,faces,expected", [
    ([.8, .0, .8], True, True),
    ([.8, .3, .8], True, None),
    ([.8, .0, .8], False, None),
    ([.0, .0, .8], True, None),
    ([.8, .0, .0], True, None),
    ([.0, .0, .0], True, None),
])
def test_two_controls_require_positive_colors_and_faces(
        monkeypatch, fractions, faces, expected):
    monkeypatch.setattr(module, "three_control_candidate", lambda image: {
        "hero_turn": None, "button_fractions": fractions})
    monkeypatch.setattr(module, "face_card_support",
                        lambda image, rect: (faces, "mock"))
    result = module.hero_turn_candidate(None)
    assert result["hero_turn"] is expected
    assert not result["strategy_eligible"]


def test_existing_three_control_behavior_is_retained(monkeypatch):
    monkeypatch.setattr(module, "three_control_candidate", lambda image: {
        "hero_turn": True, "button_fractions": [.8, .8, .8]})
    assert module.hero_turn_candidate(None)["control_layout"] == "three_active_controls"


def test_unsupported_canvas_remains_unknown(monkeypatch):
    monkeypatch.setattr(module, "three_control_candidate", lambda image: {
        "hero_turn": None, "reason": "unsupported_canvas"})
    assert module.hero_turn_candidate(None)["hero_turn"] is None
