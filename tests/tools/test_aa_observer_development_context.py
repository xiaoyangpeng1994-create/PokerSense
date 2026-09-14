import pytest

from tools.aa_observer_development_context import ObserverDevelopmentContext


def context():
    return ObserverDevelopmentContext(first_frame=10, last_frame=20,
                                      receipt_sha256="a" * 64,
                                      registry_sha256="b" * 64)


def row(frame=10, **updates):
    value = {"frame": frame, "source_receipt_sha256": "a" * 64,
             "scene_supported": True, "board_count": 0,
             "stacks": {}, "glyph_transitions": [], "special_modes": {}}
    value.update(updates)
    return value


def test_manual_context_never_promotes_legal_state():
    result = context().observe(row())
    assert result["observed_epoch"] is not None
    assert result["complete_legal_state"] is False
    assert result["strategy_eligible"] is False
    assert result["authoritative_hand_boundary"] is False


@pytest.mark.parametrize("bad", [row(9), row(11), row(21),
                                 row(source_receipt_sha256="c" * 64)])
def test_wrong_source_or_interval_rejected(bad):
    with pytest.raises(ValueError):
        context().observe(bad)


def test_overlay_clears_context_without_manual_reanchoring():
    state = context()
    state.observe(row())
    assert state.observe(row(11, special_modes={
        "block_state_updates": True}))["observed_epoch"] is None
    assert state.observe(row(12))["observed_epoch"] is None


def test_postflop_start_does_not_invent_preflop():
    assert context().observe(row(board_count=3))["observed_epoch"] is None
