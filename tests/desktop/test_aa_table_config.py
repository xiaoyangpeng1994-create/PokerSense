import pytest

from poker_engine.desktop.aa_table_config import (
    AATableConfigStore, empty_config, validate_config,
)


def complete():
    return {**empty_config(), "dealt_players": 8, "small_blind": "2",
            "big_blind": "4", "ante": "1", "straddle_amount": "8",
            "straddle_mode": "mandatory_utg", "rake_percent": "3",
            "rake_cap_bb": "2", "minimum_chip": "1",
            "rake_application": "all_pots", "rake_rounding": "exact",
            "rake_distribution": "proportional_all_pots",
            "insurance": "off", "bomb": "off", "mushroom": "off"}


def test_no_default_rules_and_never_promotes_user_declaration():
    result = validate_config(empty_config())
    assert result["pending_fields"] and result["simulation_rules"] is None
    result = validate_config(complete())
    assert result["conditional_analysis_ready"]
    assert result["simulation_rules"]["rake_percent"] == "0.03"
    assert result["simulation_rules"]["verification_status"] == "simulation"
    assert result["source"] == "MANUAL_DECLARATION"
    assert not result["live_strategy_eligible"] and not result["visual_verified"]


@pytest.mark.parametrize("field,value", [
    ("small_blind", "4"), ("rake_percent", "101"), ("big_blind", "NaN"),
    ("ante", -1), ("minimum_chip", "0"), ("straddle_amount", "2"),
    ("dealt_players", True), ("ante", "0.01"),
])
def test_contradictory_or_inexact_rules_reject(field, value):
    with pytest.raises(ValueError):
        validate_config({**complete(), field: value})


def test_special_modes_retained_and_block_conditional_analysis():
    result = validate_config({**complete(), "bomb": "on"})
    assert result["unsupported_effects"] == ["bomb"]
    assert result["simulation_rules"] is None


def test_atomic_persistence_revision_conflict_and_reset(tmp_path):
    path = tmp_path / "aa-rules.json"
    store = AATableConfigStore(path)
    first = store.get()
    saved = store.save(complete(), first["revision"])
    assert AATableConfigStore(path).get() == saved
    with pytest.raises(ValueError, match="其他页面"):
        store.save(empty_config(), first["revision"])
    reset = store.save(empty_config(), saved["revision"])
    assert not reset["conditional_analysis_ready"]
    assert AATableConfigStore(path).get() == reset
