"""Editable rule persistence, exact amounts, and unsupported-rule refusal."""

import json

import pytest
from fastapi.testclient import TestClient

from poker_engine.desktop.server import create_app
from poker_engine.desktop.table_rules import (
    load_user_table_rules, save_user_table_rules, validate_table_rules,
)


def rules(**changes):
    return dict(schema_version=1, mode="live", table_size=8,
                small_blind="2", big_blind="4", minimum_chip="1", ante="2",
                rake_percent="0.03", rake_cap_bb="2", straddle_mode="none",
                straddle_amount=None, extra_effects="") | changes


def test_rule_units_are_exact_and_cap_is_normalized_by_current_big_blind():
    result = validate_table_rules(rules())
    assert result.game_config.big_blind.value == 4
    assert str(result.game_config.rake_percent) == "0.03"
    assert result.game_config.rake_cap.value == 8
    updated = validate_table_rules(rules(small_blind="5", big_blind="10"))
    assert updated.game_config.rake_cap.value == 20


@pytest.mark.parametrize("changes,reason", (
    ({"mode": "simulation"}, "simulation_rules_only"),
    ({"ante": None}, "table_rules_unverified"),
    ({"straddle_mode": "mandatory", "straddle_amount": "8"},
     "straddle_strategy_not_supported"),
    ({"extra_effects": "游戏暴击，细节尚未明确"}, "extra_table_effects_not_supported"),
))
def test_assumed_or_unsupported_rules_cannot_enable_live_strategy(changes, reason):
    result = validate_table_rules(rules(**changes))
    assert result.game_config is None
    assert result.unavailable_reason == reason


@pytest.mark.parametrize("changes", (
    {"rake_percent": "NaN"}, {"rake_percent": "1.1"}, {"ante": "-1"},
    {"big_blind": "0"}, {"big_blind": 4}, {"table_size": 6.0},
    {"small_blind": "5"}, {"extra_effects": "x" * 501},
))
def test_invalid_rules_are_rejected_before_persistence(changes):
    with pytest.raises(ValueError):
        validate_table_rules(rules(**changes))


def test_rules_endpoint_persists_and_invalidates_revision(tmp_path, monkeypatch):
    path = tmp_path / "table-rules.json"
    monkeypatch.setenv("POKERSENSE_TABLE_RULES_PATH", str(path))
    monkeypatch.setenv("POKERSENSE_SETTINGS_PATH", str(tmp_path / "language.json"))
    client = TestClient(create_app())
    response = client.put("/table-rules", json=rules())
    assert response.status_code == 200
    first = response.json()
    assert first["rules_supported"] is True
    client.put("/settings", json={"language": "zh"})
    assert client.get("/table-rules").json()["rules"] == rules()
    updated = client.put("/table-rules", json=rules(table_size=6, ante="1")).json()
    assert updated["revision"] != first["revision"]
    assert client.get("/settings").json() == {"language": "zh"}
    before = path.read_bytes()
    assert client.put("/table-rules", json=rules(rake_percent="2")).status_code == 422
    assert path.read_bytes() == before


def test_missing_settings_use_explicit_simulation_defaults(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    monkeypatch.setenv("POKERSENSE_TABLE_RULES_PATH", str(path))
    default = tmp_path / "defaults.json"
    default.write_text(json.dumps(rules(mode="simulation")), encoding="utf-8")
    document = load_user_table_rules(default)
    assert validate_table_rules(document).unavailable_reason == "simulation_rules_only"
    save_user_table_rules(rules(big_blind="8"))
    assert load_user_table_rules(default)["big_blind"] == "8"
