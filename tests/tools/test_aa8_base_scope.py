import json
from pathlib import Path

import pytest

from tools.aa8_base_scope import scope_receipt


@pytest.mark.parametrize("modes,expected", [
    ({"insurance": "VISIBLE"}, ["insurance"]),
    ({"mushroom_trigger": "VISIBLE"}, ["mushroom"]),
    ({"critical_hit_title": {"critical_hit_animation": True}}, ["bomb"]),
])
def test_observed_modes_deferred_without_marking_test_passed(modes, expected):
    result = scope_receipt({"scene_supported": True, "special_modes": modes})
    assert result["observed_deferred_modes"] == expected
    assert result["status"] == "DEFERRED_MODE_OBSERVED"
    assert not result["automatically_exempt_from_acceptance"]
    assert not result["strategy_eligible"]


def test_no_matching_template_does_not_prove_normal_mode():
    result = scope_receipt({"scene_supported": True, "special_modes": {
        "insurance": "UNKNOWN", "mushroom_rule": "3BB",
        "critical_hit_rule": "7BB", "mushroom_trigger": "UNKNOWN"}})
    assert result["status"] == "BASE_CANDIDATE_UNVERIFIED"
    assert not result["normal_mode_verified"]
    assert result["observed_deferred_modes"] == []
    assert not result["full_visual_acceptance"]


def test_unsupported_scene_stays_blocked():
    assert scope_receipt({})["status"] == "BLOCKED_OR_UNSUPPORTED_SCENE"


def test_scope_configuration_and_runtime_receipt_agree():
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa8_candidate_v2/base_visual_scope.json")
    profile = json.loads(path.read_text(encoding="utf-8"))
    actual = scope_receipt({})
    assert profile["id"] == actual["scope_id"]
    assert profile["deferred_semantics"] == actual["deferred_semantics"]
    assert profile["retain_mode_guards"] is True
    assert profile["unknown_mode_means_inactive"] is False
    assert "capture_chain_stability" in profile["required"]
    assert "independent_whole_hand_test" in profile["required"]
