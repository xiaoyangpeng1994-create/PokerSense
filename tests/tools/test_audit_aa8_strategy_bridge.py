import json

import pytest

from tools.audit_aa8_strategy_bridge import audit


def test_saved_candidate_rows_are_audited_without_strategy(tmp_path):
    source = tmp_path / "observations.jsonl"
    source.write_text(json.dumps({
        "frame": 1,
        "source_sha256": "a" * 64,
        "scene_supported": True,
        "visual_scope": {
            "scope_id": "aa8_base_visual_v1",
            "status": "BASE_CANDIDATE_UNVERIFIED",
            "normal_mode_verified": False,
            "observed_deferred_modes": [],
        },
        "observed_state_v2": {
            "authoritative_hand_boundary": False,
            "complete_legal_state": False,
        },
        "causal_street_wagers_v2": {"canonical_verified": False},
    }) + "\n", encoding="utf-8")
    result = audit(source, 8, tmp_path / "report")
    assert result["classifications"] == {"ABSTAIN": 1}
    assert result["advice_emitted"] is False
    assert result["provider_or_equity_executed"] is False
    assert result["blocker_counts"]["legal_state_incomplete"] == 1


def test_deferred_mode_is_separate_from_base_abstention(tmp_path):
    source = tmp_path / "observations.jsonl"
    source.write_text(json.dumps({
        "frame": 2, "source_sha256": "b" * 64, "scene_supported": True,
        "visual_scope": {"observed_deferred_modes": ["insurance"]},
    }) + "\n", encoding="utf-8")
    result = audit(source, 6, tmp_path / "report")
    assert result["classifications"] == {"DEFERRED_SPECIAL_MODE": 1}
    assert result["blocker_counts"]["deferred_special_mode:insurance"] == 1


def test_player_count_is_parameterized_but_bounded(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    for count in (6, 7, 8):
        assert audit(source, count, tmp_path / str(count))["frames"] == 0
    with pytest.raises(ValueError, match="6, 7 or 8"):
        audit(source, 9, tmp_path / "forbidden")
