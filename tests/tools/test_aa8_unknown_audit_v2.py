import pytest

from tools.aa8_unknown_audit_v2 import analyze, classify


def row(frame, unknown=True, actor=None):
    return {"frame": frame, "pts_seconds": str(frame / 30), "scene_supported": True,
            "current_actor": actor, "special_modes": {}, "causal_street_wagers_v2": {
                "status": "WAGERS_UNKNOWN" if unknown else "CANDIDATE", "reason": "x"}}


def test_no_special_detection_does_not_mean_ordinary():
    assert classify(row(1, actor=3)) == "ACTION_OPPORTUNITY_MODE_UNVERIFIED"


def test_positive_insurance_wins_over_actor_cue():
    value = row(1, actor=3)
    value["special_modes"]["insurance"] = "VISIBLE"
    assert classify(value) == "POSITIVE_SPECIAL_MODE_PAUSE"


def test_center_null_not_claimed_from_ambiguous_reason():
    value = row(1)
    value["causal_street_wagers_v2"]["reason"] = (
        "collection_or_center_change_invalidates_epoch")
    result = analyze([value, row(2)])
    assert result["center_triggers"][0]["null_vs_numeric_change"] == (
        "UNKNOWN_FROM_SAVED_DATA")


def test_runs_split_on_known_frame_and_source_gap():
    result = analyze([row(1), row(2, False), row(3), row(7)])
    assert [r["frames"] for r in result["intervals"]] == [1, 1, 1]
    assert result["unknown_frames"] == 3


def test_duplicate_frame_rejected():
    with pytest.raises(ValueError):
        analyze([row(1), row(1)])


def test_current_logged_center_is_not_reported_missing():
    value = row(1)
    value["observed_center_v2"] = {"value": "115"}
    value["causal_street_wagers_v2"]["reason"] = (
        "collection_or_center_change_invalidates_epoch")
    result = analyze([value])
    assert result["center_triggers"][0]["observed_center"] == "115"
    assert not result["center_values_not_present_in_saved_observations"]
