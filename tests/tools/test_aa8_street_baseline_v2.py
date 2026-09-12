from copy import deepcopy

import pytest

from tools.aa8_street_baseline_v2 import StreetBaselineCandidate, partition_evidence


def observation():
    wagers = {str(s): None for s in range(8)}
    wagers.update({"0": "1", "1": "2", "2": "4", "4": "2"})
    return {"scene_supported": True, "unobstructed": True, "actor": 3,
            "actor_evidence": {"timer_suffix_verified": True},
            "title": "23", "center": {"value": "14"}, "wagers": wagers,
            "visibility": {s: {
                "status": "VISIBLE_COIN_CANDIDATE" if v is not None
                else "VISIBLE_EMPTY_CANDIDATE"} for s, v in wagers.items()}}


def context():
    return {"epoch": "observed_epoch", "new_street_observed": True,
            "no_earlier_actions": True, "collection_or_modal": False,
            "automated": False, "not_applicable": {"6": "empty"}}


def test_only_two_stable_frames_can_conditionally_initialize():
    state = StreetBaselineCandidate()
    assert state.observe(1, observation(), context())["status"] == "BASELINE_UNKNOWN"
    result = state.observe(2, observation(), context())
    assert result["status"] == "CONDITIONAL_BASELINE_CANDIDATE"
    assert result["street_price"] == "4"
    assert result["wagers"]["6"] == {"status": "NOT_APPLICABLE"}
    assert result["wagers"]["3"] == "0"
    assert not result["context_automated"]
    assert not result["canonical_verified"]


@pytest.mark.parametrize("key", ["new_street_observed", "no_earlier_actions", "epoch"])
def test_missing_causal_context_unknown(key):
    value = context()
    value[key] = None
    assert partition_evidence(observation(), value)["status"] == "BASELINE_UNKNOWN"


def test_coin_with_failed_ocr_not_filled_by_partition_math():
    value = observation()
    value["wagers"]["1"] = None
    assert partition_evidence(value, context())["status"] == "BASELINE_UNKNOWN"


def test_single_unknown_roi_cannot_become_zero():
    value = observation()
    value["visibility"]["5"]["status"] = "UNKNOWN"
    result = partition_evidence(value, context())
    assert result["wagers"] is None


def test_amount_partition_disagreement_abstains():
    value = observation()
    value["title"] = "24"
    assert partition_evidence(value, context())["reason"] == (
        "title_center_wager_partition_disagrees")


def test_modal_or_actor_loss_or_gap_invalidates_baseline():
    state = StreetBaselineCandidate()
    state.observe(1, observation(), context())
    state.observe(2, observation(), context())
    value = deepcopy(context())
    value["collection_or_modal"] = True
    assert state.observe(3, observation(), value)["status"] == "BASELINE_UNKNOWN"
    assert state.observe(4, observation(), context())["status"] == "BASELINE_UNKNOWN"
    assert state.observe(8, observation(), context())["status"] == "BASELINE_UNKNOWN"
