import copy

import pytest

from tools.aa8_gold_registry_v2 import inspect_gold
from tools.aa8_holdout_predict import FIELDS


def inputs():
    registry = {"role": "holdout", "audit_sha256": "audit",
                "candidate_range_seconds": ["300", "600"], "hands": [{
                    "id": "h", "first_frame": 1, "last_frame": 2,
                    "preroll_first_frame": 0, "boundaries_verified": True,
                    "opportunities_reviewed": True,
                    "action_opportunity_frames": [1, 2]}]}
    manifest = {"audit_sha256": "audit", "samples": [{
        "global_frame": f, "role": "holdout", "pts_seconds": str(301 + f / 30),
        "sha256": "a" * 64} for f in range(3)]}
    values = {"actor": 0, "street_wagers": dict.fromkeys(map(str, range(8)), "0"),
              "actions": [], "pot": "0",
              "stacks": dict.fromkeys(map(str, range(8)), "100"),
              "street": "preflop", "hand": "h", "hero_cards": ["Ah", "Kd"],
              "board_cards": [],
              "participation": dict.fromkeys(map(str, range(8)), "active"),
              "insurance": {"active": False}, "mushroom": {"active": False},
              "bomb": {"active": False}}
    labels = [{"frame": f, "hand": "h", "source_sha256": "a" * 64,
               "source_reviewed": True, "reviewer": "synthetic",
               "decision_opportunity": True, "fields": {
                   k: {"status": "KNOWN", "value": copy.deepcopy(values[k])}
                   for k in FIELDS}}
              for f in (1, 2)]
    return registry, manifest, labels


def test_complete_synthetic_gold_never_grants_visual_pass():
    result = inspect_gold(*inputs())
    assert result["status"] == "GOLD_READY_NOT_PASS"
    assert not result["full_visual_acceptance"]
    assert not result["predictions_read"]


def test_sparse_gold_not_complete_hand():
    registry, manifest, labels = inputs()
    result = inspect_gold(registry, manifest, labels[:1])
    assert result["status"] == "PARTIAL"
    assert "sparse_gold_not_full_frame_truth" in result["failures"]
    assert result["owned_frames"] == 2


def test_all_unknown_rows_are_inventory_not_truth():
    registry, manifest, labels = inputs()
    for row in labels:
        row["fields"] = {k: {"status": "UNKNOWN", "value": None} for k in FIELDS}
        row["decision_opportunity"] = None
    result = inspect_gold(registry, manifest, labels)
    assert result["gold_frames"] == result["owned_frames"] == 2
    assert result["status"] == "PARTIAL"
    assert all(c["known"] == 0 for c in result["coverage"].values())


@pytest.mark.parametrize("mutation", ["duplicate", "preroll", "hash", "hand"])
def test_source_identity_failures(mutation):
    registry, manifest, labels = inputs()
    if mutation == "duplicate":
        labels.append(copy.deepcopy(labels[0]))
    elif mutation == "preroll":
        labels[0]["frame"] = 0
    elif mutation == "hash":
        labels[0]["source_sha256"] = "b" * 64
    else:
        labels[0]["hand"] = "other"
    with pytest.raises(ValueError):
        inspect_gold(registry, manifest, labels)


def test_action_opportunity_cannot_be_masked():
    registry, manifest, labels = inputs()
    labels[0]["fields"]["actor"].update(required=False, exemption={
        "reason": "transition", "source_reviewed": True, "source_sha256": "a" * 64})
    result = inspect_gold(registry, manifest, labels)
    assert "invalid_gold_exemption:actor" in result["failures"]


def test_missing_fields_not_implicitly_optional():
    registry, manifest, labels = inputs()
    del labels[0]["fields"]["board_cards"]
    assert "missing_field_record:board_cards" in inspect_gold(
        registry, manifest, labels)["failures"]
