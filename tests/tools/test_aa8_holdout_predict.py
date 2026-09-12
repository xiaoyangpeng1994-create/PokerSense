import json

import pytest

from tools.aa8_holdout_predict import bound_file, normalize_fields, validate_rows
from tools.aa8_holdout_plan import sha


def registry():
    return {"role": "holdout", "audit_sha256": "audit",
            "allowed_start_seconds": 600., "allowed_end_seconds": 820.,
            "hands": [{"id": "blind_hand", "first_frame": 101,
                       "last_frame": 103, "preroll_first_frame": 100,
                       "boundaries_verified": True}]}


def rows():
    return {f: {"global_frame": f, "pts_seconds": str(601 + f / 30),
                "role": "holdout", "sha256": "a" * 64} for f in range(100, 104)}


def test_complete_hand_plus_preroll_metadata_only():
    assert validate_rows(rows(), registry(), "audit") == {101, 102, 103}


@pytest.mark.parametrize("change", ["dev", "gap", "outside", "nonmonotonic", "hash"])
def test_metadata_failures_are_rejected(change):
    values = rows()
    if change == "dev":
        values[101]["role"] = "development"
    elif change == "gap":
        del values[101]
    elif change == "outside":
        values[103]["pts_seconds"] = "821"
    elif change == "nonmonotonic":
        values[103]["pts_seconds"] = values[102]["pts_seconds"]
    else:
        values[103]["sha256"] = "missing"
    with pytest.raises(ValueError):
        validate_rows(values, registry(), "audit")


def test_unverified_boundaries_rejected():
    meta = registry()
    meta["hands"][0]["boundaries_verified"] = False
    with pytest.raises(ValueError, match="bounds"):
        validate_rows(rows(), meta, "audit")


def test_freeze_binding_cannot_change_or_substitute_model(tmp_path):
    model = tmp_path / "model.txt"
    model.write_text("original")
    freeze = {"files": [{"path": str(model), "sha256": sha(model)}]}
    assert bound_file(model, freeze) == model.resolve()
    model.write_text("changed")
    with pytest.raises(ValueError, match="freeze"):
        bound_file(model, freeze)


def test_normalization_never_fills_legal_or_rule_unknowns():
    row = {"scene_supported": True, "current_actor": 0,
           "pot": {"value": "474"}, "street_wagers": {"0": "63", "3": "221"},
           "special_modes": {"insurance": "VISIBLE", "mushroom_rule": "3BB",
                             "critical_hit_rule": "7BB"},
           "cards": {"hero": ["9d", "Qd"],
                     "board_slots": ["7h", "5h", "2s", "Tc", None]},
           "board_count": 4}
    fields = normalize_fields(row)
    assert fields["actor"]["value"] == 0
    assert fields["board_cards"]["value"] == ["7h", "5h", "2s", "Tc"]
    for field in ("street_wagers", "actions", "hand", "participation", "street",
                  "mushroom", "bomb"):
        assert fields[field]["status"] == "UNKNOWN"
    json.dumps(fields)


def test_unsupported_scene_cannot_promote_stale_values():
    fields = normalize_fields({"scene_supported": False, "current_actor": 0,
                               "pot": {"value": "100"}})
    assert all(v["status"] == "UNKNOWN" for v in fields.values())
