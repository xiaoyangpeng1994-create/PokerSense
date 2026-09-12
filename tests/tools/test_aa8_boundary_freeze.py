from decimal import Decimal
import json

import pytest

from tools.sample_aa8_development import boundary_permission


def inputs(tmp_path, monkeypatch):
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"id": "test"}))
    monkeypatch.setattr("tools.sample_aa8_development.validate_freeze", lambda *a: [])
    return {"ranges": [{"role": "holdout_candidate", "start_inclusive": "600",
                        "end_exclusive": "820"}]}, path


def test_frozen_coarse_boundary_review_allowed(tmp_path, monkeypatch):
    plan, path = inputs(tmp_path, monkeypatch)
    assert boundary_permission(Decimal(600), Decimal(820), Decimal(5), False,
                               plan, path)["id"] == "test"


def test_dense_refinement_is_bounded(tmp_path, monkeypatch):
    plan, path = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="three seconds"):
        boundary_permission(Decimal(600), Decimal(820), Decimal(1), True, plan, path)
    boundary_permission(Decimal(610), Decimal(612), Decimal(1), True, plan, path)


def test_other_range_and_changed_freeze_rejected(tmp_path, monkeypatch):
    plan, path = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reserved holdout"):
        boundary_permission(Decimal(0), Decimal(10), Decimal(5), False, plan, path)
    monkeypatch.setattr("tools.sample_aa8_development.validate_freeze",
                        lambda *a: ["changed"])
    with pytest.raises(ValueError, match="changed freeze"):
        boundary_permission(Decimal(600), Decimal(610), Decimal(5), False, plan, path)
