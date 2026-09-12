from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from tools.aa8_holdout_plan import audit_existing, boundary_plan, sha, validate_freeze


def freeze(tmp_path):
    files = []
    for kind in ("implementation", "model", "parameters", "training"):
        path = tmp_path / (kind + ".json")
        path.write_text("[]", encoding="utf-8")
        files.append({"path": path.name, "kind": kind, "sha256": sha(path)})
    return {"frozen_at_utc": "2020-01-01T00:00:00Z", "files": files,
            "exposures": [{"used_for": "training", "role": "development",
                           "artifact_sha256": files[-1]["sha256"]}]}


def test_freeze_hash_rechecked_not_only_declared(tmp_path):
    value = freeze(tmp_path)
    assert not validate_freeze(value, tmp_path, "2020-01-01T00:01:00Z")
    (tmp_path / "model.json").write_text("changed")
    assert "freeze_file_hash_mismatch:model.json" in validate_freeze(value, tmp_path)


@pytest.mark.parametrize("stamp", ["bad", "2020-01-01", "2999-01-01T00:00:00Z",
                                   "2020-01-01T00:00:00+08:00"])
def test_invalid_future_or_nonutc_freeze(tmp_path, stamp):
    value = freeze(tmp_path)
    value["frozen_at_utc"] = stamp
    assert validate_freeze(value, tmp_path)


def test_prediction_must_be_later_than_freeze(tmp_path):
    assert "freeze_not_before_prediction" in validate_freeze(
        freeze(tmp_path), tmp_path, "2020-01-01T00:00:00Z")


def test_sorted_unique_paths(tmp_path):
    value = freeze(tmp_path)
    value["files"].reverse()
    assert "freeze_paths_not_sorted_unique" in validate_freeze(value, tmp_path)


@pytest.mark.parametrize("role", ["holdout", "unknown", "holdout_candidate"])
def test_holdout_never_training(tmp_path, role):
    value = freeze(tmp_path)
    value["exposures"][0]["role"] = role
    result = validate_freeze(value, tmp_path)
    assert "holdout_or_unknown_exposure_used_for_tuning" in result


def test_boundary_review_cannot_be_shared_with_tuning(tmp_path):
    value = freeze(tmp_path)
    value["exposures"].append({"used_for": "boundary_review", "role": "holdout",
                               "shared_with_tuning": True})
    assert "boundary_review_leakage" in validate_freeze(value, tmp_path)


def test_real_split_plan_only_candidates(tmp_path):
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa8_recording_split_plan_20260909.json")
    result = audit_existing(path, [])
    assert result["status"] == "PARTIAL"
    assert "split_plan_has_zero_verified_independent_hands" in result["failures"]
    plan = boundary_plan(json.loads(path.read_text()), freeze(tmp_path), tmp_path,
                         datetime(2021, 1, 1, tzinfo=timezone.utc))
    assert plan["candidate_ranges"] == [
        {"start_inclusive": "600", "end_exclusive": "820", "role": "holdout_candidate"}]
    assert plan["status"] == "READY_FOR_BOUNDARY_REVIEW_ONLY"
    assert not plan["prediction_permitted"]
    assert not plan["full_visual_acceptance"]


def test_development_report_cannot_be_independent(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"verified_independent_hands": 0}))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"independent_holdout": False,
                                  "full_visual_acceptance": False,
                                  "frame_count": 2057}))
    result = audit_existing(split, [report])
    assert result["status"] == "PARTIAL"
    assert result["reports"][0]["frame_count"] == 2057
    assert not result["media_consumed"]
