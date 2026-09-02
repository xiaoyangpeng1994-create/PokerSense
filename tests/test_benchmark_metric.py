"""Tests for benchmark atomic evaluation (REVISE v12)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "benchmark_vision.py"
_spec = importlib.util.spec_from_file_location("benchmark_vision", _TOOL)
benchmark_vision = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_vision"] = benchmark_vision
_spec.loader.exec_module(benchmark_vision)

FieldMetric = benchmark_vision.FieldMetric
evaluate = benchmark_vision.evaluate


def _entry(truth, pred, status, source="real-platform", sample_id="s",
           image_sha="h"):
    """Build a full entry with dataset metadata + a single pot field."""
    return {
        "_sample_id": sample_id,
        "_source": source,
        "_image_sha": image_sha,
        "pot": {"truth": truth, "pred": pred, "status": status},
    }


# ---------- FieldMetric basics ----------

def test_negative_fp_rate_is_na_when_no_negatives():
    m = FieldMetric(correct=10, total=10, unknown=0, conflict=0,
                    negative_total=0, negative_fp=0)
    assert m.negative_fp_rate is None


def test_negative_fp_rate_real_ratio_when_negatives_present():
    m = FieldMetric(correct=10, total=10, unknown=0, conflict=0,
                    negative_total=5, negative_fp=1)
    assert m.negative_fp_rate == 0.2


def test_conflict_rate():
    m = FieldMetric(correct=0, total=10, unknown=0, conflict=3,
                    negative_total=0, negative_fp=0)
    assert m.conflict_rate == 0.3


# ---------- status-based scoring (via atomic evaluate) ----------

def test_valid_correct():
    r = evaluate([_entry("10", "10", "valid")])
    assert r.metrics["pot"].correct == 1
    assert r.metrics["pot"].unknown == 0
    assert r.metrics["pot"].conflict == 0


def test_valid_wrong_value_is_incorrect():
    r = evaluate([_entry("10", "99", "valid")])
    assert r.metrics["pot"].correct == 0


def test_unknown_with_correct_candidate_is_reject():
    r = evaluate([_entry("10", "10", "unknown")])
    assert r.metrics["pot"].correct == 0
    assert r.metrics["pot"].unknown == 1


def test_conflict_with_correct_candidate_is_conflict():
    r = evaluate([_entry("10", "10", "conflict")])
    assert r.metrics["pot"].correct == 0
    assert r.metrics["pot"].conflict == 1


def test_evaluate_negative_sample_ratio():
    entries = [
        _entry(None, None, "unknown", sample_id="a", image_sha="ha"),
        _entry(None, "99", "valid", sample_id="b", image_sha="hb"),
    ]
    r = evaluate(entries)
    assert r.metrics["pot"].negative_total == 2
    assert r.metrics["pot"].negative_fp == 1
    assert r.metrics["pot"].negative_fp_rate == 0.5


# ---------- status / field schema validation ----------

def test_evaluate_rejects_invalid_status():
    with pytest.raises(ValueError):
        evaluate([_entry("10", "10", "bogus")])


def test_evaluate_rejects_unknown_field():
    with pytest.raises(ValueError):
        evaluate([{
            "_sample_id": "s", "_source": "real-platform", "_image_sha": "h",
            "not_a_field": {"truth": "10", "pred": "10", "status": "valid"},
        }])


# ---------- dataset validation (eligibility) via atomic evaluate ----------

def test_evaluate_synthetic_not_eligible():
    r = evaluate([_entry("10", "10", "valid", source="synthetic-render")])
    assert r.verdict.eligible is False
    assert r.verdict.reason == "dataset not eligible for acceptance"
    assert r.report["acceptance_eligible"] is False
    assert r.report["fields"]["pot"]["acceptance_target_met"] is False


def test_evaluate_mixed_source_returns_not_eligible():
    # mixing real + synthetic in one dataset is rejected (no partial eligibility)
    entries = [
        _entry("10", "10", "valid", source="real-platform",
               sample_id="a", image_sha="ha"),
        _entry("10", "10", "valid", source="synthetic-render",
               sample_id="b", image_sha="hb"),
    ]
    r = evaluate(entries)
    assert r.verdict.eligible is False
    assert r.verdict.reason == "mixed sample sources"


def test_evaluate_duplicate_sample_id_not_eligible():
    entries = [
        _entry("10", "10", "valid", sample_id="r-0", image_sha="ha"),
        _entry("10", "10", "valid", sample_id="r-0", image_sha="hb"),
    ]
    r = evaluate(entries)
    assert r.verdict.eligible is False
    assert r.verdict.reason == "duplicate sample_id"


def test_evaluate_duplicate_image_content_not_eligible():
    # distinct sample_ids, but SAME image hash -> duplicate content
    entries = [
        _entry("10", "10", "valid", sample_id="r-0", image_sha="same"),
        _entry("10", "10", "valid", sample_id="r-1", image_sha="same"),
    ]
    r = evaluate(entries)
    assert r.verdict.eligible is False
    assert r.verdict.reason == "duplicate image content"


def test_evaluate_real_unique_eligible():
    entries = [
        _entry("10", "10", "valid", sample_id="r-0", image_sha="ha"),
        _entry("10", "10", "valid", sample_id="r-1", image_sha="hb"),
    ]
    r = evaluate(entries)
    assert r.verdict.eligible is True
    assert r.verdict.reason is None


def test_evaluate_rejects_bad_source():
    with pytest.raises(ValueError):
        evaluate([_entry("10", "10", "valid", source="bogus")])


def test_evaluate_rejects_missing_image_sha():
    with pytest.raises(ValueError):
        e = _entry("10", "10", "valid")
        e.pop("_image_sha")
        evaluate([e])


# ---------- report reflects eligibility ----------

def test_report_synthetic_not_acceptance_eligible():
    r = evaluate([_entry("10", "10", "valid", source="synthetic-render")])
    f = r.report["fields"]["pot"]
    assert f["acceptance_target_met"] is False
    assert f["acceptance_reason"] == "dataset not eligible for acceptance"


def test_report_eligible_small_sample_not_accepted():
    # one real sample is too little for statistical acceptance
    r = evaluate([_entry("10", "10", "valid", sample_id="r-0", image_sha="h")])
    f = r.report["fields"]["pot"]
    assert f["acceptance_target_met"] is False
    assert f["acceptance_reason"] == ("insufficient sample size for "
                                      "statistical acceptance")


# ---------- deep immutability (REVISE v13 blocker 3) ----------

def test_frozen_targets_is_read_only():
    from benchmark_vision import FROZEN_TARGETS

    with pytest.raises(TypeError):
        FROZEN_TARGETS["pot"] = 0.5  # cannot reassign


def test_evaluation_result_report_is_deeply_immutable():
    r = evaluate([_entry("10", "10", "valid", source="synthetic-render")])
    # top-level report mapping is read-only
    with pytest.raises(TypeError):
        r.report["dataset"] = "x"
    # nested field dict is read-only
    with pytest.raises(TypeError):
        r.report["fields"]["pot"]["accuracy"] = 0.0
    # metrics mapping is read-only
    with pytest.raises(TypeError):
        r.metrics["pot"] = None


def test_evaluation_result_to_dict_returns_plain_dict():
    r = evaluate([_entry("10", "10", "valid", source="synthetic-render")])
    d = r.to_dict()
    assert isinstance(d, dict)
    assert isinstance(d["fields"], dict)
    assert isinstance(d["fields"]["pot"], dict)
    # returned dict is mutable (plain)
    d["dataset"] = "changed"
    assert d["dataset"] == "changed"
