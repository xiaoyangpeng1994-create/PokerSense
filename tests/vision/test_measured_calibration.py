"""Tests for evidence-derived confidence calibration.

The point of :class:`MeasuredCalibration` is that a confidence number must
be earned by measurement rather than chosen. These tests pin the properties
that make that true, so a future edit cannot quietly reintroduce a
hand-picked constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_engine.perceptual.vision.calibration import (
    MeasuredCalibration,
    wilson_lower_bound_95,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEPOKER_CALIBRATION = (
    _REPO_ROOT / "configs" / "vision" / "wepoker" / "calibration.json"
)


def _measurement(**overrides) -> MeasuredCalibration:
    kwargs = dict(
        samples=62,
        correct=62,
        readable_score_floor=0.5,
        unreadable_score_ceiling=0.335,
        source="test measurement",
    )
    kwargs.update(overrides)
    return MeasuredCalibration(**kwargs)


def test_perfect_score_does_not_justify_perfect_confidence():
    """62/62 correct is not evidence of 100% accuracy."""
    assert wilson_lower_bound_95(62, 62) < 0.97
    assert wilson_lower_bound_95(62, 62) > 0.90


def test_more_samples_justify_more_confidence():
    """The only way to claim a higher number is to measure more."""
    small = wilson_lower_bound_95(50, 50)
    large = wilson_lower_bound_95(500, 500)
    assert large > small
    assert wilson_lower_bound_95(600, 600) > 0.995


def test_errors_lower_what_can_be_claimed():
    assert wilson_lower_bound_95(60, 62) < wilson_lower_bound_95(62, 62)


def test_zero_samples_justify_nothing():
    assert wilson_lower_bound_95(0, 0) == 0.0


@pytest.mark.parametrize(
    "correct,total",
    [(-1, 10), (11, 10), (5, -1)],
)
def test_impossible_counts_are_rejected(correct, total):
    with pytest.raises(ValueError):
        wilson_lower_bound_95(correct, total)


def test_calibrator_reports_only_the_justified_confidence():
    measured = _measurement()
    calibrator = measured.to_calibrator("card")
    readable = calibrator.calibrate(0.95).confidence
    assert readable == pytest.approx(measured.justified_confidence)
    assert readable < 1.0


def test_unreadable_scores_get_no_confidence_and_abstain():
    calibrator = _measurement().to_calibrator("card")
    # 0.335 was the highest score any non-card produced.
    assert calibrator.calibrate(0.335).confidence == 0.0
    assert calibrator.should_abstain(0.335)
    assert not calibrator.should_abstain(0.95)


def test_overlapping_readable_and_unreadable_ranges_are_rejected():
    """If non-cards score as high as cards, the two are not separable."""
    with pytest.raises(ValueError):
        _measurement(readable_score_floor=0.3, unreadable_score_ceiling=0.4)


def test_measurement_must_document_its_source():
    with pytest.raises(ValueError):
        _measurement(source="")


def test_committed_wepoker_calibration_is_self_consistent():
    """The shipped measurement must describe a separable, honest calibration."""
    data = json.loads(_WEPOKER_CALIBRATION.read_text())["card"]
    measured = MeasuredCalibration(
        samples=data["samples"],
        correct=data["correct"],
        readable_score_floor=data["readable_score_floor"],
        unreadable_score_ceiling=data["unreadable_score_ceiling"],
        source=data["source"],
    )
    assert measured.correct <= measured.samples
    assert measured.justified_confidence < 1.0
    # The claimed floor must sit inside the observed separation gap.
    assert measured.unreadable_score_ceiling < measured.readable_score_floor
