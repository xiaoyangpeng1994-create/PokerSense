"""Tests for card layout, calibration, asset manifest, protocols (Task 7B)."""

from __future__ import annotations

import json

import pytest

from poker_engine.perceptual.vision import (
    BoardSlotLayout,
    CalibrationBins,
    CardSubROI,
    ConfidenceCalibrator,
    HeroSlotLayout,
    VisionAssetManifest,
)


# ---------- CardSubROI / layouts ----------

def test_card_subroi_valid():
    r = CardSubROI(x=0.1, y=0.2, width=0.3, height=0.4)
    assert r.width == 0.3


def test_card_subroi_range_rejected():
    with pytest.raises(ValueError):
        CardSubROI(x=1.5, y=0.2, width=0.3, height=0.4)
    with pytest.raises(ValueError):
        CardSubROI(x=0.1, y=0.2, width=0.0, height=0.4)


def _sub():
    return CardSubROI(x=0.0, y=0.0, width=0.2, height=1.0)


def test_board_layout_requires_5():
    with pytest.raises(ValueError):
        BoardSlotLayout(layout_id="b", version=1, slots=(_sub(), _sub(), _sub()))
    ok = BoardSlotLayout(layout_id="b", version=1, slots=tuple(_sub()
                         for _ in range(5)))
    assert len(ok.slots) == 5


def test_hero_layout_requires_2():
    with pytest.raises(ValueError):
        HeroSlotLayout(layout_id="h", version=1, slots=(_sub(),))
    ok = HeroSlotLayout(layout_id="h", version=1, slots=(_sub(), _sub()))
    assert len(ok.slots) == 2


def test_board_layout_json_roundtrip():
    from poker_engine.perceptual.vision.card_layout import (
        board_layout_to_json,
        board_layout_from_dict,
    )

    layout = BoardSlotLayout(layout_id="b", version=3,
                             slots=tuple(_sub() for _ in range(5)))
    data = json.loads(board_layout_to_json(layout))
    rt = board_layout_from_dict(data)
    assert rt == layout


# ---------- CalibrationBins / ConfidenceCalibrator ----------

def test_calibration_bins_monotonic():
    bins = CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.2, 0.8))
    assert bins.map(0.0) == 0.2
    assert bins.map(0.4) == 0.2
    assert bins.map(0.5) == 0.8
    assert bins.map(1.0) == 0.8


def test_calibration_bins_reject_non_monotonic():
    with pytest.raises(ValueError):
        CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.8, 0.2))


def test_calibration_bins_reject_bad_edges():
    with pytest.raises(ValueError):
        CalibrationBins(edges=(0.5, 0.0, 1.0), confidence=(0.5, 0.5))


@pytest.mark.parametrize(
    "bad_edge",
    [float("nan"), float("inf"), float("-inf"), 1.5, -0.1, True],
)
def test_calibration_bins_reject_non_finite_or_out_of_range_edge(bad_edge):
    # Any edge that is NaN, +/-Inf, out of [0,1], or bool must fail fast — it
    # must never be silently accepted and thereby mis-map a raw score.
    with pytest.raises((ValueError, TypeError)):
        CalibrationBins(edges=(0.0, bad_edge, 1.0), confidence=(0.5, 0.5))


def test_calibrator_abstain_floor():
    cal = ConfidenceCalibrator(
        name="card", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.5,
    )
    assert cal.should_abstain(0.3) is True
    assert cal.should_abstain(0.6) is False
    assert cal.calibrate(0.3).confidence == 0.1


def test_calibrator_reject_bad_abstain_floor():
    with pytest.raises(ValueError):
        ConfidenceCalibrator(
            name="x", version=1,
            bins=CalibrationBins(edges=(0.0, 1.0), confidence=(0.5,)),
            abstain_floor=1.5,
        )


# ---------- VisionAssetManifest ----------

def _manifest():
    return VisionAssetManifest(
        platform_id="wpk",
        layout_id="6max",
        card_layout_version=1,
        template_set_version="sha-abc",
        calibration_version=2,
        recognizer_versions={"card": "1", "amount": "1"},
    )


def test_manifest_json_roundtrip():
    m = _manifest()
    rt = VisionAssetManifest.from_json(m.to_json())
    assert rt == m
    assert rt.sha == m.sha


def test_manifest_recognizer_versions_immutable():
    m = _manifest()
    from types import MappingProxyType

    assert isinstance(m.recognizer_versions, MappingProxyType)
    with pytest.raises(TypeError):
        m.recognizer_versions["card"] = "9"  # type: ignore[index]


def test_manifest_sha_deterministic():
    m1 = _manifest()
    m2 = _manifest()
    assert m1.sha == m2.sha
    assert m1.to_json() == m2.to_json()


def test_manifest_schema_missing_field():
    d = {
        "platform_id": "wpk",
        "layout_id": "6max",
        # missing card_layout_version
        "template_set_version": "sha-abc",
        "calibration_version": 2,
        "recognizer_versions": {"card": "1"},
    }
    with pytest.raises(ValueError):
        VisionAssetManifest.from_dict(d)


def test_manifest_schema_wrong_type():
    d = {
        "platform_id": "wpk",
        "layout_id": "6max",
        "card_layout_version": "not-an-int",  # wrong type
        "template_set_version": "sha-abc",
        "calibration_version": 2,
        "recognizer_versions": {"card": "1"},
    }
    with pytest.raises(TypeError):
        VisionAssetManifest.from_dict(d)


def test_manifest_schema_bad_recognizer_versions():
    d = {
        "platform_id": "wpk",
        "layout_id": "6max",
        "card_layout_version": 1,
        "template_set_version": "sha-abc",
        "calibration_version": 2,
        "recognizer_versions": {"card": 1},  # value not str
    }
    with pytest.raises(TypeError):
        VisionAssetManifest.from_dict(d)


def test_manifest_schema_not_mapping():
    with pytest.raises(TypeError):
        VisionAssetManifest.from_dict([1, 2, 3])  # not a mapping
