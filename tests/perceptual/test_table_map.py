"""Tests for TableMap / ROI contracts."""

from __future__ import annotations

import pytest

from poker_engine.perceptual import ROIKind, ROI, TableMap, TableMapError


def test_roi_global_slot_none():
    r = ROI(kind=ROIKind.HERO_CARDS, x=0.1, y=0.2, width=0.3, height=0.4)
    assert r.slot_id is None
    assert r.kind is ROIKind.HERO_CARDS


def test_roi_per_seat_slot_required():
    with pytest.raises(TypeError):
        ROI(kind=ROIKind.STACK, x=0.1, y=0.2, width=0.3, height=0.4)  # no slot
    r = ROI(kind=ROIKind.STACK, x=0.1, y=0.2, width=0.3, height=0.4, slot_id=2)
    assert r.slot_id == 2


def test_roi_global_slot_forbidden():
    with pytest.raises(ValueError):
        ROI(kind=ROIKind.POT, x=0.1, y=0.2, width=0.3, height=0.4, slot_id=1)


def test_roi_normalized_range():
    with pytest.raises(ValueError):
        ROI(kind=ROIKind.POT, x=1.5, y=0.2, width=0.3, height=0.4)
    with pytest.raises(ValueError):
        ROI(kind=ROIKind.POT, x=0.1, y=-0.2, width=0.3, height=0.4)
    with pytest.raises(ValueError):
        ROI(kind=ROIKind.POT, x=0.1, y=0.2, width=0.0, height=0.4)


def test_roi_reject_bool_coord():
    with pytest.raises(TypeError):
        ROI(kind=ROIKind.POT, x=True, y=0.2, width=0.3, height=0.4)


def test_roi_immutable():
    r = ROI(kind=ROIKind.POT, x=0.1, y=0.2, width=0.3, height=0.4)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        r.x = 0.9  # type: ignore[misc]


def _sample_map() -> TableMap:
    return TableMap(
        platform_id="p",
        layout_id="l",
        reference_size=(1920, 1080),
        rois=(
            ROI(kind=ROIKind.HERO_CARDS, x=0.45, y=0.75, width=0.1, height=0.06),
            ROI(kind=ROIKind.STACK, x=0.1, y=0.8, width=0.08, height=0.04, slot_id=0),
        ),
    )


def test_reference_aspect_ratio_derived():
    m = _sample_map()
    assert m.reference_aspect_ratio == 1920 / 1080
    # not a persisted field
    assert "reference_aspect_ratio" not in m.to_dict()


def test_table_map_rois_immutable():
    m = _sample_map()
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        m.rois = ()  # type: ignore[misc]


def test_table_map_invalid_reference_size():
    with pytest.raises(ValueError):
        TableMap(platform_id="p", layout_id="l", reference_size=(0, 1080))
    with pytest.raises(ValueError):
        TableMap(platform_id="p", layout_id="l", reference_size=(1920, -1080))


def test_table_map_bad_tolerance():
    with pytest.raises(ValueError):
        TableMap(
            platform_id="p", layout_id="l",
            reference_size=(1920, 1080), aspect_tolerance=1.5,
        )
    with pytest.raises(TypeError):
        TableMap(
            platform_id="p", layout_id="l",
            reference_size=(1920, 1080), aspect_tolerance=True,
        )


def test_serialization_roundtrip():
    m = _sample_map()
    data = json_roundtrip(m.to_dict())
    m2 = TableMap.from_dict(data)
    assert m2 == m
    assert m2.reference_aspect_ratio == m.reference_aspect_ratio
    assert m2.rois == m.rois


def test_serialization_json_stable():
    m = _sample_map()
    assert m.to_json() == m.to_json()


def test_unknown_roi_kind_rejected():
    data = _sample_map().to_dict()
    data["rois"][0]["kind"] = "bogus"
    with pytest.raises(ValueError):
        TableMap.from_dict(data)


def test_duplicate_roi_key_rejected():
    # duplicate (POT, None)
    with pytest.raises(TableMapError):
        TableMap(
            platform_id="p", layout_id="l", reference_size=(1920, 1080),
            rois=(
                ROI(kind=ROIKind.POT, x=0.1, y=0.1, width=0.1, height=0.1),
                ROI(kind=ROIKind.POT, x=0.5, y=0.5, width=0.1, height=0.1),
            ),
        )


def test_duplicate_per_seat_roi_key_rejected():
    # duplicate (STACK, 0)
    with pytest.raises(TableMapError):
        TableMap(
            platform_id="p", layout_id="l", reference_size=(1920, 1080),
            rois=(
                ROI(kind=ROIKind.STACK, x=0.1, y=0.1, width=0.1, height=0.1, slot_id=0),
                ROI(kind=ROIKind.STACK, x=0.5, y=0.5, width=0.1, height=0.1, slot_id=0),
            ),
        )


def test_unsupported_schema_version_rejected():
    data = _sample_map().to_dict()
    data["schema_version"] = 999
    with pytest.raises(TableMapError):
        TableMap.from_dict(data)


def json_roundtrip(d):
    import json

    return json.loads(json.dumps(d))
