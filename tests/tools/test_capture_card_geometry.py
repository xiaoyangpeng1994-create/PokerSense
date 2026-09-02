"""Tests for stage E geometry draft generation."""

from __future__ import annotations

import json

import pytest
from poker_engine.perceptual.vision.table_map import ROIKind, TableMap

from tools.capture_card_calibration.dataset import RoiMeasurement
from tools.capture_card_calibration.geometry import (
    GeometryError,
    build_indexed_slot_layout,
    build_relative_slot_layout,
    build_table_map_draft,
    normalized_roi,
    write_geometry_drafts,
)

CANVAS = (1080, 1920)
BOARD = RoiMeasurement("board_cards", 200, 800, 880, 1000, source_frame="f")


def _table_map_rows() -> list[RoiMeasurement]:
    rows = [
        RoiMeasurement("hero_cards", 400, 1500, 680, 1650),
        BOARD,
        RoiMeasurement("pot", 440, 480, 640, 540),
    ]
    rows.extend(
        RoiMeasurement(
            "stack", 40 + slot * 120, 700, 140 + slot * 120, 760, slot_id=slot
        )
        for slot in range(8)
    )
    rows.extend(
        RoiMeasurement(
            "action", 40 + slot * 120, 640, 140 + slot * 120, 690, slot_id=slot
        )
        for slot in range(8)
    )
    return rows


def _board_card_rows() -> list[RoiMeasurement]:
    return [
        RoiMeasurement("board_card", 200 + index * 136, 800,
                       200 + (index + 1) * 136, 1000)
        for index in range(5)
    ]


# --- coordinate conversion -------------------------------------------------


def test_normalized_roi_converts_pixels_to_fractions():
    x, y, width, height = normalized_roi(
        RoiMeasurement("pot", 0, 0, 540, 960), CANVAS
    )
    assert (x, y, width, height) == (0.0, 0.0, 0.5, 0.5)


def test_normalized_roi_rejects_measurements_outside_canvas():
    with pytest.raises(GeometryError, match="outside the 1080x1920 canvas"):
        normalized_roi(RoiMeasurement("pot", 0, 0, 1200, 100), CANVAS)


def test_normalized_roi_rejects_bad_canvas():
    with pytest.raises(GeometryError, match="positive int"):
        normalized_roi(RoiMeasurement("pot", 0, 0, 10, 10), (0, 100))


# --- TableMap draft --------------------------------------------------------


def test_table_map_draft_is_loadable_by_production_code():
    table_map = build_table_map_draft(
        _table_map_rows(),
        platform_id="wepoker_android_capture_card",
        layout_id="phone_a__card_b__uvc_1920x1080_30__canvas_1080x1920__v1",
        canvas=CANVAS,
    )
    assert isinstance(table_map, TableMap)
    restored = TableMap.from_json(table_map.to_json())
    assert restored == table_map


def test_table_map_draft_contains_every_slot():
    table_map = build_table_map_draft(
        _table_map_rows(),
        platform_id="p",
        layout_id="l",
        canvas=CANVAS,
    )
    kinds = [(roi.kind, roi.slot_id) for roi in table_map.rois]
    assert kinds.count((ROIKind.HERO_CARDS, None)) == 1
    assert kinds.count((ROIKind.BOARD_CARDS, None)) == 1
    assert kinds.count((ROIKind.POT, None)) == 1
    for slot in range(8):
        assert (ROIKind.STACK, slot) in kinds
        assert (ROIKind.ACTION, slot) in kinds
    assert not any(kind is ROIKind.ACTOR for kind, _ in kinds)


def test_table_map_draft_includes_optional_actor_roi():
    rows = _table_map_rows() + [
        RoiMeasurement("hero_actor", 300, 1700, 780, 1800)
    ]
    table_map = build_table_map_draft(
        rows, platform_id="p", layout_id="l", canvas=CANVAS
    )
    assert any(roi.kind is ROIKind.ACTOR for roi in table_map.rois)


def test_table_map_draft_rejects_missing_slot():
    rows = [row for row in _table_map_rows() if row.slot_id != 5]
    with pytest.raises(GeometryError, match="missing 5"):
        build_table_map_draft(rows, platform_id="p", layout_id="l", canvas=CANVAS)


def test_table_map_draft_rejects_duplicate_global_measurement():
    rows = _table_map_rows() + [RoiMeasurement("pot", 0, 0, 10, 10)]
    with pytest.raises(GeometryError, match="exactly once"):
        build_table_map_draft(rows, platform_id="p", layout_id="l", canvas=CANVAS)


def test_table_map_draft_rejects_unknown_field():
    rows = _table_map_rows() + [RoiMeasurement("potato", 0, 0, 10, 10)]
    with pytest.raises(GeometryError, match="unrecognized ROI measurement"):
        build_table_map_draft(rows, platform_id="p", layout_id="l", canvas=CANVAS)


def test_table_map_draft_rejects_empty_platform_id():
    with pytest.raises(GeometryError, match="platform_id"):
        build_table_map_draft(
            _table_map_rows(), platform_id="", layout_id="l", canvas=CANVAS
        )


def test_table_map_draft_rejects_slot_id_on_a_global_field():
    rows = [row for row in _table_map_rows() if row.field != "pot"]
    rows.append(RoiMeasurement("pot", 0, 0, 10, 10, slot_id=2))
    with pytest.raises(GeometryError, match="slot_id must be empty"):
        build_table_map_draft(rows, platform_id="p", layout_id="l", canvas=CANVAS)


# --- relative slot layouts -------------------------------------------------


def test_relative_slot_layout_is_fraction_of_parent():
    layout = build_relative_slot_layout(
        _board_card_rows(),
        kind="board",
        layout_id="l",
        parent=BOARD,
        expected_count=5,
    )
    assert layout["kind"] == "board"
    assert layout["status"] == "draft"
    first = layout["slots"][0]
    assert first["x"] == pytest.approx(0.0)
    assert first["width"] == pytest.approx(136 / 680)
    assert all(slot["height"] == pytest.approx(1.0) for slot in layout["slots"])


def test_relative_slot_layout_rejects_wrong_count():
    with pytest.raises(GeometryError, match="exactly 5"):
        build_relative_slot_layout(
            _board_card_rows()[:4],
            kind="board",
            layout_id="l",
            parent=BOARD,
            expected_count=5,
        )


def test_relative_slot_layout_rejects_slot_outside_parent():
    rows = _board_card_rows()
    rows[0] = RoiMeasurement("board_card", 0, 800, 136, 1000)
    with pytest.raises(GeometryError, match="not inside the parent"):
        build_relative_slot_layout(
            rows, kind="board", layout_id="l", parent=BOARD, expected_count=5
        )


def test_relative_slot_layout_rejects_slot_ids():
    rows = [
        RoiMeasurement("board_card", 200, 800, 336, 1000, slot_id=0)
        for _ in range(5)
    ]
    with pytest.raises(GeometryError, match="must not carry a slot_id"):
        build_relative_slot_layout(
            rows, kind="board", layout_id="l", parent=BOARD, expected_count=5
        )


# --- indexed slot layouts --------------------------------------------------


def _dealer_rows() -> list[RoiMeasurement]:
    return [
        RoiMeasurement(
            "dealer_search", 30 + slot * 120, 600, 150 + slot * 120, 680,
            slot_id=slot,
        )
        for slot in range(8)
    ]


def test_indexed_slot_layout_covers_every_slot():
    layout = build_indexed_slot_layout(
        _dealer_rows(),
        layout_id="l",
        canvas=CANVAS,
        expected_slots=tuple(range(8)),
    )
    assert [slot["slot_id"] for slot in layout["slots"]] == list(range(8))
    assert layout["status"] == "draft"


def test_indexed_slot_layout_reports_missing_slots():
    rows = [row for row in _dealer_rows() if row.slot_id != 4]
    with pytest.raises(GeometryError, match="missing slot ids: 4"):
        build_indexed_slot_layout(
            rows, layout_id="l", canvas=CANVAS, expected_slots=tuple(range(8))
        )


def test_indexed_slot_layout_rejects_duplicate_slots():
    rows = [
        RoiMeasurement("dealer_search", 0, 0, 10, 10, slot_id=0),
        RoiMeasurement("dealer_search", 0, 0, 20, 20, slot_id=0),
    ]
    with pytest.raises(GeometryError, match="duplicate slot_id"):
        build_indexed_slot_layout(
            rows, layout_id="l", canvas=CANVAS, expected_slots=(0,)
        )


def test_indexed_slot_layout_requires_slot_ids():
    with pytest.raises(GeometryError, match="require a slot_id"):
        build_indexed_slot_layout(
            [RoiMeasurement("dealer_search", 0, 0, 10, 10)],
            layout_id="l",
            canvas=CANVAS,
            expected_slots=(0,),
        )


# --- writing drafts --------------------------------------------------------


def test_write_geometry_drafts_writes_every_file(tmp_path):
    rows = (
        _table_map_rows()
        + _board_card_rows()
        + _dealer_rows()
        + [
            RoiMeasurement(
                "empty_slot", 30 + slot * 120, 560, 150 + slot * 120, 620,
                slot_id=slot,
            )
            for slot in range(1, 8)
        ]
    )
    written = write_geometry_drafts(
        rows,
        tmp_path,
        platform_id="wepoker_android_capture_card",
        layout_id="l",
        canvas=CANVAS,
    )
    assert set(written) == {
        "table_map",
        "board_slot_layout",
        "dealer_slot_layout",
        "empty_slot_layout",
    }
    payload = json.loads(written["table_map"].read_text(encoding="utf-8"))
    assert payload["platform_id"] == "wepoker_android_capture_card"
    assert payload["reference_size"] == [1080, 1920]
    TableMap.from_dict(payload)


def test_write_geometry_drafts_skips_absent_layouts(tmp_path):
    written = write_geometry_drafts(
        _table_map_rows(),
        tmp_path,
        platform_id="p",
        layout_id="l",
        canvas=CANVAS,
    )
    assert set(written) == {"table_map"}
