"""Tests for the capture-card calibration data layer: hashing, schemas,
layout_id construction and dataset I/O.
"""

from __future__ import annotations

import pytest

from tools.capture_card_calibration import dataset, hashing, layout_id, schema
from tools.capture_card_calibration.schema import (
    CompletedAction,
    DeviceAndCapture,
    FieldMetrics,
    FieldValue,
    FrameLabel,
    LabelStatus,
    Occupancy,
    Review,
    Scene,
    SchemaError,
    SlotLabel,
    Street,
)

# --- helpers ---------------------------------------------------------------


def _slot(slot_id: int, **overrides: object) -> SlotLabel:
    data = {
        "slot_id": slot_id,
        "occupancy": FieldValue.valid(Occupancy.OCCUPIED.value).to_dict(),
        "stack": FieldValue.valid(100 + slot_id).to_dict(),
        "dealer": FieldValue.valid(False).to_dict(),
        "completed_action": FieldValue.unknown().to_dict(),
        "current_actor": FieldValue.unknown().to_dict(),
    }
    data.update(overrides)
    return SlotLabel.from_dict(data)


def _label(**overrides: object) -> FrameLabel:
    kwargs = {
        "frame": "session_001__t_00000000__f_000000__abcdef123456.png",
        "sha256": "a" * 64,
        "session_id": "session_001",
        "hand_id": "session_001_hand_0001",
        "timestamp_ms": 0,
        "stable": True,
        "scene": Scene.TABLE,
        "hero_cards": FieldValue.valid(["TS", "JD"]),
        "board_cards": FieldValue.valid(["4H", "KC", "TS"]),
        "street": FieldValue.valid(Street.FLOP.value),
        "pot": FieldValue.valid(120),
        "slots": tuple(_slot(index) for index in range(8)),
        "review": Review(reviewer="tester"),
    }
    kwargs.update(overrides)
    return FrameLabel(**kwargs)


def _device(**sections: dict) -> DeviceAndCapture:
    base = {
        "phone": {"model": "Pixel 8", "android_version": "14"},
        "app": {"name": "WePoker", "version": "1.2.3"},
        "video_adapter": {"model": "Adapter X"},
        "capture_card": {"model": "Card Y"},
        "uvc": {"frame_size": [1920, 1080], "fps": 30},
        "recording": {"container": "mkv", "codec": "h264"},
    }
    base.update(sections)
    return DeviceAndCapture(**base)


def _metrics(**overrides: object) -> FieldMetrics:
    kwargs = {
        "field": "pot",
        "algorithm_version": "pot-v1",
        "threshold": 0.8,
        "train_samples": 10,
        "calibration_positive_samples": 5,
        "calibration_negative_samples": 5,
        "validation_positive_samples": 10,
        "validation_negative_samples": 10,
        "correct_valid": 9,
        "false_valid": 1,
        "unknown_on_positive": 1,
        "conflict": 0,
        "lowest_accepted_positive": 0.82,
        "highest_rejected_negative": 0.61,
        "source_sessions": ("session_001",),
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "template_sha256": "c" * 64,
    }
    kwargs.update(overrides)
    return FieldMetrics(**kwargs)


# --- hashing ---------------------------------------------------------------


def test_sha256_bytes_matches_known_vector():
    assert hashing.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_json_ignores_key_order():
    assert hashing.sha256_json({"a": 1, "b": 2}) == hashing.sha256_json(
        {"b": 2, "a": 1}
    )


def test_write_and_verify_sha256sums(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    manifest = hashing.write_sha256sums(tmp_path)

    assert manifest.is_file()
    body = manifest.read_text(encoding="utf-8")
    assert "  a.txt" in body
    assert "  nested/b.txt" in body
    assert hashing.verify_sha256sums(tmp_path) == []


def test_verify_detects_tampering(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    hashing.write_sha256sums(tmp_path)
    (tmp_path / "a.txt").write_text("tampered", encoding="utf-8")

    problems = hashing.verify_sha256sums(tmp_path)

    assert problems == ["hash mismatch: a.txt"]


def test_verify_reports_missing_manifest(tmp_path):
    assert hashing.verify_sha256sums(tmp_path) == ["missing manifest: SHA256SUMS"]


def test_verify_reports_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    hashing.write_sha256sums(tmp_path)
    (tmp_path / "a.txt").unlink()

    assert hashing.verify_sha256sums(tmp_path) == ["missing file: a.txt"]


# --- FieldValue ------------------------------------------------------------


def test_valid_field_requires_a_value():
    with pytest.raises(SchemaError, match="VALID field must carry"):
        FieldValue.valid(None)


def test_unknown_field_rejects_a_value():
    with pytest.raises(SchemaError, match="must not carry a value"):
        FieldValue(LabelStatus.UNKNOWN, 120)


def test_conflict_field_rejects_a_value():
    with pytest.raises(SchemaError, match="must not carry a value"):
        FieldValue(LabelStatus.CONFLICT, "FLOP")


def test_field_value_accepts_plain_status_string():
    assert FieldValue("VALID", 5).status is LabelStatus.VALID


def test_field_value_roundtrip():
    field = FieldValue.valid(120)
    assert FieldValue.from_dict(field.to_dict()) == field


# --- cards and enums -------------------------------------------------------


@pytest.mark.parametrize("code", ["TS", "2C", "AH", "9D"])
def test_is_card_code_accepts_valid_codes(code):
    assert schema.is_card_code(code)


@pytest.mark.parametrize("code", ["1S", "TX", "TS ", "", "ts"])
def test_is_card_code_rejects_invalid_codes(code):
    assert not schema.is_card_code(code)


def test_validate_card_list_enforces_count():
    with pytest.raises(SchemaError, match="must hold 2-2 cards"):
        schema.validate_card_list("hero", ["TS"], min_cards=2, max_cards=2)


def test_frame_label_rejects_opponent_actor():
    with pytest.raises(SchemaError, match="only be HERO"):
        _label(slots=tuple(
            _slot(0, current_actor=FieldValue.valid("VILLAIN").to_dict())
            if index == 0
            else _slot(index)
            for index in range(8)
        ))


# --- FrameLabel ------------------------------------------------------------


def test_frame_label_requires_all_eight_slots():
    with pytest.raises(SchemaError, match="must list all 8"):
        _label(slots=(_slot(0), _slot(1)))


def test_frame_label_rejects_duplicate_slot_ids():
    with pytest.raises(SchemaError, match="exactly once"):
        _label(slots=tuple(_slot(index % 7) for index in range(8)))


def test_frame_label_roundtrip():
    label = _label()
    restored = FrameLabel.from_json(label.to_json())
    assert restored == label


def test_frame_label_group_id_is_session_scoped():
    assert _label().group_id == "session_001::session_001_hand_0001"


def test_frame_label_rejects_bad_sha256():
    with pytest.raises(SchemaError, match="64 lowercase hex"):
        _label(sha256="abc")


def test_frame_label_rejects_unknown_scene():
    with pytest.raises(SchemaError):
        _label(scene="not_a_scene")


def test_frame_label_rejects_placeholder_reviewer():
    with pytest.raises(SchemaError, match="real reviewer"):
        _label(review=Review(reviewer="REPLACE_ME"))


def test_frame_label_rejects_unreviewed_method():
    with pytest.raises(SchemaError, match="review.method"):
        _label(review=Review(reviewer="tester", method="guessed_from_logic"))


def test_board_cards_accept_one_to_five_cards():
    assert _label(board_cards=FieldValue.valid(["TS"])).board_cards.status is (
        LabelStatus.VALID
    )
    with pytest.raises(SchemaError, match="1-5 cards"):
        _label(board_cards=FieldValue.valid(["TS", "JD", "4H", "KC", "AS", "2D"]))


# --- DeviceAndCapture ------------------------------------------------------


def test_device_ready_when_no_placeholders():
    assert _device().is_ready


def test_device_placeholders_are_reported_with_paths():
    device = _device(uvc={"frame_size": [1920, 1080], "pixel_format": "REPLACE_ME"})
    assert device.placeholders() == ["uvc.pixel_format"]


def test_device_require_ready_fails_closed():
    device = _device(capture_card={"model": "REPLACE_ME"})
    with pytest.raises(SchemaError, match="not frozen"):
        device.require_ready()


def test_device_require_min_sessions():
    # Floor is now 2 (owner-authorised waiver of the third session).
    with pytest.raises(SchemaError, match="at least 2 independent"):
        _device().require_min_sessions()


def test_device_roundtrip():
    device = _device(sessions=({"session_id": "session_001"},))
    assert DeviceAndCapture.from_json(device.to_json()) == device


def test_device_rejects_empty_section():
    with pytest.raises(SchemaError, match="non-empty object"):
        _device(phone={})


# --- FieldMetrics ----------------------------------------------------------


def test_metrics_zero_false_valid():
    assert _metrics(false_valid=0).zero_false_valid
    assert not _metrics(false_valid=1).zero_false_valid


def test_metrics_recall_counts_unknown_as_missed():
    metrics = _metrics(validation_positive_samples=10, unknown_on_positive=3)
    assert metrics.recall_on_validation == pytest.approx(0.7)


def test_metrics_require_source_sessions():
    with pytest.raises(SchemaError, match="source_sessions"):
        _metrics(source_sessions=())


def test_metrics_reject_placeholder_algorithm_version():
    with pytest.raises(SchemaError, match="REPLACE_ME"):
        _metrics(algorithm_version="REPLACE_ME")


def test_metrics_reject_short_hash():
    with pytest.raises(SchemaError, match="code_sha256"):
        _metrics(code_sha256="abc")


# --- layout_id -------------------------------------------------------------


def test_slugify_lowercases_and_folds_non_ascii():
    assert layout_id.slugify("Redmi K60") == "redmi_k60"
    assert layout_id.slugify("绿联 CM716") == "cm716"


def test_slugify_collapses_separators():
    assert layout_id.slugify("  A--B / C  ") == "a_b_c"


def test_slugify_number_turns_decimal_point_into_underscore():
    assert layout_id.slugify_number(30) == "30"
    assert layout_id.slugify_number(29.97) == "29_97"


def test_build_layout_id_follows_section_six():
    built = layout_id.build_layout_id(
        phone_model="Redmi K60",
        capture_card_model="CM716",
        uvc_width=1920,
        uvc_height=1080,
        fps=29.97,
        canvas_width=1080,
        canvas_height=1920,
    )
    assert built == (
        "phone_redmi_k60__card_cm716__uvc_1920x1080_29_97"
        "__canvas_1080x1920__v1"
    )
    assert layout_id.is_valid_layout_id(built)


def test_layout_id_changes_when_geometry_changes():
    common = dict(
        phone_model="Redmi K60",
        capture_card_model="CM716",
        uvc_width=1920,
        uvc_height=1080,
        fps=30,
    )
    first = layout_id.build_layout_id(
        **common, canvas_width=1080, canvas_height=1920
    )
    second = layout_id.build_layout_id(
        **common, canvas_width=1080, canvas_height=1918
    )
    assert first != second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"uvc_width": 0},
        {"canvas_height": -1},
        {"fps": 0},
        {"version": 0},
    ],
)
def test_build_layout_id_rejects_invalid_dimensions(kwargs):
    params = dict(
        phone_model="A",
        capture_card_model="B",
        uvc_width=1920,
        uvc_height=1080,
        fps=30,
        canvas_width=1080,
        canvas_height=1920,
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        layout_id.build_layout_id(**params)


def test_is_valid_layout_id_rejects_free_text():
    assert not layout_id.is_valid_layout_id("my-layout")


# --- dataset scaffolding ---------------------------------------------------


def test_create_skeleton_builds_every_directory(tmp_path):
    root = dataset.create_skeleton(tmp_path / "calib")
    for relative in dataset.DATASET_DIRECTORIES:
        assert (root / relative).is_dir()
    assert "PRIVATE" in (root / "README.md").read_text(encoding="utf-8")


def test_device_template_is_not_ready(tmp_path):
    path = dataset.write_device_template(tmp_path / "source" / "device.json")
    device = DeviceAndCapture.from_json(path.read_text(encoding="utf-8"))
    assert not device.is_ready
    assert device.placeholders()


def test_frames_jsonl_roundtrip(tmp_path):
    path = tmp_path / "frames.jsonl"
    labels = [_label(), _label(hand_id="hand_0002")]
    assert dataset.write_frames_jsonl(path, labels) == 2
    assert dataset.read_frames_jsonl(path) == labels


def test_read_frames_jsonl_reports_the_offending_line(tmp_path):
    path = tmp_path / "frames.jsonl"
    path.write_text(_label().to_json() + "\n{}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=":2: invalid label line"):
        dataset.read_frames_jsonl(path)


def test_frame_filename_follows_section_seven():
    name = dataset.frame_filename("session_001", 1234, 37, "a" * 64)
    assert name == "session_001__t_00001234__f_000037__aaaaaaaaaaaa.png"


def test_frame_manifest_roundtrip(tmp_path):
    entry = dataset.FrameEntry(
        file="a.png",
        sha256="a" * 64,
        source_video_id="session_001.mkv",
        timestamp_ms=500,
        source_frame=15,
        normalization_version="capture-card-normalization-v1",
        stable=True,
        scene=Scene.TABLE,
        group_id="session_001::hand_0001",
        reason="steady_state",
    )
    path = dataset.write_frame_manifest(tmp_path / "manifest.json", [entry])
    assert dataset.read_frame_manifest(path) == [entry]


def test_frame_entry_requires_a_sampling_reason():
    with pytest.raises(SchemaError, match="sampling reason"):
        dataset.FrameEntry(
            file="a.png",
            sha256="a" * 64,
            source_video_id="session_001.mkv",
            timestamp_ms=0,
            source_frame=0,
            normalization_version="v1",
            stable=True,
            scene=Scene.TABLE,
            group_id="g",
            reason="",
        )


def _entry(**overrides: object) -> dataset.FrameEntry:
    base = {
        "file": "session_002__t_00003500__f_000105__8e55a6c8b73b.png",
        "sha256": "a" * 64,
        "source_video_id": "session_002",
        "timestamp_ms": 3500,
        "source_frame": 105,
        "normalization_version": "capture-card-normalization-v1",
        "stable": True,
        "scene": Scene.TABLE,
        "group_id": "session_002__scene_table",
        "reason": "stable_table",
    }
    base.update(overrides)
    return dataset.FrameEntry.from_dict(base)


def test_build_frame_label_skeleton_is_all_unknown():
    label = dataset.build_frame_label_skeleton(
        _entry(), hand_id="session_002_hand_0001", reviewer="tester"
    )
    assert label.hand_id == "session_002_hand_0001"
    assert label.hero_cards.status is LabelStatus.UNKNOWN
    assert label.hero_cards.value is None
    assert label.board_cards.status is LabelStatus.UNKNOWN
    assert label.street.status is LabelStatus.UNKNOWN
    assert label.pot.status is LabelStatus.UNKNOWN
    assert len(label.slots) == 8
    assert [slot.slot_id for slot in label.slots] == list(range(8))
    for slot in label.slots:
        assert slot.occupancy.status is LabelStatus.UNKNOWN
        assert slot.stack.status is LabelStatus.UNKNOWN
        assert slot.dealer.status is LabelStatus.UNKNOWN
        assert slot.completed_action.status is LabelStatus.UNKNOWN
        assert slot.current_actor.status is LabelStatus.UNKNOWN
    assert label.review.reviewer == "tester"
    assert label.review.method == "manual_source_pixels"


def test_build_frame_label_skeleton_rejects_placeholder_reviewer():
    with pytest.raises(SchemaError, match="real reviewer"):
        dataset.build_frame_label_skeleton(
            _entry(), hand_id="h", reviewer="REPLACE_ME"
        )


def test_build_frame_label_skeleton_rejects_unstable_frame():
    with pytest.raises(SchemaError, match="unstable frame"):
        dataset.build_frame_label_skeleton(
            _entry(stable=False), hand_id="h", reviewer="tester"
        )


def test_build_frame_label_skeleton_roundtrips_through_json():
    label = dataset.build_frame_label_skeleton(
        _entry(), hand_id="session_002_hand_0001", reviewer="tester"
    )
    assert dataset.FrameLabel.from_json(label.to_json()) == label


def test_build_frame_label_skeleton_requires_real_reviewer():
    with pytest.raises(SchemaError, match="real reviewer"):
        dataset.build_frame_label_skeleton(
            _entry(), hand_id="h", reviewer="  "
        )


def test_roi_csv_roundtrip(tmp_path):
    rows = [
        dataset.RoiMeasurement("pot", 10, 20, 110, 60, source_frame="f1"),
        dataset.RoiMeasurement(
            "stack", 0, 0, 40, 20, slot_id=3, source_frame="f1", notes="n"
        ),
    ]
    path = tmp_path / "roi.csv"
    assert dataset.write_roi_measurements_csv(path, rows) == 2
    assert dataset.read_roi_measurements_csv(path) == rows


def test_roi_measurement_rejects_degenerate_rectangle():
    with pytest.raises(SchemaError, match="x1 must be > x0"):
        dataset.RoiMeasurement("pot", 10, 20, 10, 60)


def test_roi_measurement_rejects_out_of_range_slot():
    with pytest.raises(SchemaError, match="slot_id"):
        dataset.RoiMeasurement("stack", 0, 0, 4, 4, slot_id=8)


def test_roi_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "roi.csv"
    path.write_text("field,x0\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="missing columns"):
        dataset.read_roi_measurements_csv(path)


# --- action enum coverage --------------------------------------------------


def test_completed_action_enum_matches_guide():
    assert {action.value for action in CompletedAction} == {
        "FOLD",
        "CHECK",
        "CALL",
        "BET",
        "RAISE",
        "ALL_IN",
    }
