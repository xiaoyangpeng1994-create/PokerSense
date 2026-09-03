"""Tests for stage G coverage, stage H splits, stage 17 reporting and the CLI."""

from __future__ import annotations

import json

import pytest

from tools.capture_card_calibration import coverage as coverage_module
from tools.capture_card_calibration.cli import main
from tools.capture_card_calibration.coverage import (
    CoverageReport,
    evaluate_coverage,
)
from tools.capture_card_calibration.report import (
    PRODUCTION_FIELDS,
    STATUS_BLOCKED,
    STATUS_PARTIAL,
    STATUS_PASS,
    ReportInputs,
    determine_status,
    render_review_report,
)
from tools.capture_card_calibration.schema import (
    DeviceAndCapture,
    FieldMetrics,
    FieldValue,
    FrameLabel,
    Review,
    Scene,
    SlotLabel,
)
from tools.capture_card_calibration.splits import (
    DEFAULT_RATIOS,
    build_split_plan,
    detect_leakage,
    read_split_file,
    validate_split_plan,
)

# --- builders --------------------------------------------------------------


def _slot(slot_id: int, **overrides: object) -> dict:
    data = {
        "slot_id": slot_id,
        "occupancy": FieldValue.valid("OCCUPIED").to_dict(),
        "stack": FieldValue.valid(100 + slot_id).to_dict(),
        "dealer": FieldValue.valid(False).to_dict(),
        "completed_action": FieldValue.unknown().to_dict(),
        "current_actor": FieldValue.unknown().to_dict(),
    }
    data.update(overrides)
    return data


def _label(index: int, session: str = "session_001", **overrides: object):
    kwargs = {
        "frame": f"{session}__t_{index:08d}__f_{index:06d}__{index:012d}.png",
        "sha256": f"{index:064d}",
        "session_id": session,
        "hand_id": f"{session}_hand_{index // 4:04d}",
        "timestamp_ms": index * 500,
        "stable": True,
        "scene": Scene.TABLE,
        "hero_cards": FieldValue.valid(["TS", "JD"]),
        "board_cards": FieldValue.valid(["4H", "KC", "TS"]),
        "street": FieldValue.valid("FLOP"),
        "pot": FieldValue.valid(100 + index),
        "review": Review(reviewer="tester"),
    }
    kwargs["slots"] = tuple(
        SlotLabel.from_dict(_slot(i)) for i in range(8)
    )
    kwargs.update(overrides)
    return FrameLabel(**kwargs)


def _device(sessions: int = 3) -> DeviceAndCapture:
    return DeviceAndCapture(
        phone={"model": "Pixel 8", "android_version": "14"},
        app={"name": "WePoker", "version": "1.2.3"},
        video_adapter={"model": "Adapter X"},
        capture_card={"model": "Card Y"},
        uvc={"frame_size": [1920, 1080], "fps": 30},
        recording={"container": "mkv", "codec": "h264"},
        sessions=tuple(
            {"session_id": f"session_{n:03d}"} for n in range(1, sessions + 1)
        ),
    )


def _metrics(field: str = "pot", **overrides: object) -> FieldMetrics:
    kwargs = {
        "field": field,
        "algorithm_version": "algo-v1",
        "threshold": 0.8,
        "train_samples": 10,
        "calibration_positive_samples": 5,
        "calibration_negative_samples": 5,
        "validation_positive_samples": 10,
        "validation_negative_samples": 10,
        "correct_valid": 10,
        "false_valid": 0,
        "unknown_on_positive": 0,
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


def _dataset(count: int = 12, sessions: int = 3) -> list[FrameLabel]:
    labels = []
    for index in range(count):
        session = f"session_{(index % sessions) + 1:03d}"
        labels.append(_label(index, session=session))
    return labels


# --- stage G coverage ------------------------------------------------------


def test_empty_dataset_is_not_complete():
    report = evaluate_coverage([])
    assert not report.is_complete
    assert report.unmet


def test_coverage_counts_sessions_and_frames():
    report = evaluate_coverage(_dataset(12, sessions=3))
    assert report.frame_count == 12
    assert report.sessions == ("session_001", "session_002", "session_003")


def test_coverage_reports_missing_ranks_and_suits():
    report = evaluate_coverage(_dataset(4))
    hero = next(item for item in report.requirements if item.field == "hero_cards")
    assert any("ranks never observed" in line for line in hero.shortfalls)
    assert any("suits never observed" in line for line in hero.shortfalls)


def test_coverage_names_every_missing_action():
    report = evaluate_coverage(_dataset(4))
    action = next(
        item for item in report.requirements if item.field == "completed_action"
    )
    joined = " ".join(action.shortfalls)
    for name in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
        assert name in joined
    # ALL_IN is owner-waived (2026-09-04 decision, recorded in coverage.py):
    # it must NOT be named as a shortfall even when absent.
    assert "ALL_IN" not in joined


def test_coverage_counts_temporal_groups():
    labels = [
        _label(0, scene=Scene.TABLE),
        _label(1, scene=Scene.DEAL_TRANSITION),
        _label(2, scene=Scene.SIGNAL_LOSS),
    ]
    report = evaluate_coverage(labels)
    temporal = next(item for item in report.requirements if item.field == "temporal")
    assert temporal.measured_negative == 1


def test_coverage_gap_lines_are_prefixed_by_field():
    report = evaluate_coverage(_dataset(4))
    assert report.gap_lines
    assert all(": " in line for line in report.gap_lines)


def test_coverage_counts_anomaly_frames():
    labels = [
        _label(0, scene=Scene.MENU),
        _label(1, scene=Scene.SIGNAL_LOSS),
        _label(2, scene=Scene.TABLE),
    ]
    report = evaluate_coverage(labels)
    anomaly = next(
        item for item in report.requirements if item.field == "anomaly_scenes"
    )
    assert anomaly.measured_negative == 2


# --- stage H splits --------------------------------------------------------


def test_every_frame_of_a_hand_stays_in_one_split():
    labels = _dataset(24, sessions=3)
    plan = build_split_plan(labels)
    assert detect_leakage(plan) == []
    owners: dict[str, str] = {}
    for assignment in plan.assignments:
        for group in assignment.groups:
            owners[group] = assignment.name
    for label in labels:
        assert owners[label.group_id]


def test_splits_follow_the_default_ratios():
    labels = _dataset(40, sessions=3)
    plan = build_split_plan(labels)
    total = sum(len(item.frames) for item in plan.assignments)
    shares = {
        item.name: len(item.frames) / total for item in plan.assignments
    }
    for name, expected in DEFAULT_RATIOS.items():
        assert shares[name] == pytest.approx(expected, abs=0.06)


def test_splits_spread_sessions_across_every_split():
    labels = _dataset(60, sessions=3)
    plan = build_split_plan(labels)
    for assignment in plan.assignments:
        sessions = {name.split("__")[0] for name in assignment.frames}
        assert len(sessions) > 1, f"{assignment.name} drew from one session only"


def test_validate_split_plan_flags_one_sided_splits():
    labels = [
        _label(0, scene=Scene.TABLE),
        _label(1, scene=Scene.TABLE),
        _label(2, scene=Scene.TABLE),
        _label(3, scene=Scene.TABLE),
        _label(4, scene=Scene.TABLE),
    ]
    plan = build_split_plan(labels)
    problems = validate_split_plan(plan, labels)
    assert any("no real hard negatives" in item for item in problems)


def test_validate_split_plan_flags_unassigned_frames():
    labels = _dataset(6, sessions=1)
    plan = build_split_plan(labels)
    problems = validate_split_plan(plan, labels[:4])
    assert any("unknown frame" in item for item in problems)


def test_splits_write_and_read_back(tmp_path):
    plan = build_split_plan(_dataset(12, sessions=3))
    written = plan.write(tmp_path)
    assert set(written) == {
        "train",
        "calibration",
        "validation",
        "negative",
        "temporal",
    }
    frames = read_split_file(written["train"])
    assert set(frames) == set(plan.by_name("train").frames)


def test_splits_reject_empty_label_set():
    with pytest.raises(ValueError, match="empty label set"):
        build_split_plan([])


def test_splits_reject_partial_ratios():
    labels = _dataset(6, sessions=1)
    with pytest.raises(ValueError, match="ratios must cover exactly"):
        build_split_plan(labels, ratios={"train": 0.5, "calibration": 0.5})


# --- stage 17 report -------------------------------------------------------


def test_status_is_blocked_without_a_device_manifest():
    status, reasons = determine_status(ReportInputs())
    assert status == STATUS_BLOCKED
    assert reasons


def test_status_is_blocked_while_placeholders_remain():
    device = DeviceAndCapture(
        phone={"model": "REPLACE_ME"},
        app={"name": "WePoker"},
        video_adapter={"model": "A"},
        capture_card={"model": "B"},
        uvc={"frame_size": [1920, 1080]},
        recording={"container": "mkv"},
    )
    status, reasons = determine_status(ReportInputs(device=device))
    assert status == STATUS_BLOCKED
    assert any("REPLACE_ME" in line for line in reasons)


def test_status_is_blocked_when_a_stop_condition_is_raised():
    status, reasons = determine_status(
        ReportInputs(device=_device(), stop_conditions=["content boundary drift"])
    )
    assert status == STATUS_BLOCKED
    assert any("section 19" in line for line in reasons)


def test_status_is_partial_when_coverage_is_incomplete():
    report = evaluate_coverage(_dataset(6, sessions=1))
    status, reasons = determine_status(
        ReportInputs(device=_device(), coverage=report)
    )
    assert status == STATUS_PARTIAL
    assert reasons


def test_status_is_partial_without_field_metrics():
    coverage = CoverageReport(sessions=("a", "b", "c"), frame_count=1,
                              requirements=())
    status, reasons = determine_status(
        ReportInputs(device=_device(), coverage=coverage)
    )
    assert status == STATUS_PARTIAL
    assert any("no stage I metrics" in line for line in reasons)


def test_status_is_partial_when_a_field_has_false_valid():
    coverage = CoverageReport(sessions=("a", "b", "c"), frame_count=1,
                              requirements=())
    metrics = [_metrics(field=name) for name in PRODUCTION_FIELDS]
    metrics[0] = _metrics(field=PRODUCTION_FIELDS[0], false_valid=2)
    status, reasons = determine_status(
        ReportInputs(
            device=_device(),
            coverage=coverage,
            field_metrics=metrics,
            replay_evidence={"draft": True},
            seat_mapping_status="ok",
            action_consistency_status="ok",
            hashes={"x": "y"},
        )
    )
    assert status == STATUS_PARTIAL
    assert any("false VALID" in line for line in reasons)


def test_status_passes_only_with_complete_evidence():
    coverage = CoverageReport(
        sessions=("session_001", "session_002", "session_003"),
        frame_count=100,
        requirements=(),
    )
    inputs = ReportInputs(
        platform_id="wepoker_android_capture_card",
        layout_id="phone_a__card_b__uvc_1920x1080_30__canvas_1080x1920__v1",
        device=_device(),
        coverage=coverage,
        field_metrics=[_metrics(field=name) for name in PRODUCTION_FIELDS],
        replay_evidence={"draft": True},
        seat_mapping_status="2-8 players verified",
        action_consistency_status="chip/pot conservation verified",
        hashes={"configs/x.json": "a" * 64},
    )
    status, reasons = determine_status(inputs)
    assert status == STATUS_PASS
    assert reasons == []


def test_rendered_report_contains_every_section():
    text = render_review_report(ReportInputs(device=_device()))
    for heading in (
        "# Capture Card Calibration Review",
        "## 结论",
        "## 硬件与采集参数",
        "## 画面稳定性",
        "## 字段结果",
        "## 时序与重连",
        "## 座位映射与动作一致性",
        "## 性能",
        "## 未解决问题",
        "## 需要补录的精确清单",
        "## 文件和版本哈希",
    ):
        assert heading in text


def test_rendered_report_marks_unmeasured_values_as_not_recorded():
    text = render_review_report(ReportInputs(device=_device()))
    assert "_not recorded_" in text


def test_rendered_report_lists_every_production_field():
    text = render_review_report(ReportInputs(device=_device()))
    for name in PRODUCTION_FIELDS:
        assert f"| {name} |" in text


def test_rendered_report_verdict_line_matches_status():
    inputs = ReportInputs(device=_device())
    status, _ = determine_status(inputs)
    assert f"- 状态：{status}" in render_review_report(inputs)


# --- CLI -------------------------------------------------------------------


def test_cli_init_creates_a_dataset(tmp_path):
    root = tmp_path / "calib"
    assert main(["init", "--root", str(root)]) == 0
    assert (root / "labels").is_dir()
    assert (root / "source" / "device_and_capture.json").is_file()


def test_cli_coverage_reports_gaps(tmp_path, capsys):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    write_frames_jsonl(root / "labels" / "frames.jsonl", _dataset(6, sessions=1))
    assert main(["coverage", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "GAP" in out
    assert "top-up list" in out


def test_cli_coverage_strict_exits_non_zero(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    write_frames_jsonl(root / "labels" / "frames.jsonl", _dataset(6, sessions=1))
    assert main(["coverage", "--root", str(root), "--strict"]) == 1


def test_cli_splits_writes_files_even_when_degenerate(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    write_frames_jsonl(root / "labels" / "frames.jsonl", _dataset(12, sessions=3))
    # The synthetic set has no hard negatives, so validation legitimately
    # flags every split and the command exits non-zero. Files must still
    # be written; refusing to write would hide the problem instead.
    assert main(["splits", "--root", str(root)]) == 1
    assert (root / "splits" / "train.txt").is_file()
    assert (root / "splits" / "validation.txt").is_file()


def test_cli_splits_accepts_a_balanced_dataset(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    labels = [
        _label(
            index,
            session=f"session_{(index % 3) + 1:03d}",
            scene=Scene.MENU if index % 3 == 2 else Scene.TABLE,
        )
        for index in range(30)
    ]
    root = create_skeleton(tmp_path / "calib")
    write_frames_jsonl(root / "labels" / "frames.jsonl", labels)
    assert main(["splits", "--root", str(root)]) == 0


def test_cli_geometry_writes_drafts(tmp_path):
    from tools.capture_card_calibration.dataset import (
        RoiMeasurement,
        create_skeleton,
        write_roi_measurements_csv,
    )

    root = create_skeleton(tmp_path / "calib")
    rows = [
        RoiMeasurement("hero_cards", 400, 1500, 680, 1650),
        RoiMeasurement("board_cards", 200, 800, 880, 1000),
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
    write_roi_measurements_csv(root / "labels" / "roi_measurements.csv", rows)
    assert main([
        "geometry",
        "--root", str(root),
        "--layout-id",
        "phone_a__card_b__uvc_1920x1080_30__canvas_1080x1920__v1",
        "--canvas", "1080x1920",
    ]) == 0
    assert (root / "geometry" / "table_map.draft.json").is_file()


def test_cli_report_writes_a_blocked_report(tmp_path, capsys):
    from tools.capture_card_calibration.dataset import create_skeleton

    root = create_skeleton(tmp_path / "calib")
    assert main(["report", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "status: BLOCKED" in out
    assert (root / "evidence" / "review_report.md").is_file()


def test_cli_report_strict_exits_non_zero(tmp_path):
    from tools.capture_card_calibration.dataset import create_skeleton

    root = create_skeleton(tmp_path / "calib")
    assert main(["report", "--root", str(root), "--strict"]) == 1


def test_cli_hash_write_and_verify(tmp_path):
    from tools.capture_card_calibration.dataset import create_skeleton

    root = create_skeleton(tmp_path / "calib")
    (root / "labels" / "x.json").write_text("{}", encoding="utf-8")
    assert main(["hash", "--root", str(root)]) == 0
    assert main(["hash", "--root", str(root), "--verify"]) == 0
    (root / "labels" / "x.json").write_text("{ }", encoding="utf-8")
    assert main(["hash", "--root", str(root), "--verify"]) == 1


def test_cli_layout_id_prints_an_identifier(capsys):
    assert main([
        "layout-id",
        "--phone", "Redmi K60",
        "--card", "绿联 CM716",
        "--uvc", "1920x1080",
        "--fps", "30",
        "--canvas", "1080x1920",
    ]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("phone_redmi_k60__card_cm716")


def test_cli_check_labels_reports_counts(tmp_path, capsys):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    write_frames_jsonl(root / "labels" / "frames.jsonl", _dataset(8, sessions=2))
    assert main(["check-labels", "--root", str(root)]) == 0
    assert "sessions: 2" in capsys.readouterr().out


def test_cli_returns_error_code_on_invalid_labels(tmp_path):
    from tools.capture_card_calibration.dataset import create_skeleton

    root = create_skeleton(tmp_path / "calib")
    (root / "labels" / "frames.jsonl").write_text("{}\n", encoding="utf-8")
    assert main(["check-labels", "--root", str(root)]) == 1


def test_cli_manifest_reads_frame_manifest(tmp_path, capsys):
    from tools.capture_card_calibration.dataset import (
        FrameEntry,
        create_skeleton,
        write_frame_manifest,
    )

    root = create_skeleton(tmp_path / "calib")
    write_frame_manifest(
        root / "normalized" / "manifest.json",
        [
            FrameEntry(
                file="a.png",
                sha256="a" * 64,
                source_video_id="session_001.mkv",
                timestamp_ms=0,
                source_frame=0,
                normalization_version="capture-card-normalization-v1",
                stable=True,
                scene=Scene.TABLE,
                group_id="session_001::hand_0001",
                reason="steady_state",
            )
        ],
    )
    assert main(["manifest", "--root", str(root)]) == 0
    assert "frames: 1" in capsys.readouterr().out


def test_coverage_module_exposes_section_ten_minimums():
    assert coverage_module.STREET_REQUIREMENTS["FLOP"] == 20
    assert coverage_module.ACTION_REQUIREMENTS["ALL_IN"] == 6
    assert coverage_module.MIN_ANOMALY_FRAMES == 50


def test_owner_focus_excludes_3_5_headcount_bucket():
    # The owner plays ~90% of hands at a 6-8 handed table and never plays a
    # 3-5 handed table. Section 10's generic {2,3-5,6-8} is owner-authorized
    # to a focused {2,6-8} (recorded in AGENTS.md), so a 3-5 handed dataset
    # must not be required. This pins that focus so it cannot silently regress.
    assert "3-5" not in coverage_module.REQUIRED_HEADCOUNT_BUCKETS
    assert "2" in coverage_module.REQUIRED_HEADCOUNT_BUCKETS
    assert "6-8" in coverage_module.REQUIRED_HEADCOUNT_BUCKETS


def test_coverage_2_and_6_8_buckets_are_required():
    # A dataset with only a 6-8 handed table passes the head-count bucket
    # check; only missing {2,6-8} (not 3-5) is reported as a shortfall.
    labels = _dataset(12, sessions=3)
    report = evaluate_coverage(labels)
    occupancy = next(r for r in report.requirements if r.field == "occupancy")
    # The synthetic fixture is 8-handed OCCUPIED -> bucket "6-8".
    assert "2" in occupancy.shortfalls or not occupancy.met
    assert not any("3-5" in shortfall for shortfall in occupancy.shortfalls)


def test_coverage_report_serializes():
    report = evaluate_coverage(_dataset(4))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["frame_count"] == 4
    assert payload["complete"] is False
