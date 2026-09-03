"""Stage 17 acceptance report (guide sections 16 and 17).

The report is generated, never hand-written, so the verdict cannot drift
away from the evidence. Section 16 is strict:

    If any key field, mapping, source or verification evidence is missing,
    the final status must be written as PARTIAL or BLOCKED. Declaring a
    full pass is not allowed.

:func:`determine_status` implements that literally: ``PASS`` requires the
hardware manifest to be frozen, coverage to be complete, every production
field to have metrics with zero false ``VALID``, replay evidence to be
bound, and no stop condition raised. Anything less downgrades the verdict
and says why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .coverage import CoverageReport
from .schema import DeviceAndCapture, FieldMetrics, MIN_SESSIONS

STATUS_PASS = "PASS"
STATUS_PARTIAL = "PARTIAL"
STATUS_BLOCKED = "BLOCKED"

PRODUCTION_FIELDS: tuple[str, ...] = (
    "hero_cards",
    "board_cards",
    "street",
    "pot",
    "stack",
    "action",
    "dealer",
    "occupancy",
    "hero_actor",
)

_PLACEHOLDER_TEXT = "_not recorded_"


@dataclass
class ReportInputs:
    """Everything the generator knows; gaps stay gaps."""

    platform_id: str = "wepoker_android_capture_card"
    layout_id: str = _PLACEHOLDER_TEXT
    device: DeviceAndCapture | None = None
    normalization_version: str | None = None
    raw_size: tuple[int, int] | None = None
    normalized_size: tuple[int, int] | None = None
    max_boundary_drift: float | None = None
    coverage: CoverageReport | None = None
    field_metrics: Sequence[FieldMetrics] = ()
    split_summary: Mapping[str, int] | None = None
    replay_evidence: Mapping[str, Any] | None = None
    performance: Mapping[str, Any] | None = None
    seat_mapping_status: str | None = None
    action_consistency_status: str | None = None
    unresolved: Sequence[str] = ()
    stop_conditions: Sequence[str] = ()
    hashes: Mapping[str, str] = field(default_factory=dict)


def determine_status(
    inputs: ReportInputs,
) -> tuple[str, list[str]]:
    """Return ``(status, reasons)`` following the section 16 gate."""
    reasons: list[str] = []

    if inputs.stop_conditions:
        return STATUS_BLOCKED, [
            "section 19 stop condition raised: " + item
            for item in inputs.stop_conditions
        ]

    if inputs.device is None:
        return STATUS_BLOCKED, [
            "stage A hardware manifest (source/device_and_capture.json) is "
            "missing"
        ]
    placeholders = inputs.device.placeholders()
    if placeholders:
        return STATUS_BLOCKED, [
            "stage A hardware manifest still holds REPLACE_ME at: "
            + ", ".join(placeholders)
        ]

    if inputs.coverage is None:
        return STATUS_BLOCKED, [
            "no labelled dataset; stage G coverage was never evaluated"
        ]

    if not inputs.coverage.is_complete:
        reasons.extend(inputs.coverage.gap_lines)
    if len(inputs.coverage.sessions) < MIN_SESSIONS:
        reasons.append(
            f"fewer than {MIN_SESSIONS} independent capture sessions "
            f"({len(inputs.coverage.sessions)})"
        )

    measured = {item.field: item for item in inputs.field_metrics}
    missing_metrics = [f for f in PRODUCTION_FIELDS if f not in measured]
    if missing_metrics:
        reasons.append(
            "no stage I metrics for fields: " + ", ".join(missing_metrics)
        )
    for name, item in sorted(measured.items()):
        if not item.zero_false_valid:
            reasons.append(
                f"{name}: {item.false_valid} false VALID on the locked "
                "validation set (section 12 requires zero)"
            )

    if inputs.replay_evidence is None:
        reasons.append("stage K replay evidence is not bound")
    if inputs.seat_mapping_status is None:
        reasons.append("stage J seat mapping acceptance was not recorded")
    if inputs.action_consistency_status is None:
        reasons.append("stage J action consistency check was not recorded")
    if not inputs.hashes:
        reasons.append(
            "no SHA-256 evidence; rule 8 requires hashes for every artifact"
        )
    reasons.extend(f"unresolved: {item}" for item in inputs.unresolved)

    if reasons:
        return STATUS_PARTIAL, reasons
    return STATUS_PASS, []


def _size_text(size: tuple[int, int] | None) -> str:
    return f"{size[0]}x{size[1]}" if size else _PLACEHOLDER_TEXT


def render_review_report(inputs: ReportInputs) -> str:
    """Render the section 17 markdown template."""
    status, reasons = determine_status(inputs)
    lines: list[str] = ["# Capture Card Calibration Review", ""]

    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 状态：{status}")
    lines.append(f"- platform_id：{inputs.platform_id}")
    lines.append(f"- layout_id：{inputs.layout_id}")
    lines.append(
        f"- normalization version："
        f"{inputs.normalization_version or _PLACEHOLDER_TEXT}"
    )
    if inputs.coverage is not None:
        sessions = ", ".join(inputs.coverage.sessions) or _PLACEHOLDER_TEXT
        lines.append(f"- source sessions：{sessions}")
    else:
        lines.append(f"- source sessions：{_PLACEHOLDER_TEXT}")
    lines.append("")

    # --- hardware and capture parameters ---
    lines.append("## 硬件与采集参数")
    lines.append("")
    if inputs.device is None:
        lines.append(f"{_PLACEHOLDER_TEXT} — 阶段 A 尚未完成。")
    else:
        device = inputs.device
        lines.append(
            f"- 手机：{device.phone.get('model', '?')} / "
            f"Android {device.phone.get('android_version', '?')}"
        )
        lines.append(
            f"- 应用：{device.app.get('name', '?')} "
            f"{device.app.get('version', '?')} "
            f"({device.app.get('orientation', '?')})"
        )
        lines.append(
            f"- 视频输出适配器：{device.video_adapter.get('model', '?')}"
        )
        lines.append(
            f"- 采集卡：{device.capture_card.get('model', '?')} / "
            f"{device.capture_card.get('connection', '?')}"
        )
        uvc_size = device.uvc.get("frame_size") or ["?", "?"]
        lines.append(
            f"- UVC：{uvc_size[0]}x{uvc_size[1]} @ "
            f"{device.uvc.get('fps', '?')} fps, "
            f"{device.uvc.get('pixel_format', '?')}"
        )
        lines.append(
            f"- 录制：{device.recording.get('container', '?')} / "
            f"{device.recording.get('codec', '?')}"
        )
        lines.append(f"- 采集会话数：{len(device.sessions)}")
    lines.append("")

    # --- picture stability ---
    lines.append("## 画面稳定性")
    lines.append("")
    lines.append(f"- 原始尺寸：{_size_text(inputs.raw_size)}")
    lines.append(f"- 归一化尺寸：{_size_text(inputs.normalized_size)}")
    drift = inputs.max_boundary_drift
    lines.append(
        "- 最大内容边界漂移："
        + (
            f"{drift} px"
            if drift is not None
            else f"{_PLACEHOLDER_TEXT}（阶段 C 未测量）"
        )
    )
    if inputs.device is not None:
        recording = inputs.device.recording
        lines.append(
            f"- 旋转/镜像/裁剪：rotate={recording.get('rotate', '?')}, "
            f"mirror={recording.get('mirror', '?')}, "
            f"crop={recording.get('crop', '?')}"
        )
    else:
        lines.append(f"- 旋转/镜像/裁剪：{_PLACEHOLDER_TEXT}")
    lines.append("")

    # --- field results ---
    lines.append("## 字段结果")
    lines.append("")
    lines.append(
        "| 字段 | 正样本 | 负样本 | false VALID | UNKNOWN | 结论 |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    metrics = {item.field: item for item in inputs.field_metrics}
    fields_seen = list(PRODUCTION_FIELDS)
    if inputs.coverage is not None:
        for item in inputs.coverage.requirements:
            if item.field not in fields_seen:
                fields_seen.append(item.field)
    for name in fields_seen:
        metric = metrics.get(name)
        coverage_item = None
        if inputs.coverage is not None:
            coverage_item = next(
                (
                    item
                    for item in inputs.coverage.requirements
                    if item.field == name
                ),
                None,
            )
        positive = (
            metric.validation_positive_samples
            if metric
            else (coverage_item.measured_positive if coverage_item else 0)
        )
        negative = (
            metric.validation_negative_samples
            if metric
            else (coverage_item.measured_negative if coverage_item else 0)
        )
        false_valid = metric.false_valid if metric else "—"
        unknown = metric.unknown_on_positive if metric else "—"
        if metric is None:
            verdict = "未标定"
        elif not metric.zero_false_valid:
            verdict = "不通过"
        elif not coverage_item or coverage_item.met:
            verdict = "通过"
        else:
            verdict = "覆盖不足"
        lines.append(
            f"| {name} | {positive} | {negative} | {false_valid} | "
            f"{unknown} | {verdict} |"
        )
    lines.append("")

    # --- temporal and reconnect ---
    lines.append("## 时序与重连")
    lines.append("")
    if inputs.coverage is not None:
        temporal = next(
            (
                item
                for item in inputs.coverage.requirements
                if item.field == "temporal"
            ),
            None,
        )
        if temporal is None:
            lines.append(f"{_PLACEHOLDER_TEXT}")
        elif temporal.met:
            lines.append(
                f"- 时序组：{temporal.measured_positive}，"
                f"断流/重连组：{temporal.measured_negative} — 达标。"
            )
        else:
            lines.append("- 时序与重连未达标：")
            lines.extend(f"  - {item}" for item in temporal.shortfalls)
    else:
        lines.append(f"{_PLACEHOLDER_TEXT} — 尚无标注数据。")
    lines.append("")

    # --- seat mapping and action consistency ---
    lines.append("## 座位映射与动作一致性")
    lines.append("")
    lines.append(
        f"- 座位映射：{inputs.seat_mapping_status or _PLACEHOLDER_TEXT}"
    )
    lines.append(
        f"- 动作一致性："
        f"{inputs.action_consistency_status or _PLACEHOLDER_TEXT}"
    )
    lines.append("")

    # --- performance ---
    lines.append("## 性能")
    lines.append("")
    if inputs.performance:
        for key, value in inputs.performance.items():
            lines.append(f"- {key}：{value}")
    else:
        lines.append(
            f"{_PLACEHOLDER_TEXT} — 阶段 K 未实测。数字只如实记录，"
            "未经批准不得写成发布通过。"
        )
    lines.append("")

    # --- unresolved ---
    lines.append("## 未解决问题")
    lines.append("")
    if inputs.unresolved:
        lines.extend(f"- {item}" for item in inputs.unresolved)
    else:
        lines.append("- 无。")
    if reasons:
        lines.append("")
        lines.append("### 状态判定依据")
        lines.append("")
        lines.extend(f"- {item}" for item in reasons)
    lines.append("")

    # --- precise top-up list ---
    lines.append("## 需要补录的精确清单")
    lines.append("")
    top_up: list[str] = []
    if inputs.coverage is not None:
        top_up.extend(inputs.coverage.gap_lines)
    top_up.extend(
        f"阶段 I：补齐 {name} 的锁定验证集指标"
        for name in PRODUCTION_FIELDS
        if name not in metrics
    )
    if top_up:
        lines.extend(f"- {item}" for item in top_up)
    else:
        lines.append("- 无。")
    lines.append("")

    # --- hashes ---
    lines.append("## 文件和版本哈希")
    lines.append("")
    if inputs.hashes:
        lines.append("| 文件 | SHA-256 |")
        lines.append("|---|---|")
        for path in sorted(inputs.hashes):
            lines.append(f"| {path} | `{inputs.hashes[path]}` |")
        missing_hashes = [
            f for f in PRODUCTION_FIELDS if f"{f}.config" not in inputs.hashes
        ]
        if missing_hashes:
            lines.append("")
            lines.append(
                "尚未产生生产配置哈希的字段："
                + ", ".join(missing_hashes)
            )
    else:
        lines.append(
            f"{_PLACEHOLDER_TEXT} — 规则 8 要求所有配置、模板、数据和代码"
            "记录 SHA-256。"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


__all__ = [
    "PRODUCTION_FIELDS",
    "ReportInputs",
    "STATUS_BLOCKED",
    "STATUS_PARTIAL",
    "STATUS_PASS",
    "determine_status",
    "render_review_report",
]
