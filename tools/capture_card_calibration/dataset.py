"""Calibration dataset layout, label I/O and ROI measurement I/O.

Implements the delivery directory from guide section 3, the frame manifest
contract from section 7, the per-frame ground-truth file from section 9 and
the ROI measurement CSV from section 8.

Privacy note (guide rule 4): this directory is a **private working
dataset**. Raw video, full frames, nicknames and avatars must never enter
Git, the installer, or any public upload. The generated README states this
so the rule travels with the directory instead of living only in the guide.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import SCHEMA_VERSION, SLOT_COUNT
from .schema import (
    FieldValue,
    FrameLabel,
    PLACEHOLDER,
    Review,
    Scene,
    SchemaError,
    SlotLabel,
)

DATASET_DIRECTORIES: tuple[str, ...] = (
    "source/probe",
    "source/raw",
    "normalization",
    "normalized/frames",
    "labels",
    "geometry",
    "templates/cards",
    "templates/stacks",
    "templates/actions",
    "templates/dealer",
    "templates/occupancy",
    "templates/actor",
    "splits",
    "evidence/contact_sheets",
    "replay",
)

ROI_CSV_FIELDS: tuple[str, ...] = (
    "field",
    "slot_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "source_frame",
    "notes",
)

README_TEXT = """# Capture-card calibration dataset (PRIVATE)

This directory is a **private working dataset**, not a repository asset.

Per guide rule 4 the following must never enter Git, the packaged
installer, or any public upload:

- raw capture video (`source/raw/`);
- full normalized frames (`normalized/frames/`);
- anything containing nicknames or avatars.

Only tightly cropped, privacy-reviewed regression samples may ever be
promoted into the repository, and only under `configs/` or `tests/`.

## Layout

| Path | Contents |
|---|---|
| `source/` | Hardware manifest, ffprobe output, raw capture video |
| `normalization/` | Stage C normalization config |
| `normalized/` | Stage C/D canvas frames and the frame manifest |
| `labels/` | Stage F ground truth (`frames.jsonl`, `roi_measurements.csv`) |
| `geometry/` | Stage E ROI drafts and seat mapping |
| `templates/` | Stage I templates (tight crops only) |
| `splits/` | Stage H hand/session-isolated splits |
| `evidence/` | Stage I field metrics, review report, contact sheets |
| `replay/` | Stage K replay draft |
| `SHA256SUMS` | Stage 18 hash manifest |
"""


def create_skeleton(root: Path | str) -> Path:
    """Create the section 3 directory tree and its privacy README."""
    root = Path(root)
    for relative in DATASET_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")
    return root


def write_device_template(path: Path | str) -> Path:
    """Write a stage A ``device_and_capture.json`` skeleton to be filled in.

    Every unknown is a ``REPLACE_ME`` placeholder. ``DeviceAndCapture.
    require_ready()`` refuses to let stage B start until they are gone, so
    the placeholders are a gate rather than a reminder.
    """
    template: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phone": {
            "model": PLACEHOLDER,
            "android_version": PLACEHOLDER,
            "display_size": [1080, 2400],
            "font_scale": PLACEHOLDER,
            "display_scale": PLACEHOLDER,
            "dpi": PLACEHOLDER,
            "theme": PLACEHOLDER,
        },
        "app": {
            "name": "WePoker",
            "version": PLACEHOLDER,
            "language": "zh-CN",
            "table_theme": PLACEHOLDER,
            "orientation": "portrait",
        },
        "video_adapter": {
            "model": PLACEHOLDER,
            "output_resolution": PLACEHOLDER,
            "refresh_hz": PLACEHOLDER,
        },
        "capture_card": {
            "model": PLACEHOLDER,
            "firmware": PLACEHOLDER,
            "connection": "USB 3.x",
        },
        "uvc": {
            "frame_size": [1920, 1080],
            "fps": 30,
            "pixel_format": PLACEHOLDER,
            "color_space": PLACEHOLDER,
            "color_range": PLACEHOLDER,
        },
        "recording": {
            "container": "mkv",
            "codec": PLACEHOLDER,
            "bitrate": PLACEHOLDER,
            "keyframe_interval": PLACEHOLDER,
            "filters": [],
            "rotate": False,
            "mirror": False,
            "crop": None,
            "scale": None,
        },
        "sessions": [],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(template, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


# --- stage F: labels/frames.jsonl -----------------------------------------


def write_frames_jsonl(path: Path | str, labels: Iterable[FrameLabel]) -> int:
    """Write one JSON object per line; returns the number written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as sink:
        for label in labels:
            sink.write(label.to_json())
            sink.write("\n")
            count += 1
    return count


def read_frames_jsonl(path: Path | str) -> list[FrameLabel]:
    """Read ``frames.jsonl``, reporting the offending line number on error."""
    source = Path(path)
    labels: list[FrameLabel] = []
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                labels.append(FrameLabel.from_json(line))
            except (ValueError, KeyError, TypeError) as exc:
                raise SchemaError(
                    f"{source}:{number}: invalid label line: {exc}"
                ) from exc
    return labels


def build_frame_label_skeleton(
    entry: FrameEntry,
    *,
    hand_id: str,
    reviewer: str,
    notes: str = "",
) -> FrameLabel:
    """Build a valid all-``UNKNOWN`` :class:`FrameLabel` for one frame.

    Section 9 ground truth must be labelled field by field from pixels that
    are actually visible. This helper produces the legal *skeleton* — every
    field ``UNKNOWN`` (no value), all ``SLOT_COUNT`` visual slots present,
    a real reviewer — so a human (or the vision model) only has to promote
    the fields it can read to ``VALID``.

    Failure-closed: a skeleton is never a pass. ``UNKNOWN`` carries no value,
    and ``reviewer`` must name a real person, not ``REPLACE_ME``.
    """
    if entry.stable is False:
        # Section 7: only stable table frames are eligible for ground truth.
        raise SchemaError(
            f"{entry.file}: unstable frame cannot be labelled as ground truth"
        )
    if reviewer == PLACEHOLDER or not reviewer.strip():
        raise SchemaError("reviewer must name a real reviewer, not REPLACE_ME")

    slots = tuple(
        SlotLabel(
            slot_id=slot,
            occupancy=FieldValue.unknown(),
            stack=FieldValue.unknown(),
            dealer=FieldValue.unknown(),
            completed_action=FieldValue.unknown(),
            current_actor=FieldValue.unknown(),
        )
        for slot in range(SLOT_COUNT)
    )
    return FrameLabel(
        frame=entry.file,
        sha256=entry.sha256,
        session_id=entry.source_video_id,
        hand_id=hand_id,
        timestamp_ms=entry.timestamp_ms,
        stable=entry.stable,
        scene=entry.scene,
        hero_cards=FieldValue.unknown(),
        board_cards=FieldValue.unknown(),
        street=FieldValue.unknown(),
        pot=FieldValue.unknown(),
        slots=slots,
        review=Review(reviewer=reviewer, method="manual_source_pixels", notes=notes),
    )


def iter_frames_jsonl(path: Path | str) -> Iterator[FrameLabel]:
    """Stream labels without holding the whole file in memory."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield FrameLabel.from_json(line)
            except (ValueError, KeyError, TypeError) as exc:
                raise SchemaError(
                    f"{source}:{number}: invalid label line: {exc}"
                ) from exc


# --- stage D: normalized/manifest.json ------------------------------------


def frame_filename(
    session_id: str, timestamp_ms: int, source_frame: int, sha256: str
) -> str:
    """Section 7 naming: ``<session>__t_<ms>__f_<frame>__<sha12>.png``."""
    if not session_id:
        raise ValueError("session_id must be non-empty")
    if timestamp_ms < 0 or source_frame < 0:
        raise ValueError("timestamp_ms and source_frame must be non-negative")
    if len(sha256) < 12:
        raise ValueError("sha256 must be at least 12 characters")
    return (
        f"{session_id}__t_{timestamp_ms:08d}__f_{source_frame:06d}"
        f"__{sha256[:12]}.png"
    )


@dataclass(frozen=True)
class FrameEntry:
    """One row of ``normalized/manifest.json`` (section 7)."""

    file: str
    sha256: str
    source_video_id: str
    timestamp_ms: int
    source_frame: int
    normalization_version: str
    stable: bool
    scene: Scene
    group_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.file:
            raise SchemaError("frame entry requires a file name")
        if len(self.sha256) != 64:
            raise SchemaError("frame entry sha256 must be 64 hex characters")
        if not self.source_video_id:
            raise SchemaError("frame entry requires source_video_id")
        if not self.normalization_version:
            raise SchemaError("frame entry requires normalization_version")
        if not self.group_id:
            raise SchemaError("frame entry requires group_id")
        if not self.reason:
            raise SchemaError("frame entry requires a sampling reason")
        if self.timestamp_ms < 0 or self.source_frame < 0:
            raise SchemaError("timestamp_ms/source_frame must be non-negative")
        if not isinstance(self.scene, Scene):
            object.__setattr__(self, "scene", Scene(self.scene))

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "sha256": self.sha256,
            "source_video_id": self.source_video_id,
            "timestamp_ms": self.timestamp_ms,
            "source_frame": self.source_frame,
            "normalization_version": self.normalization_version,
            "stable": self.stable,
            "scene": self.scene.value,
            "group_id": self.group_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameEntry":
        return cls(
            file=data["file"],
            sha256=data["sha256"],
            source_video_id=data["source_video_id"],
            timestamp_ms=int(data["timestamp_ms"]),
            source_frame=int(data["source_frame"]),
            normalization_version=data["normalization_version"],
            stable=bool(data["stable"]),
            scene=Scene(data["scene"]),
            group_id=data["group_id"],
            reason=data["reason"],
        )


def write_frame_manifest(
    path: Path | str,
    entries: Sequence[FrameEntry],
    *,
    schema_version: int = SCHEMA_VERSION,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema_version,
        "frame_count": len(entries),
        "frames": [entry.to_dict() for entry in entries],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_frame_manifest(path: Path | str) -> list[FrameEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [FrameEntry.from_dict(row) for row in payload.get("frames", [])]


# --- stage E: labels/roi_measurements.csv ----------------------------------


@dataclass(frozen=True)
class RoiMeasurement:
    """One row of ``labels/roi_measurements.csv`` (section 8)."""

    field: str
    x0: int
    y0: int
    x1: int
    y1: int
    slot_id: int | None = None
    source_frame: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.field:
            raise SchemaError("roi measurement requires a field name")
        for name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaError(f"{name} must be an int")
            if value < 0:
                raise SchemaError(f"{name} must be non-negative")
        if self.x1 <= self.x0:
            raise SchemaError(f"x1 must be > x0 for {self.field}")
        if self.y1 <= self.y0:
            raise SchemaError(f"y1 must be > y0 for {self.field}")
        if self.slot_id is not None:
            if isinstance(self.slot_id, bool) or not isinstance(self.slot_id, int):
                raise SchemaError("slot_id must be an int or empty")
            if not 0 <= self.slot_id < SLOT_COUNT:
                raise SchemaError(
                    f"slot_id must be in [0, {SLOT_COUNT}), got {self.slot_id}"
                )

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_row(self) -> dict[str, str]:
        return {
            "field": self.field,
            "slot_id": "" if self.slot_id is None else str(self.slot_id),
            "x0": str(self.x0),
            "y0": str(self.y0),
            "x1": str(self.x1),
            "y1": str(self.y1),
            "source_frame": self.source_frame,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "RoiMeasurement":
        raw_slot = (row.get("slot_id") or "").strip()
        return cls(
            field=(row.get("field") or "").strip(),
            x0=int(row["x0"]),
            y0=int(row["y0"]),
            x1=int(row["x1"]),
            y1=int(row["y1"]),
            slot_id=int(raw_slot) if raw_slot else None,
            source_frame=(row.get("source_frame") or "").strip(),
            notes=(row.get("notes") or "").strip(),
        )


def write_roi_measurements_csv(
    path: Path | str, measurements: Iterable[RoiMeasurement]
) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(ROI_CSV_FIELDS))
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement.to_row())
            count += 1
    return count


def read_roi_measurements_csv(path: Path | str) -> list[RoiMeasurement]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f"{source}: empty ROI measurement file")
        missing = [f for f in ROI_CSV_FIELDS if f not in reader.fieldnames]
        if missing:
            raise SchemaError(
                f"{source}: missing columns: {', '.join(missing)}"
            )
        measurements = []
        for number, row in enumerate(reader, start=2):
            try:
                measurements.append(RoiMeasurement.from_row(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise SchemaError(f"{source}:{number}: invalid row: {exc}") from exc
    return measurements


__all__ = [
    "DATASET_DIRECTORIES",
    "FrameEntry",
    "README_TEXT",
    "ROI_CSV_FIELDS",
    "RoiMeasurement",
    "build_frame_label_skeleton",
    "create_skeleton",
    "frame_filename",
    "iter_frames_jsonl",
    "read_frame_manifest",
    "read_frames_jsonl",
    "read_roi_measurements_csv",
    "write_device_template",
    "write_frame_manifest",
    "write_frames_jsonl",
    "write_roi_measurements_csv",
]
