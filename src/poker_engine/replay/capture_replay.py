"""Integrity-checked real-capture Replay contract and deterministic runner."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.events import EventType
from poker_engine.core.observation import RawObservation, ValidationStatus
from poker_engine.core.serialization import deserialize
from poker_engine.core.state import PokerState, StateContext
from poker_engine.state_engine.platform_mapping import (
    CandidateMappingStatus,
    PlatformMappedStateEngine,
    PlatformSeatMapping,
)


class CaptureReplayError(ValueError):
    """Replay artifact, reference, or execution contract is invalid."""


class ReplayEvidenceKind(str, Enum):
    SYNTHETIC = "synthetic"
    REAL_CAPTURE = "real_capture"


class ReplayStage(str, Enum):
    STABLE_OBSERVATION = "stable_observation"
    RAW_FRAME = "raw_frame"


@dataclass(frozen=True)
class CalibrationReference:
    field_name: str
    path: Path
    sha256: str
    sample_count: int


@dataclass(frozen=True)
class ReplayExpectation:
    status: CandidateMappingStatus
    state_version: int
    event_types: tuple[EventType, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CaptureReplayFrame:
    frame_seq: int
    timestamp: datetime
    expected: ReplayExpectation
    observation: RawObservation | None = None
    frame_path: Path | None = None
    frame_sha256: str | None = None


@dataclass(frozen=True)
class CaptureReplay:
    replay_id: str
    artifact_path: Path
    artifact_sha256: str
    evidence_kind: ReplayEvidenceKind
    replay_stage: ReplayStage
    platform_id: str
    layout_id: str
    source_revision: str
    captured_at: datetime
    authorized: bool
    privacy_reviewed: bool
    platform_config_path: Path
    platform_config_sha256: str
    calibrations: Mapping[str, CalibrationReference]
    mapping: PlatformSeatMapping
    initial_state: PokerState
    frames: tuple[CaptureReplayFrame, ...]
    release_eligible: bool
    eligibility_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CaptureReplayReport:
    replay_id: str
    artifact_sha256: str
    frame_count: int
    status_counts: Mapping[str, int]
    field_quality_counts: Mapping[str, int]
    mismatches: tuple[str, ...]
    release_eligible: bool
    eligibility_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict:
        """Return a deterministic JSON-safe R6 quality-report payload."""
        return {
            "schema_version": 1,
            "report_type": "capture_replay_quality",
            "replay_id": self.replay_id,
            "artifact_sha256": self.artifact_sha256,
            "frame_count": self.frame_count,
            "status_counts": dict(self.status_counts),
            "field_quality_counts": dict(self.field_quality_counts),
            "passed": self.passed,
            "mismatches": list(self.mismatches),
            "release_eligible": self.release_eligible,
            "eligibility_reasons": list(self.eligibility_reasons),
        }


RawFrameRecognizer = Callable[[Path, int, datetime], RawObservation]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CaptureReplayError(f"{name} must be a lowercase SHA-256")
    if any(char not in "0123456789abcdef" for char in value):
        raise CaptureReplayError(f"{name} must be a lowercase SHA-256")
    return value


def _required_str(data: Mapping, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CaptureReplayError(f"{key} must be a non-empty string")
    return value


def _expect_keys(data: Mapping, expected: set[str], name: str) -> None:
    actual = set(data)
    if actual != expected:
        raise CaptureReplayError(
            f"{name} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _aware_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise CaptureReplayError(f"{name} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _require_aware_dt(result)
    except (TypeError, ValueError) as exc:
        raise CaptureReplayError(f"{name} must be timezone-aware") from exc
    return result


def _resolve_reference(asset_root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise CaptureReplayError(f"{name} must be a relative path")
    root = asset_root.resolve()
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise CaptureReplayError(f"{name} escapes asset_root")
    if not resolved.is_file():
        raise CaptureReplayError(f"{name} does not exist: {value}")
    return resolved


def _verified_reference(
    asset_root: Path,
    data: Mapping,
    name: str,
) -> tuple[Path, str]:
    path = _resolve_reference(asset_root, data.get("path"), f"{name}.path")
    expected = _require_sha256(data.get("sha256"), f"{name}.sha256")
    if _sha256(path) != expected:
        raise CaptureReplayError(f"{name} SHA-256 mismatch")
    return path, expected


def _json_object(path: Path, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureReplayError(f"{name} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CaptureReplayError(f"{name} must contain a JSON object")
    return value


def _verify_platform_identity(
    data: Mapping,
    platform_id: str,
    layout_id: str,
    name: str,
) -> None:
    if data.get("platform_id") != platform_id or data.get("layout_id") != layout_id:
        raise CaptureReplayError(f"{name} platform/layout mismatch")


def _slot_map(value: object, name: str) -> dict[int, int]:
    if not isinstance(value, dict):
        raise CaptureReplayError(f"{name} must be an object")
    result: dict[int, int] = {}
    for key, seat in value.items():
        try:
            slot = int(key)
        except (TypeError, ValueError) as exc:
            raise CaptureReplayError(f"{name} keys must be integers") from exc
        if str(slot) != key or not isinstance(seat, int) or isinstance(seat, bool):
            raise CaptureReplayError(f"{name} must map integer strings to seats")
        result[slot] = seat
    return result


def _parse_mapping(data: object, platform_id: str, layout_id: str):
    if not isinstance(data, dict):
        raise CaptureReplayError("seat_mapping must be an object")
    # Schema-v1 manifests predate explicit occupancy mapping. Treat its
    # absence as an empty mapping while still rejecting every unknown key.
    data = dict(data)
    data.setdefault("occupancy_slot_to_seat", {})
    data.setdefault("actor_observation_is_current", False)
    _expect_keys(data, {
        "platform_id", "layout_id", "version", "stack_slot_to_seat",
        "action_slot_to_seat", "actor_slot_to_seat",
        "dealer_slot_to_seat", "occupancy_slot_to_seat",
        "actor_observation_is_current",
    }, "seat_mapping")
    mapping_platform = _required_str(data, "platform_id")
    mapping_layout = _required_str(data, "layout_id")
    if mapping_platform != platform_id or mapping_layout != layout_id:
        raise CaptureReplayError("seat_mapping platform/layout mismatch")
    return PlatformSeatMapping(
        mapping_platform,
        mapping_layout,
        _required_str(data, "version"),
        _slot_map(data.get("stack_slot_to_seat"), "stack_slot_to_seat"),
        _slot_map(data.get("action_slot_to_seat"), "action_slot_to_seat"),
        _slot_map(data.get("actor_slot_to_seat"), "actor_slot_to_seat"),
        _slot_map(data.get("dealer_slot_to_seat"), "dealer_slot_to_seat"),
        _slot_map(
            data.get("occupancy_slot_to_seat", {}),
            "occupancy_slot_to_seat",
        ),
        data.get("actor_observation_is_current"),
    )


def _parse_expectation(data: object) -> ReplayExpectation:
    if not isinstance(data, dict):
        raise CaptureReplayError("frame.expected must be an object")
    _expect_keys(
        data, {"status", "state_version", "event_types", "reasons"},
        "frame.expected",
    )
    try:
        status = CandidateMappingStatus(data.get("status"))
        event_types = tuple(EventType(value) for value in data.get("event_types", []))
    except (TypeError, ValueError) as exc:
        raise CaptureReplayError("frame expected status/event type is invalid") from exc
    state_version = data.get("state_version")
    if not isinstance(state_version, int) or isinstance(state_version, bool):
        raise CaptureReplayError("frame expected state_version must be an int")
    reasons = tuple(data.get("reasons", []))
    if not all(isinstance(reason, str) and reason for reason in reasons):
        raise CaptureReplayError("frame expected reasons must be strings")
    return ReplayExpectation(status, state_version, event_types, reasons)


def _used_calibration_fields(observation: RawObservation) -> set[str]:
    names = set()
    aliases = {
        "stacks": "player_stacks",
        "dealer_pos": "dealer_button",
    }
    for field_name in (
        "hero_cards", "board_cards", "pot", "stacks", "bet_size",
        "action", "street", "dealer_pos", "actor",
    ):
        field = getattr(observation, field_name)
        if (
            field.validation_status is ValidationStatus.VALID
            and field.value is not None
        ):
            names.add(aliases.get(field_name, field_name))
    if any(slot.field.validation_status is ValidationStatus.VALID and
           slot.field.value is not None for slot in observation.slot_stacks):
        names.add("player_stacks")
    if any(slot.field.validation_status is ValidationStatus.VALID and
           slot.field.value is not None
           for slot in observation.slot_occupancies):
        names.add("occupancy")
    if any(slot.field.validation_status is ValidationStatus.VALID and
           slot.field.value is not None for slot in observation.slot_actions):
        names.add("action")
    return names


def load_capture_replay(
    path: str | Path,
    *,
    expected_sha256: str,
    asset_root: str | Path,
) -> CaptureReplay:
    """Load one Replay only after artifact and referenced-file verification."""
    artifact_path = Path(path).resolve()
    expected_artifact_hash = _require_sha256(
        expected_sha256, "expected_sha256"
    )
    if not artifact_path.is_file():
        raise CaptureReplayError("Replay artifact does not exist")
    actual_artifact_hash = _sha256(artifact_path)
    if actual_artifact_hash != expected_artifact_hash:
        raise CaptureReplayError("Replay artifact SHA-256 mismatch")
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureReplayError("Replay artifact is not valid UTF-8 JSON") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CaptureReplayError("unsupported Replay schema_version")
    _expect_keys(data, {
        "schema_version", "replay_id", "evidence_kind", "replay_stage",
        "platform", "source", "calibrations", "seat_mapping",
        "initial_state", "frames",
    }, "Replay")

    try:
        evidence_kind = ReplayEvidenceKind(data.get("evidence_kind"))
        replay_stage = ReplayStage(data.get("replay_stage"))
    except ValueError as exc:
        raise CaptureReplayError("invalid evidence_kind or replay_stage") from exc
    replay_id = _required_str(data, "replay_id")
    platform = data.get("platform")
    source = data.get("source")
    if not isinstance(platform, dict) or not isinstance(source, dict):
        raise CaptureReplayError("platform and source must be objects")
    _expect_keys(
        platform, {"platform_id", "layout_id", "config"}, "platform"
    )
    _expect_keys(source, {
        "authorized", "privacy_reviewed", "captured_at", "source_revision",
    }, "source")
    platform_id = _required_str(platform, "platform_id")
    layout_id = _required_str(platform, "layout_id")
    platform_reference = platform.get("config")
    if not isinstance(platform_reference, dict):
        raise CaptureReplayError("platform.config must be an object")
    _expect_keys(platform_reference, {"path", "sha256"}, "platform.config")
    root = Path(asset_root)
    platform_path, platform_hash = _verified_reference(
        root, platform_reference, "platform.config"
    )
    platform_config = _json_object(platform_path, "platform.config")
    _verify_platform_identity(
        platform_config, platform_id, layout_id, "platform.config"
    )
    mapping = _parse_mapping(data.get("seat_mapping"), platform_id, layout_id)

    calibrations_data = data.get("calibrations")
    if not isinstance(calibrations_data, dict):
        raise CaptureReplayError("calibrations must be an object")
    calibrations: dict[str, CalibrationReference] = {}
    for field_name, reference in calibrations_data.items():
        if not isinstance(field_name, str) or not field_name:
            raise CaptureReplayError("calibration field name is invalid")
        if not isinstance(reference, dict):
            raise CaptureReplayError("calibration reference must be an object")
        _expect_keys(
            reference, {"path", "sha256", "sample_count"},
            f"calibrations.{field_name}",
        )
        ref_path, ref_hash = _verified_reference(
            root, reference, f"calibrations.{field_name}"
        )
        sample_count = reference.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 0
        ):
            raise CaptureReplayError("calibration sample_count must be >= 0")
        calibration_data = _json_object(
            ref_path, f"calibrations.{field_name}"
        )
        _verify_platform_identity(
            calibration_data,
            platform_id,
            layout_id,
            f"calibrations.{field_name}",
        )
        fields = calibration_data.get("fields")
        field_data = fields.get(field_name) if isinstance(fields, dict) else None
        if not isinstance(field_data, dict):
            raise CaptureReplayError(
                f"calibrations.{field_name} does not declare its field"
            )
        if field_data.get("samples") != sample_count:
            raise CaptureReplayError(
                f"calibrations.{field_name} sample_count mismatch"
            )
        calibrations[field_name] = CalibrationReference(
            field_name, ref_path, ref_hash, sample_count
        )

    try:
        initial_state = deserialize(PokerState, data.get("initial_state"))
    except Exception as exc:
        raise CaptureReplayError("initial_state is invalid") from exc
    frames_data = data.get("frames")
    if not isinstance(frames_data, list) or not frames_data:
        raise CaptureReplayError("frames must be a non-empty array")
    frames = []
    previous_seq = -1
    previous_timestamp: datetime | None = None
    inline_used_fields: set[str] = set()
    for index, frame_data in enumerate(frames_data):
        if not isinstance(frame_data, dict):
            raise CaptureReplayError(f"frames[{index}] must be an object")
        expected_frame_keys = {
            "frame_seq", "timestamp", "expected",
            "observation" if replay_stage is ReplayStage.STABLE_OBSERVATION
            else "frame_asset",
        }
        _expect_keys(frame_data, expected_frame_keys, f"frames[{index}]")
        frame_seq = frame_data.get("frame_seq")
        if (
            not isinstance(frame_seq, int)
            or isinstance(frame_seq, bool)
            or frame_seq <= previous_seq
        ):
            raise CaptureReplayError("frame_seq must be strictly increasing")
        timestamp = _aware_datetime(frame_data.get("timestamp"), "frame.timestamp")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise CaptureReplayError("frame timestamps must be non-decreasing")
        expectation = _parse_expectation(frame_data.get("expected"))
        observation = None
        frame_path = None
        frame_hash = None
        if replay_stage is ReplayStage.STABLE_OBSERVATION:
            try:
                observation = deserialize(
                    RawObservation, frame_data.get("observation")
                )
            except Exception as exc:
                raise CaptureReplayError("frame observation is invalid") from exc
            if observation.frame_seq != frame_seq or observation.timestamp != timestamp:
                raise CaptureReplayError("frame and observation identity mismatch")
            inline_used_fields.update(_used_calibration_fields(observation))
        else:
            frame_reference = frame_data.get("frame_asset")
            if not isinstance(frame_reference, dict):
                raise CaptureReplayError("frame_asset must be an object")
            _expect_keys(
                frame_reference, {"path", "sha256"},
                f"frames[{index}].frame_asset",
            )
            frame_path, frame_hash = _verified_reference(
                root, frame_reference,
                f"frames[{index}].frame_asset",
            )
        frames.append(CaptureReplayFrame(
            frame_seq, timestamp, expectation, observation, frame_path, frame_hash
        ))
        previous_seq = frame_seq
        previous_timestamp = timestamp

    if not isinstance(source.get("authorized"), bool) or not isinstance(
        source.get("privacy_reviewed"), bool
    ):
        raise CaptureReplayError(
            "source authorization/privacy flags must be bool"
        )
    authorized = source["authorized"]
    privacy_reviewed = source["privacy_reviewed"]
    captured_at = _aware_datetime(source.get("captured_at"), "captured_at")
    source_revision = _required_str(source, "source_revision")
    eligibility_reasons: list[str] = []
    if evidence_kind is not ReplayEvidenceKind.REAL_CAPTURE:
        eligibility_reasons.append("evidence_not_real_capture")
    if replay_stage is not ReplayStage.RAW_FRAME:
        eligibility_reasons.append("replay_does_not_start_from_raw_frame")
    else:
        eligibility_reasons.append("recognizer_not_executed")
    if not authorized:
        eligibility_reasons.append("capture_not_authorized")
    if not privacy_reviewed:
        eligibility_reasons.append("privacy_review_missing")
    for field_name in sorted(inline_used_fields):
        reference = calibrations.get(field_name)
        if reference is None:
            eligibility_reasons.append(f"calibration_missing:{field_name}")
        elif reference.sample_count == 0:
            eligibility_reasons.append(f"calibration_empty:{field_name}")

    return CaptureReplay(
        replay_id,
        artifact_path,
        actual_artifact_hash,
        evidence_kind,
        replay_stage,
        platform_id,
        layout_id,
        source_revision,
        captured_at,
        authorized,
        privacy_reviewed,
        platform_path,
        platform_hash,
        MappingProxyType(calibrations),
        mapping,
        initial_state,
        tuple(frames),
        not eligibility_reasons,
        tuple(eligibility_reasons),
    )


def run_capture_replay(
    replay: CaptureReplay,
    *,
    raw_frame_recognizer: RawFrameRecognizer | None = None,
) -> CaptureReplayReport:
    """Execute each Replay frame and compare status, version, events, reasons."""
    if not isinstance(replay, CaptureReplay):
        raise TypeError("replay must be a CaptureReplay")
    if replay.replay_stage is ReplayStage.RAW_FRAME and raw_frame_recognizer is None:
        raise CaptureReplayError("raw_frame Replay requires a recognizer")
    if raw_frame_recognizer is not None and not callable(raw_frame_recognizer):
        raise TypeError("raw_frame_recognizer must be callable or None")

    engine = PlatformMappedStateEngine(replay.mapping)
    current = replay.initial_state
    statuses: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    mismatches: list[str] = []
    used_fields: set[str] = set()
    for index, frame in enumerate(replay.frames):
        if replay.replay_stage is ReplayStage.RAW_FRAME:
            assert frame.frame_path is not None and raw_frame_recognizer is not None
            observation = raw_frame_recognizer(
                frame.frame_path, frame.frame_seq, frame.timestamp
            )
            if not isinstance(observation, RawObservation):
                raise CaptureReplayError("recognizer must return RawObservation")
            if (
                observation.frame_seq != frame.frame_seq
                or observation.timestamp != frame.timestamp
            ):
                raise CaptureReplayError("recognizer returned wrong frame identity")
        else:
            assert frame.observation is not None
            observation = frame.observation
        for field_name in _used_calibration_fields(observation):
            used_fields.add(field_name)
        if replay.replay_stage is ReplayStage.RAW_FRAME:
            assert frame.frame_sha256 is not None
            evidence_fields = {
                "hero_cards": observation.hero_cards,
                "board_cards": observation.board_cards,
                "pot": observation.pot,
                "player_stacks": observation.stacks,
                "bet_size": observation.bet_size,
                "action": observation.action,
                "street": observation.street,
                "dealer_button": observation.dealer_pos,
                "actor": observation.actor,
            }
            for slot in observation.slot_stacks:
                evidence_fields[f"player_stacks[{slot.slot_id}]"] = slot.field
            for slot in observation.slot_actions:
                evidence_fields[f"action[{slot.slot_id}]"] = slot.field
            for field_name, field in evidence_fields.items():
                if (
                    field.validation_status is not ValidationStatus.VALID
                    or field.value is None
                ):
                    continue
                if field.evidence.get("frame_sha256") != frame.frame_sha256:
                    eligibility_reasons_key = (
                        f"observation_frame_hash_mismatch:{field_name}"
                    )
                    mismatches.append(
                        f"frame[{index}] seq={frame.frame_seq}: "
                        f"{eligibility_reasons_key}"
                    )
                if field.evidence.get("recognizer_revision") != (
                    replay.source_revision
                ):
                    mismatches.append(
                        f"frame[{index}] seq={frame.frame_seq}: "
                        f"recognizer_revision_mismatch:{field_name}"
                    )
        for field_name in (
            "hero_cards", "board_cards", "pot", "stacks", "bet_size",
            "action", "street", "dealer_pos", "actor",
        ):
            qualities[getattr(observation, field_name).validation_status.value] += 1
        transition = engine.transition(current, observation, StateContext(current))
        if transition.changed and transition.validation.is_valid:
            status = CandidateMappingStatus.EXACT
            current = transition.state
        elif transition.validation.is_valid:
            status = CandidateMappingStatus.NO_ACTION
        else:
            reason_set = set(transition.validation.errors)
            ambiguous_reasons = {
                "actor_missing", "conflicting_actor_slots",
                "conflicting_action_labels", "multiple_legal_action_interpretations",
            }
            status = (
                CandidateMappingStatus.AMBIGUOUS
                if reason_set & ambiguous_reasons else CandidateMappingStatus.INVALID
            )
        statuses[status.value] += 1
        actual_events = tuple(event.event_type for event in transition.events)
        actual_reasons = tuple(transition.validation.errors)
        expected = frame.expected
        prefix = f"frame[{index}] seq={frame.frame_seq}"
        if status is not expected.status:
            mismatches.append(
                f"{prefix}: status {status.value} != {expected.status.value}"
            )
        if current.state_version != expected.state_version:
            mismatches.append(
                f"{prefix}: state_version {current.state_version} != "
                f"{expected.state_version}"
            )
        if actual_events != expected.event_types:
            mismatches.append(f"{prefix}: event_types mismatch")
        if actual_reasons != expected.reasons:
            mismatches.append(f"{prefix}: reasons mismatch")

    eligibility_reasons = [
        reason for reason in replay.eligibility_reasons
        if reason != "recognizer_not_executed"
    ]
    if replay.replay_stage is ReplayStage.RAW_FRAME:
        if not used_fields:
            eligibility_reasons.append("no_valid_recognized_fields")
        for field_name in sorted(used_fields):
            reference = replay.calibrations.get(field_name)
            if reference is None:
                eligibility_reasons.append(f"calibration_missing:{field_name}")
            elif reference.sample_count == 0:
                eligibility_reasons.append(f"calibration_empty:{field_name}")
    eligibility_reasons = list(dict.fromkeys(eligibility_reasons))
    return CaptureReplayReport(
        replay.replay_id,
        replay.artifact_sha256,
        len(replay.frames),
        MappingProxyType(dict(sorted(statuses.items()))),
        MappingProxyType(dict(sorted(qualities.items()))),
        tuple(mismatches),
        not eligibility_reasons and not mismatches,
        tuple(eligibility_reasons),
    )


__all__ = [
    "CaptureReplay",
    "CaptureReplayError",
    "CaptureReplayReport",
    "ReplayEvidenceKind",
    "ReplayStage",
    "load_capture_replay",
    "run_capture_replay",
]
