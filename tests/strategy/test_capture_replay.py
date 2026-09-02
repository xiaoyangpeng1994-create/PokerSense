from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Street,
)
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from poker_engine.core.opponents import PlayerState
from poker_engine.core.serialization import serialize
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount
from poker_engine.replay import (
    CaptureReplayError,
    load_capture_replay,
    run_capture_replay,
)

from .helpers import card


NOW = datetime(2026, 8, 23, 12, tzinfo=timezone.utc)
CALIBRATED_FIELDS = (
    "hero_cards", "pot", "action", "dealer_button", "actor",
    "player_stacks",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return sha256(path)


def player(seat, *, hero=False):
    return PlayerState(
        "hero" if hero else f"p{seat}",
        seat,
        (Position.BTN, Position.SB, Position.BB)[seat],
        ChipAmount("100"),
        ChipAmount("0"),
        ChipAmount("0"),
        PlayerStatus.ACTIVE,
        True,
        hero,
        seat == 0,
    )


def initial_state():
    return PokerState(
        1,
        "capture-replay-hand",
        Street.PREFLOP,
        (card("As"), card("Kd")),
        (),
        (player(0), player(1), player(2, hero=True)),
        ChipAmount("0"),
        ChipAmount("0"),
        ChipAmount("0"),
        2,
    )


def field(value=None, status=ValidationStatus.UNKNOWN):
    return ObservationField(
        value,
        1.0,
        "replay-test-recognizer",
        {"frame_sha256": "a" * 64},
        NOW,
        status,
    )


def observation(
    *, frame_seq=100, timestamp=NOW,
    frame_sha256_value="a" * 64,
    recognizer_revision="test-revision",
):
    def at_time(value=None, status=ValidationStatus.UNKNOWN):
        return ObservationField(
            value,
            1.0,
            "replay-test-recognizer",
            {
                "frame_sha256": frame_sha256_value,
                "recognizer_revision": recognizer_revision,
            },
            timestamp,
            status,
        )

    return RawObservation(
        frame_seq,
        timestamp,
        at_time((card("As"), card("Kd")), ValidationStatus.VALID),
        at_time(),
        at_time(ChipAmount("20"), ValidationStatus.VALID),
        at_time(),
        at_time(),
        at_time(ActionType.BET, ValidationStatus.VALID),
        at_time(),
        at_time(40, ValidationStatus.VALID),
        at_time(10, ValidationStatus.VALID),
        1.0,
        (SlotObservation(
            30,
            at_time(ChipAmount("80"), ValidationStatus.VALID),
        ),),
        (),
    )


def file_ref(root, name, content=b"evidence"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": sha256(path)}


def json_ref(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)
    return {"path": name, "sha256": sha256(path)}


def recognize(path, seq, timestamp):
    return observation(
        frame_seq=seq,
        timestamp=timestamp,
        frame_sha256_value=sha256(path),
    )


def empty_observation(seq, timestamp):
    unknown = ObservationField(
        None,
        1.0,
        "replay-test-recognizer",
        {},
        timestamp,
        ValidationStatus.UNKNOWN,
    )
    return RawObservation(
        seq, timestamp, unknown, unknown, unknown, unknown, unknown,
        unknown, unknown, unknown, unknown, 1.0,
    )


def replay_manifest(
    root,
    *,
    stage="stable_observation",
    evidence_kind="synthetic",
    authorized=True,
    privacy_reviewed=True,
):
    platform_ref = json_ref(root, "configs/platform.json", {
        "schema_version": 1,
        "platform_id": "synthetic-platform",
        "layout_id": "three-seat-v1",
    })
    calibrations = {}
    for name in CALIBRATED_FIELDS:
        ref = json_ref(root, f"calibration/{name}.json", {
            "schema_version": 1,
            "platform_id": "synthetic-platform",
            "layout_id": "three-seat-v1",
            "fields": {name: {"samples": 10}},
        })
        calibrations[name] = {**ref, "sample_count": 10}
    frame = {
        "frame_seq": 100,
        "timestamp": NOW.isoformat(),
        "expected": {
            "status": "EXACT",
            "state_version": 2,
            "event_types": ["bet"],
            "reasons": [],
        },
    }
    if stage == "raw_frame":
        frame["frame_asset"] = file_ref(root, "frames/100.bin", b"frame")
    else:
        frame["observation"] = serialize(observation())
    return {
        "schema_version": 1,
        "replay_id": "replay-contract-test",
        "evidence_kind": evidence_kind,
        "replay_stage": stage,
        "platform": {
            "platform_id": "synthetic-platform",
            "layout_id": "three-seat-v1",
            "config": platform_ref,
        },
        "source": {
            "authorized": authorized,
            "privacy_reviewed": privacy_reviewed,
            "captured_at": NOW.isoformat(),
            "source_revision": "test-revision",
        },
        "calibrations": calibrations,
        "seat_mapping": {
            "platform_id": "synthetic-platform",
            "layout_id": "three-seat-v1",
            "version": "mapping-v1",
            "stack_slot_to_seat": {"30": 0, "31": 1, "32": 2},
            "action_slot_to_seat": {"20": 0, "21": 1, "22": 2},
            "actor_slot_to_seat": {"10": 0, "11": 1, "12": 2},
            "dealer_slot_to_seat": {"40": 0, "41": 1, "42": 2},
        },
        "initial_state": serialize(initial_state()),
        "frames": [frame],
    }


def load_manifest(root, manifest):
    path = root / "replay.json"
    digest = write_json(path, manifest)
    return load_capture_replay(
        path, expected_sha256=digest, asset_root=root
    )


def test_stable_observation_replay_runs_but_is_not_release_evidence(tmp_path):
    replay = load_manifest(tmp_path, replay_manifest(tmp_path))
    report = run_capture_replay(replay)
    assert report.passed
    assert report.status_counts == {"EXACT": 1}
    assert not report.release_eligible
    assert set(report.eligibility_reasons) == {
        "evidence_not_real_capture",
        "replay_does_not_start_from_raw_frame",
    }


def test_raw_frame_replay_requires_recognizer_before_eligibility(tmp_path):
    replay = load_manifest(
        tmp_path,
        replay_manifest(
            tmp_path, stage="raw_frame", evidence_kind="real_capture"
        ),
    )
    assert not replay.release_eligible
    assert replay.eligibility_reasons == ("recognizer_not_executed",)
    with pytest.raises(CaptureReplayError, match="requires a recognizer"):
        run_capture_replay(replay)


def test_hash_pinned_raw_frame_executes_and_becomes_release_eligible(tmp_path):
    replay = load_manifest(
        tmp_path,
        replay_manifest(
            tmp_path, stage="raw_frame", evidence_kind="real_capture"
        ),
    )
    report = run_capture_replay(
        replay,
        raw_frame_recognizer=recognize,
    )
    assert report.passed
    assert report.release_eligible
    assert report.eligibility_reasons == ()
    assert report.field_quality_counts["valid"] == 5
    payload = report.to_dict()
    assert payload["schema_version"] == 1
    assert payload["report_type"] == "capture_replay_quality"
    assert payload["passed"] is True
    assert payload["release_eligible"] is True
    json.dumps(payload)


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("action", "calibration_missing:action"),
        ("player_stacks", "calibration_missing:player_stacks"),
    ],
)
def test_recognized_field_without_calibration_blocks_release(
    tmp_path, field_name, reason,
):
    manifest = replay_manifest(
        tmp_path, stage="raw_frame", evidence_kind="real_capture"
    )
    del manifest["calibrations"][field_name]
    replay = load_manifest(tmp_path, manifest)
    report = run_capture_replay(
        replay,
        raw_frame_recognizer=recognize,
    )
    assert report.passed
    assert not report.release_eligible
    assert reason in report.eligibility_reasons


def test_zero_sample_calibration_blocks_release(tmp_path):
    manifest = replay_manifest(
        tmp_path, stage="raw_frame", evidence_kind="real_capture"
    )
    manifest["calibrations"]["action"]["sample_count"] = 0
    calibration_path = tmp_path / "calibration/action.json"
    write_json(calibration_path, {
        "schema_version": 1,
        "platform_id": "synthetic-platform",
        "layout_id": "three-seat-v1",
        "fields": {"action": {"samples": 0}},
    })
    manifest["calibrations"]["action"]["sha256"] = sha256(
        calibration_path
    )
    replay = load_manifest(tmp_path, manifest)
    report = run_capture_replay(
        replay,
        raw_frame_recognizer=recognize,
    )
    assert "calibration_empty:action" in report.eligibility_reasons


@pytest.mark.parametrize(
    ("authorized", "privacy", "reason"),
    [
        (False, True, "capture_not_authorized"),
        (True, False, "privacy_review_missing"),
    ],
)
def test_authorization_and_privacy_are_independent_release_gates(
    tmp_path, authorized, privacy, reason,
):
    replay = load_manifest(
        tmp_path,
        replay_manifest(
            tmp_path,
            stage="raw_frame",
            evidence_kind="real_capture",
            authorized=authorized,
            privacy_reviewed=privacy,
        ),
    )
    report = run_capture_replay(
        replay,
        raw_frame_recognizer=recognize,
    )
    assert reason in report.eligibility_reasons


def test_expected_output_mismatch_is_reported_not_silently_accepted(tmp_path):
    manifest = replay_manifest(tmp_path)
    manifest["frames"][0]["expected"]["state_version"] = 99
    replay = load_manifest(tmp_path, manifest)
    report = run_capture_replay(replay)
    assert not report.passed
    assert "state_version 2 != 99" in report.mismatches[0]


def test_artifact_hash_is_mandatory_and_verified(tmp_path):
    manifest = replay_manifest(tmp_path)
    path = tmp_path / "replay.json"
    write_json(path, manifest)
    with pytest.raises(CaptureReplayError, match="artifact SHA-256 mismatch"):
        load_capture_replay(
            path, expected_sha256="0" * 64, asset_root=tmp_path
        )
    with pytest.raises(CaptureReplayError, match="lowercase SHA-256"):
        load_capture_replay(path, expected_sha256="bad", asset_root=tmp_path)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("platform", "platform.config SHA-256 mismatch"),
        ("calibration", "calibrations.action SHA-256 mismatch"),
        ("frame", "frames[0].frame_asset SHA-256 mismatch"),
    ],
)
def test_every_external_reference_is_hash_verified(tmp_path, target, message):
    manifest = replay_manifest(
        tmp_path, stage="raw_frame", evidence_kind="real_capture"
    )
    if target == "platform":
        manifest["platform"]["config"]["sha256"] = "0" * 64
    elif target == "calibration":
        manifest["calibrations"]["action"]["sha256"] = "0" * 64
    else:
        manifest["frames"][0]["frame_asset"]["sha256"] = "0" * 64
    with pytest.raises(CaptureReplayError, match=re.escape(message)):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize("target", ("platform", "calibration"))
def test_referenced_json_must_declare_matching_platform_layout(tmp_path, target):
    manifest = replay_manifest(tmp_path)
    if target == "platform":
        path = tmp_path / "configs/platform.json"
        write_json(path, {
            "schema_version": 1,
            "platform_id": "different-platform",
            "layout_id": "three-seat-v1",
        })
        manifest["platform"]["config"]["sha256"] = sha256(path)
        message = "platform.config platform/layout mismatch"
    else:
        path = tmp_path / "calibration/action.json"
        write_json(path, {
            "schema_version": 1,
            "platform_id": "synthetic-platform",
            "layout_id": "different-layout",
            "fields": {"action": {"samples": 10}},
        })
        manifest["calibrations"]["action"]["sha256"] = sha256(path)
        message = "calibrations.action platform/layout mismatch"
    with pytest.raises(CaptureReplayError, match=re.escape(message)):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize("fault", ("missing_field", "sample_count"))
def test_calibration_file_must_prove_the_declared_field(tmp_path, fault):
    manifest = replay_manifest(tmp_path)
    path = tmp_path / "calibration/action.json"
    fields = (
        {"other": {"samples": 10}}
        if fault == "missing_field" else {"action": {"samples": 9}}
    )
    write_json(path, {
        "schema_version": 1,
        "platform_id": "synthetic-platform",
        "layout_id": "three-seat-v1",
        "fields": fields,
    })
    manifest["calibrations"]["action"]["sha256"] = sha256(path)
    with pytest.raises(CaptureReplayError, match="declare|sample_count"):
        load_manifest(tmp_path, manifest)


def test_reference_path_cannot_escape_asset_root(tmp_path):
    manifest = replay_manifest(tmp_path)
    manifest["platform"]["config"]["path"] = "../outside.json"
    with pytest.raises(CaptureReplayError, match="escapes asset_root"):
        load_manifest(tmp_path, manifest)


def test_mapping_must_match_platform_and_layout(tmp_path):
    manifest = replay_manifest(tmp_path)
    manifest["seat_mapping"]["layout_id"] = "wrong-layout"
    with pytest.raises(CaptureReplayError, match="platform/layout mismatch"):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize("location", ("top", "source", "frame"))
def test_versioned_contract_rejects_unknown_fields(tmp_path, location):
    manifest = replay_manifest(tmp_path)
    target = {
        "top": manifest,
        "source": manifest["source"],
        "frame": manifest["frames"][0],
    }[location]
    target["future_guess"] = True
    with pytest.raises(CaptureReplayError, match="fields mismatch"):
        load_manifest(tmp_path, manifest)


def test_authorization_flags_must_be_boolean(tmp_path):
    manifest = replay_manifest(tmp_path)
    manifest["source"]["authorized"] = "yes"
    with pytest.raises(CaptureReplayError, match="flags must be bool"):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize("fault", ("sequence", "timestamp"))
def test_frame_order_must_be_monotonic(tmp_path, fault):
    manifest = replay_manifest(tmp_path)
    second = json.loads(json.dumps(manifest["frames"][0]))
    second["frame_seq"] = 101
    second["observation"]["frame_seq"] = 101
    second_time = NOW + timedelta(seconds=1)
    second["timestamp"] = second_time.isoformat()
    second["observation"]["timestamp"] = second_time.isoformat()
    if fault == "sequence":
        second["frame_seq"] = 100
    else:
        second["timestamp"] = (NOW - timedelta(seconds=1)).isoformat()
        second["observation"]["timestamp"] = second["timestamp"]
    manifest["frames"].append(second)
    with pytest.raises(CaptureReplayError, match="strictly|non-decreasing"):
        load_manifest(tmp_path, manifest)


def test_raw_frame_cannot_embed_precomputed_observation(tmp_path):
    manifest = replay_manifest(
        tmp_path, stage="raw_frame", evidence_kind="real_capture"
    )
    manifest["frames"][0]["observation"] = serialize(observation())
    with pytest.raises(CaptureReplayError, match="fields mismatch"):
        load_manifest(tmp_path, manifest)


@pytest.mark.parametrize("fault", ("frame_hash", "revision"))
def test_recognized_evidence_must_bind_frame_and_recognizer_revision(
    tmp_path, fault,
):
    replay = load_manifest(
        tmp_path,
        replay_manifest(
            tmp_path, stage="raw_frame", evidence_kind="real_capture"
        ),
    )

    def recognizer(path, seq, timestamp):
        return observation(
            frame_seq=seq,
            timestamp=timestamp,
            frame_sha256_value=(
                "0" * 64 if fault == "frame_hash" else sha256(path)
            ),
            recognizer_revision=(
                "wrong-revision" if fault == "revision" else "test-revision"
            ),
        )

    report = run_capture_replay(replay, raw_frame_recognizer=recognizer)
    assert not report.passed
    assert not report.release_eligible
    assert any(fault.split("_")[0] in item for item in report.mismatches)


def test_raw_replay_with_no_valid_recognized_fields_is_not_evidence(tmp_path):
    manifest = replay_manifest(
        tmp_path, stage="raw_frame", evidence_kind="real_capture"
    )
    manifest["frames"][0]["expected"] = {
        "status": "NO_ACTION",
        "state_version": 1,
        "event_types": [],
        "reasons": [],
    }
    replay = load_manifest(tmp_path, manifest)
    report = run_capture_replay(
        replay,
        raw_frame_recognizer=lambda path, seq, timestamp: empty_observation(
            seq, timestamp
        ),
    )
    assert report.passed
    assert not report.release_eligible
    assert "no_valid_recognized_fields" in report.eligibility_reasons


@pytest.mark.parametrize("wrong", ("type", "sequence", "timestamp"))
def test_recognizer_output_identity_is_verified(tmp_path, wrong):
    replay = load_manifest(
        tmp_path,
        replay_manifest(
            tmp_path, stage="raw_frame", evidence_kind="real_capture"
        ),
    )

    def recognizer(path, seq, timestamp):
        if wrong == "type":
            return None
        if wrong == "sequence":
            return observation(frame_seq=seq + 1, timestamp=timestamp)
        return observation(frame_seq=seq, timestamp=timestamp + timedelta(seconds=1))

    with pytest.raises(CaptureReplayError, match="return|identity"):
        run_capture_replay(replay, raw_frame_recognizer=recognizer)
