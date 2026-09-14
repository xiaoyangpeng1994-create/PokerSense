import ast
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import aa_passive_capture_intake_v1 as intake
from tools.aa_passive_capture_intake_v1 import (
    CONFIRMATION,
    FFMPEG_CONTRACT,
    GIB,
    build_dry_run_signoff,
    build_plan,
    run_authorized_capture,
    validate_authorization,
    validate_finalization_receipt,
    verify_finalization_receipt_from_recording,
)


NOW = datetime(2026, 9, 14, 10, 0, tzinfo=timezone.utc)


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authorization(root, **updates):
    value = {
        "schema_version": 1,
        "status": "AUTHORIZED_ONCE_FOR_PASSIVE_CAPTURE",
        "authorization_id": "aa-auth-dryrun01",
        "nonce": "a" * 64,
        "authorized_by": "owner",
        "authorized_at_utc": "2026-09-14T09:00:00+00:00",
        "expires_at_utc": "2026-09-14T11:00:00+00:00",
        "operator_confirmation": CONFIRMATION,
        "phase": "hardware_dry_run",
        "platform_id": "aa_poker",
        "capture_kind": "PASSIVE_VIDEO_ONLY",
        "session_id": "aa-live-dryrun01",
        "source_recording_group_id": "aa-group-dryrun01",
        "output_root": root.resolve().as_posix(),
        "requested_duration_seconds": 10,
        "device": {
            "friendly_name": "UGREEN 25854",
            "backend": "dshow",
            "video_size": [1920, 1080],
            "framerate": 30,
            "input_codec": "mjpeg",
            "phone_model": None,
            "android_version": None,
            "app_version": None,
            "orientation": None,
            "video_adapter_model": None,
            "capture_card_model": None,
            "capture_card_firmware": None,
            "capture_card_serial": None,
            "device_instance_id": None,
            "usb_vid_pid": None,
            "host_os": None,
            "driver_version": None,
            "uvc_color_space": None,
            "uvc_color_range": None,
            "dshow_input_name": None,
            "pnp_instance_suffix": None,
            "interface_number": None,
            "pnp_service": None,
            "pnp_class": None,
            "pnp_class_guid": None,
            "driver_device_id": None,
            "driver_provider": None,
            "driver_inf": None,
            "normalization_sha256": None,
            "layout_sha256": None,
            "hardware_fingerprint_sha256": None,
        },
        "limits": {
            "segment_seconds": 60,
            "size_limit_bytes": 20 * GIB,
            "minimum_free_bytes": 20 * GIB,
            "minimum_start_free_bytes": 25 * GIB,
            "graceful_stop_seconds": 15,
            "wall_clock_grace_seconds": 20,
        },
        "privacy": {
            "raw_media_private": True,
            "public_upload_allowed": False,
            "git_eligible": False,
            "audio_enabled": False,
            "post_capture_segment_hashing": True,
            "full_frames_in_public_bundle": False,
            "nicknames_in_metadata": False,
            "avatars_in_metadata": False,
            "direct_account_identifiers_in_metadata": False,
            "chat_or_room_identifiers_in_metadata": False,
            "privacy_review_status": "PENDING",
            "privacy_review_evidence_sha256": None,
            "retention_policy_id": "private-session-manual-retention-v1",
        },
        "rule_declaration": {
            "table_size": 8,
            "fields": {name: {
                "status": "UNKNOWN", "value": None, "evidence_sha256": [],
            } for name in (
                "small_blind", "big_blind", "ante", "ante_mode",
                "straddle_mode", "straddle_amount", "rake_percent",
                "rake_cap_bb", "rake_application", "rake_rounding",
                "rake_distribution", "minimum_chip")},
            "verification_status": "UNKNOWN",
            "author_review": None,
            "independent_review": None,
            "rule_fingerprint": None,
        },
        "identity_declaration": {
            "method": "session_pseudonym_v1",
            "direct_nickname_storage": False,
            "pseudonym_secret_in_manifest": False,
            "stable_player_keys_available": False,
            "mapping_artifact_sha256": None,
            "verification_status": "UNKNOWN",
            "avatar_storage": False,
            "direct_account_identifier_storage": False,
            "chat_or_room_identifier_storage": False,
            "privacy_review_evidence_sha256": None,
        },
        "forbidden_capabilities": {
            "recognition": False,
            "strategy": False,
            "provider": False,
            "equity": False,
            "advice": False,
            "live_control": False,
            "automated_input": False,
            "network": False,
            "audio": False,
            "emulator": False,
            "adb": False,
        },
        "prior_hardware_receipt_sha256": None,
        "prior_dry_run_finalization_sha256": None,
        "prior_dry_run_plan_sha256": None,
        "prior_dry_run_authorization_sha256": None,
    }
    value.update(updates)
    return value


def bind_hardware(auth, *, driver="driver-v1"):
    device = auth["device"]
    device.update({
        "phone_model": "Samsung test phone",
        "android_version": "test-android",
        "app_version": "aa-test-version",
        "orientation": "portrait",
        "video_adapter_model": "test-adapter",
        "capture_card_model": "UGREEN 25854",
        "capture_card_firmware": "test-firmware",
        "capture_card_serial": None,
        "device_instance_id": (
            "USB\\VID_1234&PID_ABCD&MI_00\\TEST-SUFFIX"),
        "usb_vid_pid": "VID_1234&PID_ABCD",
        "pnp_instance_suffix": "TEST-SUFFIX",
        "interface_number": "00",
        "pnp_service": "usbvideo",
        "pnp_class": "Camera",
        "pnp_class_guid": "TEST-CLASS-GUID",
        "driver_device_id": (
            "USB\\VID_1234&PID_ABCD&MI_00\\TEST-SUFFIX"),
        "host_os": "test-windows",
        "driver_version": driver,
        "driver_provider": "Microsoft",
        "driver_inf": "usbvideo.inf",
        "uvc_color_space": "test-color-space",
        "uvc_color_range": "test-color-range",
        "dshow_input_name": (
            "@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_00#"
            "test-suffix#{class}\\global"),
        "normalization_sha256": "1" * 64,
        "layout_sha256": "2" * 64,
    })
    payload = {key: device[key] for key in sorted(device)
               if key != "hardware_fingerprint_sha256"}
    device["hardware_fingerprint_sha256"] = intake._canonical_sha(payload)
    return device["hardware_fingerprint_sha256"]


def plan_file(tmp_path, auth=None, *, now=NOW):
    auth = auth or authorization(tmp_path)
    auth_path = write(tmp_path / "authorization.json", auth)
    plan = build_plan(auth, sha(auth_path), private_root=tmp_path, now=now)
    path = write(tmp_path / "plan.json", plan)
    return auth_path, path, plan


def fake_recording(target, duration, *, drop_frames=0, exit_code=0,
                   stop_reason="duration_or_source_end", end_pts="10",
                   frame_count=None):
    target.mkdir()
    segment = target / "segment_0000.mkv"
    segment.write_bytes(b"private-video-fixture")
    (target / "segments.csv").write_text(
        "segment_0000.mkv,0," + end_pts + "\n", encoding="utf-8")
    (target / "progress.log").write_text(
        "frame=" + str(frame_count if frame_count is not None else int(
            Decimal(end_pts) * 30)) + "\ndrop_frames=" + str(drop_frames)
        + "\ndup_frames=0\nout_time=00:00:" + end_pts
        + "\nprogress=end\n",
        encoding="utf-8")
    (target / "capture.log").write_text("", encoding="utf-8")
    write(target / "status.json", {
        "state": "stopped" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "device": "UGREEN 25854",
        "expected_platform": "AA_phone_8seat_unvalidated",
        "duration_limit_seconds": duration,
        "size_limit_bytes": 20 * GIB,
        "minimum_free_bytes": 20 * GIB,
        "audio": False,
        "recognition_running": False,
        "strategy_running": False,
        "ffmpeg_path": "TEST_ONLY_INJECTED_RECORD_FUNCTION",
        "ffmpeg_sha256": hashlib.sha256(
            b"TEST_ONLY_INJECTED_RECORD_FUNCTION").hexdigest(),
        "dshow_input_name": (
            "@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_00#"
            "test-suffix#{class}\\global"),
        "command_sha256": intake._canonical_sha(intake._build_ffmpeg_command(
            target, duration, "TEST_ONLY_INJECTED_RECORD_FUNCTION",
            "@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_00#"
            "test-suffix#{class}\\global")),
        "started_utc": "2026-09-14T10:00:00+00:00",
        "ended_utc": "2026-09-14T10:00:10+00:00",
        "stop_reason": stop_reason,
        "forced_termination": False,
        "segment_files": 1,
        "recorded_bytes": segment.stat().st_size,
    })


def test_plan_allows_capture_only_and_keeps_every_promotion_gate_closed(tmp_path):
    auth = authorization(tmp_path)
    path = write(tmp_path / "authorization.json", auth)
    plan = build_plan(auth, sha(path), private_root=tmp_path, now=NOW)
    assert plan["status"] == "READY_FOR_AUTHORIZED_HARDWARE_DRY_RUN"
    assert plan["capture_permitted"] is True
    assert plan["ffmpeg_contract"] == FFMPEG_CONTRACT
    assert plan["source_partition"] == "UNASSIGNED_QUARANTINE"
    assert plan["blockers_for_calibration"]
    for key in (
            "ready_for_offline_review", "ready_for_calibration",
            "model_fit_executed", "strategy_eligible", "advice_emitted",
            "live_control"):
        assert plan[key] is False


@pytest.mark.parametrize("field,value", [
    ("status", "NOT_DATA_REQUIRES_ONE_TIME_USER_AUTHORIZATION"),
    ("operator_confirmation", "继续开发"),
    ("capture_kind", "LIVE_ANALYSIS"),
    ("platform_id", "unknown"),
    ("requested_duration_seconds", True),
    ("requested_duration_seconds", 31),
    ("session_id", "../old-recording"),
])
def test_missing_wrong_or_overbroad_authorization_is_rejected(
        tmp_path, field, value):
    auth = authorization(tmp_path)
    auth[field] = value
    with pytest.raises(ValueError):
        validate_authorization(auth, private_root=tmp_path, now=NOW)


@pytest.mark.parametrize("capability", [
    "recognition", "strategy", "provider", "equity", "advice",
    "live_control", "automated_input", "network", "audio", "emulator",
    "adb",
])
def test_every_forbidden_capability_is_fail_closed(tmp_path, capability):
    auth = authorization(tmp_path)
    auth["forbidden_capabilities"][capability] = True
    with pytest.raises(ValueError, match="cannot_enable"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)


def test_expired_authorization_is_rejected(tmp_path):
    auth = authorization(
        tmp_path, expires_at_utc="2026-09-14T09:30:00+00:00")
    with pytest.raises(ValueError, match="expired"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)


def test_rule_identity_and_privacy_status_cannot_contradict_values(tmp_path):
    auth = authorization(tmp_path)
    auth["rule_declaration"]["fields"]["small_blind"]["value"] = "1"
    with pytest.raises(ValueError, match="unknown_rule_field"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)
    auth = authorization(tmp_path)
    auth["rule_declaration"]["fields"]["small_blind"][
        "status"] = "OPERATOR_DECLARED"
    with pytest.raises(ValueError, match="requires_value"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)
    auth = authorization(tmp_path)
    auth["identity_declaration"].update(
        stable_player_keys_available=True,
        mapping_artifact_sha256="e" * 64)
    with pytest.raises(ValueError, match="identity_status"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)
    auth = authorization(tmp_path)
    auth["privacy"]["avatars_in_metadata"] = True
    with pytest.raises(ValueError, match="private_no_audio"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)
    auth = authorization(tmp_path)
    bind_hardware(auth)
    auth["device"]["driver_version"] = "changed-after-fingerprint"
    with pytest.raises(ValueError, match="fingerprint_differs"):
        validate_authorization(auth, private_root=tmp_path, now=NOW)


def test_not_data_template_cannot_create_a_plan(tmp_path):
    template = Path(__file__).resolve().parents[2] / "configs" / "reproduction" / (
        "aa_passive_capture_authorization_v1.NOT-DATA.json")
    output = tmp_path / "plan.json"
    run = subprocess.run([
        sys.executable, "-m", "tools.aa_passive_capture_intake_v1", "plan",
        "--authorization", str(template), "--authorization-sha256",
        sha(template), "--output", str(output)], capture_output=True, text=True)
    assert run.returncode == 2
    assert not output.exists()


def test_development_capture_requires_a_distinct_signed_dry_run(tmp_path):
    dry_auth = authorization(tmp_path)
    bind_hardware(dry_auth)
    dry_auth_path, dry_plan_path, dry_plan = plan_file(tmp_path, dry_auth)
    receipt = run_authorized_capture(
        dry_plan_path, sha(dry_plan_path), dry_auth_path, sha(dry_auth_path),
        private_root=tmp_path,
        record_function=fake_recording, now=NOW)
    receipt_path = write(tmp_path / "dry-receipt-copy.json", receipt)
    signoff = build_dry_run_signoff(
        receipt, sha(receipt_path), reviewer="owner",
        signed_at_utc="2026-09-14T10:01:00+00:00")
    signoff_path = write(tmp_path / "dry-signoff.json", signoff)
    auth = authorization(
        tmp_path,
        authorization_id="aa-auth-session02",
        nonce="b" * 64,
        phase="development_capture",
        session_id="aa-live-session02",
        source_recording_group_id="aa-group-session02",
        authorized_at_utc="2026-09-14T10:01:30+00:00",
        requested_duration_seconds=60,
        prior_hardware_receipt_sha256=sha(signoff_path),
        prior_dry_run_finalization_sha256=sha(receipt_path),
        prior_dry_run_plan_sha256=sha(dry_plan_path),
        prior_dry_run_authorization_sha256=sha(dry_auth_path),
    )
    bind_hardware(auth)
    auth_path = write(tmp_path / "authorization-session02.json", auth)
    with pytest.raises(ValueError, match="finalization"):
        build_plan(
            auth, sha(auth_path), private_root=tmp_path,
            prior_hardware_receipt=signoff,
            prior_hardware_receipt_sha256=sha(signoff_path),
            now=datetime(2026, 9, 14, 10, 2, tzinfo=timezone.utc))
    plan = build_plan(
        auth, sha(auth_path), private_root=tmp_path,
        prior_hardware_receipt=signoff,
        prior_hardware_receipt_sha256=sha(signoff_path),
        prior_dry_run_finalization=receipt,
        prior_dry_run_finalization_sha256=sha(receipt_path),
        prior_dry_run_plan=dry_plan,
        prior_dry_run_plan_sha256=sha(dry_plan_path),
        prior_dry_run_authorization=dry_auth,
        prior_dry_run_authorization_sha256=sha(dry_auth_path),
        now=datetime(2026, 9, 14, 10, 2, tzinfo=timezone.utc))
    assert plan["status"] == "READY_FOR_AUTHORIZED_PASSIVE_CAPTURE"
    assert plan["strategy_eligible"] is False
    mismatched = copy.deepcopy(auth)
    bind_hardware(mismatched, driver="different-driver")
    mismatched_path = write(tmp_path / "authorization-mismatch.json", mismatched)
    with pytest.raises(ValueError, match="hardware_must_match"):
        build_plan(
            mismatched, sha(mismatched_path), private_root=tmp_path,
            prior_hardware_receipt=signoff,
            prior_hardware_receipt_sha256=sha(signoff_path),
            prior_dry_run_finalization=receipt,
            prior_dry_run_finalization_sha256=sha(receipt_path),
            prior_dry_run_plan=dry_plan,
            prior_dry_run_plan_sha256=sha(dry_plan_path),
            prior_dry_run_authorization=dry_auth,
            prior_dry_run_authorization_sha256=sha(dry_auth_path),
            now=datetime(2026, 9, 14, 10, 2, tzinfo=timezone.utc))
    (tmp_path / dry_auth["session_id"] / "status.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_plan(
            auth, sha(auth_path), private_root=tmp_path,
            prior_hardware_receipt=signoff,
            prior_hardware_receipt_sha256=sha(signoff_path),
            prior_dry_run_finalization=receipt,
            prior_dry_run_finalization_sha256=sha(receipt_path),
            prior_dry_run_plan=dry_plan,
            prior_dry_run_plan_sha256=sha(dry_plan_path),
            prior_dry_run_authorization=dry_auth,
            prior_dry_run_authorization_sha256=sha(dry_auth_path),
            now=datetime(2026, 9, 14, 10, 2, tzinfo=timezone.utc))


def test_wrong_plan_hash_never_calls_recorder_or_creates_session(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)
    calls = []
    with pytest.raises(ValueError, match="external_sha256"):
        run_authorized_capture(
            plan_path, "0" * 64, auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=lambda *args: calls.append(args), now=NOW)
    assert not calls
    assert not (tmp_path / plan["authorization"]["session_id"]).exists()
    assert not (tmp_path / ".aa-passive-capture-v1-authorizations").exists()


def test_observed_pnp_device_must_match_manifest_before_recorder_call(
        tmp_path, monkeypatch):
    auth = authorization(tmp_path)
    bind_hardware(auth)
    auth_path, plan_path, plan = plan_file(tmp_path, auth)
    observed = intake._device_probe_for_test(auth)
    observed["device_instance_id"] = "DIFFERENT\\INSTANCE"
    monkeypatch.setattr(intake, "_device_probe_for_test", lambda value: observed)
    calls = []
    with pytest.raises(ValueError, match="observed_capture_device_differs"):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=lambda *args: calls.append(args), now=NOW)
    assert not calls
    assert not (tmp_path / plan["authorization"]["session_id"]).exists()


def test_plan_and_authorization_sources_must_be_direct_private_json(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_plan = outside / "plan.json"
    outside_plan.write_bytes(plan_path.read_bytes())
    calls = []
    with pytest.raises(ValueError, match="direct_private_json"):
        run_authorized_capture(
            outside_plan, sha(outside_plan), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=lambda *args: calls.append(args), now=NOW)
    assert not calls
    assert not (tmp_path / plan["authorization"]["session_id"]).exists()


def test_valid_capture_is_one_shot_hashed_and_quarantined(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path,
        record_function=fake_recording, now=NOW)
    validate_finalization_receipt(receipt)
    assert receipt["status"] == "CAPTURE_FINALIZED_UNREVIEWED", receipt[
        "blockers"]
    assert receipt["source_integrity_status"] == (
        "SEGMENT_BYTES_HASHED_NOT_DECODED")
    assert receipt["segments"][0]["sha256"] == hashlib.sha256(
        b"private-video-fixture").hexdigest()
    assert receipt["source_partition"] == "UNASSIGNED_QUARANTINE"
    assert receipt["data_readiness"] == "BLOCKED"
    assert receipt["strategy_eligible"] is False
    assert not (tmp_path / ".aa-passive-capture-v1.device.lock").exists()
    with pytest.raises(ValueError, match="directory_must_be_new"):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=fake_recording, now=NOW)
    ledger = tmp_path / ".aa-passive-capture-v1-authorizations"
    assert len(list(ledger.glob("*.json"))) == 1
    original = (tmp_path / plan["authorization"]["session_id"]
                / "capture-finalization-receipt.json")
    copied_root = tmp_path / "aa-live-wrongcopy01"
    copied_root.mkdir()
    copied = copied_root / "capture-finalization-receipt.json"
    copied.write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="path_and_identity"):
        intake._load_private_session_receipt(copied, sha(copied), tmp_path)


def test_formal_recorder_executes_the_preflight_pinned_ffmpeg_binary(
        tmp_path, monkeypatch):
    commands = []
    current = datetime.now(timezone.utc)
    ffmpeg = tmp_path / "ffmpeg-test.exe"
    ffmpeg.write_bytes(b"test-ffmpeg-binary")
    ffmpeg_info = ffmpeg.stat()

    class Input:
        def close(self):
            pass

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            output = Path(command[-1].replace("%04d", "0000"))
            output.write_bytes(b"formal-recorder-segment")
            csv_path = Path(command[command.index("-segment_list") + 1])
            csv_path.write_text(
                "segment_0000.mkv,0,10\n", encoding="utf-8")
            progress = Path(command[command.index("-progress") + 1])
            progress.write_text(
                "frame=300\ndrop_frames=0\ndup_frames=0\n"
                "out_time=00:00:10.000000\nprogress=end\n",
                encoding="utf-8")
            self.stdin = Input()
            self.pid = 321
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    monkeypatch.setattr(
        intake, "_probe_ffmpeg_binary", lambda: {
            "path": ffmpeg.as_posix(), "sha256": sha(ffmpeg),
            "size_bytes": ffmpeg_info.st_size,
            "mtime_ns": ffmpeg_info.st_mtime_ns})
    monkeypatch.setattr(
        intake, "_probe_windows_capture_device", lambda ffmpeg_identity: {
            "probe_method": "windows_cim_pnp_dshow_v2",
            "friendly_name": "UGREEN 25854",
            "device_instance_id": (
                "USB\\VID_0000&PID_0000&MI_00\\TEST-SUFFIX"),
            "usb_vid_pid": "VID_0000&PID_0000",
            "pnp_instance_suffix": "TEST-SUFFIX",
            "interface_number": "00",
            "pnp_service": "usbvideo",
            "pnp_class": "Camera",
            "pnp_class_guid": "TEST-CLASS-GUID",
            "driver_device_id": (
                "USB\\VID_0000&PID_0000&MI_00\\TEST-SUFFIX"),
            "driver_version": "TEST-DRIVER",
            "driver_provider": "Microsoft",
            "driver_inf": "usbvideo.inf",
            "dshow_input_name": (
                "@device_pnp_\\\\?\\usb#vid_0000&pid_0000&mi_00#"
                "test-suffix#{class}\\global"),
            "status": "OK"})
    monkeypatch.setattr(
        intake.shutil, "disk_usage",
        lambda path: SimpleNamespace(free=30 * GIB))
    monkeypatch.setattr(intake.subprocess, "Popen", Process)
    monkeypatch.setattr(intake.time, "sleep", lambda seconds: None)
    auth = authorization(
        tmp_path,
        authorized_at_utc=(current - timedelta(minutes=1)).isoformat(),
        expires_at_utc=(current + timedelta(hours=1)).isoformat())
    auth_path, plan_path, _ = plan_file(tmp_path, auth, now=current)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, now=current)
    ledger = json.loads(next((tmp_path / (
        ".aa-passive-capture-v1-authorizations")).glob("*.json")).read_text())
    assert Path(ledger["ffmpeg_path"]).is_file()
    assert intake._hash_file_snapshot(
        Path(ledger["ffmpeg_path"]), GIB)[0] == ledger["ffmpeg_sha256"]
    assert receipt["status"] == "CAPTURE_FINALIZED_UNREVIEWED", receipt[
        "blockers"]
    assert receipt["forced_termination"] is False
    assert receipt["ffmpeg_binary_sha256"] == sha(ffmpeg)
    assert commands[0][0][0] == ffmpeg.as_posix()
    assert any(item.startswith("video=@device_pnp_") for item in commands[0][0])
    assert commands[0][1]["shell"] is False
    assert receipt["segments"][0]["sha256"] == hashlib.sha256(
        b"formal-recorder-segment").hexdigest()


def test_directshow_alternative_name_must_be_unique_and_device_bound():
    text = (
        '[dshow @ x] "UGREEN 25854" (video)\n'
        '[dshow @ x]   Alternative name '
        '"@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_00#'
        'test-suffix#{class}\\global"\n')
    expected = (
        "@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_00#"
        "test-suffix#{class}\\global")
    assert intake._parse_dshow_input_name(text, "UGREEN 25854") == expected
    with pytest.raises(ValueError, match="unique_dshow"):
        intake._parse_dshow_input_name(text + text, "UGREEN 25854")


def test_windows_probe_selects_only_usbvideo_from_composite_device(monkeypatch):
    calls = []
    cim = json.dumps({
        "friendly_name": "UGREEN 25854",
        "device_instance_id": (
            "USB\\VID_2B89&PID_5854&MI_00\\6&UNIT&0&0000"),
        "status": "OK",
        "pnp_service": "usbvideo",
        "pnp_class": "Camera",
        "pnp_class_guid": "{test-guid}",
        "driver_device_id": (
            "USB\\VID_2B89&PID_5854&MI_00\\6&UNIT&0&0000"),
        "driver_version": "10.0.1",
        "driver_provider": "Microsoft",
        "driver_inf": "usbvideo.inf",
    })
    dshow = (
        '[in#0 @ x] "UGREEN 25854" (video)\n'
        '[in#0 @ x]   Alternative name '
        '"@device_pnp_\\\\?\\usb#vid_2b89&pid_5854&mi_00#'
        '6&unit&0&0000#{class}\\global"\n'
        '[in#0 @ x] "数字音频接口 (UGREEN 25854)" (audio)\n')

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr=dshow)
        return SimpleNamespace(returncode=0, stdout=cim, stderr="")

    monkeypatch.setattr(intake.os, "name", "nt")
    monkeypatch.setattr(intake.shutil, "which", lambda name: "powershell.exe")
    monkeypatch.setattr(intake.subprocess, "run", run)
    result = intake._probe_windows_capture_device({"path": "ffmpeg.exe"})
    assert "Service -eq 'usbvideo'" in calls[1][-1]
    assert result["device_instance_id"].endswith("6&UNIT&0&0000")
    assert result["usb_vid_pid"] == "VID_2B89&PID_5854"
    assert result["interface_number"] == "00"
    assert result["pnp_service"] == "usbvideo"
    assert result["dshow_input_name"].startswith("@device_pnp_")


@pytest.mark.parametrize("field,value", [
    ("pnp_service", "usbaudio"),
    ("pnp_class", "MEDIA"),
    ("driver_device_id", "USB\\VID_1234&PID_ABCD&MI_02\\TEST-SUFFIX"),
    ("interface_number", "02"),
    ("pnp_instance_suffix", "OTHER"),
    ("dshow_input_name", (
        "@device_pnp_\\\\?\\usb#vid_1234&pid_abcd&mi_02#"
        "test-suffix#{class}\\global")),
])
def test_observed_device_rejects_audio_or_mismatched_interface(field, value):
    observed = intake._device_probe_for_test(authorization(Path.cwd()))
    observed[field] = value
    with pytest.raises(ValueError):
        intake._validate_device_probe(observed)


def test_dry_run_signoff_recomputes_receipt_from_recording(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=fake_recording, now=NOW)
    verify_finalization_receipt_from_recording(receipt, plan, sha(plan_path))
    receipt["segments"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest_hash|differs_from_recording"):
        verify_finalization_receipt_from_recording(
            receipt, plan, sha(plan_path))


def test_dry_run_signoff_requires_named_reviewer_after_capture_end(tmp_path):
    auth = authorization(tmp_path)
    bind_hardware(auth)
    auth_path, plan_path, _ = plan_file(tmp_path, auth)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=fake_recording, now=NOW)
    with pytest.raises(ValueError, match="signoff"):
        build_dry_run_signoff(
            receipt, "1" * 64, reviewer="REPLACE_ME",
            signed_at_utc="2026-09-14T10:01:00+00:00")
    with pytest.raises(ValueError, match="follow_capture_end"):
        build_dry_run_signoff(
            receipt, "1" * 64, reviewer="owner",
            signed_at_utc="2026-09-14T09:59:00+00:00")


def test_short_capture_cannot_be_signed_off_as_hardware_dry_run(tmp_path):
    auth = authorization(tmp_path)
    bind_hardware(auth)
    auth_path, plan_path, _ = plan_file(tmp_path, auth)

    def too_short(target, duration):
        fake_recording(target, duration, end_pts="4.999")

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=too_short, now=NOW)
    assert receipt["status"] == "CAPTURE_FINALIZED_UNREVIEWED"
    with pytest.raises(ValueError, match="successful_hardware_dry_run"):
        build_dry_run_signoff(
            receipt, "1" * 64, reviewer="owner",
            signed_at_utc="2026-09-14T10:01:00+00:00")


@pytest.mark.parametrize("mutation", [
    "segments", "metadata", "integrity", "exit", "forced", "blockers",
])
def test_success_receipt_cannot_hide_missing_or_failed_capture_evidence(
        tmp_path, mutation):
    auth = authorization(tmp_path)
    bind_hardware(auth)
    auth_path, plan_path, _ = plan_file(tmp_path, auth)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=fake_recording, now=NOW)
    tampered = copy.deepcopy(receipt)
    if mutation == "segments":
        tampered["segments"] = []
    elif mutation == "metadata":
        tampered["metadata_files"] = []
    elif mutation == "integrity":
        tampered["source_integrity_status"] = "NOT_HASHED"
    elif mutation == "exit":
        tampered["exit_code"] = 99
    elif mutation == "forced":
        tampered["forced_termination"] = True
    else:
        tampered["blockers"] = []
    with pytest.raises(ValueError):
        validate_finalization_receipt(tampered)
    with pytest.raises(ValueError):
        build_dry_run_signoff(
            tampered, "1" * 64, reviewer="owner",
            signed_at_utc="2026-09-14T10:01:00+00:00")


def test_nonce_replay_is_rejected_before_second_recorder_call(tmp_path):
    first_auth, first_path, _ = plan_file(tmp_path)
    run_authorized_capture(
        first_path, sha(first_path), first_auth, sha(first_auth),
        private_root=tmp_path,
        record_function=fake_recording, now=NOW)
    auth = authorization(
        tmp_path, authorization_id="aa-auth-second02",
        session_id="aa-live-second02", source_recording_group_id="aa-group-second02")
    auth_path = write(tmp_path / "authorization-second.json", auth)
    second_plan = build_plan(auth, sha(auth_path), private_root=tmp_path, now=NOW)
    second_path = write(tmp_path / "plan-second.json", second_plan)
    calls = []
    with pytest.raises(FileExistsError):
        run_authorized_capture(
            second_path, sha(second_path), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=lambda *args: calls.append(args), now=NOW)
    assert not calls
    assert not (tmp_path / "aa-live-second02").exists()


def test_existing_device_lock_blocks_before_recorder_call(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)
    (tmp_path / ".aa-passive-capture-v1.device.lock").write_text("occupied")
    calls = []
    with pytest.raises(FileExistsError):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=lambda *args: calls.append(args), now=NOW)
    assert not calls
    assert not (tmp_path / plan["authorization"]["session_id"]).exists()


def test_unchanged_device_lock_is_released_after_recorder_exception(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def failed(target, duration):
        target.mkdir()
        raise RuntimeError("injected recorder failure")

    with pytest.raises(RuntimeError, match="injected recorder failure"):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path, record_function=failed, now=NOW)
    assert not (tmp_path / ".aa-passive-capture-v1.device.lock").exists()
    failure = json.loads((tmp_path / plan["authorization"]["session_id"]
                          / "capture-attempt-failure.json").read_text())
    assert failure["status"] == "FAILED_QUARANTINED"
    assert failure["automatic_retry"] is False
    assert failure["files_deleted"] is False


def test_device_lock_changed_after_creation_is_preserved(tmp_path):
    auth_path, plan_path, _ = plan_file(tmp_path)
    lock = tmp_path / ".aa-passive-capture-v1.device.lock"

    def changed(target, duration):
        fake_recording(target, duration)
        with lock.open("ab") as stream:
            stream.write(b"externally-changed")

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=changed, now=NOW)
    assert receipt["status"] == "CAPTURE_FINALIZED_UNREVIEWED"
    assert lock.exists()
    assert lock.read_bytes().endswith(b"externally-changed")


def test_drop_or_unclean_stop_is_retained_as_failed_quarantine(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def failed(target, duration):
        fake_recording(target, duration, drop_frames=1)

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path,
        record_function=failed, now=NOW)
    assert receipt["status"] == "FAILED_QUARANTINED"
    assert "ffmpeg_reported_dropped_frames" in receipt["blockers"]
    target = tmp_path / plan["authorization"]["session_id"]
    assert (target / "segment_0000.mkv").exists()
    assert (target / "capture-finalization-receipt.json").exists()
    assert receipt["ready_for_offline_review"] is False
    assert not (tmp_path / ".aa-passive-capture-v1.device.lock").exists()
    validate_finalization_receipt(receipt)
    missing_reason = copy.deepcopy(receipt)
    missing_reason["blockers"].remove("ffmpeg_reported_dropped_frames")
    with pytest.raises(ValueError, match="visible_blocker_mismatch"):
        validate_finalization_receipt(missing_reason)


def test_failed_receipt_requires_a_specific_capture_failure(tmp_path):
    auth_path, plan_path, _ = plan_file(tmp_path)
    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=fake_recording, now=NOW)
    receipt["status"] = "FAILED_QUARANTINED"
    receipt["source_integrity_status"] = (
        "FAILED_OR_PARTIAL_SEGMENTS_PRESERVED")
    with pytest.raises(ValueError, match="specific_failure_blocker"):
        validate_finalization_receipt(receipt)
    receipt["blockers"].append("made_up_failure")
    with pytest.raises(ValueError, match="unknown_capture_failure_blocker"):
        validate_finalization_receipt(receipt)


def test_recorded_pts_cannot_exceed_authorized_duration(tmp_path):
    auth_path, plan_path, _ = plan_file(tmp_path)

    def overlong(target, duration):
        fake_recording(target, duration)
        (target / "segments.csv").write_text(
            "segment_0000.mkv,0,10.101\n", encoding="utf-8")

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=overlong, now=NOW)
    assert receipt["status"] == "FAILED_QUARANTINED"
    assert "recording_pts_exceed_authorized_duration" in receipt["blockers"]


@pytest.mark.parametrize("frames", [1, 1000])
def test_frame_density_must_match_declared_30fps_timeline(tmp_path, frames):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def wrong_density(target, duration):
        fake_recording(target, duration, frame_count=frames)

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=wrong_density, now=NOW)
    assert receipt["status"] == "FAILED_QUARANTINED"
    assert "ffmpeg_frame_density_differs_from_30fps_timeline" in receipt[
        "blockers"]
    validate_finalization_receipt(receipt)
    verify_finalization_receipt_from_recording(
        receipt, plan, sha(plan_path))


@pytest.mark.parametrize("failure", ["missing_out_time", "bad_time_order"])
def test_generated_quality_failure_receipt_remains_fully_verifiable(
        tmp_path, failure):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def broken(target, duration):
        fake_recording(target, duration)
        if failure == "missing_out_time":
            progress = (target / "progress.log").read_text(encoding="utf-8")
            progress = "\n".join(
                line for line in progress.splitlines()
                if not line.startswith("out_time=")) + "\n"
            (target / "progress.log").write_text(progress, encoding="utf-8")
        else:
            status = json.loads((target / "status.json").read_text())
            status["ended_utc"] = status["started_utc"]
            write(target / "status.json", status)

    receipt = run_authorized_capture(
        plan_path, sha(plan_path), auth_path, sha(auth_path),
        private_root=tmp_path, record_function=broken, now=NOW)
    assert receipt["status"] == "FAILED_QUARANTINED"
    validate_finalization_receipt(receipt)
    verify_finalization_receipt_from_recording(
        receipt, plan, sha(plan_path))


def test_runtime_exception_preserves_files_and_writes_failure_receipt(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def crash(target, duration):
        target.mkdir()
        (target / "segment_0000.mkv").write_bytes(b"partial")
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path,
            record_function=crash, now=NOW)
    target = tmp_path / plan["authorization"]["session_id"]
    assert (target / "segment_0000.mkv").read_bytes() == b"partial"
    failure = json.loads((target / "capture-attempt-failure.json").read_text())
    assert failure["status"] == "FAILED_QUARANTINED"
    assert failure["automatic_retry"] is False
    assert failure["files_deleted"] is False


def test_negative_ffmpeg_progress_is_rejected_as_malformed_metadata(tmp_path):
    auth_path, plan_path, plan = plan_file(tmp_path)

    def malformed(target, duration):
        fake_recording(target, duration, frame_count=-1)

    with pytest.raises(ValueError, match="invalid_progress_frame"):
        run_authorized_capture(
            plan_path, sha(plan_path), auth_path, sha(auth_path),
            private_root=tmp_path, record_function=malformed, now=NOW)
    target = tmp_path / plan["authorization"]["session_id"]
    assert (target / "capture-attempt-failure.json").exists()
    assert not (target / "capture-finalization-receipt.json").exists()


def test_module_import_graph_has_no_analysis_or_device_backend_dependency():
    source = Path(__file__).resolve().parents[2] / "tools" / (
        "aa_passive_capture_intake_v1.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = (
        "poker_engine.desktop", "poker_engine.strategy", "cv2", "adb",
        "tools.record_aa_capture_test", "tools.aa_record_session",
    )
    assert not any(name.startswith(banned) for name in imported)
