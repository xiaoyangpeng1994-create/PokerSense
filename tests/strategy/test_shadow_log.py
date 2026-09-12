import hashlib
import json

import pytest

from poker_engine.strategy.shadow_log import (
    ShadowWalFollower,
    ShadowWalWriter,
    analyze_shadow_wal,
    optimization_flags,
    summarize_aa8_observation,
)


SHA = "a" * 64


def append(writer, frame, *, status="ABSTAIN", blockers=("dealer_not_canonical",)):
    return writer.append(
        source_frame_seq=frame,
        source_sha256=SHA,
        source_ref=f"C:/evidence/{frame}.png",
        pts_seconds=str(frame / 30),
        vision={"current_actor": 4, "wager_status": "WAGERS_UNKNOWN",
                "known_stack_count": 7, "hero_cards": ["As", "Kd"],
                "street": "preflop", "stack_vector_complete": False},
        strategy_status=status,
        blockers=blockers,
        stage_timings_ms={"strategy_input_gate": .25},
    )


def test_append_only_hash_chain_and_analysis(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-1", {"source": "test"})
    first = append(writer, 10)
    second = append(writer, 11, status="WAITING", blockers=("hero_not_actor",))
    receipt = writer.close()
    assert first != second
    assert receipt.records == 2
    assert receipt.wal_sha256 == hashlib.sha256(
        writer.wal_path.read_bytes()
    ).hexdigest()
    report = analyze_shadow_wal(writer.wal_path)
    assert report["hash_chain_valid"]
    assert report["statuses"] == {"ABSTAIN": 1, "WAITING": 1}
    assert report["advice_emitted"] is False
    assert len(report["optimization_examples"][
        "actor_visible_but_strategy_blocked"]) == 1
    timing = report["stage_timing_summary"]["strategy_input_gate"]
    assert timing == {
        "count": 2, "p50_ms": .25, "p95_ms": .25,
        "p99_ms": .25, "max_ms": .25,
    }
    queue = report["optimization_queue"]
    assert queue[0]["priority"] == "P0"
    assert queue[0]["frames"] == 1
    wager = next(item for item in queue if item["flag"] == "street_wager_unknown")
    assert wager["frames"] == 2
    assert wager["longest_intervals"][0]["frames"] == 2


def test_nonmonotonic_frame_fails_closed_without_receipt(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-2", {})
    append(writer, 10)
    with pytest.raises(ValueError, match="strictly increasing"):
        append(writer, 10)
    with pytest.raises(RuntimeError, match="closed or failed"):
        append(writer, 11)
    with pytest.raises(RuntimeError, match="cannot produce receipt"):
        writer.close()
    assert not writer.receipt_path.exists()


def test_tamper_is_detected(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-3", {})
    append(writer, 10)
    writer.close()
    text = writer.wal_path.read_text(encoding="utf-8").replace(
        '"source_frame_seq":10', '"source_frame_seq":12'
    )
    writer.wal_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        analyze_shadow_wal(writer.wal_path)


def test_invalid_timing_is_rejected_even_with_recomputed_hash(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-timing", {})
    append(writer, 10)
    writer.close()
    row = json.loads(writer.wal_path.read_text(encoding="utf-8"))
    row["stage_timings_ms"]["strategy_input_gate"] = -1
    from poker_engine.strategy.shadow_log import _digest
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    row["record_sha256"] = _digest(body)
    writer.wal_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stage timing"):
        analyze_shadow_wal(writer.wal_path)


@pytest.mark.parametrize("session", ("../escape", "bad name", "", "x" * 81))
def test_session_path_is_bounded(session, tmp_path):
    with pytest.raises(ValueError, match="session_id"):
        ShadowWalWriter(tmp_path, session, {})


def test_summary_omits_image_pixels_and_player_names():
    row = {
        "scene_supported": True,
        "current_actor": 4,
        "actor_evidence": {"reason": "timer", "image": [[1, 2, 3]]},
        "player_name": "private nickname",
        "image": [[1, 2, 3]],
        "cards": {"hero": ["As", "Kd"]},
        "stacks": {"0": {"value": "100"}, "1": {"value": None}},
        "observed_state_v2": {
            "street_candidate": "preflop", "observed_epoch": "h1",
            "unallocated_positive_cash": [{"amount": "10"}],
            "participants": {"1": {"state": "empty"}},
        },
        "causal_street_wagers_v2": {
            "status": "WAGERS_UNKNOWN", "reason": "unknown",
        },
        "visual_scope": {"status": "BASE_CANDIDATE_UNVERIFIED"},
    }
    summary = summarize_aa8_observation(row)
    assert summary["known_stack_count"] == 1
    assert summary["known_or_not_applicable_stack_count"] == 2
    assert summary["stack_vector_complete"] is False
    assert summary["unallocated_positive_cash_count"] == 1
    assert "image" not in summary and "player_name" not in summary
    assert "image" not in summary["actor_evidence"]
    assert summary["dealer_seat_candidate"] is None
    assert summary["hand_commitments_candidate"] is None
    assert summary["action_line_candidate"] is None


def test_optimization_flags_keep_special_modes_out_of_base_tuning():
    flags = optimization_flags(
        {"current_actor": 4, "known_stack_count": 0,
         "stack_vector_complete": False, "wager_status": "WAGERS_UNKNOWN",
         "hero_cards": None, "street": None},
        "DEFERRED_SPECIAL_MODE", ("deferred_special_mode:insurance",),
    )
    assert flags == ("deferred_special_mode_not_base_tuning",)


def test_candidate_progress_is_distinct_from_missing_and_canonical():
    flags = optimization_flags(
        {"current_actor": None, "known_stack_count": 8,
         "stack_vector_complete": True, "hero_cards": ["As", "Kd"],
         "street": "preflop", "dealer_seat_candidate": 7,
         "hand_commitments_candidate": {"0": "2"},
         "action_line_candidate": "raise"},
        "ABSTAIN", (
            "dealer_not_canonical", "hand_commitments_not_canonical",
            "actions_not_complete_or_canonical",
        ),
    )
    assert flags == (
        "dealer_candidate_requires_independent_validation",
        "hand_ledger_requires_reconciliation_and_validation",
        "action_line_requires_completeness_validation",
    )


def test_postflop_provider_gap_is_not_mislabeled_visual_action_gap():
    flags = optimization_flags(
        {"current_actor": None, "known_stack_count": 8,
         "stack_vector_complete": True, "hero_cards": ["As", "Kd"],
         "street": "flop", "dealer_seat_candidate": 7,
         "hand_commitments_candidate": {"0": "2"},
         "action_line_candidate": None},
        "ABSTAIN", ("actions_not_complete_or_canonical",),
    )
    assert flags == ("postflop_strategy_provider_not_released",)


def test_manifest_and_receipt_cannot_be_overwritten(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-4", {})
    append(writer, 1)
    writer.close()
    with pytest.raises(FileExistsError):
        ShadowWalWriter(tmp_path, "session-4", {})
    receipt = json.loads(writer.receipt_path.read_text(encoding="utf-8"))
    assert receipt["records"] == 1


def test_manifest_cannot_override_no_advice_safety_fields(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-5", {
        "mode": "live", "image_pixels_stored": True, "schema_version": 99,
    })
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "offline_shadow_no_advice"
    assert manifest["image_pixels_stored"] is False
    assert manifest["schema_version"] == 1
    writer.abort()
    assert not writer.receipt_path.exists()


def test_follower_reads_growing_wal_with_bounded_metrics(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-follow", {})
    follower = ShadowWalFollower(writer.wal_path)
    append(writer, 10)
    assert follower.poll() == 1
    assert follower.poll() == 0
    append(writer, 11)
    assert follower.poll() == 1
    snapshot = follower.snapshot()
    assert snapshot["records"] == 2
    assert snapshot["last_source_frame_seq"] == 11
    assert snapshot["partial_line_bytes"] == 0
    assert snapshot["failed"] is False
    assert snapshot["stage_timing_histograms"]["strategy_input_gate"] == {
        "count": 2, "p50_upper_bound_ms": .25,
        "p95_upper_bound_ms": .25, "p99_upper_bound_ms": .25,
        "max_ms": .25,
    }
    writer.close()


def test_follower_buffers_incomplete_last_line(tmp_path):
    source = ShadowWalWriter(tmp_path, "source-complete", {})
    append(source, 10)
    source.close()
    payload = source.wal_path.read_bytes()
    target = tmp_path / "growing.jsonl"
    midpoint = len(payload) // 2
    target.write_bytes(payload[:midpoint])
    follower = ShadowWalFollower(target)
    assert follower.poll() == 0
    assert follower.snapshot()["partial_line_bytes"] == midpoint
    with target.open("ab") as stream:
        stream.write(payload[midpoint:])
    assert follower.poll() == 1
    assert follower.snapshot()["partial_line_bytes"] == 0


def test_follower_fails_permanently_on_tampered_record(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-follow-tamper", {})
    append(writer, 10)
    writer.close()
    value = writer.wal_path.read_text(encoding="utf-8").replace(
        '"source_frame_seq":10', '"source_frame_seq":12'
    )
    writer.wal_path.write_text(value, encoding="utf-8")
    follower = ShadowWalFollower(writer.wal_path)
    with pytest.raises(ValueError, match="hash chain"):
        follower.poll()
    assert follower.snapshot()["failed"]
    with pytest.raises(RuntimeError, match="has failed"):
        follower.poll()


def test_follower_detects_truncation(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-follow-truncate", {})
    append(writer, 10)
    follower = ShadowWalFollower(writer.wal_path)
    assert follower.poll() == 1
    writer.abort()
    writer.wal_path.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        follower.poll()


def test_shadow_math_can_be_logged_but_actionable_output_is_rejected(tmp_path):
    writer = ShadowWalWriter(tmp_path, "session-shadow-math", {})
    writer.append(
        source_frame_seq=1, source_sha256=SHA, source_ref="C:/evidence/1.png",
        pts_seconds="0.1", vision={}, strategy_status="READY", blockers=(),
        stage_timings_ms={"equity": 2.0}, provider_executed=True,
        equity_executed=True, shadow_result={
            "status": "SIMULATION", "method": "exact",
            "gross_expected_chips": "10", "configured_net_expected_chips": "9",
            "rake": "1", "rule_fingerprint": "b" * 64,
        },
    )
    writer.close()
    report = analyze_shadow_wal(writer.wal_path)
    assert report["provider_executions"] == 1
    assert report["equity_executions"] == 1
    assert report["shadow_result_statuses"] == {"SIMULATION": 1}
    follower = ShadowWalFollower(writer.wal_path)
    assert follower.poll() == 1
    assert follower.snapshot()["equity_executions"] == 1

    blocked = ShadowWalWriter(tmp_path, "session-actionable", {})
    with pytest.raises(ValueError, match="actionable or unknown"):
        blocked.append(
            source_frame_seq=1, source_sha256=SHA,
            source_ref="C:/evidence/1.png", pts_seconds=None, vision={},
            strategy_status="READY", blockers=(), stage_timings_ms={},
            shadow_result={"preferred_action": "raise"},
        )
    blocked.abort()
