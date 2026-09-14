from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path

import pytest

from poker_engine.strategy.decision_opportunities_v1 import (
    audit_decision_opportunities,
)
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2


def rule():
    profile = AARuleProfileV2.from_dict({
        "schema_version": 2, "table_size": 6, "small_blind": "1",
        "big_blind": "2", "ante": "0", "ante_mode": "none",
        "straddle_mode": "none", "straddle_amount": "0",
        "rake_percent": "0", "rake_cap_bb": "0",
        "rake_application": "all_pots", "rake_rounding": "exact",
        "rake_distribution": "proportional_all_pots", "minimum_chip": "1",
        "verification_status": "simulation", "source": "synthetic-reviewed-rule",
    })
    return {
        "rule_profile_id": "aa-reviewed-rules",
        "rule_fingerprint": profile.fingerprint,
        "table_size": 6, "small_blind": "1", "big_blind": "2",
        "ante": "0", "ante_mode": "none", "straddle_mode": "none",
        "straddle_amount": "0", "rake_percent": "0", "rake_cap_bb": "0",
        "rake_application": "all_pots", "rake_rounding": "exact",
        "rake_distribution": "proportional_all_pots", "minimum_chip": "1",
        "verification_status": "simulation", "source": "synthetic-reviewed-rule",
        "source_sha256": "b" * 64, "independent_review_status": "APPROVED",
    }


def evidence(seed, start=1):
    return {
        "actor_frame": start, "actor_pts": str(start),
        "actor_sha256": seed * 64,
        "menu_frame": start + 1, "menu_pts": str(start + 1),
        "menu_sha256": seed * 64,
        "state_frame": start + 2, "state_pts": str(start + 2),
        "state_sha256": seed * 64,
        "action_onset_frame": start + 3, "action_onset_pts": str(start + 3),
        "action_onset_sha256": seed * 64,
        "action_confirmation_frame": start + 4,
        "action_confirmation_pts": str(start + 4),
        "action_confirmation_sha256": seed * 64,
        "review_bundle_sha256": "f" * 64,
    }


def snapshot():
    return {
        "dealer_seat": 0, "board": [],
        "actor_status": "ACTIVE", "actor_stack": "100",
        "actor_street_committed": "0", "actor_hand_committed": "0",
        "current_bet": "6", "pot_before": "9", "to_call": "6",
        "minimum_raise_increment": "4", "betting_reopened": True,
        "public_history_through_sequence": [],
        "state_quality": "REVIEWED_PREDECISION",
        "decision_price_fraction": {"numerator": 2, "denominator": 5},
    }


def menu():
    return [
        {"action": "fold", "min_amount": "0", "max_amount": "0",
         "amount_semantics": "none"},
        {"action": "call", "min_amount": "6", "max_amount": "6",
         "amount_semantics": "additional"},
        {"action": "raise", "min_amount": "10", "max_amount": "100",
         "amount_semantics": "total_street"},
    ]


def special():
    return {"insurance": "ABSENT_REVIEWED", "mushroom": "ABSENT_REVIEWED",
            "bomb": "ABSENT_REVIEWED", "buyin_overlay": "ABSENT_REVIEWED",
            "block_state_updates": False, "evidence_sha256": "9" * 64}


def document():
    declared_rule = rule()
    fingerprint = declared_rule["rule_fingerprint"]
    sessions, hands, ledger, details = [], [], [], []
    participant_sessions = {}
    for index, (sid, split, seat, action, seed) in enumerate((
            ("train-session", "training", 4, "fold", "1"),
            ("validation-session", "validation", 4, "fold", "2")), start=1):
        hid, oid = sid + "-hand", sid + "-opportunity"
        sessions.append({
            "session_id": sid, "source_recording_group_id": sid + "-recording",
            "split": split, "platform_id": "synthetic-aa",
            "rule_profile_id": "aa-reviewed-rules",
            "rule_fingerprint": fingerprint,
            "source_recording_sha256": ("8" if index == 1 else "9") * 64,
            "source_audit_sha256": seed * 64,
            "sample_manifest_sha256": (str(index + 2) * 64),
            "first_frame": 0, "last_frame": 30, "start_pts": "0",
            "end_pts_exclusive": "31", "complete_hand_ids": [hid],
            "censored_edge_intervals": [], "coverage_status": "INDEPENDENT_COMPLETE",
            "coverage_evidence_sha256": (str(index + 4) * 64),
        })
        seat_map = {}
        for physical_seat in range(8):
            if physical_seat >= 6:
                player = "EMPTY"
            elif physical_seat == seat:
                player = "stable-opponent"
            elif physical_seat == 3:
                player = "hero-player"
            else:
                player = f"{sid}-seat-{physical_seat}"
            seat_map[str(physical_seat)] = player
            if player != "EMPTY":
                participant_sessions.setdefault(player, set()).add(sid)
        hands.append({
            "hand_id": hid, "session_id": sid, "start_frame": 0,
            "last_gameplay_frame": 30, "last_gameplay_pts": "30",
            "end_frame": 30, "next_start_frame": None,
            "temporal_complete": True, "dealer_seat": 0,
            "occupied_seats": list(range(6)), "seat_player_map": seat_map,
            "table_size": 6, "rule_profile_id": "aa-reviewed-rules",
            "rule_fingerprint": fingerprint, "observed_optional_straddle": None,
            "special_mode_summary": {
                name: "ABSENT_REVIEWED" for name in (
                    "insurance", "mushroom", "bomb", "buyin_overlay")},
            "public_action_timeline_sha256": (str(index + 6) * 64),
            "expected_opportunity_count": 6,
            "coverage_status": "INDEPENDENT_COMPLETE",
            "timeline_binding_status": "INDEPENDENTLY_VERIFIED",
        })
        hero_identity = {
            "opportunity_id": sid + "-hero-bet",
            "canonical_opportunity_id": sid + ":hero:1",
            "session_id": sid, "hand_id": hid, "decision_sequence": 1,
            "actor_player_id": "hero-player", "actor_seat": 3,
            "is_hero": True,
        }
        hero_snapshot = snapshot()
        hero_snapshot.update(
            current_bet="2", pot_before="3", to_call="2",
            minimum_raise_increment="2")
        hero_seed = str(index + 2)
        hero_observed = {
            "action": "raise", "amount": "6",
            "amount_semantics": "total_street"}
        ledger.append(hero_identity)
        details.append({
            **hero_identity, "position": "UTG", "street": "preflop",
            "table_size": 6, "active_count": 6,
            "pot_eligible_seats": list(range(6)),
            "pending_action_seats": [3, 4, 5, 0, 1, 2],
            "rule_fingerprint": fingerprint,
            "predecision_snapshot": hero_snapshot,
            "evidence": evidence(hero_seed, 1), "legal_menu": [
                {"action": "fold", "min_amount": "0", "max_amount": "0",
                 "amount_semantics": "none"},
                {"action": "call", "min_amount": "2", "max_amount": "2",
                 "amount_semantics": "additional"},
                {"action": "raise", "min_amount": "4", "max_amount": "100",
                 "amount_semantics": "total_street"}],
            "observed_action": hero_observed, "special_mode_state": special(),
            "rule_binding_status": "VERIFIED", "row_status": "REVIEWED_COMPLETE",
            "audit_status": "reviewed", "unknown_reasons": [],
            "author_review": "APPROVED", "independent_review": "APPROVED",
        })
        identity = {
            "opportunity_id": oid, "canonical_opportunity_id": sid + ":1",
            "session_id": sid, "hand_id": hid, "decision_sequence": 2,
            "actor_player_id": "stable-opponent", "actor_seat": seat,
            "is_hero": False,
        }
        ledger.append(identity)
        opponent_snapshot = snapshot()
        opponent_snapshot["public_history_through_sequence"] = [{
            "decision_sequence": 1, "actor_player_id": "hero-player",
            "actor_seat": 3, "street": "preflop", **hero_observed,
            "source_sha256": hero_seed * 64}]
        details.append({
            **identity, "position": "HJ", "street": "preflop", "table_size": 6,
            "active_count": 6, "pot_eligible_seats": list(range(6)),
            "pending_action_seats": [4, 5, 0, 1, 2],
            "rule_fingerprint": fingerprint,
            "predecision_snapshot": opponent_snapshot,
            "evidence": evidence(seed, 6), "legal_menu": menu(),
            "observed_action": {
                "action": action, "amount": "0" if action == "fold" else "6",
                "amount_semantics": "none" if action == "fold" else "additional"},
            "special_mode_state": special(), "rule_binding_status": "VERIFIED",
            "row_status": "REVIEWED_COMPLETE", "audit_status": "reviewed",
            "unknown_reasons": [], "author_review": "APPROVED",
            "independent_review": "APPROVED",
        })
        history = [deepcopy(opponent_snapshot["public_history_through_sequence"][0]), {
            "decision_sequence": 2, "actor_player_id": "stable-opponent",
            "actor_seat": seat, "street": "preflop", "action": action,
            "amount": "0", "amount_semantics": "none",
            "source_sha256": seed * 64}]
        pot_eligible = set(range(6))
        pot_eligible.remove(seat)
        pending = [5, 0, 1, 2]
        positions = {5: "CO", 0: "BTN", 1: "SB", 2: "BB"}
        extra_seeds = "5678" if index == 1 else "abcd"
        for sequence_number, (actor_seat, action_seed) in enumerate(
                zip((5, 0, 1, 2), extra_seeds), start=3):
            player = seat_map[str(actor_seat)]
            extra_identity = {
                "opportunity_id": f"{sid}-tail-{sequence_number}",
                "canonical_opportunity_id": f"{sid}:tail:{sequence_number}",
                "session_id": sid, "hand_id": hid,
                "decision_sequence": sequence_number,
                "actor_player_id": player, "actor_seat": actor_seat,
                "is_hero": False}
            committed = Decimal("1") if actor_seat == 1 else (
                Decimal("2") if actor_seat == 2 else Decimal(0))
            to_call = Decimal("6") - committed
            price = Fraction(to_call) / (
                Fraction(Decimal("9")) + Fraction(to_call))
            extra_snapshot = snapshot()
            extra_snapshot.update(
                actor_stack=str(Decimal("100") - committed),
                actor_street_committed=str(committed),
                actor_hand_committed=str(committed), to_call=str(to_call),
                public_history_through_sequence=deepcopy(history),
                decision_price_fraction={"numerator": price.numerator,
                                         "denominator": price.denominator})
            extra_menu = [
                {"action": "fold", "min_amount": "0", "max_amount": "0",
                 "amount_semantics": "none"},
                {"action": "call", "min_amount": str(to_call),
                 "max_amount": str(to_call), "amount_semantics": "additional"},
                {"action": "raise", "min_amount": "10", "max_amount": "100",
                 "amount_semantics": "total_street"}]
            extra_observed = {
                "action": "fold", "amount": "0", "amount_semantics": "none"}
            ledger.append(extra_identity)
            details.append({
                **extra_identity, "position": positions[actor_seat],
                "street": "preflop", "table_size": 6,
                "active_count": len(pot_eligible),
                "pot_eligible_seats": sorted(pot_eligible),
                "pending_action_seats": list(pending),
                "rule_fingerprint": fingerprint,
                "predecision_snapshot": extra_snapshot,
                "evidence": evidence(action_seed, 1 + 5 * (sequence_number - 1)),
                "legal_menu": extra_menu, "observed_action": extra_observed,
                "special_mode_state": special(), "rule_binding_status": "VERIFIED",
                "row_status": "REVIEWED_COMPLETE", "audit_status": "reviewed",
                "unknown_reasons": [], "author_review": "APPROVED",
                "independent_review": "APPROVED"})
            history.append({
                "decision_sequence": sequence_number, "actor_player_id": player,
                "actor_seat": actor_seat, "street": "preflop", **extra_observed,
                "source_sha256": action_seed * 64})
            pot_eligible.remove(actor_seat)
            pending.pop(0)
    return {
        "schema_version": 1, "dataset_id": "synthetic-contract-v1",
        "source_kind": "synthetic",
        "dataset_scope": {
            "calibration_unit": "per_stable_opponent",
            "target_player_ids": ["stable-opponent"],
            "hero_player_id": "hero-player",
            "included_streets": ["preflop"], "included_active_counts": [6],
            "included_modes": ["ordinary"],
            "contiguous_session_policy": (
                "all_complete_hands_and_explicit_censored_edges"),
            "all_table_decisions_required": True},
        "platform_binding": {"platform_id": "synthetic-aa",
                             "capture_path": "synthetic", "emulator_used": False},
        "rule_profiles": [declared_rule],
        "split_protocol": {
            "frozen_before_label_review": True,
            "training_session_ids": ["train-session"],
            "validation_session_ids": ["validation-session"],
            "split_evidence_sha256": "c" * 64},
        "coverage_review": {
            "status": "INDEPENDENT_COMPLETE", "author_reviewer": "author",
            "independent_reviewer": "reviewer", "evidence_sha256": "d" * 64,
            "expected_opportunity_count": 12, "listed_opportunity_count": 12},
        "sessions": sessions,
        "participants": [{
            "player_id": player,
            "identity_scope": "cross_session" if len(session_ids) > 1
            else "same_session", "session_ids": sorted(session_ids),
            "identity_status": "VERIFIED", "identity_evidence_sha256": "e" * 64,
            "independent_review_status": "APPROVED"}
            for player, session_ids in sorted(participant_sessions.items())],
        "hands": hands, "opportunity_ledger": ledger, "opportunities": details,
    }


def as_physical(data):
    declared = data["rule_profiles"][0]
    profile = AARuleProfileV2.from_dict({
        "schema_version": 2,
        **{key: declared[key] for key in (
            "table_size", "small_blind", "big_blind", "ante", "ante_mode",
            "straddle_mode", "straddle_amount", "rake_percent", "rake_cap_bb",
            "rake_application", "rake_rounding", "rake_distribution",
            "minimum_chip")},
        "verification_status": "live_verified", "source": "physical-review",
    })
    declared.update(verification_status="live_verified", source="physical-review",
                    rule_fingerprint=profile.fingerprint)
    data["source_kind"] = "reviewed_physical_capture_card"
    data["platform_binding"] = {
        "platform_id": "aa_poker", "capture_path": "physical_phone_capture_card",
        "emulator_used": False}
    for session in data["sessions"]:
        session["platform_id"] = "aa_poker"
        session["rule_fingerprint"] = profile.fingerprint
    for hand in data["hands"]:
        hand["rule_fingerprint"] = profile.fingerprint
    for row in data["opportunities"]:
        row["rule_fingerprint"] = profile.fingerprint
    return data


def test_complete_synthetic_contract_never_becomes_real_or_runs_fit():
    report = audit_decision_opportunities(document())
    assert report["engineering_contract_status"] == "PASS"
    assert report["data_readiness"] == "NOT_REAL_DATA"
    assert report["eligible_target_count"] == 2 and not report["blockers"]
    assert not report["model_fit_executed"] and report["selection"] is None
    assert not report["strategy_eligible"] and not report["advice_emitted"]


def test_complete_physical_declaration_still_needs_artifact_verification():
    report = audit_decision_opportunities(as_physical(document()))
    assert report["data_readiness"] == (
        "DECLARED_COMPLETE_NEEDS_ARTIFACT_VERIFICATION")
    assert not report["ready_for_offline_calibration"]
    assert not report["model_fit_executed"] and not report["live_use"]


def test_missing_detail_is_retained_in_original_ledger_order_and_blocks():
    data = document()
    missing = data["opportunities"].pop(0)["opportunity_id"]
    report = audit_decision_opportunities(data)
    assert report["missing_ids"] == [missing]
    assert report["records"][0]["reasons"] == ["decision_detail_missing"]
    assert report["data_readiness"] == "BLOCKED"


@pytest.mark.parametrize("field,value,reason", [
    ("legal_menu", None, "complete_legal_menu_missing"),
    ("actor_player_id", None, "stable_actor_identity_missing"),
    ("position", None, "position_or_street_unknown"),
    ("active_count", None, "active_and_pot_eligible_players_unknown"),
    ("audit_status", "candidate", "row_not_author_reviewed"),
])
def test_unknown_target_rows_are_retained_and_forbid_complete_case_fit(
        field, value, reason):
    data = document()
    data["opportunities"][1][field] = value
    if field in data["opportunity_ledger"][1]:
        data["opportunity_ledger"][1][field] = value
    report = audit_decision_opportunities(data)
    assert reason in report["records"][1]["reasons"]
    assert report["eligible_target_count"] == 1
    assert report["data_readiness"] == "BLOCKED"


def test_same_recording_group_cannot_cross_train_validation():
    data = document()
    data["sessions"][1]["source_recording_group_id"] = data[
        "sessions"][0]["source_recording_group_id"]
    with pytest.raises(ValueError, match="recording_group"):
        audit_decision_opportunities(data)


def test_same_player_cannot_change_seat_inside_one_hand():
    data = document()
    clone = deepcopy(data["opportunity_ledger"][0])
    clone.update(opportunity_id="train-second", canonical_opportunity_id="train:2",
                 decision_sequence=2, actor_seat=2)
    detail = deepcopy(data["opportunities"][0])
    detail.update(clone)
    data["opportunity_ledger"].insert(1, clone)
    data["opportunities"].insert(1, detail)
    data["hands"][0]["expected_opportunity_count"] = 7
    data["coverage_review"]["expected_opportunity_count"] = 13
    data["coverage_review"]["listed_opportunity_count"] = 13
    with pytest.raises(ValueError, match="changed_seat"):
        audit_decision_opportunities(data)


def test_predecision_evidence_must_be_strictly_before_action_onset():
    data = document()
    data["opportunities"][0]["evidence"]["state_frame"] = 4
    report = audit_decision_opportunities(data)
    assert "decision_frame_causality_invalid" in report[
        "records"][0]["reasons"]


def test_decision_price_fraction_is_recomputed_exactly():
    data = document()
    data["opportunities"][0]["predecision_snapshot"][
        "decision_price_fraction"] = {"numerator": 333, "denominator": 1000}
    report = audit_decision_opportunities(data)
    assert "decision_price_fraction_not_exact_or_canonical" in report[
        "records"][0]["reasons"]


def test_menu_and_observed_action_semantics_fail_closed():
    data = document()
    data["opportunities"][1]["legal_menu"][1].update(
        min_amount="19", max_amount="19")
    data["opportunities"][1]["observed_action"] = {
        "action": "bet", "amount": "40", "amount_semantics": "total_street"}
    report = audit_decision_opportunities(data)
    reasons = report["records"][1]["reasons"]
    assert "call_amount_differs_from_to_call" in reasons
    assert "observed_action_not_exactly_in_legal_menu" in reasons


def test_special_mode_unknown_cannot_be_called_ordinary():
    data = document()
    data["opportunities"][0]["special_mode_state"]["bomb"] = "UNKNOWN"
    report = audit_decision_opportunities(data)
    assert "special_mode_not_reviewed_ordinary" in report[
        "records"][0]["reasons"]


def test_rule_fingerprint_mixing_blocks_row_and_session():
    data = document()
    data["opportunities"][0]["rule_fingerprint"] = "8" * 64
    data["sessions"][0]["rule_fingerprint"] = "8" * 64
    report = audit_decision_opportunities(data)
    assert any("rule_binding" in item for item in report["blockers"])
    assert "opportunity_table_or_rule_binding_mismatch" in report[
        "records"][0]["reasons"]


def test_duplicate_canonical_opportunity_is_rejected():
    data = document()
    data["opportunity_ledger"][1]["canonical_opportunity_id"] = data[
        "opportunity_ledger"][0]["canonical_opportunity_id"]
    with pytest.raises(ValueError, match="duplicate_physical"):
        audit_decision_opportunities(data)


def test_author_only_complete_declaration_still_needs_independent_review():
    data = document()
    data["coverage_review"]["status"] = "AUTHOR_DECLARED_COMPLETE"
    data["coverage_review"]["independent_reviewer"] = None
    for row in data["opportunities"]:
        row["independent_review"] = "PENDING"
    for participant in data["participants"]:
        participant["independent_review_status"] = "PENDING"
    report = audit_decision_opportunities(data)
    assert report["data_readiness"] == "NOT_REAL_DATA"
    assert report["review_pending"] and not report["blockers"]


def test_hand_decision_sequence_must_be_contiguous():
    data = document()
    data["opportunity_ledger"][0]["decision_sequence"] = 2
    data["opportunities"][0]["decision_sequence"] = 2
    report = audit_decision_opportunities(data)
    assert any("decision_sequence" in item for item in report["blockers"])


@pytest.mark.parametrize("key", [
    "source_recording_sha256", "source_audit_sha256", "sample_manifest_sha256",
])
def test_each_source_hash_cannot_cross_split_under_renamed_group(key):
    data = document()
    second = data["sessions"][1]
    first = data["sessions"][0]
    second[key] = first[key]
    with pytest.raises(ValueError, match="same_.*cross"):
        audit_decision_opportunities(data)


def test_actor_must_match_hand_seat_map_and_identity_sessions():
    data = document()
    mapping = data["hands"][0]["seat_player_map"]
    mapping["4"], mapping["2"] = mapping["2"], mapping["4"]
    report = audit_decision_opportunities(data)
    assert "actor_differs_from_hand_seat_player_map" in report[
        "records"][1]["reasons"]
    data = document()
    participant = next(row for row in data["participants"]
                       if row["player_id"] == "stable-opponent")
    participant["session_ids"] = ["train-session"]
    participant["identity_scope"] = "same_session"
    report = audit_decision_opportunities(data)
    assert "actor_session_not_in_identity_scope" in report[
        "records"][7]["reasons"]


def test_pts_causality_and_hand_frame_bounds_are_both_required():
    data = document()
    data["opportunities"][0]["evidence"]["state_pts"] = "9"
    report = audit_decision_opportunities(data)
    assert "decision_pts_causality_invalid" in report["records"][0]["reasons"]
    data = document()
    for key in ("actor_frame", "menu_frame", "state_frame", "action_onset_frame",
                "action_confirmation_frame"):
        data["opportunities"][0]["evidence"][key] += 100
    report = audit_decision_opportunities(data)
    assert "decision_evidence_frame_outside_hand" in report[
        "records"][0]["reasons"]


def test_snapshot_status_dealer_board_and_to_call_are_cross_checked():
    for field, value, reason in (
            ("actor_status", "ALL_IN", "predecision_state_not_reviewed_complete"),
            ("dealer_seat", 1, "predecision_state_not_reviewed_complete"),
            ("board", ["2c"], "board_count_differs_from_street"),
            ("to_call", "19", "to_call_differs_from_current_bet_and_commitment")):
        data = document()
        data["opportunities"][0]["predecision_snapshot"][field] = value
        report = audit_decision_opportunities(data)
        assert reason in report["records"][0]["reasons"]


def test_menu_requires_aggression_and_all_in_must_equal_stack():
    data = document()
    data["opportunities"][1]["legal_menu"].pop()
    report = audit_decision_opportunities(data)
    assert "facing_bet_menu_missing_aggressive_option" in report[
        "records"][1]["reasons"]
    data = document()
    data["opportunities"][1]["legal_menu"].append({
        "action": "all_in", "min_amount": "1", "max_amount": "1",
        "amount_semantics": "additional"})
    report = audit_decision_opportunities(data)
    assert "all_in_amount_differs_from_actor_stack" in report[
        "records"][1]["reasons"]


def test_nonreopened_betting_cannot_offer_raise_or_large_all_in():
    data = document()
    data["opportunities"][1]["predecision_snapshot"]["betting_reopened"] = False
    report = audit_decision_opportunities(data)
    assert "nonreopened_betting_forbids_aggressive_option" in report[
        "records"][1]["reasons"]


def test_hand_special_mode_cannot_be_hidden_by_ordinary_row():
    data = document()
    data["hands"][0]["special_mode_summary"]["bomb"] = "PRESENT"
    report = audit_decision_opportunities(data)
    assert "row_special_mode_conflicts_hand_summary" in report[
        "records"][0]["reasons"]


def test_zero_price_fraction_must_be_canonical_zero_over_one():
    data = document()
    row = data["opportunities"][0]
    row["predecision_snapshot"].update(
        current_bet="0", to_call="0",
        decision_price_fraction={"numerator": 0, "denominator": 999})
    row["legal_menu"] = [
        {"action": "check", "min_amount": "0", "max_amount": "0",
         "amount_semantics": "none"},
        {"action": "bet", "min_amount": "2", "max_amount": "100",
         "amount_semantics": "total_street"}]
    row["observed_action"] = {
        "action": "check", "amount": "0", "amount_semantics": "none"}
    report = audit_decision_opportunities(data)
    assert "decision_price_fraction_not_exact_or_canonical" in report[
        "records"][0]["reasons"]


def test_complete_hand_cannot_self_report_zero_opportunities():
    data = document()
    data["opportunity_ledger"] = data["opportunity_ledger"][6:]
    data["opportunities"] = data["opportunities"][6:]
    data["hands"][0]["expected_opportunity_count"] = 0
    data["coverage_review"]["expected_opportunity_count"] = 6
    data["coverage_review"]["listed_opportunity_count"] = 6
    report = audit_decision_opportunities(data)
    assert any("hand_expected" in item for item in report["blockers"])
    assert "train_and_validation_need_complete_target_opportunities" in report[
        "blockers"]


def test_censored_edges_require_exact_window_edge_contract():
    data = document()
    data["sessions"][0]["censored_edge_intervals"] = [{"middle": "anything"}]
    report = audit_decision_opportunities(data)
    assert any("censored" in item for item in report["blockers"])


def test_hero_cannot_be_target_and_same_session_scope_means_one_session():
    data = document()
    data["dataset_scope"]["target_player_ids"].append("hero-player")
    report = audit_decision_opportunities(data)
    assert "hero_cannot_be_opponent_model_target" in report["blockers"]
    data = document()
    participant = next(row for row in data["participants"]
                       if row["player_id"] == "stable-opponent")
    participant["identity_scope"] = "same_session"
    report = audit_decision_opportunities(data)
    assert any("participant_session_scope" in item for item in report["blockers"])


def test_invalid_blinds_cannot_preserve_rule_fingerprint_eligibility():
    data = document()
    data["rule_profiles"][0]["small_blind"] = "100"
    report = audit_decision_opportunities(data)
    assert any("rule" in item for item in report["blockers"])


def test_position_and_pot_seats_must_match_rule_bound_hand():
    data = document()
    data["opportunities"][1]["position"] = "BTN"
    data["opportunities"][1]["pot_eligible_seats"] = [1, 7]
    report = audit_decision_opportunities(data)
    reasons = report["records"][1]["reasons"]
    assert "position_differs_from_rule_bound_hand" in reasons
    assert "active_and_pot_eligible_players_unknown" in reasons


def test_legal_menu_ranges_must_equal_rule_bound_snapshot():
    data = document()
    raise_action = data["opportunities"][1]["legal_menu"][2]
    raise_action.update(min_amount="60", max_amount="80")
    report = audit_decision_opportunities(data)
    assert "legal_menu_differs_from_rule_bound_snapshot" in report[
        "records"][1]["reasons"]
    data = document()
    data["opportunities"][0]["legal_menu"] = [
        {"action": "check", "min_amount": "0", "max_amount": "0",
         "amount_semantics": "none"},
        {"action": "all_in", "min_amount": "100", "max_amount": "100",
         "amount_semantics": "additional"}]
    report = audit_decision_opportunities(data)
    assert "legal_menu_differs_from_rule_bound_snapshot" in report[
        "records"][0]["reasons"]


def test_public_history_and_minimum_raise_must_match_prior_ledger_actions():
    data = document()
    history = data["opportunities"][1]["predecision_snapshot"][
        "public_history_through_sequence"]
    history[0]["actor_player_id"] = "train-session-seat-2"
    report = audit_decision_opportunities(data)
    assert "public_history_differs_from_prior_ledger_actions" in report[
        "records"][1]["reasons"]
    data = document()
    row = data["opportunities"][1]
    row["predecision_snapshot"]["minimum_raise_increment"] = "1"
    row["legal_menu"][2]["min_amount"] = "21"
    report = audit_decision_opportunities(data)
    assert "bet_and_raise_state_differs_from_public_history" in report[
        "records"][1]["reasons"]


def test_hand_rule_must_equal_its_session_rule():
    data = document()
    original = data["rule_profiles"][0]
    profile = AARuleProfileV2.from_dict({
        "schema_version": 2,
        **{key: original[key] for key in (
            "table_size", "small_blind", "big_blind", "ante", "ante_mode",
            "straddle_mode", "straddle_amount", "rake_percent", "rake_cap_bb",
            "rake_application", "rake_rounding", "rake_distribution",
            "minimum_chip", "verification_status")},
        "source": "synthetic-rule-b"})
    second = deepcopy(original)
    second.update(rule_profile_id="rule-b", rule_fingerprint=profile.fingerprint,
                  source="synthetic-rule-b", source_sha256="7" * 64)
    data["rule_profiles"].append(second)
    data["hands"][0]["rule_profile_id"] = "rule-b"
    data["hands"][0]["rule_fingerprint"] = profile.fingerprint
    for row in data["opportunities"][:2]:
        row["rule_fingerprint"] = profile.fingerprint
    report = audit_decision_opportunities(data)
    assert "hand_rule_binding_mismatch:train-session-hand" in report["blockers"]


def test_last_hand_next_boundary_is_not_free_text():
    data = document()
    data["hands"][0]["next_start_frame"] = 999
    report = audit_decision_opportunities(data)
    assert "last_hand_next_boundary_mismatch:train-session" in report["blockers"]


def test_each_per_opponent_target_needs_both_session_splits():
    data = document()
    second_target = "validation-session-seat-1"
    data["dataset_scope"]["target_player_ids"].append(second_target)
    report = audit_decision_opportunities(data)
    assert ("each_target_needs_cross_session_train_validation:" + second_target
            in report["blockers"])


def test_action_confirmation_must_be_within_last_gameplay_frame():
    data = document()
    data["hands"][0]["last_gameplay_frame"] = 9
    report = audit_decision_opportunities(data)
    assert "action_evidence_after_last_gameplay_frame" in report[
        "records"][1]["reasons"]


def test_next_predecision_must_follow_previous_action_confirmation():
    data = document()
    evidence_row = data["opportunities"][1]["evidence"]
    for key, value in (("actor_frame", 2), ("menu_frame", 3),
                       ("state_frame", 4), ("action_onset_frame", 5),
                       ("action_confirmation_frame", 6)):
        evidence_row[key] = value
        evidence_row[key.replace("frame", "pts")] = str(value)
    report = audit_decision_opportunities(data)
    assert "next_predecision_not_after_previous_confirmation" in report[
        "records"][1]["reasons"]


def test_pot_and_street_must_follow_public_action_trajectory():
    data = document()
    row = data["opportunities"][1]
    row["predecision_snapshot"]["pot_before"] = "6000"
    row["predecision_snapshot"]["decision_price_fraction"] = {
        "numerator": 1, "denominator": 1001}
    report = audit_decision_opportunities(data)
    assert "public_chip_trajectory_differs_from_snapshot" in report[
        "records"][1]["reasons"]
    data = document()
    first = data["opportunities"][0]
    first["street"] = "flop"
    first["predecision_snapshot"]["board"] = ["2c", "3d", "4h"]
    data["dataset_scope"]["included_streets"].append("flop")
    report = audit_decision_opportunities(data)
    assert "street_advances_before_action_round_closed" in report[
        "records"][0]["reasons"]


def test_action_round_tail_cannot_be_omitted_or_keep_folded_player_active():
    data = document()
    keep = {row["opportunity_id"] for row in (
        data["opportunity_ledger"][0], data["opportunity_ledger"][1],
        data["opportunity_ledger"][6], data["opportunity_ledger"][7])}
    data["opportunity_ledger"] = [row for row in data["opportunity_ledger"]
                                  if row["opportunity_id"] in keep]
    data["opportunities"] = [row for row in data["opportunities"]
                             if row["opportunity_id"] in keep]
    for hand in data["hands"]:
        hand["expected_opportunity_count"] = 2
    data["coverage_review"]["expected_opportunity_count"] = 4
    data["coverage_review"]["listed_opportunity_count"] = 4
    report = audit_decision_opportunities(data)
    assert any("action_round" in item for item in report["blockers"])
    data = document()
    data["opportunities"][2]["pot_eligible_seats"] = list(range(6))
    data["opportunities"][2]["active_count"] = 6
    report = audit_decision_opportunities(data)
    assert "pot_eligible_players_differ_from_action_trajectory" in report[
        "records"][2]["reasons"]


def test_actor_must_be_first_pending_seat():
    data = document()
    row = data["opportunities"][0]
    row["actor_player_id"] = "stable-opponent"
    row["actor_seat"] = 4
    row["is_hero"] = False
    for key in ("actor_player_id", "actor_seat", "is_hero"):
        data["opportunity_ledger"][0][key] = row[key]
    report = audit_decision_opportunities(data)
    assert "actor_or_pending_order_differs_from_trajectory" in report[
        "records"][0]["reasons"]


def test_unacted_player_cannot_hide_raise_right_after_full_raise():
    data = document()
    row = data["opportunities"][1]
    row["predecision_snapshot"]["betting_reopened"] = False
    row["legal_menu"].pop()
    report = audit_decision_opportunities(data)
    assert "betting_reopened_differs_from_action_trajectory" in report[
        "records"][1]["reasons"]


def test_short_all_in_does_not_reopen_prior_actor_raise_right():
    data = document()
    bb = data["opportunities"][5]
    bb["predecision_snapshot"]["actor_stack"] = "5"
    bb["legal_menu"] = [
        {"action": "fold", "min_amount": "0", "max_amount": "0",
         "amount_semantics": "none"},
        {"action": "call", "min_amount": "4", "max_amount": "4",
         "amount_semantics": "additional"},
        {"action": "all_in", "min_amount": "5", "max_amount": "5",
         "amount_semantics": "additional"}]
    bb["observed_action"] = {
        "action": "all_in", "amount": "5", "amount_semantics": "additional"}
    identity = {
        "opportunity_id": "train-hero-short-allin-response",
        "canonical_opportunity_id": "train:hero:short-allin-response",
        "session_id": "train-session", "hand_id": "train-session-hand",
        "decision_sequence": 7, "actor_player_id": "hero-player",
        "actor_seat": 3, "is_hero": True}
    history = deepcopy(bb["predecision_snapshot"][
        "public_history_through_sequence"])
    history.append({
        "decision_sequence": 6, "actor_player_id": bb["actor_player_id"],
        "actor_seat": 2, "street": "preflop", **bb["observed_action"],
        "source_sha256": bb["evidence"]["action_confirmation_sha256"]})
    response_snapshot = snapshot()
    response_snapshot.update(
        actor_stack="94", actor_street_committed="6",
        actor_hand_committed="6", current_bet="7", pot_before="14",
        to_call="1", minimum_raise_increment="4", betting_reopened=False,
        public_history_through_sequence=history,
        decision_price_fraction={"numerator": 1, "denominator": 15})
    detail = {
        **identity, "position": "UTG", "street": "preflop", "table_size": 6,
        "active_count": 2, "pot_eligible_seats": [2, 3],
        "pending_action_seats": [3],
        "rule_fingerprint": data["rule_profiles"][0]["rule_fingerprint"],
        "predecision_snapshot": response_snapshot, "evidence": evidence("e", 31),
        "legal_menu": [
            {"action": "fold", "min_amount": "0", "max_amount": "0",
             "amount_semantics": "none"},
            {"action": "call", "min_amount": "1", "max_amount": "1",
             "amount_semantics": "additional"}],
        "observed_action": {
            "action": "call", "amount": "1", "amount_semantics": "additional"},
        "special_mode_state": special(), "rule_binding_status": "VERIFIED",
        "row_status": "REVIEWED_COMPLETE", "audit_status": "reviewed",
        "unknown_reasons": [], "author_review": "APPROVED",
        "independent_review": "APPROVED"}
    data["opportunity_ledger"].insert(6, identity)
    data["opportunities"].insert(6, detail)
    data["hands"][0].update(
        last_gameplay_frame=35, last_gameplay_pts="35", end_frame=35,
        expected_opportunity_count=7)
    data["sessions"][0].update(
        last_frame=35, end_pts_exclusive="36")
    data["coverage_review"]["expected_opportunity_count"] = 13
    data["coverage_review"]["listed_opportunity_count"] = 13
    clean = audit_decision_opportunities(data)
    assert not clean["blockers"], [
        (index, row["reasons"]) for index, row in enumerate(clean["records"])
        if row["reasons"]]
    data["opportunities"][6]["predecision_snapshot"]["betting_reopened"] = True
    report = audit_decision_opportunities(data)
    assert "betting_reopened_differs_from_action_trajectory" in report[
        "records"][6]["reasons"]


def test_shipped_examples_are_explicitly_synthetic_or_not_data():
    root = Path(__file__).parents[2] / "configs" / "strategy" / "examples"
    synthetic = json.loads((
        root / "decision-opportunities-synthetic-example-v1.json").read_text(
            encoding="utf-8"))
    template = json.loads((
        root / "decision-opportunities-reviewed-template-NOT-DATA-v1.json").read_text(
            encoding="utf-8"))
    assert audit_decision_opportunities(synthetic)["data_readiness"] == (
        "NOT_REAL_DATA")
    report = audit_decision_opportunities(template)
    assert report["data_readiness"] == "BLOCKED"
    assert report["source_kind"] == "template_not_data"
