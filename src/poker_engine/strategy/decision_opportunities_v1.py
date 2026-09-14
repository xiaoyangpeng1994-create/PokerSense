"""Fail-closed audit contract for complete public decision opportunities.

This module validates declarations and preserves every ledger row.  It does not
authenticate reviewers, fit a model, infer ranges, or emit poker advice.
"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

from .aa_rules_v2 import AARuleProfileV2, build_forced_bet_plan


SHA256 = re.compile(r"[0-9a-f]{64}")
STREETS = ("preflop", "flop", "turn", "river")
POSITIONS = {"SB", "BB", "UTG", "UTG1", "UTG2", "MP", "LJ", "HJ", "CO", "BTN"}
MODES = ("insurance", "mushroom", "bomb", "buyin_overlay")
ROOT_KEYS = {
    "schema_version", "dataset_id", "source_kind", "dataset_scope",
    "platform_binding", "rule_profiles", "split_protocol", "coverage_review",
    "sessions", "participants", "hands", "opportunity_ledger", "opportunities",
}
SCOPE_KEYS = {
    "calibration_unit", "target_player_ids", "hero_player_id", "included_streets",
    "included_active_counts", "included_modes", "contiguous_session_policy",
    "all_table_decisions_required",
}
PLATFORM_KEYS = {"platform_id", "capture_path", "emulator_used"}
RULE_KEYS = {
    "rule_profile_id", "rule_fingerprint", "table_size", "small_blind",
    "big_blind", "ante", "ante_mode", "straddle_mode", "straddle_amount",
    "rake_percent", "rake_cap_bb", "rake_application", "rake_rounding",
    "rake_distribution", "minimum_chip", "verification_status", "source",
    "source_sha256", "independent_review_status",
}
SPLIT_KEYS = {
    "frozen_before_label_review", "training_session_ids",
    "validation_session_ids", "split_evidence_sha256",
}
COVERAGE_KEYS = {
    "status", "author_reviewer", "independent_reviewer", "evidence_sha256",
    "expected_opportunity_count", "listed_opportunity_count",
}
SESSION_KEYS = {
    "session_id", "source_recording_group_id", "split", "platform_id",
    "rule_profile_id", "rule_fingerprint", "source_audit_sha256",
    "source_recording_sha256", "sample_manifest_sha256", "first_frame",
    "last_frame", "start_pts",
    "end_pts_exclusive", "complete_hand_ids", "censored_edge_intervals",
    "coverage_status", "coverage_evidence_sha256",
}
PARTICIPANT_KEYS = {
    "player_id", "identity_scope", "session_ids", "identity_status",
    "identity_evidence_sha256", "independent_review_status",
}
HAND_KEYS = {
    "hand_id", "session_id", "start_frame", "last_gameplay_frame",
    "last_gameplay_pts", "end_frame", "next_start_frame", "temporal_complete",
    "dealer_seat", "seat_player_map",
    "occupied_seats", "table_size", "rule_profile_id", "rule_fingerprint",
    "observed_optional_straddle", "special_mode_summary",
    "public_action_timeline_sha256", "expected_opportunity_count",
    "coverage_status", "timeline_binding_status",
}
LEDGER_KEYS = {
    "opportunity_id", "canonical_opportunity_id", "session_id", "hand_id",
    "decision_sequence", "actor_player_id", "actor_seat", "is_hero",
}
OPPORTUNITY_KEYS = LEDGER_KEYS | {
    "position", "street", "table_size", "active_count", "pot_eligible_seats",
    "pending_action_seats", "rule_fingerprint", "predecision_snapshot",
    "evidence", "legal_menu", "observed_action", "special_mode_state",
    "rule_binding_status", "row_status", "audit_status", "unknown_reasons",
    "author_review", "independent_review",
}
SNAPSHOT_KEYS = {
    "dealer_seat", "board", "actor_status", "actor_stack",
    "actor_street_committed", "actor_hand_committed", "current_bet",
    "pot_before", "to_call", "minimum_raise_increment",
    "betting_reopened", "public_history_through_sequence", "state_quality",
    "decision_price_fraction",
}
EVIDENCE_KEYS = {
    "actor_frame", "actor_pts", "actor_sha256", "menu_frame", "menu_pts",
    "menu_sha256", "state_frame", "state_pts", "state_sha256",
    "action_onset_frame", "action_onset_pts", "action_onset_sha256",
    "action_confirmation_frame", "action_confirmation_pts",
    "action_confirmation_sha256", "review_bundle_sha256",
}
SPECIAL_KEYS = set(MODES) | {"block_state_updates", "evidence_sha256"}
ACTION_KEYS = {"action", "min_amount", "max_amount", "amount_semantics"}
OBSERVED_ACTION_KEYS = {"action", "amount", "amount_semantics"}
CENSOR_KEYS = {"first_frame", "last_frame", "reason"}
HISTORY_KEYS = {
    "decision_sequence", "actor_player_id", "actor_seat", "street", "action",
    "amount", "amount_semantics", "source_sha256",
}
CARD = re.compile(r"(?:[2-9TJQKA][cdhs])")


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("exact_" + name + "_fields_required")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _decimal(value):
    if not isinstance(value, str):
        return None
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def _add(reasons, reason):
    if reason not in reasons:
        reasons.append(reason)


def _list(value, predicate):
    return isinstance(value, list) and all(predicate(item) for item in value)


def _rotate(values, first):
    values = list(values)
    index = values.index(first)
    return values[index:] + values[:index]


def _after(values, seat):
    values = list(values)
    index = values.index(seat) + 1
    return values[index:] + values[:index]


def _rule_reasons(rule, source_kind):
    _exact(rule, RULE_KEYS, "rule_profile")
    reasons = []
    if not _text(rule["rule_profile_id"]):
        _add(reasons, "missing_rule_profile_id")
    if not _hash(rule["rule_fingerprint"]):
        _add(reasons, "unknown_rule_fingerprint")
    if type(rule["table_size"]) is not int or rule["table_size"] not in (6, 7, 8):
        _add(reasons, "unknown_rule_table_size")
    money = (
        "small_blind", "big_blind", "ante", "straddle_amount", "rake_percent",
        "rake_cap_bb", "minimum_chip",
    )
    if any(_decimal(rule[key]) is None for key in money):
        _add(reasons, "unknown_or_invalid_rule_amount")
    profile = None
    try:
        profile = AARuleProfileV2.from_dict({
            "schema_version": 2,
            **{key: rule[key] for key in (
                "table_size", "small_blind", "big_blind", "ante", "ante_mode",
                "straddle_mode", "straddle_amount", "rake_percent", "rake_cap_bb",
                "rake_application", "rake_rounding", "rake_distribution",
                "minimum_chip", "verification_status", "source")},
        })
    except (ValueError, TypeError, InvalidOperation):
        _add(reasons, "unknown_or_invalid_aa_rule_profile_v2")
    if profile is not None and profile.fingerprint != rule["rule_fingerprint"]:
        _add(reasons, "rule_fingerprint_not_recomputed_from_profile")
    expected_verification = (
        "simulation" if source_kind == "synthetic" else "live_verified")
    if rule["verification_status"] != expected_verification:
        _add(reasons, "rule_verification_status_wrong_for_source_kind")
    if rule["independent_review_status"] != "APPROVED":
        _add(reasons, "rule_profile_not_independently_reviewed")
    if not _text(rule["source"]) or not _hash(rule["source_sha256"]):
        _add(reasons, "missing_rule_source_hash")
    return reasons, profile


def _action_menu(value, snapshot, rule, reasons):
    to_call = _decimal(snapshot.get("to_call")) if isinstance(snapshot, dict) else None
    stack = (_decimal(snapshot.get("actor_stack"))
             if isinstance(snapshot, dict) else None)
    committed = (_decimal(snapshot.get("actor_street_committed"))
                 if isinstance(snapshot, dict) else None)
    current_bet = (_decimal(snapshot.get("current_bet"))
                   if isinstance(snapshot, dict) else None)
    raise_increment = (_decimal(snapshot.get("minimum_raise_increment"))
                       if isinstance(snapshot, dict) else None)
    reopened = snapshot.get("betting_reopened") if isinstance(snapshot, dict) else None
    chip = _decimal(rule["minimum_chip"]) if isinstance(rule, dict) else None
    big_blind = _decimal(rule["big_blind"]) if isinstance(rule, dict) else None
    if not isinstance(value, list) or not value:
        _add(reasons, "complete_legal_menu_missing")
        return None
    menu, kinds = [], []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != ACTION_KEYS:
            _add(reasons, "invalid_legal_action_shape")
            return None
        kind, semantics = raw["action"], raw["amount_semantics"]
        low, high = _decimal(raw["min_amount"]), _decimal(raw["max_amount"])
        if (kind not in ("fold", "check", "call", "bet", "raise", "all_in")
                or low is None or high is None or low > high):
            _add(reasons, "invalid_legal_action_amount")
            return None
        expected = "none" if kind in ("fold", "check") else (
            "additional" if kind in ("call", "all_in") else "total_street")
        if semantics != expected:
            _add(reasons, "invalid_legal_action_amount_semantics")
        if kind in ("fold", "check") and (low or high):
            _add(reasons, "zero_action_has_nonzero_amount")
        if (kind == "call" and to_call is not None
                and (low != to_call or high != to_call)):
            _add(reasons, "call_amount_differs_from_to_call")
        if chip is not None and (low % chip or high % chip):
            _add(reasons, "legal_action_not_aligned_to_minimum_chip")
        if kind == "all_in" and stack is not None and (low != stack or high != stack):
            _add(reasons, "all_in_amount_differs_from_actor_stack")
        if (kind in ("bet", "raise") and stack is not None
                and committed is not None and high > committed + stack):
            _add(reasons, "aggressive_action_exceeds_available_stack")
        if (kind == "bet" and big_blind is not None and low < big_blind):
            _add(reasons, "bet_below_rule_minimum")
        if (kind == "raise" and current_bet is not None
                and raise_increment is not None
                and low < current_bet + raise_increment):
            _add(reasons, "raise_below_minimum_reopen_target")
        menu.append((kind, low, high, semantics))
        kinds.append(kind)
    if len(set(kinds)) != len(kinds):
        _add(reasons, "duplicate_legal_action_kind")
    kind_set = set(kinds)
    if to_call is None:
        _add(reasons, "legal_menu_cannot_bind_unknown_to_call")
    elif to_call == 0:
        if "check" not in kind_set or kind_set & {"fold", "call", "raise"}:
            _add(reasons, "incomplete_or_inconsistent_unopened_menu")
        if stack is not None and stack > 0 and not kind_set & {"bet", "all_in"}:
            _add(reasons, "unopened_menu_missing_aggressive_option")
    elif ("fold" not in kind_set or not kind_set & {"call", "all_in"}
          or kind_set & {"check", "bet"}):
        _add(reasons, "incomplete_or_inconsistent_facing_bet_menu")
    if to_call is not None and stack is not None and to_call > 0:
        if stack < to_call and ("all_in" not in kind_set or "call" in kind_set):
            _add(reasons, "short_stack_menu_must_use_all_in_not_call")
        if stack >= to_call and "call" not in kind_set:
            _add(reasons, "facing_bet_menu_missing_call")
        if reopened is True and stack > to_call and not kind_set & {"raise", "all_in"}:
            _add(reasons, "facing_bet_menu_missing_aggressive_option")
        if reopened is False and stack > to_call and kind_set & {"raise", "all_in"}:
            _add(reasons, "nonreopened_betting_forbids_aggressive_option")
    if all(value is not None for value in (
            to_call, stack, committed, current_bet, raise_increment,
            chip, big_blind)) and len(menu) == len(set(kinds)):
        maximum_total = committed + stack
        expected = {}
        if to_call == 0:
            expected["check"] = (Decimal(0), Decimal(0), "none")
            if stack > 0:
                if maximum_total >= big_blind:
                    expected["bet"] = (big_blind, maximum_total, "total_street")
                else:
                    expected["all_in"] = (stack, stack, "additional")
        else:
            expected["fold"] = (Decimal(0), Decimal(0), "none")
            if stack <= to_call:
                expected["all_in"] = (stack, stack, "additional")
            else:
                expected["call"] = (to_call, to_call, "additional")
                if reopened is True:
                    minimum_raise_to = current_bet + raise_increment
                    if maximum_total >= minimum_raise_to:
                        expected["raise"] = (
                            minimum_raise_to, maximum_total, "total_street")
                    else:
                        expected["all_in"] = (stack, stack, "additional")
        actual = {kind: (low, high, semantics)
                  for kind, low, high, semantics in menu}
        if actual != expected:
            _add(reasons, "legal_menu_differs_from_rule_bound_snapshot")
    return menu


def _observed(value, menu, reasons):
    if not isinstance(value, dict) or set(value) != OBSERVED_ACTION_KEYS:
        _add(reasons, "observed_action_missing")
        return
    amount = _decimal(value["amount"])
    if amount is None or menu is None:
        _add(reasons, "observed_action_amount_or_menu_missing")
        return
    matches = [row for row in menu if row[0] == value["action"]
               and row[1] <= amount <= row[2]
               and row[3] == value["amount_semantics"]]
    if len(matches) != 1:
        _add(reasons, "observed_action_not_exactly_in_legal_menu")


def _snapshot(value, reasons, hand, street, decision_sequence, plan, profile):
    if not isinstance(value, dict) or set(value) != SNAPSHOT_KEYS:
        _add(reasons, "predecision_snapshot_missing")
        return
    amounts = {}
    for key in ("actor_stack", "actor_street_committed", "actor_hand_committed",
                "current_bet", "pot_before", "to_call", "minimum_raise_increment"):
        amounts[key] = _decimal(value[key])
        if amounts[key] is None:
            _add(reasons, "unknown_predecision_money:" + key)
    pot, to_call = amounts["pot_before"], amounts["to_call"]
    fraction = value["decision_price_fraction"]
    if (pot is not None and to_call is not None and pot + to_call > 0
            and isinstance(fraction, dict)
            and set(fraction) == {"numerator", "denominator"}
            and type(fraction["numerator"]) is int
            and type(fraction["denominator"]) is int
            and fraction["denominator"] > 0):
        actual = Fraction(fraction["numerator"], fraction["denominator"])
        expected = Fraction(to_call) / (Fraction(pot) + Fraction(to_call))
        if (actual != expected or actual.numerator != fraction["numerator"]
                or actual.denominator != fraction["denominator"]):
            _add(reasons, "decision_price_fraction_not_exact_or_canonical")
    else:
        _add(reasons, "decision_price_fraction_missing")
    history = value["public_history_through_sequence"]
    if (value["dealer_seat"] != hand["dealer_seat"]
            or not isinstance(value["board"], list)
            or value["actor_status"] != "ACTIVE"
            or not isinstance(history, list)
            or type(value["betting_reopened"]) is not bool
            or value["state_quality"] != "REVIEWED_PREDECISION"):
        _add(reasons, "predecision_state_not_reviewed_complete")
    board_count = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(street)
    if isinstance(value["board"], list) and len(value["board"]) != board_count:
        _add(reasons, "board_count_differs_from_street")
    if (isinstance(value["board"], list)
            and (len(set(value["board"])) != len(value["board"])
                 or any(not isinstance(card, str) or CARD.fullmatch(card) is None
                        for card in value["board"]))):
        _add(reasons, "board_cards_invalid_or_duplicate")
    if isinstance(history, list):
        valid_history = True
        for index, item in enumerate(history, start=1):
            if (not isinstance(item, dict) or set(item) != HISTORY_KEYS
                    or item["decision_sequence"] != index
                    or not _text(item["actor_player_id"])
                    or type(item["actor_seat"]) is not int
                    or item["street"] not in STREETS
                    or item["action"] not in (
                        "fold", "check", "call", "bet", "raise", "all_in")
                    or _decimal(item["amount"]) is None
                    or item["amount_semantics"] not in (
                        "none", "additional", "total_street")
                    or not _hash(item["source_sha256"])):
                valid_history = False
                break
        if (not valid_history or type(decision_sequence) is not int
                or len(history) != decision_sequence - 1):
            _add(reasons, "public_history_not_exactly_prior_decisions")
    required = ("actor_stack", "actor_street_committed", "actor_hand_committed",
                "current_bet", "to_call")
    if all(amounts[key] is not None for key in required):
        if amounts["actor_stack"] <= 0:
            _add(reasons, "actor_has_no_stack_to_decide")
        if amounts["actor_street_committed"] > amounts["actor_hand_committed"]:
            _add(reasons, "street_commitment_exceeds_hand_commitment")
        expected_call = max(amounts["current_bet"]
                            - amounts["actor_street_committed"], Decimal(0))
        if amounts["to_call"] != expected_call:
            _add(reasons, "to_call_differs_from_current_bet_and_commitment")
        if amounts["minimum_raise_increment"] is None or amounts[
                "minimum_raise_increment"] <= 0:
            _add(reasons, "minimum_raise_increment_not_positive")
        aggressive = []
        has_all_in = False
        if isinstance(history, list):
            for item in history:
                if (isinstance(item, dict) and item.get("street") == street
                        and item.get("action") == "all_in"):
                    has_all_in = True
                if (isinstance(item, dict) and item.get("street") == street
                        and item.get("action") in ("bet", "raise")
                        and item.get("amount_semantics") == "total_street"):
                    amount = _decimal(item.get("amount"))
                    if amount is not None:
                        aggressive.append(amount)
        if has_all_in:
            pass
        elif aggressive:
            base = (plan.current_bet if street == "preflop" and plan is not None
                    else Decimal(0))
            derived_increment = aggressive[-1] - (
                aggressive[-2] if len(aggressive) > 1 else base)
            if (amounts["current_bet"] != aggressive[-1]
                    or amounts["minimum_raise_increment"] != derived_increment):
                _add(reasons, "bet_and_raise_state_differs_from_public_history")
        elif street == "preflop" and plan is not None:
            if (amounts["current_bet"] != plan.current_bet
                    or amounts["minimum_raise_increment"] != plan.current_bet):
                _add(reasons, "preflop_bet_state_differs_from_forced_bets")
        elif amounts["current_bet"] != 0:
            _add(reasons, "current_bet_not_explained_by_public_history")
        elif profile is not None and amounts[
                "minimum_raise_increment"] != profile.big_blind:
            _add(reasons, "unopened_minimum_bet_differs_from_big_blind")


def _evidence(value, reasons, hand, session):
    if not isinstance(value, dict) or set(value) != EVIDENCE_KEYS:
        _add(reasons, "exact_decision_evidence_missing")
        return
    pairs = (
        ("actor_frame", "actor_pts", "actor_sha256"),
        ("menu_frame", "menu_pts", "menu_sha256"),
        ("state_frame", "state_pts", "state_sha256"),
        ("action_onset_frame", "action_onset_pts", "action_onset_sha256"),
        ("action_confirmation_frame", "action_confirmation_pts",
         "action_confirmation_sha256"),
    )
    frames, points = {}, {}
    for frame_key, pts_key, sha_key in pairs:
        frames[frame_key] = value[frame_key]
        points[pts_key] = _decimal(value[pts_key])
        if (type(value[frame_key]) is not int or points[pts_key] is None
                or not _hash(value[sha_key])):
            _add(reasons, "missing_or_invalid_evidence:" + frame_key)
    if all(type(frame) is int for frame in frames.values()):
        onset = frames["action_onset_frame"]
        confirmation = frames["action_confirmation_frame"]
        if (not all(frames[key] < onset for key in (
                "actor_frame", "menu_frame", "state_frame"))
                or onset > confirmation):
            _add(reasons, "decision_frame_causality_invalid")
        if any(not hand["start_frame"] <= frame <= hand["end_frame"]
               for frame in frames.values()):
            _add(reasons, "decision_evidence_frame_outside_hand")
    if all(point is not None for point in points.values()):
        onset_pts = points["action_onset_pts"]
        confirmation_pts = points["action_confirmation_pts"]
        if (not all(points[key] < onset_pts for key in (
                "actor_pts", "menu_pts", "state_pts"))
                or onset_pts > confirmation_pts):
            _add(reasons, "decision_pts_causality_invalid")
        start = _decimal(session["start_pts"])
        end = _decimal(session["end_pts_exclusive"])
        if (start is None or end is None
                or any(not start <= point < end for point in points.values())):
            _add(reasons, "decision_evidence_pts_outside_session")
    if not _hash(value["review_bundle_sha256"]):
        _add(reasons, "missing_review_bundle_hash")


def _special(value, reasons):
    if not isinstance(value, dict) or set(value) != SPECIAL_KEYS:
        _add(reasons, "special_mode_state_missing")
        return
    allowed = {"ABSENT_REVIEWED", "PRESENT", "UNKNOWN"}
    if any(value[key] not in allowed for key in MODES):
        _add(reasons, "invalid_special_mode_state")
    if (value["block_state_updates"] is not False
            or any(value[key] != "ABSENT_REVIEWED" for key in MODES)):
        _add(reasons, "special_mode_not_reviewed_ordinary")
    if not _hash(value["evidence_sha256"]):
        _add(reasons, "missing_special_mode_evidence")


def audit_decision_opportunities(data):
    """Audit a complete opportunity ledger without fitting or filtering rows."""
    _exact(data, ROOT_KEYS, "decision_opportunity_dataset")
    if (data["schema_version"] != 1 or type(data["schema_version"]) is not int
            or not _text(data["dataset_id"])):
        raise ValueError("versioned_dataset_identity_required")
    if data["source_kind"] not in (
            "reviewed_physical_capture_card", "synthetic", "template_not_data"):
        raise ValueError("explicit_dataset_source_kind_required")
    blockers, review_pending = [], []
    scope, platform = data["dataset_scope"], data["platform_binding"]
    _exact(scope, SCOPE_KEYS, "dataset_scope")
    _exact(platform, PLATFORM_KEYS, "platform_binding")
    if (scope["calibration_unit"] not in ("per_stable_opponent", "declared_population")
            or not _list(scope["target_player_ids"], _text)
            or len(set(scope["target_player_ids"])) != len(scope["target_player_ids"])
            or (scope["hero_player_id"] is not None
                and not _text(scope["hero_player_id"]))
            or not _list(scope["included_streets"], lambda v: v in STREETS)
            or not _list(scope["included_active_counts"],
                         lambda v: type(v) is int and 2 <= v <= 8)
            or scope["included_modes"] != ["ordinary"]
            or scope["contiguous_session_policy"] != (
                "all_complete_hands_and_explicit_censored_edges")
            or scope["all_table_decisions_required"] is not True):
        raise ValueError("exact_all_opportunities_scope_required")
    if not scope["target_player_ids"]:
        blockers.append("target_players_not_declared")
    if (not _text(platform["platform_id"])
            or platform["capture_path"] not in (
                "physical_phone_capture_card", "synthetic", "not_data")
            or type(platform["emulator_used"]) is not bool):
        raise ValueError("explicit_platform_capture_binding_required")
    if (data["source_kind"] == "reviewed_physical_capture_card"
            and (platform["capture_path"] != "physical_phone_capture_card"
                 or platform["emulator_used"])):
        raise ValueError("physical_dataset_cannot_use_emulator")

    rules, rule_objects = {}, {}
    if not isinstance(data["rule_profiles"], list) or not data["rule_profiles"]:
        blockers.append("rule_profiles_missing")
    else:
        for rule in data["rule_profiles"]:
            reasons, profile = _rule_reasons(rule, data["source_kind"])
            rid = rule["rule_profile_id"]
            if not _text(rid) or rid in rules:
                raise ValueError("unique_rule_profile_ids_required")
            rules[rid] = rule
            if profile is not None:
                rule_objects[rid] = profile
            blockers.extend("rule:" + rid + ":" + reason for reason in reasons)

    split = data["split_protocol"]
    _exact(split, SPLIT_KEYS, "split_protocol")
    train, validation = split["training_session_ids"], split["validation_session_ids"]
    if (not _list(train, _text) or not _list(validation, _text)
            or len(set(train)) != len(train)
            or len(set(validation)) != len(validation)):
        raise ValueError("unique_session_split_lists_required")
    if set(train) & set(validation):
        raise ValueError("training_validation_session_overlap")
    if not train or not validation:
        blockers.append("whole_session_train_validation_split_missing")
    if split["frozen_before_label_review"] is not True:
        blockers.append("split_not_frozen_before_label_review")
    if not _hash(split["split_evidence_sha256"]):
        blockers.append("split_evidence_hash_missing")

    sessions = {}
    groups_by_split = {"training": set(), "validation": set()}
    source_hashes_by_split = {
        name: {key: set() for key in (
            "source_recording_sha256", "source_audit_sha256",
            "sample_manifest_sha256")}
        for name in ("training", "validation")}
    if not isinstance(data["sessions"], list) or not data["sessions"]:
        raise ValueError("nonempty_session_inventory_required")
    for session in data["sessions"]:
        _exact(session, SESSION_KEYS, "session")
        sid = session["session_id"]
        if not _text(sid) or sid in sessions:
            raise ValueError("unique_session_ids_required")
        sessions[sid] = session
        expected_split = "training" if sid in train else (
            "validation" if sid in validation else "unassigned")
        if session["split"] != expected_split:
            blockers.append("session_split_unassigned_or_mismatched:" + sid)
        if expected_split in groups_by_split:
            group = session["source_recording_group_id"]
            if not _text(group):
                blockers.append("recording_group_missing:" + sid)
            else:
                groups_by_split[expected_split].add(group)
            for key in source_hashes_by_split[expected_split]:
                if _hash(session[key]):
                    source_hashes_by_split[expected_split][key].add(session[key])
        if (session["platform_id"] != platform["platform_id"]
                or session["rule_profile_id"] not in rules
                or (session["rule_profile_id"] in rules
                    and session["rule_fingerprint"] != rules[
                        session["rule_profile_id"]]["rule_fingerprint"])):
            blockers.append("session_platform_or_rule_binding_mismatch:" + sid)
        if any(not _hash(session[key]) for key in (
                "source_recording_sha256", "source_audit_sha256",
                "sample_manifest_sha256",
                "coverage_evidence_sha256")):
            blockers.append("session_source_or_coverage_hash_missing:" + sid)
        if (type(session["first_frame"]) is not int
                or type(session["last_frame"]) is not int
                or session["last_frame"] < session["first_frame"]
                or _decimal(session["start_pts"]) is None
                or _decimal(session["end_pts_exclusive"]) is None
                or (_decimal(session["start_pts"]) is not None
                    and _decimal(session["end_pts_exclusive"]) is not None
                    and _decimal(session["start_pts"])
                    >= _decimal(session["end_pts_exclusive"]))
                or not _list(session["complete_hand_ids"], _text)
                or not isinstance(session["censored_edge_intervals"], list)):
            blockers.append("session_window_or_hand_inventory_invalid:" + sid)
        if session["coverage_status"] == "AUTHOR_DECLARED_COMPLETE":
            review_pending.append("session_coverage_review_pending:" + sid)
        elif session["coverage_status"] != "INDEPENDENT_COMPLETE":
            blockers.append("session_opportunity_coverage_incomplete:" + sid)
    if groups_by_split["training"] & groups_by_split["validation"]:
        raise ValueError("same_recording_group_crosses_train_validation")
    for key in source_hashes_by_split["training"]:
        if (source_hashes_by_split["training"][key]
                & source_hashes_by_split["validation"][key]):
            raise ValueError("same_" + key + "_cross_train_validation")
    if set(sessions) != set(train) | set(validation):
        blockers.append("split_does_not_cover_exact_session_inventory")

    participants = {}
    if not isinstance(data["participants"], list):
        raise ValueError("participant_inventory_list_required")
    for participant in data["participants"]:
        _exact(participant, PARTICIPANT_KEYS, "participant")
        pid = participant["player_id"]
        if not _text(pid) or pid in participants:
            raise ValueError("unique_stable_player_ids_required")
        participants[pid] = participant
        identity_scope = participant["identity_scope"]
        participant_sessions = participant["session_ids"]
        if (identity_scope not in (
                "same_session", "cross_session", "declared_population")
                or not _list(participant_sessions, _text)
                or not set(participant_sessions) <= set(sessions)
                or identity_scope == "same_session" and len(participant_sessions) != 1
                or identity_scope == "cross_session" and len(
                    participant_sessions) < 2):
            blockers.append("participant_session_scope_invalid:" + pid)
        if (participant["identity_status"] != "VERIFIED"
                or not _hash(participant["identity_evidence_sha256"])):
            blockers.append("stable_player_identity_unverified:" + pid)
        if participant["independent_review_status"] != "APPROVED":
            review_pending.append("participant_identity_review_pending:" + pid)
    if not set(scope["target_player_ids"]) <= set(participants):
        blockers.append("target_player_missing_from_participant_inventory")
    if (scope["hero_player_id"] is not None
            and scope["hero_player_id"] not in participants):
        blockers.append("hero_player_missing_from_participant_inventory")
    if scope["hero_player_id"] in scope["target_player_ids"]:
        blockers.append("hero_cannot_be_opponent_model_target")

    hands, hand_plans = {}, {}
    hands_by_session = {sid: [] for sid in sessions}
    participant_appearances = {pid: set() for pid in participants}
    if not isinstance(data["hands"], list):
        raise ValueError("hand_inventory_list_required")
    for hand in data["hands"]:
        _exact(hand, HAND_KEYS, "hand")
        hid = hand["hand_id"]
        if not _text(hid) or hid in hands:
            raise ValueError("unique_hand_ids_required")
        hands[hid] = hand
        session = sessions.get(hand["session_id"])
        rule = rules.get(hand["rule_profile_id"])
        if session is None:
            raise ValueError("hand_references_unknown_session")
        if (rule is None or hand["rule_fingerprint"] != rule["rule_fingerprint"]
                or hand["table_size"] != rule["table_size"]
                or hand["rule_profile_id"] != session["rule_profile_id"]
                or hand["rule_fingerprint"] != session["rule_fingerprint"]):
            blockers.append("hand_rule_binding_mismatch:" + hid)
        frames = [hand[key] for key in (
            "start_frame", "last_gameplay_frame", "end_frame")]
        if (any(type(value) is not int for value in frames)
                or not frames[0] <= frames[1] <= frames[2]
                or type(hand["temporal_complete"]) is not bool
                or frames[0] < session["first_frame"]
                or frames[2] > session["last_frame"]):
            blockers.append("hand_boundary_invalid:" + hid)
        last_pts = _decimal(hand["last_gameplay_pts"])
        session_start = _decimal(session["start_pts"])
        session_end = _decimal(session["end_pts_exclusive"])
        if (last_pts is None or session_start is None or session_end is None
                or last_pts < session_start or last_pts >= session_end):
            blockers.append("hand_last_gameplay_pts_invalid:" + hid)
        if hand["temporal_complete"] is not True:
            blockers.append("incomplete_hand_must_be_censored_not_registered:" + hid)
        hands_by_session[hand["session_id"]].append(hand)
        occupied = hand["occupied_seats"]
        mapping = hand["seat_player_map"] if isinstance(
            hand["seat_player_map"], dict) else {}
        occupied = hand["occupied_seats"] if isinstance(
            hand["occupied_seats"], list) else []
        if (not isinstance(occupied, list)
                or any(type(seat) is not int or not 0 <= seat < 8
                       for seat in occupied)
                or occupied != sorted(set(occupied))
                or len(occupied) != hand["table_size"]):
            blockers.append("occupied_seats_unknown_or_invalid:" + hid)
            occupied = []
        if not isinstance(mapping, dict) or set(mapping) != {
                str(i) for i in range(8)}:
            blockers.append("complete_seat_player_map_missing:" + hid)
        else:
            values = []
            for seat in range(8):
                value = mapping[str(seat)]
                if seat in occupied:
                    if value not in participants:
                        blockers.append("occupied_seat_identity_missing:" + hid)
                    else:
                        values.append(value)
                        participant_appearances[value].add(hand["session_id"])
                elif value != "EMPTY":
                    blockers.append("unoccupied_seat_not_explicit_empty:" + hid)
            if len(values) != len(set(values)):
                raise ValueError("same_player_occupies_multiple_seats")
        modes = hand["special_mode_summary"]
        if (not isinstance(modes, dict) or set(modes) != set(MODES)
                or any(value not in ("ABSENT_REVIEWED", "PRESENT", "UNKNOWN")
                       for value in modes.values())):
            blockers.append("hand_special_mode_summary_invalid:" + hid)
        elif any(value == "UNKNOWN" for value in modes.values()):
            blockers.append("hand_special_mode_summary_unreviewed:" + hid)
        if not _hash(hand["public_action_timeline_sha256"]):
            blockers.append("hand_timeline_hash_missing:" + hid)
        if hand["timeline_binding_status"] != "INDEPENDENTLY_VERIFIED":
            blockers.append("hand_timeline_artifact_not_verified:" + hid)
        if hand["coverage_status"] == "AUTHOR_DECLARED_COMPLETE":
            review_pending.append("hand_coverage_review_pending:" + hid)
        elif hand["coverage_status"] != "INDEPENDENT_COMPLETE":
            blockers.append("hand_timeline_or_coverage_incomplete:" + hid)
        profile = rule_objects.get(hand["rule_profile_id"])
        if profile is not None and occupied and type(hand["dealer_seat"]) is int:
            try:
                optional = hand["observed_optional_straddle"]
                if optional is not None and type(optional) is not int:
                    raise ValueError("optional straddler must be integer or null")
                hand_plans[hid] = build_forced_bet_plan(
                    profile, tuple(occupied), hand["dealer_seat"],
                    observed_optional_straddler=optional)
            except (ValueError, TypeError):
                blockers.append("hand_position_or_straddle_plan_invalid:" + hid)
    for sid, session in sessions.items():
        if set(session["complete_hand_ids"]) != {
                hid for hid, hand in hands.items()
                if hand["session_id"] == sid and hand["temporal_complete"]}:
            blockers.append("session_complete_hand_inventory_mismatch:" + sid)
        ordered_hands = hands_by_session[sid]
        if ordered_hands != sorted(ordered_hands, key=lambda item: item["start_frame"]):
            blockers.append("hands_not_in_session_time_order:" + sid)
        for first, second in zip(ordered_hands, ordered_hands[1:]):
            if (first["end_frame"] >= second["start_frame"]
                    or first["next_start_frame"] != second["start_frame"]):
                blockers.append("hands_overlap_or_next_boundary_mismatch:" + sid)
        intervals = session["censored_edge_intervals"]
        valid_intervals = []
        for interval in intervals:
            if (not isinstance(interval, dict) or set(interval) != CENSOR_KEYS
                    or type(interval["first_frame"]) is not int
                    or type(interval["last_frame"]) is not int
                    or interval["first_frame"] > interval["last_frame"]
                    or interval["first_frame"] < session["first_frame"]
                    or interval["last_frame"] > session["last_frame"]
                    or interval["reason"] not in (
                        "leading_censored_context", "trailing_incomplete_hand")):
                blockers.append("invalid_censored_edge_interval:" + sid)
            else:
                valid_intervals.append(interval)
        if valid_intervals != sorted(
                valid_intervals, key=lambda item: item["first_frame"]):
            blockers.append("censored_edges_not_ordered:" + sid)
        for interval in valid_intervals:
            if any(not (interval["last_frame"] < hand["start_frame"]
                        or interval["first_frame"] > hand["end_frame"])
                   for hand in ordered_hands):
                blockers.append("censored_edge_overlaps_complete_hand:" + sid)
        if ordered_hands:
            allowed = []
            if ordered_hands[0]["start_frame"] > session["first_frame"]:
                allowed.append((session["first_frame"],
                                ordered_hands[0]["start_frame"] - 1,
                                "leading_censored_context"))
            if ordered_hands[-1]["end_frame"] < session["last_frame"]:
                allowed.append((ordered_hands[-1]["end_frame"] + 1,
                                session["last_frame"],
                                "trailing_incomplete_hand"))
            actual = [(item["first_frame"], item["last_frame"], item["reason"])
                      for item in valid_intervals]
            if actual != allowed:
                blockers.append("censored_edges_do_not_match_window_edges:" + sid)
            trailing = next((item for item in allowed
                             if item[2] == "trailing_incomplete_hand"), None)
            expected_next = trailing[0] if trailing is not None else None
            if ordered_hands[-1]["next_start_frame"] != expected_next:
                blockers.append("last_hand_next_boundary_mismatch:" + sid)
    for pid, participant in participants.items():
        if set(participant["session_ids"]) != participant_appearances[pid]:
            blockers.append("participant_sessions_differ_from_hand_appearances:" + pid)

    coverage = data["coverage_review"]
    _exact(coverage, COVERAGE_KEYS, "coverage_review")
    if (type(coverage["expected_opportunity_count"]) is not int
            or type(coverage["listed_opportunity_count"]) is not int
            or coverage["expected_opportunity_count"] < 0
            or coverage["listed_opportunity_count"] < 0):
        raise ValueError("integer_coverage_counts_required")
    if (coverage["status"] not in (
            "INCOMPLETE", "AUTHOR_DECLARED_COMPLETE", "INDEPENDENT_COMPLETE")
            or not _text(coverage["author_reviewer"])
            or not _hash(coverage["evidence_sha256"])):
        blockers.append("all_opportunities_coverage_not_reviewed")
    if coverage["status"] == "INCOMPLETE":
        blockers.append("all_opportunities_census_incomplete")
    elif coverage["status"] == "AUTHOR_DECLARED_COMPLETE":
        review_pending.append("all_opportunities_independent_review_pending")
    elif (not _text(coverage["independent_reviewer"])
          or coverage["independent_reviewer"] == coverage["author_reviewer"]):
        blockers.append("independent_coverage_reviewer_missing")

    ledger = data["opportunity_ledger"]
    details = data["opportunities"]
    if not isinstance(ledger, list) or not isinstance(details, list):
        raise ValueError("ledger_and_opportunity_lists_required")
    ledger_map, canonical, hand_rows = {}, set(), {}
    for item in ledger:
        _exact(item, LEDGER_KEYS, "opportunity_ledger_row")
        oid, cid = item["opportunity_id"], item["canonical_opportunity_id"]
        if not _text(oid) or oid in ledger_map:
            raise ValueError("unique_opportunity_ids_required")
        if not _text(cid) or cid in canonical:
            raise ValueError("duplicate_physical_opportunity_forbidden")
        canonical.add(cid)
        if item["session_id"] not in sessions or item["hand_id"] not in hands:
            raise ValueError("opportunity_ledger_unknown_session_or_hand")
        if hands[item["hand_id"]]["session_id"] != item["session_id"]:
            raise ValueError("opportunity_hand_crosses_session")
        ledger_map[oid] = item
        hand_rows.setdefault(item["hand_id"], []).append(item)
    if (coverage["listed_opportunity_count"] != len(ledger)
            or coverage["expected_opportunity_count"] != len(ledger)):
        blockers.append("coverage_count_differs_from_opportunity_ledger")
    for hid, hand in hands.items():
        rows = hand_rows.get(hid, [])
        sequences = [row["decision_sequence"] for row in rows]
        if (any(type(value) is not int for value in sequences)
                or sequences != list(range(1, len(rows) + 1))):
            blockers.append("hand_decision_sequence_not_contiguous:" + hid)
        expected = hand["expected_opportunity_count"]
        if (type(expected) is not int or expected != len(rows)
                or hand["temporal_complete"] and expected < 1):
            blockers.append("hand_expected_opportunity_count_mismatch:" + hid)
    expected_ledger_order = []
    for session in data["sessions"]:
        for hand in hands_by_session[session["session_id"]]:
            expected_ledger_order.extend(hand_rows.get(hand["hand_id"], []))
    if ledger != expected_ledger_order:
        blockers.append("ledger_not_in_session_hand_decision_order")

    detail_ids = []
    for raw in details:
        _exact(raw, OPPORTUNITY_KEYS, "decision_opportunity")
        oid = raw["opportunity_id"]
        if oid in detail_ids or oid not in ledger_map:
            raise ValueError("duplicate_or_unledgered_opportunity_detail")
        detail_ids.append(oid)
        if {key: raw[key] for key in LEDGER_KEYS} != ledger_map[oid]:
            raise ValueError("opportunity_detail_identity_differs_from_ledger")
    if detail_ids != [row["opportunity_id"] for row in ledger if row[
            "opportunity_id"] in set(detail_ids)]:
        raise ValueError("opportunity_details_not_in_ledger_order")
    details_by_id = {raw["opportunity_id"]: raw for raw in details}

    records, missing, eligible = [], [], 0
    eligible_by_split = {"training": 0, "validation": 0}
    eligible_by_player = {
        pid: {"training": 0, "validation": 0}
        for pid in scope["target_player_ids"]}
    seats_in_hand = {}
    evidence_identities = set()
    onset_by_hand = {}
    prior_actions_by_hand = {}
    trajectories = {}
    for hid, plan in hand_plans.items():
        occupied = hands[hid]["occupied_seats"]
        clockwise = _rotate(occupied, hands[hid]["dealer_seat"])
        trajectories[hid] = {
            "street": "preflop", "pot": plan.expected_total,
            "current_bet": plan.current_bet,
            "street_committed": dict(plan.contributions),
            "hand_committed": dict(plan.contributions), "stacks": {},
            "clockwise": clockwise,
            "pot_eligible": set(occupied), "all_in": set(),
            "pending": _rotate(clockwise, plan.first_actor_seat),
            "raise_rights": set(occupied),
            "minimum_raise_increment": plan.current_bet,
            "terminal": False, "round_closed": False, "valid": True,
        }
    previous_confirmation = {}
    for ledger_row in ledger:
        oid = ledger_row["opportunity_id"]
        raw = details_by_id.get(oid)
        if raw is None:
            missing.append(oid)
            records.append({"opportunity_id": oid, "input": deepcopy(ledger_row),
                            "eligible": False, "in_model_scope": None,
                            "reasons": ["decision_detail_missing"]})
            continue
        reasons = list(raw["unknown_reasons"]) if _list(
            raw["unknown_reasons"], _text) else ["unknown_reasons_invalid"]
        pid, seat, hid = raw["actor_player_id"], raw["actor_seat"], raw["hand_id"]
        if not _text(pid) or pid not in participants:
            _add(reasons, "stable_actor_identity_missing")
        if type(seat) is not int or not 0 <= seat < raw["table_size"]:
            _add(reasons, "physical_actor_seat_invalid")
        identity = (hid, pid) if _text(pid) else None
        if identity is not None and identity in seats_in_hand and seats_in_hand[
                identity] != seat:
            raise ValueError("same_player_changed_seat_inside_hand")
        if identity is not None:
            seats_in_hand[identity] = seat
        hand = hands[hid]
        session = sessions[raw["session_id"]]
        mapping = hand["seat_player_map"]
        if (type(seat) is int and isinstance(mapping, dict)
                and mapping.get(str(seat)) != pid):
            _add(reasons, "actor_differs_from_hand_seat_player_map")
        if (_text(pid) and pid in participants
                and raw["session_id"] not in participants[pid]["session_ids"]):
            _add(reasons, "actor_session_not_in_identity_scope")
        expected_hero = (_text(pid) and pid == scope["hero_player_id"])
        if type(raw["is_hero"]) is not bool or raw["is_hero"] != expected_hero:
            _add(reasons, "hero_identity_or_flag_mismatch")
        if (type(raw["table_size"]) is not int or raw["table_size"] not in (6, 7, 8)
                or raw["table_size"] != hand["table_size"]
                or raw["rule_fingerprint"] != hand["rule_fingerprint"]):
            _add(reasons, "opportunity_table_or_rule_binding_mismatch")
        if raw["position"] not in POSITIONS or raw["street"] not in STREETS:
            _add(reasons, "position_or_street_unknown")
        trajectory = trajectories.get(hid)
        if (trajectory is not None and trajectory["valid"]
                and raw["street"] in STREETS):
            prior_street = STREETS.index(trajectory["street"])
            current_street = STREETS.index(raw["street"])
            if current_street < prior_street:
                _add(reasons, "street_moves_backwards")
                trajectory["valid"] = False
            elif current_street > prior_street:
                if (current_street != prior_street + 1
                        or trajectory["pending"]
                        or trajectory["terminal"]):
                    _add(reasons, "street_advances_before_action_round_closed")
                    trajectory["valid"] = False
                else:
                    trajectory["street"] = raw["street"]
                    trajectory["street_committed"] = {
                        seat_id: Decimal(0) for seat_id in range(8)}
                    trajectory["current_bet"] = Decimal(0)
                    actionable = trajectory["pot_eligible"] - trajectory["all_in"]
                    trajectory["pending"] = [
                        value for value in _after(
                            trajectory["clockwise"], hand["dealer_seat"])
                        if value in actionable]
                    trajectory["raise_rights"] = set(actionable)
                    profile = rule_objects.get(hand["rule_profile_id"])
                    trajectory["minimum_raise_increment"] = (
                        profile.big_blind if profile is not None else Decimal(0))
                    trajectory["round_closed"] = False
        pot_seats = raw["pot_eligible_seats"]
        pending_seats = raw["pending_action_seats"]
        if (type(raw["active_count"]) is not int
                or not isinstance(pot_seats, list)
                or any(type(value) is not int for value in pot_seats)
                or pot_seats != sorted(set(pot_seats))
                or len(pot_seats) != raw["active_count"]
                or seat not in pot_seats
                or any(not 0 <= value < 8 for value in pot_seats)
                or any(value not in occupied for value in pot_seats)
                or any(mapping.get(str(value)) not in participants
                       for value in pot_seats)):
            _add(reasons, "active_and_pot_eligible_players_unknown")
        if (not isinstance(pending_seats, list)
                or any(type(value) is not int for value in pending_seats)
                or pending_seats != list(dict.fromkeys(pending_seats))
                or seat not in pending_seats):
            _add(reasons, "pending_action_order_unknown")
        if trajectory is not None and trajectory["valid"]:
            if (pot_seats != sorted(trajectory["pot_eligible"])
                    or raw["active_count"] != len(trajectory["pot_eligible"])):
                _add(reasons, "pot_eligible_players_differ_from_action_trajectory")
            if (pending_seats != trajectory["pending"] or not pending_seats
                    or seat != trajectory["pending"][0]):
                _add(reasons, "actor_or_pending_order_differs_from_trajectory")
                trajectory["valid"] = False
        plan = hand_plans.get(hid)
        if (plan is None or type(seat) is not int
                or plan.positions.get(seat) is None
                or plan.positions[seat].value != raw["position"]):
            _add(reasons, "position_differs_from_rule_bound_hand")
        if raw["rule_binding_status"] != "VERIFIED":
            _add(reasons, "row_rule_binding_unverified")
        if raw["audit_status"] != "reviewed":
            _add(reasons, "row_not_author_reviewed")
        if raw["author_review"] != "APPROVED":
            _add(reasons, "author_review_missing")
        if raw["independent_review"] != "APPROVED":
            review_pending.append("row_independent_review_pending:" + oid)
        _evidence(raw["evidence"], reasons, hand, session)
        evidence = raw["evidence"]
        if (isinstance(evidence, dict)
                and type(evidence.get("action_onset_frame")) is int
                and type(evidence.get("action_confirmation_frame")) is int
                and (evidence["action_onset_frame"] > hand["last_gameplay_frame"]
                     or evidence["action_confirmation_frame"]
                     > hand["last_gameplay_frame"])):
            _add(reasons, "action_evidence_after_last_gameplay_frame")
        last_gameplay_pts = _decimal(hand["last_gameplay_pts"])
        if (isinstance(evidence, dict) and last_gameplay_pts is not None
                and (_decimal(evidence.get("action_onset_pts")) is not None
                     and _decimal(evidence["action_onset_pts"]) > last_gameplay_pts
                     or _decimal(evidence.get("action_confirmation_pts")) is not None
                     and _decimal(evidence["action_confirmation_pts"])
                     > last_gameplay_pts)):
            _add(reasons, "action_evidence_pts_after_last_gameplay")
        prior_confirmation = previous_confirmation.get(hid)
        if prior_confirmation is not None and isinstance(evidence, dict):
            previous_frame, previous_pts = prior_confirmation
            pre_frames = [evidence.get(key) for key in (
                "actor_frame", "menu_frame", "state_frame")]
            pre_points = [_decimal(evidence.get(key)) for key in (
                "actor_pts", "menu_pts", "state_pts")]
            if (any(type(frame) is not int or frame <= previous_frame
                    for frame in pre_frames)
                    or any(point is None or point <= previous_pts
                           for point in pre_points)):
                _add(reasons, "next_predecision_not_after_previous_confirmation")
        evidence_identity = tuple(evidence.get(key) for key in (
            "actor_frame", "menu_frame", "state_frame", "action_onset_frame",
            "action_confirmation_frame", "actor_sha256", "menu_sha256",
            "state_sha256", "action_onset_sha256", "action_confirmation_sha256",
        )) if isinstance(evidence, dict) else None
        if evidence_identity is not None and all(
                value is not None for value in evidence_identity):
            if evidence_identity in evidence_identities:
                raise ValueError("duplicate_decision_evidence_for_new_canonical_id")
            evidence_identities.add(evidence_identity)
        onset = (evidence.get("action_onset_frame")
                 if isinstance(evidence, dict) else None)
        if type(onset) is int:
            onset_by_hand.setdefault(hid, []).append(onset)
        if (isinstance(evidence, dict)
                and type(evidence.get("action_confirmation_frame")) is int
                and _decimal(evidence.get("action_confirmation_pts")) is not None):
            previous_confirmation[hid] = (
                evidence["action_confirmation_frame"],
                _decimal(evidence["action_confirmation_pts"]))
        _snapshot(raw["predecision_snapshot"], reasons, hand, raw["street"],
                  raw["decision_sequence"], plan,
                  rule_objects.get(hand["rule_profile_id"]))
        snapshot = raw["predecision_snapshot"]
        history = (snapshot.get("public_history_through_sequence")
                   if isinstance(snapshot, dict) else None)
        expected_history = prior_actions_by_hand.get(hid, [])
        if history != expected_history:
            _add(reasons, "public_history_differs_from_prior_ledger_actions")
        if trajectory is None or not trajectory["valid"]:
            _add(reasons, "public_chip_trajectory_unavailable")
        elif isinstance(snapshot, dict):
            declared = {
                key: _decimal(snapshot.get(key)) for key in (
                    "pot_before", "current_bet", "actor_street_committed",
                    "actor_hand_committed", "minimum_raise_increment")}
            expected = {
                "pot_before": trajectory["pot"],
                "current_bet": trajectory["current_bet"],
                "actor_street_committed": trajectory[
                    "street_committed"].get(seat),
                "actor_hand_committed": trajectory["hand_committed"].get(seat),
                "minimum_raise_increment": trajectory[
                    "minimum_raise_increment"]}
            if any(declared[key] != expected[key] for key in expected):
                _add(reasons, "public_chip_trajectory_differs_from_snapshot")
            expected_reopened = seat in trajectory["raise_rights"]
            if snapshot.get("betting_reopened") is not expected_reopened:
                _add(reasons, "betting_reopened_differs_from_action_trajectory")
            declared_stack = _decimal(snapshot.get("actor_stack"))
            if seat in trajectory["stacks"]:
                if declared_stack != trajectory["stacks"][seat]:
                    _add(reasons, "actor_stack_differs_from_public_trajectory")
            elif declared_stack is not None:
                trajectory["stacks"][seat] = declared_stack
        rule = rules.get(hand["rule_profile_id"])
        menu = _action_menu(raw["legal_menu"], snapshot, rule, reasons)
        _observed(raw["observed_action"], menu, reasons)
        _special(raw["special_mode_state"], reasons)
        hand_modes = hand["special_mode_summary"]
        row_modes = raw["special_mode_state"]
        if (isinstance(hand_modes, dict) and isinstance(row_modes, dict)
                and any(hand_modes[mode] == "PRESENT"
                        and row_modes.get(mode) != "PRESENT" for mode in MODES)):
            _add(reasons, "row_special_mode_conflicts_hand_summary")
        model_scope = (_text(pid) and pid in scope["target_player_ids"]
                       and raw["street"] in scope["included_streets"]
                       and raw["active_count"] in scope["included_active_counts"])
        if raw["row_status"] != "REVIEWED_COMPLETE" and model_scope:
            _add(reasons, "target_row_not_reviewed_complete")
        if raw["row_status"] not in (
                "REVIEWED_COMPLETE", "UNKNOWN", "DEFERRED_SPECIAL_MODE",
                "OUTSIDE_PREREGISTERED_MODEL_SCOPE", "CENSORED_BOUNDARY"):
            _add(reasons, "invalid_row_status")
        if raw["row_status"] == "REVIEWED_COMPLETE" and reasons:
            _add(reasons, "row_claims_complete_with_blockers")
        observed = raw["observed_action"]
        if trajectory is not None and trajectory["valid"]:
            if not isinstance(observed, dict) or set(observed) != OBSERVED_ACTION_KEYS:
                trajectory["valid"] = False
            else:
                amount = _decimal(observed["amount"])
                kind = observed["action"]
                if amount is None or kind not in (
                        "fold", "check", "call", "bet", "raise", "all_in"):
                    trajectory["valid"] = False
                else:
                    old = trajectory["street_committed"].get(seat, Decimal(0))
                    previous_bet = trajectory["current_bet"]
                    previous_increment = trajectory["minimum_raise_increment"]
                    additional = (amount - old if kind in ("bet", "raise")
                                  else amount if kind in ("call", "all_in")
                                  else Decimal(0))
                    if additional < 0:
                        _add(reasons, "observed_action_reduces_commitment")
                        trajectory["valid"] = False
                    else:
                        trajectory["street_committed"][seat] = old + additional
                        trajectory["hand_committed"][seat] = trajectory[
                            "hand_committed"].get(seat, Decimal(0)) + additional
                        trajectory["pot"] += additional
                        if seat in trajectory["stacks"]:
                            if additional > trajectory["stacks"][seat]:
                                _add(reasons, "observed_action_exceeds_tracked_stack")
                                trajectory["valid"] = False
                            else:
                                trajectory["stacks"][seat] -= additional
                        if kind in ("bet", "raise"):
                            trajectory["current_bet"] = amount
                        elif kind == "all_in":
                            trajectory["current_bet"] = max(
                                trajectory["current_bet"], old + additional)
                        new_total = trajectory["street_committed"][seat]
                        aggressive = kind in ("bet", "raise", "all_in") and (
                            new_total > previous_bet)
                        if kind == "fold":
                            trajectory["pot_eligible"].discard(seat)
                        if kind == "all_in":
                            trajectory["all_in"].add(seat)
                        actionable = (trajectory["pot_eligible"]
                                      - trajectory["all_in"])
                        if aggressive:
                            raise_size = new_total - previous_bet
                            full_raise = raise_size >= previous_increment
                            trajectory["pending"] = [
                                value for value in _after(
                                    trajectory["clockwise"], seat)
                                if value in actionable and value != seat]
                            if full_raise:
                                trajectory["raise_rights"] = set(actionable) - {seat}
                                trajectory["minimum_raise_increment"] = raise_size
                            else:
                                trajectory["raise_rights"].discard(seat)
                        else:
                            trajectory["pending"] = trajectory["pending"][1:]
                            trajectory["raise_rights"].discard(seat)
                        trajectory["terminal"] = (
                            len(trajectory["pot_eligible"]) <= 1
                            or len(actionable) <= 1 and not trajectory["pending"])
                        trajectory["round_closed"] = not trajectory["pending"]
        if model_scope and not reasons:
            eligible += 1
            if session["split"] in eligible_by_split:
                eligible_by_split[session["split"]] += 1
                eligible_by_player[pid][session["split"]] += 1
        records.append({"opportunity_id": oid, "input": deepcopy(raw),
                        "eligible": bool(model_scope and not reasons),
                        "in_model_scope": bool(model_scope), "reasons": reasons})
        observed = raw["observed_action"]
        evidence = raw["evidence"]
        if (isinstance(observed, dict) and set(observed) == OBSERVED_ACTION_KEYS
                and isinstance(evidence, dict)
                and _hash(evidence.get("action_confirmation_sha256"))):
            prior_actions_by_hand.setdefault(hid, []).append({
                "decision_sequence": raw["decision_sequence"],
                "actor_player_id": pid, "actor_seat": seat,
                "street": raw["street"], "action": observed["action"],
                "amount": observed["amount"],
                "amount_semantics": observed["amount_semantics"],
                "source_sha256": evidence["action_confirmation_sha256"],
            })
    if missing:
        blockers.append("missing_details_in_all_opportunities_ledger")
    if any(record["reasons"] for record in records):
        blockers.append("unknown_or_unaudited_rows_forbid_complete_case_fit")
    if eligible == 0:
        blockers.append("zero_complete_target_opportunities")
    if any(eligible_by_split[key] == 0 for key in eligible_by_split):
        blockers.append("train_and_validation_need_complete_target_opportunities")
    if scope["calibration_unit"] == "per_stable_opponent":
        for pid, counts in eligible_by_player.items():
            participant = participants.get(pid)
            if (participant is None or participant["identity_scope"] != "cross_session"
                    or any(counts[key] == 0 for key in counts)):
                blockers.append(
                    "each_target_needs_cross_session_train_validation:" + pid)
    for hid, onsets in onset_by_hand.items():
        if onsets != sorted(onsets) or len(onsets) != len(set(onsets)):
            blockers.append("action_onsets_not_strictly_ordered:" + hid)
    for hid, trajectory in trajectories.items():
        if (trajectory["valid"] and not trajectory["terminal"]
                and not (trajectory["street"] == "river"
                         and trajectory["round_closed"])):
            blockers.append("hand_action_round_or_terminal_not_closed:" + hid)

    blockers = list(dict.fromkeys(blockers))
    review_pending = list(dict.fromkeys(review_pending))
    if blockers:
        readiness = "BLOCKED"
    elif data["source_kind"] != "reviewed_physical_capture_card":
        readiness = "NOT_REAL_DATA"
    else:
        readiness = "DECLARED_COMPLETE_NEEDS_ARTIFACT_VERIFICATION"
    return {
        "engineering_contract_status": "PASS",
        "data_readiness": readiness,
        "source_kind": data["source_kind"],
        "dataset_id": data["dataset_id"],
        "ledger_count": len(ledger), "detail_count": len(details),
        "missing_ids": missing, "eligible_target_count": eligible,
        "blockers": blockers, "review_pending": review_pending,
        "records": records,
        "train_session_count": len(train),
        "validation_session_count": len(validation),
        "eligible_by_split": eligible_by_split,
        "eligible_by_player": eligible_by_player,
        "ready_for_offline_calibration": False,
        "selection": None, "calibration": None, "range_model": None,
        "model_fit_executed": False, "strategy_eligible": False,
        "advice_emitted": False, "live_use": False,
        "provenance_unverified": True,
    }
