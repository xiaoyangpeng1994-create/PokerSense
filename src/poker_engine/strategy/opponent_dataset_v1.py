"""Audit public decision opportunities before finite opponent-model selection.

No record is dropped to create a complete-case sample. Identity and coverage
remain explicit, unverified reviewer declarations; no range is learned here.
"""

from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import re

from .response_model_calibration_v1 import (
    DecisionObservation, ModelCandidate, _observations, freeze_candidates,
    select_candidate, validate_selection,
)
from .threeway_river_v1 import ResponseModel, RiverAction


POSITIONS = {"SB", "BB", "UTG", "UTG1", "UTG2", "MP", "LJ", "HJ", "CO", "BTN"}
IDENTITY_KEYS = {"opportunity_id", "session_id", "hand_id", "opponent_id"}
DECISION_KEYS = IDENTITY_KEYS | {
    "source_kind", "source_hash", "seat_id", "table_size", "active_count",
    "position", "street", "pot", "to_call", "legal_actions", "observed_action",
    "audit_status", "unknown_reasons",
    "context",
}


def _fields(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("exact_" + name + "_fields_required")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _money(value):
    if not isinstance(value, str):
        raise ValueError("exact_decimal_string_required")
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise ValueError("finite_nonnegative_amount_required")
    return number


def _context(value):
    _fields(value, {"platform_id", "rule_fingerprint"}, "population_context")
    if not _text(value["platform_id"]) or not _hash(value["rule_fingerprint"]):
        raise ValueError("explicit_platform_and_rule_identity_required")


def _weights(value):
    if not isinstance(value, dict):
        raise ValueError("weight_mapping_required")
    return tuple((k, Fraction(_money(v))) for k, v in value.items())


def candidates_from_dict(data):
    _fields(data, {"schema_version", "opponent_id", "candidate_scope", "candidates",
                   "context"},
            "candidate_document")
    _context(data["context"])
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or not _text(data["opponent_id"])
            or data["candidate_scope"] != "public_legal_menu_and_price_only"
            or not isinstance(data["candidates"], list)):
        raise ValueError("explicit_public_only_candidates_required")
    candidates = []
    for item in data["candidates"]:
        _fields(item, {"candidate_id", "weights", "price_multipliers"}, "candidate")
        if not isinstance(item["price_multipliers"], list):
            raise ValueError("explicit_price_bands_required")
        prices = []
        for band in item["price_multipliers"]:
            _fields(band, {"upper_ratio", "weights"}, "price_band")
            prices.append((Fraction(_money(band["upper_ratio"])),
                           _weights(band["weights"])))
        candidates.append(ModelCandidate(item["candidate_id"], (
            ResponseModel(0, _weights(item["weights"]), (), tuple(prices)),)))
    return freeze_candidates(tuple(candidates))


def _action(data):
    _fields(data, {"kind", "target"}, "public_action")
    if not isinstance(data["kind"], str):
        raise ValueError("action_kind_string_required")
    return RiverAction(0, data["kind"], _money(data["target"]))


def audit_opponent_dataset(data, candidates, *, candidates_sha256):
    """Return all input rows and blockers; never select a convenient subset."""
    frozen = candidates_from_dict(candidates)
    _fields(data, {"schema_version", "source_kind", "opponent_id", "coverage",
                   "protocol", "opportunities", "decisions", "context"}, "dataset")
    _context(data["context"])
    if data["context"] != candidates["context"]:
        raise ValueError("candidate_and_dataset_population_context_mismatch")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["source_kind"] not in ("synthetic", "reviewed_public_decisions")
            or not _text(data["opponent_id"])
            or data["opponent_id"] != candidates["opponent_id"]):
        raise ValueError("one_explicit_opponent_and_source_kind_required")
    coverage, protocol = data["coverage"], data["protocol"]
    _fields(coverage, {"status", "reviewer", "evidence_hash"}, "coverage")
    _fields(protocol, {"training_sessions", "validation_sessions", "candidates_sha256"},
            "protocol")
    if (not _hash(candidates_sha256)
            or protocol["candidates_sha256"] != candidates_sha256):
        raise ValueError("candidate_file_differs_from_predeclared_protocol")
    groups = []
    for name in ("training_sessions", "validation_sessions"):
        group = protocol[name]
        if (not isinstance(group, list) or not group or not all(map(_text, group))
                or len(set(group)) != len(group)):
            raise ValueError("unique_explicit_session_partitions_required")
        groups.append(set(group))
    if groups[0] & groups[1]:
        raise ValueError("whole_session_split_overlap")
    if (not isinstance(data["opportunities"], list)
            or not isinstance(data["decisions"], list)
            or not 1 <= len(data["opportunities"]) <= 10000
            or len(data["decisions"]) > 10000):
        raise ValueError("bounded_all_decisions_inclusion_ledger_required")
    blockers = []
    expected_status = ("synthetic_complete" if data["source_kind"] == "synthetic"
                       else "reviewed_complete")
    if (coverage["status"] != expected_status or not _text(coverage["reviewer"])
            or not _hash(coverage["evidence_hash"])):
        blockers.append("all_decisions_coverage_not_reviewed")
    ledger, hand_sessions = {}, {}
    for opportunity in data["opportunities"]:
        _fields(opportunity, IDENTITY_KEYS, "opportunity")
        if not all(map(_text, opportunity.values())):
            raise ValueError("explicit_opportunity_identity_required")
        oid, sid, hid = (opportunity[k] for k in (
            "opportunity_id", "session_id", "hand_id"))
        if oid in ledger:
            raise ValueError("duplicate_opportunity_id")
        if opportunity["opponent_id"] != data["opponent_id"]:
            raise ValueError("opponent_identity_conflation_forbidden")
        if sid not in groups[0] | groups[1]:
            raise ValueError("unassigned_session_in_opportunity_ledger")
        if hid in hand_sessions and hand_sessions[hid] != sid:
            raise ValueError("hand_identity_crosses_sessions")
        hand_sessions[hid] = sid
        ledger[oid] = opportunity
    if {o["session_id"] for o in ledger.values()} != groups[0] | groups[1]:
        blockers.append("declared_partition_has_no_opportunities")
    records, observations, seen, sources = [], [], set(), set()
    seats_by_hand = {}
    for raw in data["decisions"]:
        _fields(raw, DECISION_KEYS, "decision")
        oid = raw["opportunity_id"]
        if not _text(oid) or oid in seen:
            raise ValueError("duplicate_or_invalid_decision_id")
        seen.add(oid)
        if oid not in ledger:
            raise ValueError("decision_not_in_inclusion_ledger")
        if {k: raw[k] for k in IDENTITY_KEYS} != ledger[oid]:
            raise ValueError("decision_identity_differs_from_opportunity")
        if raw["source_kind"] != data["source_kind"]:
            raise ValueError("mixed_or_relabelled_source_kinds_forbidden")
        if (not isinstance(raw["unknown_reasons"], list)
                or not all(map(_text, raw["unknown_reasons"]))):
            raise ValueError("explicit_unknown_reason_list_required")
        reasons = list(raw["unknown_reasons"])
        if raw["context"] != data["context"]:
            reasons.append("row_platform_or_rule_context_missing_or_mismatch")
        if raw["audit_status"] != "reviewed":
            reasons.append("decision_not_reviewed")
        identity = (raw["session_id"], raw["hand_id"], raw["opponent_id"])
        seat = raw["seat_id"]
        if (type(raw["table_size"]) is not int or raw["table_size"] not in (6, 7, 8)
                or type(seat) is not int or not 0 <= seat < raw["table_size"]):
            reasons.append("unknown_or_invalid_table_seat")
        elif identity in seats_by_hand and seats_by_hand[identity] != seat:
            raise ValueError("same_opponent_changed_seat_inside_hand")
        else:
            seats_by_hand[identity] = seat
        if (type(raw["active_count"]) is not int or raw["active_count"] != 3
                or raw["street"] != "river"):
            reasons.append("outside_declared_three_active_river_scope")
        if not isinstance(raw["position"], str) or raw["position"] not in POSITIONS:
            reasons.append("unknown_or_invalid_position")
        source = raw["source_hash"]
        if not _hash(source):
            reasons.append("missing_decision_evidence_hash")
        elif source in sources:
            raise ValueError("duplicate_decision_evidence_hash")
        else:
            sources.add(source)
        observation = None
        try:
            pot, to_call = _money(raw["pot"]), _money(raw["to_call"])
            if pot <= 0:
                raise ValueError("positive_public_pot_required")
            if not isinstance(raw["legal_actions"], list):
                raise ValueError("complete_public_legal_menu_required")
            observation = DecisionObservation(
                "synthetic" if data["source_kind"] == "synthetic"
                else "reviewed_all_decisions",
                raw["session_id"], raw["hand_id"], oid, source, 0, None,
                tuple(map(_action, raw["legal_actions"])),
                Fraction(to_call) / (Fraction(pot) + Fraction(to_call)),
                _action(raw["observed_action"]),
            )
            _observations((observation,), frozen)
        except (ValueError, TypeError, ArithmeticError) as exc:
            reasons.append(str(exc))
        if not reasons and observation is not None:
            observations.append(observation)
        records.append({"opportunity_id": oid, "input": deepcopy(raw),
                        "eligible": not reasons, "reasons": reasons})
    missing = sorted(set(ledger) - seen)
    if missing:
        blockers.append("missing_decisions_in_all_opportunities_ledger")
    if any(not r["eligible"] for r in records):
        blockers.append("unknown_or_unaudited_rows_forbid_complete_case_calibration")
    for oid in missing:
        records.append({"opportunity_id": oid, "input": deepcopy(ledger[oid]),
                        "eligible": False, "reasons": ["decision_missing"]})
    audit = {
        "status": "BLOCKED" if blockers else "DECLARED_COMPLETE_NEEDS_REVIEW",
        "source_kind": data["source_kind"], "opponent_id": data["opponent_id"],
        "context": deepcopy(data["context"]),
        "candidate_parameters_sha256": frozen.sha256,
        "opportunity_count": len(ledger), "decision_count": len(data["decisions"]),
        "eligible_count": len(observations), "missing_ids": missing,
        "blockers": blockers, "records": records, "coverage": deepcopy(coverage),
        "protocol": deepcopy(protocol), "provenance_unverified": True,
        "assumptions": [
            "opponent_identity_and_coverage_are_declarations_not_authenticated",
            "physical_seat_is_retained_but_model_seat0_means_named_opponent",
            "position_table_size_active_count_are_retained_not_fitted_features",
            "public_legal_menu_price_only_no_private_cards_or_range_estimation",
            "no_selective_complete_case_or_showdown_only_fit",
        ],
    }
    return audit, frozen, tuple(observations)


def calibrate_opponent_dataset(data, candidates, *, candidates_sha256):
    audit, frozen, observations = audit_opponent_dataset(
        data, candidates, candidates_sha256=candidates_sha256)
    result = {"audit": audit, "selection": None, "calibration": None,
              "strategy_eligible": False, "advice_emitted": False,
              "range_model": None}
    if audit["blockers"]:
        return result
    train_sessions = set(data["protocol"]["training_sessions"])
    train = tuple(r for r in observations if r.session_id in train_sessions)
    validation = tuple(r for r in observations if r.session_id not in train_sessions)
    try:
        selection = select_candidate(frozen, train)
        calibration = validate_selection(frozen, selection, validation)
    except (ValueError, TypeError, ArithmeticError) as exc:
        audit["status"] = "BLOCKED"
        audit["blockers"].append("calibration_contract:" + str(exc))
        return result
    result.update(selection=selection, calibration=calibration)
    return result
