"""TARGET-S real-hand confirmation: contract, isolation and derivation tests.

Everything here is synthetic metadata written into a temporary append-only
store. No raw media is read, no frame is decoded, no solver or PHH export runs.

The isolation tests are the load-bearing ones: they prove that machine
candidates, #31 annotation rows and the existing manual-hypothesis hand form
(with its own ``human_confirmed`` provenance) can none of them produce a
TARGET-S acceptance receipt.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from poker_engine.desktop import aa_hand_input
import poker_engine.desktop.aa_real_hand_confirmation as module
from poker_engine.desktop.aa_real_hand_confirmation import (
    ACCEPTED, DERIVATION_STACK_DELTA, MARKER_RIVER_START, NOT_READY,
    PROV_DERIVED, PROV_HUMAN, STATUS_CONFIRMED, STATUS_CONFLICT,
    STATUS_UNCONFIRMED, STACK_DELTA_ASSERTIONS, TARGET_S_REQUIRED_FACTS,
    ConfirmationStore, EvidenceBindingError, FactValueError,
    IdentityMismatchError, RealHandConfirmationError as ConfirmationError,
    RevisionError, canonical, confirm_fact, confirmed_view,
    derive_hand_committed_by_stack_delta, target_s_acceptance,
    validate_target_s_receipt,
)

HAND = "observed_deal_synthetic_0001"
EPOCH = "epoch-synthetic-0001"
SESSION = "capture-synthetic-0001"
SOURCE = {"media_sha256": "a" * 64, "recording_id": "aa-live-controlled-0001"}
OTHER_SOURCE = {"media_sha256": "b" * 64,
                "recording_id": "aa-live-controlled-0002"}

HERO = 1
ACTIVE = [1, 3, 5]
FOLDED = [0, 2, 4]
SEAT_SET = [0, 1, 2, 3, 4, 5]
BOARD = ["Ks", "Qh", "7h", "9h", "2c"]
HERO_CARDS = ["6d", "Kc"]
OPENING = {str(seat): 200 for seat in SEAT_SET}
RIVER = {str(seat): 183 if seat in ACTIVE else 196 for seat in SEAT_SET}
POT = {"raw": "63", "value": 63}
RULES = {
    "table_size": 6, "small_blind": 1, "big_blind": 2, "ante": 2,
    "ante_mode": "per_dealt_player", "straddle_amount": 4,
    "straddle_mode": "optional_explicit_utg", "revision": "rules-synthetic-1",
}


def evidence_descriptor(marker, *, hand_id=HAND, epoch=EPOCH, session=SESSION,
                        source=SOURCE, start=None, end=None):
    first = {"HAND_START": 100, "RIVER_START": 200, "PAYOUT": 300}[marker]
    start = first if start is None else start
    end = start + 2 if end is None else end
    return {
        "schema_version": module.EVIDENCE_SCHEMA_VERSION,
        "hand_id": hand_id, "observed_epoch": epoch,
        "capture_session_id": session, "source_ref": deepcopy(source),
        "marker": marker, "window_start_frame": start, "window_end_frame": end,
        "frames": [{"source_frame": frame, "content_sha256": "c" * 64}
                   for frame in range(start, end + 1)],
        "stable": True, "boundary_continuity": True,
    }


DESCRIPTORS = {marker: evidence_descriptor(marker) for marker in module.MARKERS}
ASSERTIONS = {
    **{key: True for key in STACK_DELTA_ASSERTIONS},
    "hand_start_evidence_digest": module.digest(DESCRIPTORS["HAND_START"]),
    "river_start_evidence_digest": module.digest(DESCRIPTORS["RIVER_START"]),
}


def sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def store(tmp_path, name="store"):
    return ConfirmationStore(Path(tmp_path) / name)


def confirm(target, fact_key, value, *, hand_id=HAND, epoch=EPOCH,
            session=SESSION, source=SOURCE, marker=None,
            frame=None, evidence=None, reviewer="reviewer-a",
            supersedes=None, audit_note=None, descriptor=None,
            snapshot_digest=None):
    marker = module.FACT_MARKERS[fact_key] if marker is None else marker
    descriptor = (evidence_descriptor(marker, hand_id=hand_id, epoch=epoch,
                                      session=session, source=source)
                  if descriptor is None else descriptor)
    frame = descriptor["window_start_frame"] if frame is None else frame
    return confirm_fact(
        target, hand_id=hand_id, observed_epoch=epoch,
        capture_session_id=session, source_ref=source, marker=marker,
        source_frame=frame,
        evidence_digest=module.digest(descriptor) if evidence is None else evidence,
        evidence_descriptor=descriptor, snapshot_digest=snapshot_digest,
        fact_key=fact_key, value=value, reviewer=reviewer,
        supersedes=supersedes, audit_note=audit_note,
    )


def write_raw_rows(target, rows):
    """Append rows bypassing ``confirm_fact``, keeping the digest chain valid.

    Used to stage states the API refuses to create: a second live head, for
    example, is exactly what the revision rules exist to prevent.
    """
    previous = None
    path = target.path / module.CONFIRMATION_FILE
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                previous = json.loads(raw).get("record_digest")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            envelope = {"record": row, "record_digest": module.digest(row),
                        "prev_digest": previous}
            handle.write(module.canonical(envelope) + "\n")
            previous = envelope["record_digest"]


FULL_FACTS = {
    "ended_hand_confirmed": True,
    "hero_seat": HERO,
    "hero_cards": list(HERO_CARDS),
    "board_cards": list(BOARD),
    "seat_set": list(SEAT_SET),
    "active_seats": list(ACTIVE),
    "all_in_seats": [],
    "folded_seats": list(FOLDED),
    "action_order": [1, 3, 5],
    "dealer_button_seat": 5,
    "opening_stacks": dict(OPENING),
    "river_start_stacks": dict(RIVER),
    "stack_delta_assertions": dict(ASSERTIONS),
    "street_wagers_zero": True,
    "pot_display": dict(POT),
    "hero_is_first_river_actor": True,
    "table_rules": dict(RULES),
    "other_fees": {"value": 0, "confirmed": True},
    "straddle_posted_this_hand": True,
    "no_side_pot": True,
    "no_pending_action": True,
}


def confirm_all(target, overrides=None, *, skip=()):
    facts = dict(FULL_FACTS)
    facts.update(overrides or {})
    written = {}
    for key in TARGET_S_REQUIRED_FACTS:
        if key in skip or key not in facts:
            continue
        written[key] = confirm(target, key, facts[key])
    return written


# --- 1-3: a confirmation is refused unless it is bound to evidence ----------

def test_missing_evidence_digest_is_refused(tmp_path):
    target = store(tmp_path)
    with pytest.raises(EvidenceBindingError):
        confirm(target, "hero_seat", HERO, evidence="not-a-digest")


def test_observed_epoch_mismatch_is_refused(tmp_path):
    target = store(tmp_path)
    confirm(target, "hero_seat", HERO)
    with pytest.raises(IdentityMismatchError):
        confirm(target, "hero_cards", list(HERO_CARDS),
                epoch="epoch-synthetic-0002")


def test_source_recording_mismatch_is_refused(tmp_path):
    target = store(tmp_path)
    confirm(target, "hero_seat", HERO)
    with pytest.raises(IdentityMismatchError):
        confirm(target, "hero_cards", list(HERO_CARDS), source=OTHER_SOURCE)


# --- 4-6: append-only revision integrity -----------------------------------

def test_revision_appends_and_supersedes_without_overwriting(tmp_path):
    target = store(tmp_path)
    first = confirm(target, "hero_seat", HERO)
    second = confirm(target, "hero_seat", 3, supersedes=first["confirmation_id"])
    rows = target.read_confirmations()
    assert len(rows) == 2
    assert second["revision"] == first["revision"] + 1
    assert first["value"] == HERO
    assert second["supersedes"] == first["confirmation_id"]
    assert confirmed_view(target, HAND)["confirmed"]["hero_seat"] == 3


def test_stale_supersedes_is_refused(tmp_path):
    target = store(tmp_path)
    confirm(target, "hero_seat", HERO)
    with pytest.raises(RevisionError):
        confirm(target, "hero_seat", 3, supersedes="rhc-does-not-exist")


def test_two_live_heads_leave_the_fact_conflicted(tmp_path):
    target = store(tmp_path)
    first = confirm(target, "hero_seat", HERO)
    duplicate = dict(first)
    duplicate["confirmation_id"] = "rhc-second-head"
    duplicate["supersedes"] = None
    duplicate["revision"] = 2
    duplicate["value"] = 3
    write_raw_rows(target, [duplicate])
    assert first["confirmation_id"] != duplicate["confirmation_id"]
    view = confirmed_view(target, HAND)
    assert "hero_seat" not in view["confirmed"]
    assert view["conflicts"]
    assert "one_revision_one_root_required" in view["conflicts"][0]["reason"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


# --- 7-9: nothing but a bound human confirmation is a fact -----------------

def test_free_text_note_is_never_a_fact_value(tmp_path):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "hero_seat", "looks like seat 1")
    row = confirm(target, "hero_seat", HERO, audit_note="checked twice")
    assert row["audit_note"] == "checked twice"
    assert row["value"] == HERO
    assert row["provenance"] == PROV_HUMAN


def test_machine_candidate_cannot_become_a_confirmed_fact(tmp_path):
    target = store(tmp_path)
    candidate = {"interpretation_status": "PRICE_DERIVED_CANDIDATE",
                 "kind": "bet", "target": "20", "slot": 1}
    with pytest.raises(FactValueError):
        confirm(target, "hero_cards", candidate)
    assert "PRICE_DERIVED_CANDIDATE" not in (
        STATUS_CONFIRMED, STATUS_UNCONFIRMED, STATUS_CONFLICT)
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_annotation_record_cannot_become_a_real_hand_fact(tmp_path):
    import tools.aa_annotation_records as annotations

    target = store(tmp_path)
    row = {"revision_id": "rev-synthetic-1", "field_id": "pot_display",
           "slot": 1, "label": {"status": annotations.STATUS_KNOWN,
                                "value": "55"}}
    with pytest.raises(FactValueError):
        confirm(target, "pot_display", row)
    assert STATUS_CONFIRMED not in (annotations.STATUS_KNOWN,
                                    annotations.STATUS_UNKNOWN,
                                    annotations.STATUS_NOT_APPLICABLE)
    # A #31 annotation store is not an acceptance store by any path.
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_manual_hypothesis_human_confirmed_is_not_acceptance(tmp_path):
    """The old form keeps working, and still cannot yield a TARGET-S receipt."""
    assert aa_hand_input.AAHandInput().template()["scope"] == (
        module.MANUAL_HYPOTHESIS_SCOPE)
    assert module.SCOPE != module.MANUAL_HYPOTHESIS_SCOPE
    assert not hasattr(aa_hand_input, "target_s_acceptance")

    block = {
        "ended_hand_confirmed": {"value": True,
                                 "provenance": PROV_HUMAN},
        "hero_seat": {"value": HERO, "provenance": PROV_HUMAN},
        "hero_cards": {"value": list(HERO_CARDS), "provenance": PROV_HUMAN},
        "board_cards": {"value": list(BOARD), "provenance": PROV_HUMAN},
        "action_order": {"value": [1, 3, 5], "provenance": PROV_HUMAN},
        "seats": {"value": [
            {"seat_id": 1, "stack": "183", "hand_committed": "17",
             "status": "ACTIVE"},
            {"seat_id": 3, "stack": "183", "hand_committed": "17",
             "status": "ACTIVE"},
            {"seat_id": 5, "stack": "183", "hand_committed": "17",
             "status": "ACTIVE"},
        ], "provenance": PROV_HUMAN},
        "history": {"value": [], "provenance": PROV_HUMAN},
        "pot_display": {"value": "55", "provenance": PROV_HUMAN},
        "table_rules": {"value": {
            "table_size": 6, "small_blind": 1, "big_blind": 2, "ante": 2,
            "verification_status": "simulation",
        }, "provenance": PROV_HUMAN},
    }
    normalised = aa_hand_input.normalise_facts(block)
    assert normalised["hero_seat"] == HERO
    assert ACCEPTED not in canonical(normalised)
    assert aa_hand_input.facts_hashes(block, {}, normalised)["scope"] == (
        module.MANUAL_HYPOTHESIS_SCOPE)

    # The same facts, with no confirmation rows, are still not accepted:
    # acceptance lives in the append-only store, not in a form submission.
    target = store(tmp_path)
    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == NOT_READY
    assert receipt["manual_hypothesis_scope"] == module.MANUAL_HYPOTHESIS_SCOPE


# --- 11-15: the TARGET-S structural gates ----------------------------------

def test_hero_not_first_to_act_is_not_ready(tmp_path):
    target = store(tmp_path)
    confirm_all(target, {"action_order": [3, 1, 5]})
    view = confirmed_view(target, HAND)
    assert view["status"] == "NOT_READY"
    assert "hero_is_not_first_in_the_action_order" in view["blocking"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_hero_first_to_act_must_be_confirmed_not_assumed(tmp_path):
    target = store(tmp_path)
    confirm_all(target, skip=("hero_is_first_river_actor",))
    view = confirmed_view(target, HAND)
    assert "hero_is_first_river_actor" in view["unconfirmed"]
    assert "history" in view["unconfirmed"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_exactly_three_active_players(tmp_path):
    target = store(tmp_path)
    confirm_all(target)
    view = confirmed_view(target, HAND)
    assert view["checks"]["exactly_three_active"] is True
    assert view["confirmed"]["active_seats"] == ACTIVE


def test_all_in_is_not_counted_as_active(tmp_path):
    target = store(tmp_path)
    confirm_all(target, {"all_in_seats": [5]})
    view = confirmed_view(target, HAND)
    assert "an_all_in_seat_was_counted_as_active" in view["blocking"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


@pytest.mark.parametrize("active", [[1, 3], [1, 3, 5, 7]])
def test_active_count_other_than_three_is_refused(tmp_path, active):
    target = store(tmp_path)
    confirm_all(target, {"active_seats": list(active),
                         "action_order": list(active)})
    view = confirmed_view(target, HAND)
    assert "river_start_must_have_exactly_three_active_players" in (
        view["blocking"])
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_side_pot_is_refused(tmp_path):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "no_side_pot", False)
    # Commitments that disagree are caught even when the box was ticked.
    confirm_all(target, {"river_start_stacks": {**RIVER, "3": 180},
                         "pot_display": {"raw": "66", "value": 66}})
    view = confirmed_view(target, HAND)
    assert any("side_pot" in item for item in view["blocking"])
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


# --- 16-24: the stack-delta commitment derivation, condition by condition ---

def _derive(**overrides):
    payload = {"opening_stacks": dict(OPENING), "river_start_stacks": dict(RIVER),
               "assertions": dict(ASSERTIONS), "pot_display": dict(POT),
               "seats": list(SEAT_SET),
               "opening_evidence_digest": module.digest(DESCRIPTORS["HAND_START"]),
               "river_evidence_digest": module.digest(DESCRIPTORS["RIVER_START"])}
    payload.update(overrides)
    return derive_hand_committed_by_stack_delta(**payload)


def test_stack_snapshot_identity_mismatch_is_unconfirmed():
    result = _derive(river_start_stacks={"1": 183, "3": 183, "5": 183})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("seat_identity_mismatch" in item for item in result["reasons"])


def test_rebuy_or_topup_flag_is_unconfirmed():
    assertions = dict(ASSERTIONS)
    assertions["no_rebuy_or_topup"] = False
    result = _derive(assertions=assertions)
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("no_rebuy_or_topup" in item for item in result["reasons"])


def test_chip_return_flag_is_unconfirmed():
    assertions = dict(ASSERTIONS)
    assertions["no_chip_return"] = False
    result = _derive(assertions=assertions)
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("no_chip_return" in item for item in result["reasons"])


def test_payout_before_river_flag_is_unconfirmed():
    assertions = dict(ASSERTIONS)
    assertions["no_payout_before_river"] = False
    result = _derive(assertions=assertions)
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("no_payout_before_river" in item for item in result["reasons"])


def test_chip_side_fee_flag_is_unconfirmed():
    assertions = dict(ASSERTIONS)
    assertions["no_chip_side_fee_or_rake"] = False
    result = _derive(assertions=assertions)
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("no_chip_side_fee_or_rake" in item for item in result["reasons"])


def test_precision_mismatch_is_unconfirmed():
    result = _derive(opening_stacks={"0": 200, "1": 200.5, "3": 200, "5": 200})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("opening_stacks_invalid" in item for item in result["reasons"])


def test_negative_stack_delta_is_unconfirmed():
    result = _derive(river_start_stacks={**RIVER, "1": 210})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("stack_delta_negative" in item for item in result["reasons"])


def test_pot_mismatch_is_unconfirmed():
    result = _derive(pot_display={"raw": "54", "value": 54})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("stack_delta_pot_mismatch" in item for item in result["reasons"])


def test_pot_agrees_so_the_derivation_is_confirmed():
    result = _derive()
    assert result["status"] == STATUS_CONFIRMED
    assert result["per_seat"] == {"0": 4, "1": 17, "2": 4, "3": 17,
                                  "4": 4, "5": 17}
    assert result["total"] == 63
    # A derived value must say it is derived, not that a human stated it.
    assert result["provenance"] == PROV_DERIVED
    assert result["derivation"] == DERIVATION_STACK_DELTA


# --- 25-28: marker-state gates ---------------------------------------------

def test_nonzero_street_wager_is_refused(tmp_path):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "street_wagers_zero", False)
    confirm_all(target, skip=("street_wagers_zero",))
    view = confirmed_view(target, HAND)
    assert "street_wagers_are_not_all_zero_at_river_start" in view["blocking"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_pending_action_or_animation_is_refused(tmp_path):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "no_pending_action", False)
    confirm_all(target, skip=("no_pending_action",))
    view = confirmed_view(target, HAND)
    assert "a_pending_action_or_animation_is_still_unresolved" in (
        view["blocking"])
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_unobserved_optional_straddle_is_refused(tmp_path):
    target = store(tmp_path)
    confirm_all(target, skip=("straddle_posted_this_hand",))
    view = confirmed_view(target, HAND)
    assert any("optional_straddle" in item for item in view["blocking"])
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_unknown_other_fees_are_not_defaulted_to_zero(tmp_path):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "other_fees", {"value": 0, "confirmed": False})
    confirm_all(target, skip=("other_fees",))
    view = confirmed_view(target, HAND)
    assert "other_fees" in view["unconfirmed"]
    assert "other_fees" not in view["confirmed"]
    assert view["confirmed"].get("other_fees") is None
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


# --- 29-30: the acceptance receipt -----------------------------------------

def test_full_valid_synthetic_hand_is_target_s_confirmed(tmp_path):
    target = store(tmp_path)
    confirm_all(target)
    view = confirmed_view(target, HAND)
    assert view["status"] == "READY"
    assert view["blocking"] == []
    assert view["derived"]["history"] == []
    assert view["derived"]["history_empty_because"] == "hero_is_first_river_actor"
    assert view["derived"]["hand_committed"] == {"0": 4, "1": 17, "2": 4,
                                                 "3": 17, "4": 4, "5": 17}

    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == ACCEPTED
    for key in ("confirmation_bundle_digest", "river_start_snapshot_digest",
                "derived_commitment_digest", "rules_digest", "facts_digest",
                "receipt_digest"):
        assert isinstance(receipt[key], str) and len(receipt[key]) == 64
    assert receipt["hand_id"] == HAND
    assert receipt["observed_epoch"] == EPOCH
    assert receipt["source_digest"]


def test_receipt_states_phh_and_strategy_are_not_assessed(tmp_path):
    target = store(tmp_path)
    confirm_all(target)
    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == ACCEPTED
    assert receipt["phh_ready"] is False
    assert receipt["strategy_assessed"] is False
    assert receipt["target_p_phh_ready"] is False
    assert receipt["target"] == "TARGET_S"


def test_confirmed_view_never_trusts_a_stored_verdict(tmp_path):
    """A store that lost one fact must fall back to NOT_READY."""
    target = store(tmp_path)
    confirm_all(target, skip=("river_start_stacks",))
    view = confirmed_view(target, HAND)
    assert view["status"] == "NOT_READY"
    assert "river_start_stacks" in view["unconfirmed"]
    assert "hand_committed" in view["unconfirmed"]
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


def test_store_integrity_failure_is_refused(tmp_path):
    target = store(tmp_path)
    confirm(target, "hero_seat", HERO)
    path = target.path / module.CONFIRMATION_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    envelope = json.loads(lines[0])
    envelope["record"]["value"] = 7
    path.write_text(module.canonical(envelope) + "\n", encoding="utf-8")
    with pytest.raises(module.StoreIntegrityError):
        target.read_confirmations()


def test_errors_share_one_base(tmp_path):
    for error in (EvidenceBindingError, IdentityMismatchError, RevisionError,
                  FactValueError, module.StoreIntegrityError,
                  module.StoreLockError):
        assert issubclass(error, ConfirmationError)


def assert_not_ready(target):
    view = confirmed_view(target, HAND)
    assert view["status"] == "NOT_READY"
    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == NOT_READY
    assert not validate_target_s_receipt(target, HAND, receipt)
    return view


def rewrite_rows(target, rows):
    """Re-seal synthetic corrupt histories, never touch real evidence."""
    (target.path / module.CONFIRMATION_FILE).write_text("", encoding="utf-8")
    write_raw_rows(target, rows)


def confirm_with_windows(target, descriptors):
    facts = deepcopy(FULL_FACTS)
    facts["stack_delta_assertions"].update(
        hand_start_evidence_digest=module.digest(descriptors["HAND_START"]),
        river_start_evidence_digest=module.digest(descriptors["RIVER_START"]))
    return {key: confirm(target, key, facts[key],
                         descriptor=descriptors[module.FACT_MARKERS[key]])
            for key in TARGET_S_REQUIRED_FACTS}


# P0-1: evidence, revision and ancestry are part of acceptance identity.
@pytest.mark.parametrize("key", ["board_cards", "table_rules"])
@pytest.mark.parametrize("change", ["reviewer", "frame", "audit_note"])
def test_same_value_revision_invalidates_old_receipt(tmp_path, key, change):
    target = store(tmp_path)
    rows = confirm_all(target)
    before = target_s_acceptance(target, HAND)
    assert validate_target_s_receipt(target, HAND, before)
    values = {"reviewer": "reviewer-b", "frame": rows[key]["source_frame"] + 1,
              "audit_note": "same value independently rechecked"}
    confirm(target, key, FULL_FACTS[key], supersedes=rows[key]["confirmation_id"],
            **{change: values[change]})
    after = target_s_acceptance(target, HAND)
    assert after["status"] == ACCEPTED
    assert before["facts_digest"] == after["facts_digest"]
    assert before["confirmation_bundle_digest"] != after["confirmation_bundle_digest"]
    assert before["receipt_digest"] != after["receipt_digest"]
    assert not validate_target_s_receipt(target, HAND, before)
    assert validate_target_s_receipt(target, HAND, after)


def test_new_river_evidence_requires_coherent_window_and_interval(tmp_path):
    target = store(tmp_path)
    rows = confirm_all(target)
    before = target_s_acceptance(target, HAND)
    descriptor = deepcopy(DESCRIPTORS[MARKER_RIVER_START])
    descriptor["frames"][0]["content_sha256"] = "d" * 64
    for key in TARGET_S_REQUIRED_FACTS:
        if module.FACT_MARKERS[key] != MARKER_RIVER_START:
            continue
        value = deepcopy(FULL_FACTS[key])
        if key == "stack_delta_assertions":
            value["river_start_evidence_digest"] = module.digest(descriptor)
        confirm(target, key, value, descriptor=descriptor,
                supersedes=rows[key]["confirmation_id"])
        if key == "hero_seat":
            assert_not_ready(target)
    after = target_s_acceptance(target, HAND)
    assert after["status"] == ACCEPTED
    assert after["river_start_snapshot_digest"] == module.digest(descriptor)
    assert after["river_start_snapshot_digest"] != before["river_start_snapshot_digest"]
    assert not validate_target_s_receipt(target, HAND, before)


def test_resealed_ancestry_change_is_not_the_same_receipt(tmp_path):
    target = store(tmp_path)
    rows = confirm_all(target)
    confirm(target, "hero_seat", HERO,
            supersedes=rows["hero_seat"]["confirmation_id"])
    before = target_s_acceptance(target, HAND)
    history = target.read_confirmations()
    next(row for row in history if row["confirmation_id"] ==
         rows["hero_seat"]["confirmation_id"])["reviewer"] = "different-ancestor"
    rewrite_rows(target, history)
    assert target_s_acceptance(target, HAND)["status"] == ACCEPTED
    assert not validate_target_s_receipt(target, HAND, before)


@pytest.mark.parametrize("field,value", [
    ("schema", "aa-real-hand-confirmation-v1"),
    ("strategy_assessed", True), ("target_p_phh_ready", True),
    ("implementation_version", "older"), ("receipt_digest", "0" * 64),
])
def test_receipt_validation_compares_whole_current_receipt(tmp_path, field, value):
    target = store(tmp_path)
    confirm_all(target)
    receipt = target_s_acceptance(target, HAND)
    receipt[field] = value
    assert not validate_target_s_receipt(target, HAND, receipt)


# P0-2: roles, identity, coherent windows, chronology and explicit interval.
@pytest.mark.parametrize("key,marker", [
    ("opening_stacks", "PAYOUT"), ("opening_stacks", "RIVER_START"),
    ("board_cards", "PAYOUT"), ("hero_is_first_river_actor", "PAYOUT"),
    ("table_rules", "RIVER_START"), ("ended_hand_confirmed", "RIVER_START"),
])
def test_fact_marker_roles_are_enforced_at_public_write(tmp_path, key, marker):
    target = store(tmp_path)
    with pytest.raises(EvidenceBindingError, match="fact_marker_role_mismatch"):
        confirm(target, key, FULL_FACTS[key], marker=marker)
    assert_not_ready(target)


@pytest.mark.parametrize("field,value", [
    ("hand_id", "new-hand"), ("epoch", "new-epoch"),
    ("session", "new-session"), ("source", OTHER_SOURCE),
])
def test_copied_descriptor_cannot_be_relabelled(tmp_path, field, value):
    target = store(tmp_path)
    with pytest.raises(IdentityMismatchError, match="evidence_identity_mismatch"):
        confirm(target, "hero_seat", HERO,
                descriptor=DESCRIPTORS["RIVER_START"], **{field: value})
    assert_not_ready(target)


@pytest.mark.parametrize("marker,start,end", [
    ("HAND_START", 500, 502), ("HAND_START", 198, 200),
    ("PAYOUT", 190, 192), ("PAYOUT", 202, 204),
])
def test_marker_chronology_uses_full_window_bounds(tmp_path, marker, start, end):
    target = store(tmp_path)
    windows = deepcopy(DESCRIPTORS)
    windows[marker] = evidence_descriptor(marker, start=start, end=end)
    confirm_with_windows(target, windows)
    view = assert_not_ready(target)
    assert any("marker_chronology_invalid" in item for item in view["blocking"])
    assert "hand_committed" not in view["derived"]
    assert "history" not in view["derived"]


def test_changing_opening_boundary_requires_new_interval_attestation(tmp_path):
    target = store(tmp_path)
    rows = confirm_all(target)
    before = target_s_acceptance(target, HAND)
    descriptor = evidence_descriptor("HAND_START", start=90, end=93)
    for key in module.HAND_START_FACTS:
        confirm(target, key, FULL_FACTS[key], descriptor=descriptor,
                supersedes=rows[key]["confirmation_id"])
    view = assert_not_ready(target)
    assert "stack_delta_interval_binding_mismatch" in (
        view["unconfirmed"]["hand_committed"])
    assert not validate_target_s_receipt(target, HAND, before)
    assertions = dict(ASSERTIONS, hand_start_evidence_digest=module.digest(descriptor))
    confirm(target, "stack_delta_assertions", assertions,
            supersedes=rows["stack_delta_assertions"]["confirmation_id"])
    assert target_s_acceptance(target, HAND)["status"] == ACCEPTED


def test_multiframe_interhand_opening_window_is_valid(tmp_path):
    target = store(tmp_path)
    windows = deepcopy(DESCRIPTORS)
    windows["HAND_START"] = evidence_descriptor("HAND_START", start=0, end=4)
    confirm_with_windows(target, windows)
    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == ACCEPTED
    assert receipt["river_start_snapshot_digest"] == module.digest(
        windows["RIVER_START"])


@pytest.mark.parametrize("mutation", [
    "unstable", "discontinuous", "missing_content", "outside_frame",
    "duplicate_frame", "missing_endpoint", "wrong_digest", "snapshot_digest",
])
def test_descriptor_cannot_be_unbound_or_silently_repaired(tmp_path, mutation):
    target = store(tmp_path)
    descriptor = deepcopy(DESCRIPTORS["RIVER_START"])
    kwargs = {}
    if mutation == "unstable":
        descriptor["stable"] = False
    elif mutation == "discontinuous":
        descriptor["boundary_continuity"] = False
    elif mutation == "missing_content":
        descriptor["frames"][0]["content_sha256"] = None
    elif mutation == "outside_frame":
        kwargs["frame"] = 199
    elif mutation == "duplicate_frame":
        descriptor["frames"].append(descriptor["frames"][-1])
    elif mutation == "missing_endpoint":
        descriptor["frames"].pop()
    elif mutation == "wrong_digest":
        kwargs["evidence"] = "f" * 64
    else:
        kwargs["snapshot_digest"] = "f" * 64
    with pytest.raises(EvidenceBindingError):
        confirm(target, "hero_seat", HERO, descriptor=descriptor, **kwargs)
    assert_not_ready(target)


# P0-3: envelope checksums cannot authorize invalid semantic recovery.
@pytest.mark.parametrize("graph", [
    [("r1", 1, "r2"), ("r2", 2, "r1"), ("r3", 3, None)],
    [("r1", 1, None), ("r3", 3, "r1")],
    [("r9", 9, None)],
    [("r1", 1, None), ("r3", 3, "r1"), ("r2", 2, "r3"), ("r4", 4, "r2")],
    [("r1", 1, None), ("r2", 2, "missing")],
    [("r1", 1, None), ("r2", 2, "r1"), ("r3", 3, "r1")],
    [("r1", 1, None), ("r2", 1, "r1")],
    [("r1", 1, None), ("r1", 2, "r1")],
    [("r2", 2, "r1"), ("r1", 1, None)],
])
def test_checksum_valid_invalid_graph_blocks_read_accept_and_append(tmp_path, graph):
    target = store(tmp_path)
    originals = confirm_all(target)
    before = target_s_acceptance(target, HAND)
    rows = [row for row in target.read_confirmations()
            if row["fact_key"] != "hero_seat"]
    rows.extend(dict(originals["hero_seat"], confirmation_id=identifier,
                     revision=revision, supersedes=parent)
                for identifier, revision, parent in graph)
    rewrite_rows(target, rows)
    assert_not_ready(target)
    assert not validate_target_s_receipt(target, HAND, before)
    with pytest.raises(module.StoreIntegrityError):
        target.read_confirmations()
    contents = (target.path / module.CONFIRMATION_FILE).read_bytes()
    with pytest.raises(module.StoreIntegrityError):
        confirm(target, "hero_seat", HERO, supersedes=graph[-1][0])
    assert (target.path / module.CONFIRMATION_FILE).read_bytes() == contents


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("field,value", [
    ("provenance", "PRICE_DERIVED_CANDIDATE"), ("evidence_digest", None),
    ("reviewer", ""), ("revision", None), ("revision", True),
    ("recorded_at", "yesterday"), ("value", True),
    ("source_digest", "0" * 64), ("source_ref", OTHER_SOURCE),
    ("observed_epoch", "another-epoch"), ("fact_key", "free_text"),
])
def test_invalid_live_and_superseded_rows_are_not_hidden(
        tmp_path, historical, field, value):
    target = store(tmp_path)
    originals = confirm_all(target)
    if historical:
        confirm(target, "hero_seat", HERO,
                supersedes=originals["hero_seat"]["confirmation_id"])
    before = target_s_acceptance(target, HAND)
    rows = target.read_confirmations()
    selected = next(row for row in rows if row["confirmation_id"] ==
                    originals["hero_seat"]["confirmation_id"])
    selected[field] = value
    rewrite_rows(target, rows)
    assert_not_ready(target)
    assert not validate_target_s_receipt(target, HAND, before)


def test_true_stale_parent_and_cross_fact_parent_are_refused(tmp_path):
    target = store(tmp_path)
    originals = confirm_all(target)
    first = originals["hero_seat"]
    confirm(target, "hero_seat", HERO, supersedes=first["confirmation_id"])
    for parent in (first["confirmation_id"],
                   originals["board_cards"]["confirmation_id"]):
        with pytest.raises(RevisionError, match="stale_supersedes"):
            confirm(target, "hero_seat", HERO, supersedes=parent)


def test_v1_store_is_preserved_but_cannot_accept_or_append(tmp_path):
    target = store(tmp_path)
    rows = confirm_all(target)
    legacy = dict(rows["hero_seat"], schema_version="aa-real-hand-confirmation-v1")
    rewrite_rows(target, [legacy])
    before = (target.path / module.CONFIRMATION_FILE).read_bytes()
    assert_not_ready(target)
    with pytest.raises(module.StoreIntegrityError, match="unknown_schema_version"):
        confirm(target, "hero_seat", HERO, supersedes=legacy["confirmation_id"])
    assert (target.path / module.CONFIRMATION_FILE).read_bytes() == before


def test_duplicate_json_keys_are_rejected_before_deserialization_loses_them(tmp_path):
    target = store(tmp_path)
    confirm_all(target)
    path = target.path / module.CONFIRMATION_FILE
    text = path.read_text(encoding="utf-8")
    text = text.replace('"hero_seat"', '"hero_seat","fact_key":"hero_seat"', 1)
    path.write_text(text, encoding="utf-8")
    assert_not_ready(target)
    with pytest.raises(module.StoreIntegrityError):
        target.read_confirmations()


# P0-4/5: complete statuses and the deliberately narrow threeway contract.
@pytest.mark.parametrize("overrides", [
    {"folded_seats": [0, 1, 2, 4]},
    {"folded_seats": list(range(6))},
    {"folded_seats": [0, 2]},
    {"folded_seats": [0, 2, 4, 7]},
    {"seat_set": [0, 1, 3, 5]},
    {"dealer_button_seat": 7},
])
def test_incomplete_or_contradictory_participant_states_block(tmp_path, overrides):
    target = store(tmp_path)
    confirm_all(target, overrides)
    assert_not_ready(target)


@pytest.mark.parametrize("seat", ACTIVE)
def test_zero_stack_active_is_impossible_even_when_delta_matches(tmp_path, seat):
    target = store(tmp_path)
    confirm_all(target, {"opening_stacks": {**OPENING, str(seat): 17},
                         "river_start_stacks": {**RIVER, str(seat): 0}})
    view = assert_not_ready(target)
    assert view["derivations"]["hand_committed"]["status"] == STATUS_CONFIRMED
    assert "every_active_seat_requires_positive_remaining_stack" in view["blocking"]


def test_later_fold_revision_invalidates_existing_ready_receipt(tmp_path):
    target = store(tmp_path)
    rows = confirm_all(target)
    receipt = target_s_acceptance(target, HAND)
    confirm(target, "folded_seats", [0, 1, 2, 4],
            supersedes=rows["folded_seats"]["confirmation_id"])
    view = assert_not_ready(target)
    assert view["confirmed"]["folded_seats"] == [0, 1, 2, 4]
    assert not validate_target_s_receipt(target, HAND, receipt)


@pytest.mark.parametrize("level", [4, 17])
def test_disjoint_short_or_equal_allin_is_always_out_of_scope(tmp_path, level):
    target = store(tmp_path)
    confirm_all(target, {
        "all_in_seats": [0], "folded_seats": [2, 4],
        "opening_stacks": {**OPENING, "0": level},
        "river_start_stacks": {**RIVER, "0": 0},
        "pot_display": {"raw": str(59 + level), "value": 59 + level},
    })
    view = assert_not_ready(target)
    assert view["derivations"]["hand_committed"]["status"] == STATUS_CONFIRMED
    assert "any_all_in_seat_is_outside_target_s_scope" in view["blocking"]


@pytest.mark.parametrize("count", [6, 7, 8])
def test_complete_six_to_eight_dealt_seats_positive_controls(tmp_path, count):
    target = store(tmp_path)
    confirm_all(target, {
        "seat_set": list(range(count)),
        "folded_seats": [seat for seat in range(count) if seat not in ACTIVE],
        "opening_stacks": {str(seat): 200 for seat in range(count)},
        "river_start_stacks": {str(seat): 183 if seat in ACTIVE else 196
                               for seat in range(count)},
        "pot_display": {"raw": str(4 * count + 39), "value": 4 * count + 39},
        "table_rules": dict(RULES, table_size=count),
    })
    receipt = target_s_acceptance(target, HAND)
    assert receipt["status"] == ACCEPTED
    assert validate_target_s_receipt(target, HAND, receipt)


# P0-6: contradictory raw monetary identities/assertions are never discarded.
@pytest.mark.parametrize("bad_key", ["01", " 1", "+1", "1.0", 1.9, True, None])
def test_noncanonical_monetary_seat_keys_reject_without_coercion(tmp_path, bad_key):
    target = store(tmp_path)
    value = {**OPENING, bad_key: 999}
    with pytest.raises(FactValueError, match="noncanonical_seat"):
        confirm(target, "opening_stacks", value)
    assert _derive(opening_stacks=value)["status"] == STATUS_UNCONFIRMED
    assert_not_ready(target)


def test_integer_and_string_duplicate_seat_is_not_last_wins(tmp_path):
    target = store(tmp_path)
    value = {**OPENING, 1: 999}
    with pytest.raises(FactValueError, match="duplicate_seat"):
        confirm(target, "opening_stacks", value)
    assert _derive(opening_stacks=value)["status"] == STATUS_UNCONFIRMED
    assert_not_ready(target)


@pytest.mark.parametrize("change", [
    "false", "missing", "unknown_false", "unknown_true"])
def test_insurance_and_exact_c4_assertion_schema(tmp_path, change):
    target = store(tmp_path)
    assertions = dict(ASSERTIONS)
    if change == "false":
        assertions["no_insurance"] = False
    elif change == "missing":
        del assertions["no_insurance"]
    else:
        assertions["unknown_movement"] = change == "unknown_true"
    with pytest.raises(FactValueError):
        confirm(target, "stack_delta_assertions", assertions)
    assert _derive(assertions=assertions)["status"] == STATUS_UNCONFIRMED
    assert_not_ready(target)


@pytest.mark.parametrize("key", module.STACK_DELTA_BINDINGS)
def test_interval_digest_missing_or_mismatch_blocks_derivation(tmp_path, key):
    target = store(tmp_path)
    assertions = dict(ASSERTIONS)
    assertions[key] = "f" * 64
    confirm_all(target, {"stack_delta_assertions": assertions})
    assert_not_ready(target)
    assert _derive(assertions=assertions)["status"] == STATUS_UNCONFIRMED


@pytest.mark.parametrize("rules", [
    dict(RULES, min_bet=99), dict(RULES, table_size=5),
    dict(RULES, table_size=9), dict(RULES, small_blind=0),
    dict(RULES, ante_mode="none"), dict(RULES, straddle_mode="none"),
])
def test_rules_do_not_allow_alternate_or_inconsistent_truths(tmp_path, rules):
    target = store(tmp_path)
    with pytest.raises(FactValueError):
        confirm(target, "table_rules", rules)
    assert_not_ready(target)


@pytest.mark.parametrize("kind", ["machine", "annotation", "manual", "review_note"])
def test_other_namespaces_cannot_supply_acceptance_store(kind):
    candidate = {"kind": kind, "status": "CONFIRMED", "value": FULL_FACTS,
                 "provenance": PROV_HUMAN}
    with pytest.raises(ConfirmationError, match="confirmation_store_required"):
        target_s_acceptance(candidate, HAND)
    assert not validate_target_s_receipt(candidate, HAND,
                                         {"schema": module.SCHEMA_VERSION})


def test_valid_permutation_with_impossible_clockwise_order_is_rejected(tmp_path):
    target = store(tmp_path)
    confirm_all(target, {"action_order": [1, 5, 3]})
    view = assert_not_ready(target)
    assert view["checks"]["action_order_matches_active"] is True
    assert view["checks"]["hero_acts_first"] is True
    assert "action_order_disagrees_with_dealer_and_clockwise_active_seats" in (
        view["blocking"])
    assert "history" not in view["derived"]


def test_utf8_corruption_is_typed_not_ready_and_never_rewritten(tmp_path):
    target = store(tmp_path)
    path = target.path / module.CONFIRMATION_FILE
    path.write_bytes(b"\xff")
    assert_not_ready(target)
    with pytest.raises(module.StoreIntegrityError, match="invalid_utf8"):
        target.read_confirmations()
    with pytest.raises(module.StoreIntegrityError, match="invalid_utf8"):
        confirm(target, "hero_seat", HERO)
    assert path.read_bytes() == b"\xff"
