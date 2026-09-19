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
)

HAND = "observed_deal_synthetic_0001"
EPOCH = "epoch-synthetic-0001"
SESSION = "capture-synthetic-0001"
SOURCE = {"media_sha256": "a" * 64, "recording_id": "aa-live-controlled-0001"}
OTHER_SOURCE = {"media_sha256": "b" * 64,
                "recording_id": "aa-live-controlled-0002"}

HERO = 1
ACTIVE = [1, 3, 5]
FOLDED = [0]
SEAT_SET = [0, 1, 3, 5]
BOARD = ["Ks", "Qh", "7h", "9h", "2c"]
HERO_CARDS = ["6d", "Kc"]
OPENING = {"0": 200, "1": 200, "3": 200, "5": 200}
RIVER = {"0": 196, "1": 183, "3": 183, "5": 183}
POT = {"raw": "55", "value": 55}
RULES = {
    "table_size": 6, "small_blind": 1, "big_blind": 2, "ante": 2,
    "ante_mode": "per_dealt_player", "straddle_amount": 4,
    "straddle_mode": "optional_explicit_utg", "revision": "rules-synthetic-1",
}
ASSERTIONS = {key: True for key in STACK_DELTA_ASSERTIONS}


def sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def store(tmp_path, name="store"):
    return ConfirmationStore(Path(tmp_path) / name)


def confirm(target, fact_key, value, *, hand_id=HAND, epoch=EPOCH,
            session=SESSION, source=SOURCE, marker=MARKER_RIVER_START,
            frame=15033, evidence=None, reviewer="reviewer-a",
            supersedes=None, audit_note=None):
    return confirm_fact(
        target, hand_id=hand_id, observed_epoch=epoch,
        capture_session_id=session, source_ref=source, marker=marker,
        source_frame=frame,
        evidence_digest=evidence or sha(f"{hand_id}:{fact_key}"),
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
    written["folded_seats"] = confirm(target, "folded_seats", list(FOLDED))
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
    assert view["conflicts"][0]["reason"] == "multiple_live_heads"
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
    confirm_all(target, {"river_start_stacks": {"0": 196, "1": 183, "3": 180,
                                                "5": 183},
                         "pot_display": {"raw": "58", "value": 58}})
    view = confirmed_view(target, HAND)
    assert any("side_pot" in item for item in view["blocking"])
    assert target_s_acceptance(target, HAND)["status"] == NOT_READY


# --- 16-24: the stack-delta commitment derivation, condition by condition ---

def _derive(**overrides):
    payload = {"opening_stacks": dict(OPENING), "river_start_stacks": dict(RIVER),
               "assertions": dict(ASSERTIONS), "pot_display": dict(POT),
               "seats": list(SEAT_SET)}
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
    result = _derive(river_start_stacks={"0": 196, "1": 210, "3": 183,
                                         "5": 183})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("stack_delta_negative" in item for item in result["reasons"])


def test_pot_mismatch_is_unconfirmed():
    result = _derive(pot_display={"raw": "54", "value": 54})
    assert result["status"] == STATUS_UNCONFIRMED
    assert any("stack_delta_pot_mismatch" in item for item in result["reasons"])


def test_pot_agrees_so_the_derivation_is_confirmed():
    result = _derive()
    assert result["status"] == STATUS_CONFIRMED
    assert result["per_seat"] == {"0": 4, "1": 17, "3": 17, "5": 17}
    assert result["total"] == 55
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
    assert view["derived"]["hand_committed"] == {"0": 4, "1": 17, "3": 17,
                                                 "5": 17}

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
