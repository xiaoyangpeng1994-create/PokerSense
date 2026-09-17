from copy import deepcopy

import pytest

from poker_engine.desktop.aa_semantics import AAObservationSemantics


def row(frame, *, price="4", own="0", stack="100", street="preflop"):
    wagers = dict.fromkeys(map(str, range(8)), "0")
    wagers.update({"1": price, "3": own, "6": {"status": "NOT_APPLICABLE"}})
    return {
        "frame": frame, "source_frame": 1000 + frame, "source_id": "source-a",
        "source_sha256": "a" * 64, "scene_supported": True, "special_modes": {},
        "observed_state_v2": {"observed_epoch": "hand-a", "street_candidate": street,
                              "participants": {}, "pending_actions": 0,
                              "unallocated_positive_cash": []},
        "causal_street_wagers_v2": {
            "status": "OBSERVED_STREET_WAGERS_CANDIDATE", "wagers": wagers,
            "street_price": price, "title_center_ledger_reconciled": True},
        "stacks": {"3": {"value": stack}}, "pot": {"value": "43"},
        "observed_actions_v2": [], "current_actor": 3,
    }


def action(glyph="aggressive", amount="20", before="100", after="80"):
    return {"frame": 2, "confirmed_at": 2, "slot": 3, "glyph": glyph,
            "kind": glyph, "epoch": "hand-a", "street": "preflop",
            "amount": amount, "cash_evidence": {
                "kind": "balance_decrease", "seat": 3, "street": "preflop",
                "first_frame": 1, "confirmed_frame": 2, "amount": amount,
                "balance_before": before, "balance_after": after}}


def run_case(pre, current_action, mutate=None):
    subject = AAObservationSemantics()
    original = deepcopy(pre)
    subject.observe(pre)
    middle = row(1)
    if mutate:
        mutate(middle)
    subject.observe(middle)
    final = row(2, stack=current_action["cash_evidence"]["balance_after"])
    final["observed_actions_v2"] = [current_action]
    before = deepcopy(final)
    result = subject.observe(final)["interpreted_actions"][0]
    assert pre == original and final == before
    assert not result["legal_action_verified"] and not result["strategy_eligible"]
    return result


@pytest.mark.parametrize("price,own,before,debit,after,glyph,kind,target,short", [
    ("4", "0", "100", "20", "80", "aggressive", "raise", "20", False),
    ("0", "0", "100", "58", "42", "aggressive", "bet", "58", False),
    ("0", "0", "214", "214", "0", "all_in", "bet", "214", False),
    ("214", "0", "120", "120", "0", "all_in", "call", "120", True),
    ("20", "2", "100", "18", "82", "call", "call", "20", False),
    ("20", "0", "25", "25", "0", "all_in", "raise", "25", False),
])
def test_price_based_semantics_and_all_in_is_separate(
        price, own, before, debit, after, glyph, kind, target, short):
    result = run_case(row(0, price=price, own=own, stack=before),
                      action(glyph, debit, before, after))
    assert result["semantic_kind"] == kind
    assert result["target_total"] == target
    assert result["all_in"] == (after == "0")
    assert result["short_all_in_call"] == short
    assert result["predecision_evidence"]["source_frame"] == 1000
    assert result["source_confirmation_frame"] == 1002


@pytest.mark.parametrize("change", [
    lambda r: r["causal_street_wagers_v2"].update(status="WAGERS_UNKNOWN"),
    lambda r: r["causal_street_wagers_v2"]["wagers"].update({"7": None}),
    lambda r: r["observed_state_v2"].update(street_candidate=None),
    lambda r: r["observed_state_v2"].update(observed_epoch="other-hand"),
    lambda r: r["special_modes"].update(insurance="VISIBLE"),
    lambda r: r["stacks"]["3"].update(value="99"),
    lambda r: r["causal_street_wagers_v2"].update(street_price="3"),
])
def test_unknown_or_conflicting_pre_state_cannot_be_repaired_from_future(change):
    pre = row(0)
    change(pre)
    result = run_case(pre, action())
    assert result["semantic_kind"] is None
    assert result["interpretation_status"] == "UNKNOWN"


def test_source_change_cannot_bridge_predecision_prices():
    result = run_case(row(0), action(), lambda r: r.update(source_id="new-source"))
    assert result["semantic_kind"] is None


def test_intervening_overlay_blocks_cash_interpretation():
    result = run_case(row(0), action(),
                      lambda r: r["special_modes"].update(insurance="VISIBLE"))
    assert result["semantic_kind"] is None


def test_visible_check_is_retained_without_inventing_street_during_animation():
    subject = AAObservationSemantics()
    current = row(0, street=None)
    current["observed_actions_v2"] = [{
        "frame": 0, "confirmed_at": 0, "slot": 3, "epoch": "hand-a",
        "street": "turn", "kind": "check", "glyph": "check", "amount": "0"}]
    result = subject.observe(current)["interpreted_actions"][0]
    assert result["semantic_kind"] == "check"
    assert result["semantic_street"] is None
    assert not result["legal_action_verified"]


@pytest.mark.parametrize("act", [
    action("all_in", "20", "100", "80"),
    action("call", "20", "100", "80"),
    action("aggressive", "4", "100", "96"),
    action("aggressive", "20", "100", "79"),
])
def test_cash_and_visible_label_conflicts_block(act):
    assert run_case(row(0), act)["semantic_kind"] is None


def terminal_row(frame, *, pot="623", credit=False, clear=False, epoch="h"):
    r = row(frame, street="river")
    r["current_actor"] = None
    r["pot"] = {"value": pot}
    r["cards"] = {"hero": None if clear else ["5d", "6d"],
                  "board_slots": [None]*5 if clear else ["5h", "6c", "6s", "Tc", "3d"]}
    r["hand_transition"] = {"center_deal": {"visible": None}}
    r["observed_state_v2"].update(
        observed_epoch=epoch,
        participants={"1": {"epoch": epoch, "state": "all_in"},
                      "4": {"epoch": epoch, "state": "all_in"},
                      "5": {"epoch": epoch, "state": "folded"}},
        unallocated_positive_cash=[{"epoch": epoch, "seat": 4, "amount": "516",
                                    "confirmed_frame": 1}] if credit else [])
    r["hand_ledger_v2"] = {"epoch": epoch, "observed_total": "629",
                           "displayed_pot": pot,
                           "unallocated_difference": "6" if pot == "623" else "629",
                           "opening_evidence": {
                               "debits": {"1": "2", "4": "2", "5": "2"}}}
    return r


def test_settlement_keeps_historical_difference_and_suppresses_current629():
    subject = AAObservationSemantics()
    subject.observe(terminal_row(0))
    assert subject.observe(terminal_row(1, credit=True))["hand_phase"]["phase"] == (
        "SETTLEMENT_CANDIDATE")
    pending = subject.observe(terminal_row(
        2, pot="0", credit=True, clear=True))["hand_phase"]
    assert pending["phase"] == "POT_CLEAR_PENDING" and pending["current_ledger"] is None
    result = subject.observe(terminal_row(
        3, pot="0", credit=True, clear=True))["hand_phase"]
    assert result["phase"] == "WAITING_NEXT_HAND_CANDIDATE"
    assert result["historical_ledger"]["observed_total"] == "629"
    assert result["historical_ledger"]["unallocated_difference"] == "6"
    assert result["visible_credits"][0]["amount"] == "516"
    assert not result["canonical_verified"] and not result["settlement_rules_verified"]
    fresh = subject.observe(terminal_row(4, epoch="new"))["hand_phase"]
    assert fresh["phase"] == "OBSERVING"


@pytest.mark.parametrize("mutation", [
    lambda r: r["observed_state_v2"]["participants"]["4"].update(state="active"),
    lambda r: r["cards"].update(board_slots=[None]*5),
    lambda r: r["special_modes"].update(insurance="VISIBLE"),
    lambda r: r["observed_state_v2"]["participants"].pop("5"),
])
def test_zero_pot_and_credit_without_terminal_evidence_do_not_mean_settled(mutation):
    subject = AAObservationSemantics()
    initial = terminal_row(0)
    mutation(initial)
    subject.observe(initial)
    for frame in (1, 2, 3):
        result = subject.observe(terminal_row(frame, pot="0", credit=True, clear=True))
    assert result["hand_phase"]["phase"] == "OBSERVING"
    assert result["hand_phase"]["current_ledger"] is not None


def test_discontinuity_drops_settlement_evidence():
    subject = AAObservationSemantics()
    subject.observe(terminal_row(0))
    subject.observe(terminal_row(1, credit=True))
    result = subject.observe(terminal_row(5, pot="0", credit=True, clear=True))
    assert result["hand_phase"]["phase"] == "OBSERVING"


def test_overlay_cannot_reuse_earlier_terminal_proof():
    subject = AAObservationSemantics()
    subject.observe(terminal_row(0))
    modal = terminal_row(1, credit=True)
    modal["special_modes"] = {"insurance": "VISIBLE"}
    assert subject.observe(modal)["hand_phase"]["phase"] == "SUSPENDED"
    for frame in (2, 3):
        result = subject.observe(terminal_row(frame, pot="0", credit=True, clear=True))
    assert result["hand_phase"]["phase"] == "OBSERVING"


def test_renewed_action_invalidates_terminal_proof():
    subject = AAObservationSemantics()
    subject.observe(terminal_row(0))
    unexpected = terminal_row(1, pot="0", credit=True, clear=True)
    unexpected["current_actor"] = 4
    result = subject.observe(unexpected)["hand_phase"]
    assert result["phase"] == "SUSPENDED"
    assert result["current_ledger"] is None
