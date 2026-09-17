"""PHH shadow export: the fail-closed contract, and a working oracle run.

The happy path proves PokerKit can be driven end to end from a PokerSense-shaped
facts document. Everything else proves the refusal: a PHH required field that no
confirmed evidence supports must NOT be written as a zero, a False or a
placeholder, because PokerKit (and any consumer downstream of it) would then
read a fabricated value as a fact.
"""

from poker_engine.desktop.aa_phh_shadow import (
    FIELD_MAP,
    PHH_REQUIRED,
    POKERKIT_NT_REQUIRED,
    shadow_export,
    shadow_ingest,
    shadow_round_trip,
)


def declared_rules(**overrides):
    rules = {"small_blind": "50", "big_blind": "100", "ante": "0",
             "straddle_amount": None, "ante_mode": "none",
             "straddle_mode": "none"}
    rules.update(overrides)
    return rules


def action(slot, text, status="CONFIRMED"):
    return {"slot": slot, "kind": "action", "phh": text,
            "interpretation_status": status}


def confirmed_hand(**overrides):
    """A three-handed no-limit hand whose every PHH field is confirmed.

    ``seat_order`` maps physical seats onto PHH player numbers, and the
    expected replay is fixed: seat 4 posts the small blind out of 1000, seat 7
    posts the big blind, seat 1 raises, seat 4 folds, seat 7 calls, and seat 1
    is pushed the 650 pot -- a settlement-time balance delta of +650, which is
    what PokerSense records as a credit. That is the number the oracle must
    reproduce on its own.
    """
    facts = {
        "seats": 3,
        "seat_order": ["4", "7", "1"],
        "seat_order_confirmed": True,
        "table_rules": declared_rules(),
        "starting_stacks": {"4": "1000", "7": "1000", "1": "1000"},
        "actions": [
            action("4", "d dh p1 AsKs"),
            action("7", "d dh p2 7d7h"),
            action("1", "d dh p3 QcJd"),
            action("1", "p3 cbr 300"),
            action("4", "p1 f"),
            action("7", "p2 cc"),
            action("7", "d db Ah2c9s"),
            action("7", "p2 cc"),
            action("1", "p3 cc"),
            action("7", "d db Td"),
            action("7", "p2 cc"),
            action("1", "p3 cc"),
            action("7", "d db 3h"),
            action("7", "p2 cc"),
            action("1", "p3 cc"),
            action("7", "p2 sm QcJd"),
            action("1", "p3 sm AsKs"),
        ],
        "settlement": [{"seat": "1", "amount": "650", "confirmed_frame": 900}],
    }
    facts.update(overrides)
    return facts


def test_a_full_confirmed_hand_exports_and_replays_through_poker_kit():
    report = shadow_export(confirmed_hand())
    assert report["status"] == "EXPORTED"
    assert report["missing_required"] == []
    assert report["phh"].startswith("variant = 'NT'")
    assert report["poker_kit"]["phh_text_identity"] is True
    assert report["poker_kit"]["all_actions_accepted"] is True
    assert report["poker_kit"]["hand_finished"] is True


def test_the_oracle_reproduces_the_payout_the_table_actually_paid():
    """The differential is the point: an independent engine must agree."""
    report = shadow_export(confirmed_hand())
    checks = {check["check"]: check for check in report["differential"]}
    payout = checks["payout_per_seat"]
    assert payout["status"] == "MATCH"
    # The chips pull (+650), not the net payoff (+350): the credit PokerSense
    # records is a settlement-time balance delta, and the two differ whenever
    # the credited seat also contributed to the pot.
    assert payout["poker_kit"] == {"1": 650}
    assert payout["poker_sense"] == {"1": 650}
    assert payout["poker_kit_net_payoffs"] == {"1": 350, "4": -50, "7": -300}
    assert checks["pot_commitment_conservation"]["status"] == "MATCH"
    assert checks["pot_commitment_conservation"]["poker_kit"] == {
        "chips_in": 3000, "chips_out": 3000, "difference": 0}


def test_an_unconfirmed_ante_is_never_written_as_a_zero_ante():
    """``antes = [0, 0, 0]`` is a FACT, not a default. Absent must refuse."""
    facts = confirmed_hand()
    facts["table_rules"] = declared_rules(ante=None, ante_mode="unknown")
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert report["reason"] == "unconfirmed_required_fields"
    assert "antes" in report["missing_required"]
    assert "phh" not in report


def test_an_undeclared_blind_blocks_the_export():
    facts = confirmed_hand()
    facts["table_rules"] = declared_rules(big_blind=None)
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert set(report["missing_required"]) >= {"blinds_or_straddles", "min_bet"}


def test_a_per_hand_straddle_is_not_written_from_the_table_rules():
    """A declared straddle AMOUNT is not evidence that a straddle was posted.

    Under ``optional_explicit_utg`` the table rules pin the amount but the
    posting itself is a per-hand observation. Defaulting the straddle seat to 0
    would restate a straddled hand as an unstraddled one, so the field must
    stay unconfirmed.
    """
    facts = confirmed_hand()
    facts["table_rules"] = declared_rules(straddle_mode="optional_explicit_utg",
                                          straddle_amount="200")
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert "blinds_or_straddles" in report["missing_required"]
    assert "phh" not in report


def test_a_mandatory_straddle_is_written_positionally():
    """``mandatory_utg`` IS a declaration, so it belongs in the PHH.

    Only the mapping is asserted: a straddle changes who opens preflop, so this
    fixture's action order is not a legal straddled sequence. That is a PokerKit
    legality question rather than a mapping question, and it is not what this
    test is about.
    """
    facts = confirmed_hand()
    facts["table_rules"] = declared_rules(straddle_mode="mandatory_utg",
                                          straddle_amount="200")
    report = shadow_export(facts)
    blind = {field["field"]: field for field in report["fields"]}[
        "blinds_or_straddles"]
    assert blind["status"] == "CONFIRMED"
    assert blind["value"] == [50, 100, 200]
    assert "blinds_or_straddles = [50, 100, 200]" in report["phh"]


def test_an_unconfirmed_positional_order_blocks_the_export():
    """PHH is indexed by player number; an unconfirmed button moves chips."""
    facts = confirmed_hand(seat_order_confirmed=False)
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert "seat_order" in report["missing_required"]
    assert "phh" not in report


def test_one_missing_opening_stack_blocks_the_export():
    facts = confirmed_hand(starting_stacks={"4": "1000", "7": "1000"})
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert "starting_stacks" in report["missing_required"]


def test_a_candidate_action_interpretation_is_never_promoted_to_a_fact():
    """The live pipeline emits PRICE_DERIVED_CANDIDATE, not confirmed actions."""
    facts = confirmed_hand()
    facts["actions"] = [
        action("4", "d dh p1 AsKs"),
        action("7", "p2 cc", status="PRICE_DERIVED_CANDIDATE"),
    ]
    report = shadow_export(facts)
    assert report["status"] == "NOT_EXPORTABLE"
    assert "actions" in report["missing_required"]
    assert report["rejected_action_interpretations"] == [
        {"seat": "7", "kind": "action", "reason": "PRICE_DERIVED_CANDIDATE"}]
    assert "phh" not in report


def test_the_shadow_never_claims_to_replace_the_production_state_machine():
    report = shadow_export(confirmed_hand())
    assert report["replaces_production_state_machine"] is False


def test_an_unmakeable_comparison_is_reported_as_not_comparable():
    """Silence is not agreement: a check that cannot run must say so."""
    report = shadow_export(confirmed_hand())
    checks = {check["check"]: check for check in report["differential"]}
    forced = checks["forced_bets"]
    assert forced["status"] == "NOT_COMPARABLE"
    assert "declared table rules" in forced["reason"]


def test_a_wrong_positional_order_moves_the_payout_and_is_caught():
    """The differential is not vacuous: break the order and it must MISMATCH."""
    facts = confirmed_hand(seat_order=["1", "4", "7"])
    report = shadow_export(facts)
    assert report["status"] == "EXPORTED"
    payout = {c["check"]: c for c in report["differential"]}["payout_per_seat"]
    assert payout["status"] == "MISMATCH"
    assert payout["poker_kit"] == {"7": 650}
    assert payout["poker_sense"] == {"1": 650}


def test_the_field_map_covers_every_required_phh_field():
    mapped = {row[0] for row in FIELD_MAP}
    assert set(PHH_REQUIRED) <= mapped
    for _, evidence, rule in FIELD_MAP:
        assert evidence and rule


# --- the gate is anchored to the installed library, not to our belief ------


def test_our_gate_matches_the_fields_pokerkit_actually_requires():
    """``POKERKIT_NT_REQUIRED`` is checked against PokerKit itself.

    If a future PokerKit adds a required field, this fails instead of our gate
    quietly gating on a stale list. It also pins the one field that is NOT
    PokerKit's: ``seat_order`` is a PokerSense precondition we impose on top.
    """
    from pokerkit import HandHistory

    required = set(HandHistory.required_field_names["NT"])
    assert set(POKERKIT_NT_REQUIRED) == required - {"variant"}
    assert "seat_order" not in required


# --- the PHH has to carry the physical seats -------------------------------


def test_the_phh_carries_the_physical_seats():
    """Without the optional ``seats`` header a PHH is only positionally indexed.

    ``p2`` would then be unmappable back onto a seat number, so the reverse
    direction could never restore ``seat_order``.
    """
    report = shadow_export(confirmed_hand())
    assert report["status"] == "EXPORTED"
    assert report["poker_kit"]["physical_seats_carried"] == ["4", "7", "1"]
    assert "seats = ['4', '7', '1']" in report["phh"]


def test_a_phh_without_seats_cannot_restore_the_positional_order():
    """The reverse adapter says so rather than inventing seat numbers."""
    document = shadow_export(confirmed_hand())["phh"]
    stripped = "\n".join(line for line in document.splitlines()
                         if not line.startswith("seats ="))
    assert "seats =" not in stripped
    reverse = shadow_ingest(stripped)
    assert reverse["status"] == "INGESTED"
    assert reverse["facts"]["seat_order"] is None
    assert reverse["facts"]["seat_order_confirmed"] is False


# --- the side-pot split is read before the chips move ---------------------


def side_pot_hand(**overrides):
    """A VALID side-pot hand: 1000 / 300 / 1000, the short stack jams.

    ``p3`` raises to 300, ``p1`` re-raises to 600, ``p2`` calls all-in for its
    last 300, ``p3`` calls. That is a main pot of 900 (three seats) and a side
    pot of 600 (``p1`` and ``p3`` only). ``p1`` wins both: it is pushed 1500,
    which is the settlement-time balance delta this hand records.
    """
    facts = confirmed_hand()
    facts["starting_stacks"] = {"4": "1000", "7": "300", "1": "1000"}
    facts["actions"] = [
        action("4", "d dh p1 AsKs"),
        action("7", "d dh p2 7d7h"),
        action("1", "d dh p3 QcJd"),
        action("1", "p3 cbr 300"),
        action("4", "p1 cbr 600"),
        action("7", "p2 cc"),
        action("1", "p3 cc"),
        action("4", "d db Ah2c9s"),
        action("4", "p1 cc"),
        action("1", "p3 cc"),
        action("4", "d db Td"),
        action("4", "p1 cc"),
        action("1", "p3 cc"),
        action("4", "d db 3h"),
        action("4", "p1 cc"),
        action("1", "p3 cc"),
        action("4", "p1 sm AsKs"),
        action("7", "p2 sm 7d7h"),
        action("1", "p3 sm QcJd"),
    ]
    facts["settlement"] = [{"seat": "4", "amount": "1500",
                            "confirmed_frame": 900}]
    facts.update(overrides)
    return facts


def test_a_side_pot_is_split_and_the_split_is_checked():
    report = shadow_export(side_pot_hand())
    assert report["status"] == "EXPORTED"
    checks = {c["check"]: c for c in report["differential"]}
    split = checks["side_pot_split"]
    assert split["status"] == "MATCH"
    assert split["problems"] == []
    # The split has to be read BEFORE the chips move: the finished state has an
    # empty pot, and an empty pot would make this check a tautology.
    assert split["poker_kit"]["total"] == 1500
    assert split["poker_kit"]["pots"] == [
        {"amount": 900, "eligible_seats": ["4", "7", "1"]},
        {"amount": 600, "eligible_seats": ["4", "1"]},
    ]
    assert split["poker_kit"]["contributed_per_seat"] == {
        "4": 600, "7": 300, "1": 600}
    assert checks["payout_per_seat"]["status"] == "MATCH"
    assert checks["pot_commitment_conservation"]["status"] == "MATCH"


def test_the_side_pot_item_never_pretends_to_be_a_two_engine_comparison():
    """PokerSense records no pot levels, so this cannot be a PokerSense check."""
    split = {c["check"]: c for c in
             shadow_export(side_pot_hand())["differential"]}["side_pot_split"]
    assert split["comparable_against_poker_sense"] is False
    assert "no pot levels" in split["poker_sense"]["evidence"]


def test_a_plain_single_pot_hand_is_not_reported_as_a_side_pot_mismatch():
    """Regression: the first version of this check failed on its own fixture.

    It recovered the pot level as ``amount / eligible_count``, which is not what
    a level is, and reported ``MISMATCH`` on this hand -- the small blind folded
    for 50 while two seats paid 300, so dividing 650 by two gave a level of 325
    that nobody had reached. A check that fails a correct hand reports the
    failure as loudly as it reports the truth, so it is pinned here.
    """
    split = {c["check"]: c for c in
             shadow_export(confirmed_hand())["differential"]}["side_pot_split"]
    assert split["status"] == "MATCH"
    assert split["problems"] == []
    assert split["poker_kit"]["pots"] == [
        {"amount": 650, "eligible_seats": ["7", "1"]}]
    assert split["poker_kit"]["contributed_per_seat"] == {
        "4": 50, "7": 300, "1": 300}
    assert split["poker_kit"]["folded_seats"] == ["4"]


def call_then_fold_hand():
    """A seat calls the full 300 and THEN folds on the flop.

    Its 300 is in the pot but it is not eligible for any of it, so the pot's
    eligible set is two seats while three seats paid. This is the case that
    distinguishes "paid into the pot" from "eligible for the pot".
    """
    facts = confirmed_hand()
    facts["actions"] = [
        action("4", "d dh p1 AsKs"),
        action("7", "d dh p2 7d7h"),
        action("1", "d dh p3 QcJd"),
        action("1", "p3 cbr 300"),
        action("4", "p1 cc"),
        action("7", "p2 cc"),
        action("4", "d db Ah2c9s"),
        action("4", "p1 f"),
        action("7", "p2 cc"),
        action("1", "p3 cc"),
        action("4", "d db Td"),
        action("7", "p2 cc"),
        action("1", "p3 cc"),
        action("4", "d db 3h"),
        action("7", "p2 cc"),
        action("1", "p3 cc"),
        action("7", "p2 sm 7d7h"),
        action("1", "p3 sm QcJd"),
    ]
    facts["settlement"] = [{"seat": "7", "amount": "900",
                            "confirmed_frame": 900}]
    return facts


def test_a_seat_that_calls_then_folds_paid_in_but_is_not_eligible():
    report = shadow_export(call_then_fold_hand())
    assert report["status"] == "EXPORTED"
    split = {c["check"]: c for c in report["differential"]}["side_pot_split"]
    assert split["status"] == "MATCH"
    assert split["poker_kit"]["contributed_per_seat"] == {
        "4": 300, "7": 300, "1": 300}
    assert split["poker_kit"]["folded_seats"] == ["4"]
    # All three paid 300, but only the two live seats are eligible.
    assert split["poker_kit"]["pots"] == [
        {"amount": 900, "eligible_seats": ["7", "1"]}]


# --- reverse direction ----------------------------------------------------


def test_the_round_trip_restores_every_field_it_sent():
    result = shadow_round_trip(confirmed_hand())
    assert result["status"] == "ROUND_TRIP_MATCH"
    assert result["mismatched_fields"] == []
    assert {row["field"] for row in result["checks"]} == {
        "variant", "antes", "blinds_or_straddles", "min_bet",
        "starting_stacks", "actions", "seats"}


def test_the_round_trip_refuses_when_the_forward_export_refuses():
    result = shadow_round_trip(confirmed_hand(actions=[]))
    assert result["status"] == "ROUND_TRIP_NOT_RUN"
    assert result["forward_status"] == "NOT_EXPORTABLE"
    assert "phh" not in result


def test_the_reverse_adapter_can_never_satisfy_the_forward_gate():
    """A PHH we wrote must not become evidence that the gate accepts.

    The reconstructed actions carry ``PHH_ROUND_TRIP``, not ``CONFIRMED``, so
    re-exporting the ingested facts refuses. Otherwise the reverse direction
    would be a way to launder unconfirmed facts past the gate.
    """
    reverse = shadow_ingest(shadow_export(confirmed_hand())["phh"])
    assert reverse["status"] == "INGESTED"
    statuses = {item["interpretation_status"]
                for item in reverse["facts"]["actions"]}
    assert statuses == {"PHH_ROUND_TRIP"}
    again = shadow_export(reverse["facts"])
    assert again["status"] == "NOT_EXPORTABLE"
    assert "actions" in again["missing_required"]
    assert "phh" not in again


def test_an_illegal_action_list_is_a_named_refusal_not_a_traceback():
    """An illegal sequence must surface as ``POKERKIT_REJECTED``.

    PokerKit raises ``ValueError: Unable to repair the hand history`` here. The
    construction therefore has to sit inside the guard, or the failure escapes
    as a traceback and the action-legality item never reports anything.
    """
    facts = confirmed_hand()
    facts["actions"] = [
        action("4", "d dh p1 AsKs"),
        action("7", "d dh p2 7d7h"),
        action("1", "d dh p3 QcJd"),
        action("1", "p3 cbr 300"),
        action("4", "p1 cbr 100"),  # a raise DOWN: illegal after 300
    ]
    report = shadow_export(facts)
    assert report["status"] == "POKERKIT_REJECTED"
    assert "ValueError" in report["reason"]
