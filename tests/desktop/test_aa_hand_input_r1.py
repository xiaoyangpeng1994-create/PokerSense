"""U1-R1 regressions: no guessing on import, legal Hero history, row amounts.

The snapshot fixture here was corrected in U1-R2: it used a made-up ledger status
(`HAND_COMMITMENTS_OBSERVED_CANDIDATE`) and carried no river-start evidence, so it
passed for the wrong reason. It now uses the real token and the real evidence
fields, and the "no start evidence" variants are negatives in
`test_aa_hand_input_r2.py`.
"""

import json

import pytest

from poker_engine.desktop import aa_hand_input as module
from poker_engine.strategy.threeway_river_v1 import analyze_threeway_river
from tools.analyze_threeway_river import scenario_from_dict


RULES = {**json.load(open(
    "configs/strategy/examples/threeway-river-response-manual.json",
    encoding="utf-8"))["rules"], "rake_percent": "0", "rake_cap_bb": "0"}
EPOCH = "20260916T101530-aaaaaaaaaaaa"


def seats_block(rows=None):
    return rows or [
        {"seat_id": 0, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 1, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 2, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 3, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 4, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 5, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    ]


def facts(**overrides):
    base = module.blank_facts()
    base["ended_hand_confirmed"] = module.field(True, "human_confirmed")
    base["hero_seat"] = module.field(0, "human_confirmed")
    base["hero_cards"] = module.field(["Qs", "Qd"], "human_confirmed")
    base["board_cards"] = module.field(["2c", "4d", "7h", "9s", "Jc"],
                                       "human_confirmed")
    base["action_order"] = module.field([1, 2, 0], "human_confirmed")
    base["seats"] = module.field(seats_block(), "human_confirmed")
    base["history"] = module.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0"}], "human_confirmed")
    base["pot_display"] = module.field("130", "human_confirmed")
    base["table_rules"] = module.field(dict(RULES), "human_confirmed")
    base.update(overrides)
    return base


def assumptions(**overrides):
    base = {
        "range_source": "manual_unvalidated",
        "ranges": [
            {"seat_id": 1, "combos": [{"combo": "JhJd", "weight": "1"},
                                      {"combo": "TcTd", "weight": "3"}]},
            {"seat_id": 2, "combos": [{"combo": "7c7s", "weight": "1"},
                                      {"combo": "KhTh", "weight": "3"}]}],
        "models": [
            {"seat_id": 1, "key": "check", "weight": "1"},
            {"seat_id": 1, "key": "bet", "weight": "2"},
            {"seat_id": 1, "key": "call", "weight": "9"},
            {"seat_id": 1, "key": "fold", "weight": "1"},
            {"seat_id": 2, "key": "check", "weight": "1"},
            {"seat_id": 2, "key": "call", "weight": "9"},
            {"seat_id": 2, "key": "fold", "weight": "1"}],
        "aggression_targets": ["20", "40", "80"], "max_aggressions": 2,
        "other_fees": {"value": "0", "provenance": "assumed"},
    }
    base.update(overrides)
    return base


def snapshot(**overrides):
    payload = {
        "frame": 1500,
        "cards": {"hero": ["5d", "6d"],
                  "board_slots": ["5h", "6c", "6s", "Tc", "3d"]},
        "pot": {"value": "90"},
        "hand_ledger_v2": {
            "frame": 1500, "epoch": EPOCH,
            "status": "OBSERVED_HAND_COMMITMENTS_CANDIDATE",
            "hand_commitments": {"0": "20", "1": "20", "2": "20",
                                 "3": "10", "4": "10", "5": "10"},
            "observed_total": "90", "displayed_pot": "90",
            "unallocated_difference": "0",
            "opening_evidence": {"frame": 20, "first_frame": 18,
                                 "debits": {"0": "20", "1": "20", "2": "20",
                                            "3": "10", "4": "10", "5": "10"},
                                 "excluded_na_slots": [],
                                 "authoritative_boundary": True},
            "applied_action_count": 4, "taint_reasons": [],
            "complete_and_canonical_verified": False, "strategy_eligible": False},
        "causal_street_wagers_v2": {
            "status": "OBSERVED_STREET_WAGERS_CANDIDATE",
            "title_center_ledger_reconciled": True,
            "wagers": {str(seat): "0" for seat in range(8)},
            "street_price": "0"},
        "observed_state_v2": {
            "observed_epoch": EPOCH, "street_candidate": "river",
            "pending_actions": 0,
            "participants": {
                seat: {"state": state, "epoch": EPOCH} for seat, state in
                ((0, "active"), (1, "active"), (2, "active"),
                 (3, "folded"), (4, "folded"), (5, "folded"))}},
        "stacks": {str(seat): {"value": "200"} for seat in range(6)},
        "current_actor": 0,
    }
    payload.update(overrides)
    return payload


def test_snapshot_never_guesses_hero_action_order_or_seat_status():
    """R1-B: an absent hero_seat/action_order stays unknown, not 'observed'."""
    loose = snapshot()
    loose.pop("current_actor")
    loose["hand_ledger_v2"] = {"status": "HAND_COMMITMENTS_UNKNOWN",
                               "hand_commitments": None}
    from_snapshot, gaps = module.facts_from_snapshot(loose, source="rec-1")
    assert from_snapshot["hero_seat"]["provenance"] == "unknown"
    assert from_snapshot["hero_seat"]["value"] is None
    assert from_snapshot["action_order"]["provenance"] == "unknown"
    assert from_snapshot["action_order"]["value"] is None
    assert from_snapshot["seats"]["provenance"] == "unknown"
    joined = " ".join(gaps)
    assert "Hero" in joined and "行动顺序" in joined
    assert "各座" in joined


def test_snapshot_preserves_folded_and_ambiguous_participant_states():
    """Legacy raw state remains a candidate, never qualified current facts."""
    clear, gaps = module.facts_from_snapshot(snapshot(), source="rec-1")
    assert clear["seats"]["value"] is None
    assert clear["seats"]["provenance"] == "unknown"
    states = clear["seats"]["candidate"]["participants"]
    assert states[3]["state"] == "folded"
    assert states[0]["state"] == "active"
    assert clear["hero_seat"]["provenance"] == "unknown"
    assert clear["action_order"]["provenance"] == "unknown"

    no_evidence = snapshot()
    no_evidence.pop("causal_street_wagers_v2")
    withheld, withheld_gaps = module.facts_from_snapshot(no_evidence,
                                                         source="rec-1b")
    assert withheld["seats"]["provenance"] == "unknown"
    assert any("起点" in gap for gap in withheld_gaps), withheld_gaps

    ambiguous = snapshot()
    ambiguous["observed_state_v2"] = {**ambiguous["observed_state_v2"],
                                      "participants": {
        "0": {"state": "active"}, "1": {"state": "unknown"},
        "2": {"state": "active"}, "3": {"state": "folded"},
        "4": {"state": "folded"}, "5": {"state": "folded"}}}
    fuzzy, fuzzy_gaps = module.facts_from_snapshot(ambiguous, source="rec-2")
    assert fuzzy["seats"]["provenance"] == "unknown"
    assert any("关键字段" in gap for gap in fuzzy_gaps)


def test_unknown_history_is_not_the_same_as_no_history():
    """R1-B: null (unknown) must be refused; [] (confirmed none) is allowed."""
    unknown_history = facts(history=module.field(None, "unknown"))
    with pytest.raises(module.HandInputError, match="公开历史"):
        module.check_support(unknown_history, assumptions())
    empty = facts(history=module.field([], "human_confirmed"),
                  action_order=module.field([0, 1, 2], "human_confirmed"),
                  pot_display=module.field("90", "human_confirmed"))
    checked = module.check_support(empty, assumptions())
    assert checked["reasons"] == []
    document = module.build_document(empty, assumptions())[0]
    result = analyze_threeway_river(scenario_from_dict(document),
                                    max_joint_assignments=128, max_nodes=20000)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons


def test_other_fees_must_be_confirmed_instead_of_defaulted_to_zero():
    """R1-B: an unknown fee state cannot silently become a zero fee."""
    for block in ({"value": None, "provenance": "unknown"},
                  {"value": "0", "provenance": "unknown"}):
        with pytest.raises(module.HandInputError, match="额外费用"):
            module.check_support(facts(), assumptions(other_fees=block))
    confirmed = module.check_support(facts(), assumptions(
        other_fees={"value": "0", "provenance": "human_confirmed"}))
    assert confirmed["reasons"] == []
    assumed = module.check_support(facts(), assumptions(
        other_fees={"value": "0", "provenance": "assumed"}))
    assert assumed["reasons"] == []


def test_duplicate_combo_or_weight_rows_are_refused():
    """R1-B: duplicates must not be silently overwritten in a dict."""
    with pytest.raises(module.HandInputError, match="重复"):
        module.check_support(facts(), assumptions(ranges=[
            {"seat_id": 1, "combos": [{"combo": "JhJd", "weight": "1"},
                                      {"combo": "JhJd", "weight": "2"}]},
            {"seat_id": 2, "combos": [{"combo": "7c7s", "weight": "1"}]}]))
    with pytest.raises(module.HandInputError, match="重复"):
        module.check_support(facts(), assumptions(models=[
            {"seat_id": 1, "key": "call", "weight": "1"},
            {"seat_id": 1, "key": "call", "weight": "2"},
            {"seat_id": 2, "key": "call", "weight": "1"}]))
    with pytest.raises(module.HandInputError, match="重复"):
        module.check_support(facts(), assumptions(ranges=[
            {"seat_id": 1, "combos": [{"combo": "JhJd", "weight": "1"}]},
            {"seat_id": 1, "combos": [{"combo": "TcTd", "weight": "1"}]},
            {"seat_id": 2, "combos": [{"combo": "7c7s", "weight": "1"}]}]))


def test_equal_amounts_are_compared_by_value_not_by_text():
    """R1-B: '20' and '20.0' are the same contribution, not a side pot."""
    rows = seats_block()
    rows[1] = {**rows[1], "hand_committed": "20.0"}
    checked = module.check_support(facts(seats=module.field(rows,
                                                            "human_confirmed")),
                                   assumptions())
    assert checked["reasons"] == []


def test_hero_check_then_facing_a_bet_is_supported_and_computes():
    """R1-C: a legal Hero check must not be read as an opponent response."""
    hero_history = facts(
        action_order=module.field([0, 1, 2], "human_confirmed"),
        history=module.field([
            {"actor": 0, "kind": "check", "target": "0"},
            {"actor": 1, "kind": "bet", "target": "20"},
            {"actor": 2, "kind": "call", "target": "0"}], "human_confirmed"),
        pot_display=module.field("130", "human_confirmed"))
    checked = module.check_support(hero_history, assumptions())
    assert checked["reasons"] == [], checked["reasons"]
    document = module.build_document(hero_history, assumptions())[0]
    result = analyze_threeway_river(scenario_from_dict(document),
                                    max_joint_assignments=128, max_nodes=20000)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    assert result.joint_assignments == 4


def test_illegal_history_order_or_wrong_final_actor_is_refused():
    """R1-C: the history is replayed through the kernel's own legal transitions."""
    out_of_order = facts(history=module.field([
        {"actor": 2, "kind": "call", "target": "0"},
        {"actor": 1, "kind": "bet", "target": "20"}], "human_confirmed"))
    out_reasons = module.check_support(out_of_order, assumptions())["reasons"]
    assert any("公开历史" in reason for reason in out_reasons), out_reasons
    ends_early = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0"},
        {"actor": 0, "kind": "call", "target": "0"}], "human_confirmed"),
        pot_display=module.field("150", "human_confirmed"))
    early_reasons = module.check_support(ends_early, assumptions())["reasons"]
    assert any("Hero" in reason for reason in early_reasons), early_reasons
    off_grid = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "25"}], "human_confirmed"))
    grid_reasons = module.check_support(off_grid, assumptions())["reasons"]
    assert any("尺寸网格" in reason for reason in grid_reasons), grid_reasons


def test_build_reports_row_amounts_and_raise_options():
    """R1-D support: river-start stack, committed, street and raise-to amounts."""
    checked = module.check_support(facts(), assumptions())
    rows = {row["seat_id"]: row for row in checked["row_amounts"]}
    assert rows[0]["stack"] == "200"
    assert rows[0]["hand_committed"] == "20"
    assert rows[0]["street_wager"] == "0"
    assert rows[0]["status"] == "ACTIVE"
    assert rows[3]["status"] == "FOLDED"
    assert checked["root_actions"][0]["kind"] in ("fold", "call")
    raise_option = next(option for option in checked["root_actions"]
                        if option["kind"] == "raise")
    assert raise_option["target"] in ("40", "80")
    assert raise_option["additional_chips"] == raise_option["target"]
    assert "加注到" in raise_option["reading"]
    call_option = next(option for option in checked["root_actions"]
                       if option["kind"] == "call")
    assert call_option["additional_chips"] == "20"
    assert "追加" in call_option["reading"]
