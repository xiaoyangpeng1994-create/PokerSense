"""U1-R2 regressions for the import/snapshot side.

Two groups, both from review `pullrequestreview-5222526862`:

* the snapshot must carry explicit river-start evidence before its CURRENT
  stacks/commitments/participant states may become observed river-start facts
  (the R1 tests used a snapshot with no start evidence as a positive case, which
  is exactly what the review rejected);
* the exact-amount replay and the size-less row controls.
"""

import json

import pytest

from poker_engine.desktop import aa_hand_input as module


RULES = {**json.load(open(
    "configs/strategy/examples/threeway-river-response-manual.json",
    encoding="utf-8"))["rules"], "rake_percent": "0", "rake_cap_bb": "0"}
EPOCH = "20260916T101530-aaaaaaaaaaaa"


def participants(**overrides):
    states = {"0": "active", "1": "active", "2": "active",
              "3": "folded", "4": "folded", "5": "folded"}
    states.update(overrides)
    return {seat: {"state": state, "epoch": EPOCH} for seat, state in states.items()}


def wagers(**overrides):
    """The river street wagers: all zero at the river start."""
    vector = {str(seat): "0" for seat in range(8)}
    for seat, value in overrides.items():
        vector[seat] = value
    return {"status": "OBSERVED_STREET_WAGERS_CANDIDATE",
            "title_center_ledger_reconciled": True, "wagers": vector,
            "street_price": "0", "context_automated": True,
            "canonical_verified": False, "strategy_eligible": False}


def ledger(**overrides):
    block = {
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
        "complete_and_canonical_verified": False, "strategy_eligible": False,
    }
    block.update(overrides)
    return block


def payload(**overrides):
    """A payload that carries the full river-start evidence."""
    block = {
        "frame": 1500,
        "cards": {"hero": ["5d", "6d"],
                  "board_slots": ["5h", "6c", "6s", "Tc", "3d"]},
        "pot": {"value": "90"},
        "hand_ledger_v2": ledger(),
        "causal_street_wagers_v2": wagers(),
        "observed_state_v2": {"observed_epoch": EPOCH, "street_candidate": "river",
                              "participants": participants(),
                              "pending_actions": 0},
        "stacks": {str(seat): {"value": "200"} for seat in range(6)},
        "current_actor": 0,
    }
    block.update(overrides)
    return block


def facts(**overrides):
    base = module.blank_facts()
    base["ended_hand_confirmed"] = module.field(True, "human_confirmed")
    base["hero_seat"] = module.field(0, "human_confirmed")
    base["hero_cards"] = module.field(["Qs", "Qd"], "human_confirmed")
    base["board_cards"] = module.field(["2c", "4d", "7h", "9s", "Jc"],
                                       "human_confirmed")
    base["action_order"] = module.field([1, 2, 0], "human_confirmed")
    base["seats"] = module.field([
        {"seat_id": 0, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 1, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 2, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 3, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 4, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 5, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    ], "human_confirmed")
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


# --- river-start evidence -------------------------------------------------

def test_a_payload_with_full_river_start_evidence_yields_observed_seats():
    from_snapshot, gaps = module.facts_from_snapshot(payload(), source="rec-A")
    assert from_snapshot["seats"]["provenance"] == "observed"
    states = {row["seat_id"]: row["status"] for row in from_snapshot["seats"]["value"]}
    assert states == {0: "ACTIVE", 1: "ACTIVE", 2: "ACTIVE",
                      3: "FOLDED", 4: "FOLDED", 5: "FOLDED"}
    assert from_snapshot["seats"]["candidate"] is not None


@pytest.mark.parametrize("label,mutate", [
    ("缺少河牌起点证据（无 epoch）",
     lambda p: p["observed_state_v2"].pop("observed_epoch")),
    ("不是河牌（街候选为 turn）", lambda p: p.update(observed_state_v2={
        **p["observed_state_v2"], "street_candidate": "turn"})),
    ("账本被污染", lambda p: p["hand_ledger_v2"].update(
        taint_reasons=["source_frame_gap"])),
    ("账本状态不是干净候选", lambda p: p["hand_ledger_v2"].update(
        status="HAND_COMMITMENTS_SUSPENDED")),
    ("账本属于另一手（epoch 不一致）", lambda p: p["hand_ledger_v2"].update(
        epoch="20260916T101600-bbbbbbbbbbbb")),
    ("本街下注额未知", lambda p: p["causal_street_wagers_v2"].update(
        status="WAGERS_UNKNOWN")),
    ("本街下注与中控对不上", lambda p: p["causal_street_wagers_v2"].update(
        title_center_ledger_reconciled=False)),
    ("已经是街中快照（河里已有下注）", lambda p: p["causal_street_wagers_v2"].update(
        wagers={**p["causal_street_wagers_v2"]["wagers"], "1": "20"},
        street_price="20")),
    ("参与状态不明确", lambda p: p["observed_state_v2"].update(participants={
        **participants(), "1": {"state": "unknown", "epoch": EPOCH}})),
])
def test_current_state_without_start_evidence_is_not_a_river_start_fact(
        label, mutate):
    """The R1 positive case is now a negative one: no start proof, no facts."""
    broken = payload()
    mutate(broken)
    from_snapshot, gaps = module.facts_from_snapshot(broken, source="rec-A")
    assert from_snapshot["seats"]["provenance"] == "unknown", label
    assert from_snapshot["seats"]["value"] is None, label
    assert any("起点" in gap or "河牌" in gap or "不明确" in gap or "另一手" in gap
               for gap in gaps), (label, gaps)


def test_start_stacks_and_commitments_never_replace_a_missing_ledger():
    """Current stacks cannot stand in for a river-start commitment level."""
    broken = payload(hand_ledger_v2=ledger(hand_commitments=None,
                                           status="HAND_COMMITMENTS_UNKNOWN"))
    from_snapshot, gaps = module.facts_from_snapshot(broken, source="rec-A")
    assert from_snapshot["seats"]["provenance"] == "unknown"
    assert any("已投入" in gap for gap in gaps), gaps


# --- exact amounts and size-less actions ---------------------------------

def test_the_replay_compares_amounts_by_value_not_by_text():
    """Grid 20 and history 20.0 are the same amount, not a mismatch."""
    off = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "20.0"},
        {"actor": 2, "kind": "call", "target": "0"}], "human_confirmed"))
    checked = module.check_support(off, assumptions())
    assert checked["reasons"] == [], checked["reasons"]


def test_size_less_actions_need_no_declared_target():
    """check/call/fold carry the kernel's own zero, not a user-entered sentinel."""
    cases = {
        # A check at the river start is legal and carries target 0.
        "check": ([{"actor": 1, "kind": "check", "target": "0"}], "90"),
        # A call only exists facing a bet, and still carries target 0.
        "call": ([{"actor": 1, "kind": "bet", "target": "20"},
                  {"actor": 2, "kind": "call", "target": "0"}], "130"),
        # A fold exists only facing a bet, and still carries target 0.
        "fold": ([{"actor": 1, "kind": "bet", "target": "20"},
                  {"actor": 2, "kind": "fold", "target": "0"}], "110"),
    }
    for kind, (history, pot) in cases.items():
        reasons = module.check_support(
            facts(history=module.field(history, "human_confirmed"),
                  pot_display=module.field(pot, "human_confirmed")),
            assumptions())["reasons"]
        assert not any("合法行动里不存在" in reason for reason in reasons), \
            (kind, reasons)


def test_an_optional_call_amount_must_match_the_replayed_amount():
    """A declared call cost is reconciled against the replay, not trusted."""
    agrees = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0", "call_amount": "20"}],
        "human_confirmed"))
    assert module.check_support(agrees, assumptions())["reasons"] == []
    disagrees = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0", "call_amount": "15"}],
        "human_confirmed"))
    reasons = module.check_support(disagrees, assumptions())["reasons"]
    assert any("跟注" in reason and "20" in reason for reason in reasons), reasons


def test_a_blank_seat_id_is_refused_instead_of_becoming_seat_zero():
    """An empty id must not turn into 0 through int('')/float('')."""
    with pytest.raises(module.HandInputError, match="座位号"):
        module.check_support(facts(seats=module.field([
            {"seat_id": None, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 1, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 2, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"}], "human_confirmed")), assumptions())
    with pytest.raises(module.HandInputError, match="座位号"):
        module.check_support(facts(history=module.field([
            {"actor": "", "kind": "bet", "target": "20"}], "human_confirmed")),
            assumptions())


def test_unknown_history_is_not_confirmed_empty_history():
    """`unknown` (unfilled) and an explicitly confirmed `[]` must stay distinct."""
    with pytest.raises(module.HandInputError, match="公开历史"):
        module.check_support(facts(history=module.field(None, "unknown")),
                             assumptions())
    confirmed = module.check_support(
        facts(history=module.field([], "human_confirmed"),
              action_order=module.field([0, 1, 2], "human_confirmed"),
              pot_display=module.field("90", "human_confirmed")),
        assumptions())
    assert confirmed["reasons"] == [], confirmed["reasons"]
