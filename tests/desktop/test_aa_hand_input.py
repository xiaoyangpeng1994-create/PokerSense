"""U1: facts/assumptions adapter, support checks and the real kernel path."""

import json

from fastapi.testclient import TestClient
import pytest

from poker_engine.desktop import aa_server
from poker_engine.desktop import aa_hand_input as module
from poker_engine.strategy.threeway_river_v1 import analyze_threeway_river
from tools.analyze_threeway_river import scenario_from_dict


HEADERS = {"X-AA-Live": "1"}
SEATS = [
    {"seat_id": 0, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
    {"seat_id": 1, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
    {"seat_id": 2, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
    {"seat_id": 3, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    {"seat_id": 4, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    {"seat_id": 5, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
]


def rules_document(**changes):
    rules = {**json.load(open(
        "configs/strategy/examples/threeway-river-response-manual.json",
        encoding="utf-8"))["rules"], "rake_percent": "0", "rake_cap_bb": "0"}
    rules.update(changes)
    return rules


def facts(**overrides):
    base = module.blank_facts()
    base["ended_hand_confirmed"] = module.field(True, "human_confirmed")
    base["hero_seat"] = module.field(0, "human_confirmed")
    base["hero_cards"] = module.field(["Qs", "Qd"], "human_confirmed")
    base["board_cards"] = module.field(["2c", "4d", "7h", "9s", "Jc"],
                                       "human_confirmed")
    base["action_order"] = module.field([1, 2, 0], "human_confirmed")
    base["seats"] = module.field([dict(row) for row in SEATS], "human_confirmed")
    base["history"] = module.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0"}], "observed")
    base["pot_display"] = module.field("130", "observed")
    base["table_rules"] = module.field(rules_document(), "observed")
    for key, value in overrides.items():
        base[key] = value
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
            {"seat_id": 1, "weights": {"fold": "1", "check": "1", "call": "9",
                                       "bet": "2"}},
            {"seat_id": 2, "weights": {"fold": "1", "check": "1", "call": "9"}}],
        "aggression_targets": ["20", "40", "80"], "max_aggressions": 2,
        "other_fees": "0",
    }
    base.update(overrides)
    return base


def run_kernel(fact_block=None, assumption_block=None):
    document, _ = module.build_document(fact_block or facts(),
                                        assumption_block or assumptions())
    result = analyze_threeway_river(scenario_from_dict(document),
                                    max_joint_assignments=128, max_nodes=20000)
    return document, result


def test_supported_input_reaches_the_real_kernel_and_measures_capacity():
    document, result = run_kernel()
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    assert result.joint_assignments == 4 and result.nodes == 17
    assert document["mode"] == "manual_hypothesis"
    measured = module.capacity(facts(), assumptions())
    assert measured["declared_combo_product"] == 4
    assert measured["legal_joint_combos"] == 4
    assert measured["to_call"] == "20" and measured["implied_pot"] == "130"
    assert measured["unit"] == "chips"


def test_hand_computed_amount_口径_matches_the_kernel():
    """Independent arithmetic: one losing and one beating combo, no rake."""
    fact_block = facts(hero_cards=module.field(["Qs", "Qd"], "human_confirmed"))
    assumption_block = assumptions(
        ranges=[
            {"seat_id": 1, "combos": [{"combo": "TcTd", "weight": "1"},
                                      {"combo": "KhKd", "weight": "1"}]},
            {"seat_id": 2, "combos": [{"combo": "3c3d", "weight": "1"}]}],
        models=[
            {"seat_id": 1, "weights": {"fold": "1", "check": "1", "call": "1",
                                       "bet": "1"}},
            {"seat_id": 2, "weights": {"fold": "1", "check": "1", "call": "1"}}],
        aggression_targets=["20", "40"], max_aggressions=2)
    document, result = run_kernel(fact_block, assumption_block)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    values = {value.action.kind: value.ev for value in result.root_actions}
    # Two equally weighted joint deals: hero wins the 150 pot (cost 20 -> +130)
    # when seat 1 holds TT, and loses the 20 call when seat 1 holds KK.
    # (130 + (-20)) / 2 = 55 exactly; folding is 0. Hand arithmetic on purpose.
    assert values["fold"] == 0
    assert values["call"] == 55


def test_changing_one_declared_amount_recomputes_instead_of_reusing_a_sample():
    first_document, first = run_kernel()
    changed = facts(history=module.field([
        {"actor": 1, "kind": "bet", "target": "40"},
        {"actor": 2, "kind": "call", "target": "0"}], "observed"),
        pot_display=module.field("170", "observed"))
    second_document, second = run_kernel(changed)
    assert first_document != second_document
    assert [str(value.ev) for value in first.root_actions] != [
        str(value.ev) for value in second.root_actions]
    assert (module.digest(first_document) != module.digest(second_document))


def test_duplicate_cards_and_blocked_combos_are_refused():
    bad = facts(hero_cards=module.field(["Qs", "Qd"], "human_confirmed"),
                board_cards=module.field(["Qs", "4d", "7h", "9s", "Jc"],
                                         "human_confirmed"))
    with pytest.raises(module.HandInputError, match="重复"):
        module.check_support(bad, assumptions())
    clash = assumptions(ranges=[
        {"seat_id": 1, "combos": [{"combo": "2c2d", "weight": "1"}]},
        {"seat_id": 2, "combos": [{"combo": "KhTh", "weight": "1"}]}])
    checked = module.check_support(facts(), clash)
    assert any("牌阻断" in reason for reason in checked["reasons"])


def test_wrong_action_order_and_unknown_actor_are_refused():
    with pytest.raises(module.HandInputError, match="行动顺序"):
        module.check_support(
            facts(action_order=module.field([1, 2, 3], "human_confirmed")),
            assumptions())
    with pytest.raises(module.HandInputError, match="行动顺序"):
        module.build_document(
            facts(action_order=module.field([1, 2, 3], "human_confirmed")),
            assumptions())
    with pytest.raises(module.HandInputError, match="不在本手"):
        module.check_support(
            facts(history=module.field([{"actor": 7, "kind": "check",
                                         "target": "0"}], "observed")),
            assumptions())


def test_pot_and_call_conflicts_are_refused_without_averaging():
    checked = module.check_support(
        facts(pot_display=module.field("999", "observed")), assumptions())
    assert any("显示底池" in reason for reason in checked["reasons"])
    checked = module.check_support(facts(), assumptions())
    assert checked["to_call"] == "20" and checked["current_bet"] == "20"
    assert checked["implied_pot"] == "130"
    assert checked["reasons"] == []


def test_missing_range_or_fee_assumption_is_refused():
    with pytest.raises(module.HandInputError, match="两名 ACTIVE 对手"):
        module.check_support(facts(), assumptions(ranges=[
            {"seat_id": 1, "combos": [{"combo": "JhJd", "weight": "1"}]}]))
    with pytest.raises(module.HandInputError, match="额外费用"):
        module.check_support(facts(), assumptions(other_fees="5"))
    with pytest.raises(module.HandInputError, match="响应权重"):
        module.check_support(facts(), assumptions(models=[
            {"seat_id": 1, "weights": {"fold": "0"}}]))


def test_allin_non_threeway_and_overbudget_are_refused():
    allin = facts(seats=module.field([
        {"seat_id": 0, "stack": "30", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 1, "stack": "30", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 2, "stack": "30", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 3, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 4, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 5, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    ], "human_confirmed"))
    checked = module.check_support(allin, assumptions())
    assert any("全下" in reason for reason in checked["reasons"])
    with pytest.raises(module.HandInputError, match="三名 ACTIVE"):
        module.check_support(facts(seats=module.field([
            {"seat_id": 0, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 1, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 2, "stack": "200", "hand_committed": "20",
             "status": "FOLDED"}], "human_confirmed")), assumptions())
    wide = assumptions(ranges=[
        {"seat_id": 1, "combos": [{"combo": f"{rank}s{rank}d", "weight": "1"}
                                  for rank in
                                  ("2", "3", "4", "5", "6", "7", "8", "9", "T",
                                   "J", "Q", "K", "A")]},
        {"seat_id": 2, "combos": [{"combo": f"{rank}h{rank}c", "weight": "1"}
                                  for rank in
                                  ("2", "3", "4", "5", "6", "7", "8", "9", "T",
                                   "J", "Q", "K", "A")]}])
    checked = module.check_support(facts(), wide)
    assert any("169" in reason and "128" in reason for reason in checked["reasons"])


def test_history_must_have_probability_and_stay_inside_the_size_grid():
    checked = module.check_support(facts(), assumptions(models=[
        {"seat_id": 1, "weights": {"fold": "1", "check": "1", "call": "1"}},
        {"seat_id": 2, "weights": {"fold": "1", "check": "1", "call": "1"}}]))
    assert any("概率为 0" in reason for reason in checked["reasons"])
    checked = module.check_support(facts(), assumptions(
        aggression_targets=["40", "80"]))
    assert any("尺寸网格" in reason for reason in checked["reasons"])


def test_snapshot_facts_never_invent_a_missing_ledger():
    payload = {
        "cards": {"hero": ["5d", "6d"], "board_slots": ["5h", "6c", "6s", "Tc", "3d"]},
        "pot": {"value": "623", "raw_text": "623"},
        "hand_ledger_v2": {"status": "HAND_COMMITMENTS_UNKNOWN",
                           "hand_commitments": None,
                           "taint_reasons": ["opening_post_vector_incomplete"]},
        "observed_state_v2": {"street_candidate": "river",
                              "participants": {"1": {"state": "unknown"},
                                               "5": {"state": "active"},
                                               "6": {"state": "waiting"}}},
        "current_actor": None,
        "stacks": {"0": {"value": "255"}},
    }
    snapshot, gaps = module.facts_from_snapshot(payload, source="rec-1")
    assert snapshot["hero_cards"]["value"] == ["5d", "6d"]
    assert snapshot["hero_cards"]["provenance"] == "observed"
    assert snapshot["seats"]["value"] is None
    assert snapshot["seats"]["provenance"] == "unknown"
    assert snapshot["pot_display"]["value"] == "623"
    assert any("HAND_COMMITMENTS_UNKNOWN" in gap for gap in gaps)
    assert any("1 名 active" in gap for gap in gaps)
    assert any("当前行动者" in gap for gap in gaps)
    with pytest.raises(module.HandInputError):
        module.build_document(snapshot, assumptions())


def test_ended_hand_confirmation_is_required():
    with pytest.raises(module.HandInputError, match="已结束"):
        module.build_document(
            facts(ended_hand_confirmed=module.field(False, "observed")),
            assumptions())


def test_build_endpoint_returns_document_capacity_and_hashes(tmp_path):
    client_app = aa_server.create_app(tmp_path / "profile.json",
                                      session=Server(),
                                      records_dir=tmp_path / "records")
    with TestClient(client_app) as client:
        template = client.get("/api/hand-input/template").json()
        assert template["facts"]["ended_hand_confirmed"]["provenance"] == "unknown"
        response = client.post("/api/hand-input/build", headers=HEADERS, json={
            "facts": facts(), "assumptions": assumptions(),
            "rules_source": "document", "rules_revision": 0})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is True and body["reasons"] == []
        assert body["capacity"]["legal_joint_combos"] == 4
        assert len(body["hashes"]["input_sha256"]) == 64
        assert body["hashes"]["scope"] == "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE"
        # The built document is what the existing analysis chain consumes.
        # The built document is exactly what the existing kernel consumes.
        result = analyze_threeway_river(
            scenario_from_dict(body["document"]), max_joint_assignments=128,
            max_nodes=20000)
        assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
        assert result.joint_assignments == 4

        stale = {"facts": facts(), "assumptions": assumptions(),
                 "rules_source": "table", "rules_revision": 999}
        rejected = client.post("/api/hand-input/build", headers=HEADERS,
                               json=stale)
        assert rejected.status_code == 400
        assert "本桌规则" in rejected.json()["detail"]

        bad = client.post("/api/hand-input/build", headers=HEADERS, json={
            "facts": facts(pot_display=module.field("999", "observed")),
            "assumptions": assumptions(), "rules_source": "document",
            "rules_revision": 0})
        assert bad.status_code == 200
        assert bad.json()["ok"] is False and bad.json()["document"] is None
        assert any("显示底池" in reason for reason in bad.json()["reasons"])


class Server:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        if name == "stop":
            return lambda *args, **kwargs: None
        raise AssertionError(f"hand input touched the live session: {name}")


def test_facts_endpoint_reads_only_the_saved_structured_snapshot(tmp_path):
    payload = {"cards": {"hero": ["5d", "6d"],
                         "board_slots": ["5h", "6c", "6s", "Tc", "3d"]},
               "pot": {"value": "623"},
               "hand_ledger_v2": {"status": "HAND_COMMITMENTS_UNKNOWN",
                                  "hand_commitments": None},
               "observed_state_v2": {"participants": {"1": {"state": "unknown"}}},
               "current_actor": None}
    record = {"issue": {"issue_id": "20260916T000000-aaaaaaaaaaaa",
                        "saved_at": "2026-09-16T00:00:00+00:00",
                        "preview_sha256": "0" * 64,
                        "observation": {"source_frame": 12, "payload": payload}}}

    class Review:
        def get(self, issue_id):
            return record

        def close(self):
            return None

    calls = []
    session = Server()
    client_app = aa_server.create_app(
        tmp_path / "profile.json", session=session,
        records_dir=tmp_path / "records", review_service=Review())
    with TestClient(client_app) as client:
        facts_body = client.get(
            "/api/hand-input/facts/20260916T000000-aaaaaaaaaaaa").json()
    calls.extend(session.calls)
    assert facts_body["source"]["scope"] == "SAVED_STRUCTURED_SNAPSHOT_NO_MEDIA_NO_OCR"
    assert facts_body["facts"]["hero_cards"]["value"] == ["5d", "6d"]
    assert facts_body["facts"]["seats"]["value"] is None
    assert any("各座本手已投入" in gap for gap in facts_body["gaps"])
    assert facts_body["strategy_eligible"] is False
    assert calls == ["stop"]
