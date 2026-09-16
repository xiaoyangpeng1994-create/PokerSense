"""Product-flow interaction tests for the hand-entry form (U1 / U1-R1).

Three layers, all against the real backend:

* the shipped ``ui/aa-live/hand_input.js`` driven inside a minimal DOM stub
  (fill an ended hand with the per-row controls, verify it, hit every refusal);
* the same javascript plus the shipped ``ui/aa-live/analysis.js`` driven all the
  way through a real ``/api/analysis`` request to a real kernel result, then
  invalidated by an input change and by a late response, then recomputed;
* the HTTP contract itself, so the document the form builds is checked against
  the kernel entry point without the browser in the way.

The DOM stub is not a browser; layout and a real click-through stay a separate
step.
"""

import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest
import uvicorn

from poker_engine.desktop import aa_server


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "tests" / "js" / "hand_input_dom_stub_test.mjs"
FLOW_HARNESS = REPO_ROOT / "tests" / "js" / "analysis_flow_test.mjs"
FLOW_SERVER = REPO_ROOT / "tests" / "js" / "analysis_flow_server.py"
NODE = shutil.which("node")
TABLE_RULES = {
    "table_label": "手工核对", "dealt_players": 6,
    "small_blind": "2", "big_blind": "4", "ante": "4", "straddle_amount": "0",
    "rake_percent": "0", "rake_cap_bb": "0", "minimum_chip": "1",
    "straddle_mode": "none", "rake_application": "all_pots",
    "rake_rounding": "exact", "rake_distribution": "proportional_all_pots",
    "insurance": "off", "bomb": "off", "mushroom": "off",
}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def hand_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hand-input-ui")
    rules_path = directory / "table-rules.json"
    app = aa_server.create_app(directory / "profile.json",
                               records_dir=directory / "records",
                               rules_path=rules_path)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the local AA server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def flow_server(tmp_path_factory):
    """A dedicated process, because the analysis worker uses spawn."""
    import urllib.error
    import urllib.request

    directory = tmp_path_factory.mktemp("analysis-flow")
    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(FLOW_SERVER), "--work", str(directory),
         "--port", str(port)],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
    try:
        line = process.stdout.readline().strip()
        assert line, (f"the flow server printed nothing; stderr: "
                      f"{process.stderr.read()[-1500:]}")
        base = json.loads(line)["base"]
        # The launcher announces its port before uvicorn binds it, so poll for
        # an actual answer rather than trusting the line (this raced on macOS).
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(base + "/api/rules", timeout=5).read()
                break
            except urllib.error.HTTPError:
                break
            except OSError:
                if time.time() >= deadline:
                    raise AssertionError(
                        f"the flow server never answered on {base}; stderr: "
                        f"{process.stderr.read()[-1500:]}") from None
                time.sleep(0.2)
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def post_rules(base):
    import urllib.request
    current = json.loads(urllib.request.urlopen(
        base + "/api/rules", timeout=10).read().decode("utf-8"))
    request = urllib.request.Request(
        base + "/api/rules",
        data=json.dumps({"document": TABLE_RULES,
                         "revision": current["revision"]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-AA-Live": "1"})
    return json.loads(urllib.request.urlopen(request, timeout=10).read().decode(
        "utf-8"))


def run_harness(harness, base, timeout=240):
    proc = subprocess.run([NODE, str(harness), base], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    checks = [json.loads(line) for line in proc.stdout.splitlines()
              if line.strip().startswith("{")]
    failures = [item for item in checks if not item.get("ok")]
    assert proc.returncode == 0, (
        f"failed checks: {failures}\nstderr: {proc.stderr[-2000:]}")
    verdict = checks[-1]
    assert verdict["name"] == "verdict" and verdict["ok"] is True, verdict
    return checks, verdict


@pytest.mark.skipif(
    NODE is None,
    reason="node is unavailable, so the shipped JS cannot be exercised here")
def test_hand_input_flow_drives_the_shipped_javascript(hand_server):
    saved = post_rules(hand_server)
    assert saved["conditional_analysis_ready"] is True, saved
    checks, verdict = run_harness(HARNESS, hand_server)
    assert verdict["failed"] == 0 and verdict["passed"] >= 20
    names = {item["name"] for item in checks}
    for required in (
            "form_wiring_ran", "row_controls_are_seeded",
            "an_untouched_seeded_row_is_not_input",
            "verified_input_is_accepted", "capacity_is_measured",
            "identity_hashes_are_shown", "document_is_the_kernel_shape",
            "no_policy_value_is_shown_here",
            "changing_a_field_invalidates_the_verified_input",
            "editing_a_row_invalidates_the_verified_input",
            "pot_conflict_is_refused_in_chinese",
            "duplicate_card_is_refused",
            "unknown_fees_are_refused_instead_of_zeroed",
            "history_without_probability_is_refused",
            "history_off_the_declared_grid_is_refused",
            "wrong_action_order_is_refused", "non_three_way_is_refused",
            "a_missing_hero_seat_is_a_gap_not_a_guess",
            "compute_hands_the_document_to_the_existing_chain",
            "compute_binds_the_input_identity_to_the_panel",
            "no_innerhtml_used",
            "only_hand_input_and_analysis_endpoints"):
        assert required in names, f"missing check: {required}"


@pytest.mark.skipif(
    NODE is None,
    reason="node is unavailable, so the shipped JS cannot be exercised here")
def test_input_reaches_the_real_kernel_and_survives_invalidation(flow_server):
    """input → kernel → result → change input / late response → recompute."""
    saved = post_rules(flow_server)
    assert saved["conditional_analysis_ready"] is True, saved
    checks, verdict = run_harness(FLOW_HARNESS, flow_server, timeout=420)
    assert verdict["failed"] == 0 and verdict["passed"] >= 14
    names = {item["name"] for item in checks}
    for required in (
            "both_shipped_scripts_loaded",
            "the_form_accepts_a_filled_ended_hand",
            "compute_hands_the_document_to_the_analysis_panel",
            "the_panel_really_posted_the_analysis",
            "the_real_kernel_returned_a_complete_result",
            "the_result_is_rendered_from_the_real_report",
            "the_report_input_matches_the_verified_form_input",
            "nothing_claims_this_is_live_advice",
            "the_export_carries_the_form_input_identity",
            "changing_the_form_clears_the_rendered_result",
            "changing_the_form_also_drops_the_panel_identity",
            "the_abandoned_input_is_a_different_document",
            "editing_during_a_running_analysis_clears_it",
            "the_late_report_is_refused_not_displayed",
            "the_late_report_really_was_for_the_abandoned_input",
            "the_changed_hand_verifies_as_a_different_input",
            "recomputing_produces_a_result_for_the_new_input",
            "the_new_result_differs_from_the_first",
            "the_second_report_is_bound_to_the_second_input",
            "a_result_whose_input_identity_moved_is_invalidated"):
        assert required in names, f"missing check: {required}"


def test_page_serves_the_form_and_keeps_safe_text_nodes():
    source = (REPO_ROOT / "ui" / "aa-live" / "hand_input.js").read_text(
        encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source and "handElement" in source
    assert "handGroupRanges" in source
    page = (REPO_ROOT / "ui" / "aa-live" / "index.html").read_text(
        encoding="utf-8")
    assert "/hand_input.js" in page
    for anchor in ('id="hand-input-panel"', 'id="hand-rows-seats"',
                   'id="hand-rows-history"', 'id="hand-rows-ranges"',
                   'id="hand-rows-weights"', 'id="hand-add-seat"',
                   'id="hand-add-history"', 'id="hand-add-range"',
                   'id="hand-add-weight"', 'id="hand-fees"',
                   'id="hand-import-record"', 'id="hand-csv"',
                   'id="hand-csv-target"', 'id="hand-csv-apply"',
                   'id="hand-build"', 'id="hand-compute"', 'id="hand-gaps"'):
        assert anchor in page, anchor
    # The old free-text CSV textareas are gone: rows are the normal path.
    for retired in ('id="hand-seats"', 'id="hand-history"',
                    'id="hand-ranges"', 'id="hand-weights"'):
        assert retired not in page, retired
    analysis = (REPO_ROOT / "ui" / "aa-live" / "analysis.js").read_text(
        encoding="utf-8")
    assert "analysisExpectedInput" in analysis


def hand_facts(**overrides):
    """A complete, unambiguous, human-confirmed ended hand."""
    facts = {
        "source": None, "source_kind": "manual_form",
        "ended_hand_confirmed": {"value": True, "provenance": "human_confirmed",
                                 "candidate": None},
        "hero_seat": {"value": 0, "provenance": "human_confirmed",
                      "candidate": None},
        "hero_cards": {"value": ["Qs", "Qd"], "provenance": "human_confirmed",
                       "candidate": None},
        "board_cards": {"value": ["2c", "4d", "7h", "9s", "Jc"],
                        "provenance": "human_confirmed", "candidate": None},
        "action_order": {"value": [1, 2, 0], "provenance": "human_confirmed",
                         "candidate": None},
        "seats": {"value": [
            {"seat_id": 0, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 1, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 2, "stack": "200", "hand_committed": "20",
             "status": "ACTIVE"},
            {"seat_id": 3, "stack": "200", "hand_committed": "10",
             "status": "FOLDED"},
            {"seat_id": 4, "stack": "200", "hand_committed": "10",
             "status": "FOLDED"},
            {"seat_id": 5, "stack": "200", "hand_committed": "10",
             "status": "FOLDED"}], "provenance": "human_confirmed",
            "candidate": None},
        "history": {"value": [{"actor": 1, "kind": "bet", "target": "40"},
                              {"actor": 2, "kind": "call", "target": "0"}],
                    "provenance": "human_confirmed", "candidate": None},
        "pot_display": {"value": "170", "provenance": "human_confirmed",
                        "candidate": None},
        "table_rules": {"value": None, "provenance": "unknown", "candidate": None},
        "observed_at": None,
    }
    facts.update(overrides)
    return facts


def hand_assumptions(**overrides):
    assumptions = {
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
        "aggression_targets": ["40", "80"], "max_aggressions": 2,
        "other_fees": {"value": "0", "provenance": "human_confirmed"},
    }
    assumptions.update(overrides)
    return assumptions


def build(base, facts=None, assumptions=None):
    """Post one build request.

    Returns ``(status, payload)``. A refusal that the static pre-checks make
    arrives as a 400 whose body is the Chinese explanation; a gap the kernel or
    the support check finds arrives as 200 with ``ok: False`` and ``reasons``.
    """
    import urllib.error
    import urllib.request

    current = json.loads(urllib.request.urlopen(
        base + "/api/rules", timeout=10).read().decode("utf-8"))
    request = urllib.request.Request(
        base + "/api/hand-input/build",
        data=json.dumps({"facts": facts or hand_facts(),
                         "assumptions": assumptions or hand_assumptions(),
                         "rules_source": "table",
                         "rules_revision": current["revision"]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-AA-Live": "1"})
    try:
        response = urllib.request.urlopen(request, timeout=20)
        return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, {"detail": error.read().decode("utf-8")}


def test_built_input_reaching_the_kernel_uses_the_saved_table_rules(hand_server):
    """The same request the form makes, then the real kernel chain."""
    from poker_engine.strategy.threeway_river_v1 import analyze_threeway_river
    from tools.analyze_threeway_river import scenario_from_dict

    status, body = build(hand_server)
    assert status == 200 and body["ok"] is True, body
    assert body["document"]["rules"]["big_blind"] == "4"
    assert body["document"]["rules"]["ante"] == "4"
    assert body["capacity"]["to_call"] == "40"
    assert body["capacity"]["legal_joint_combos"] == 4
    assert body["amounts"]["implied_pot"] == "170"
    assert [row["seat_id"] for row in body["amounts"]["rows"]] == [0, 1, 2, 3, 4, 5]
    result = analyze_threeway_river(scenario_from_dict(body["document"]),
                                    max_joint_assignments=128, max_nodes=20000)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    assert result.joint_assignments == 4
    assert result.strategy_eligible is False and result.advice_emitted is False
    assert body["hashes"]["implementation_version"] == "aa-hand-input-v1"
    # The form's identity is the same canonical digest the analysis panel's
    # backend computes for the document it sends, so a result can be tied to the
    # exact input the human verified rather than to two unrelated hashes.
    from poker_engine.desktop.aa_analysis import _json_bytes
    assert body["hashes"]["input_sha256"] == hashlib.sha256(
        _json_bytes(body["document"], limit=1 << 20, name="probe")).hexdigest()


def test_a_hero_decision_history_is_accepted_and_root_actions_are_listed(
        hand_server):
    """A legal history that stops on Hero returns the kernel's own root actions."""
    facts = hand_facts(
        history={"value": [{"actor": 1, "kind": "check", "target": "0"},
                           {"actor": 2, "kind": "bet", "target": "40"}],
                 "provenance": "human_confirmed", "candidate": None},
        pot_display={"value": "130", "provenance": "human_confirmed",
                     "candidate": None})
    assumptions = hand_assumptions(models=[
        {"seat_id": 1, "key": "check", "weight": "9"},
        {"seat_id": 1, "key": "bet", "weight": "1"},
        {"seat_id": 1, "key": "call", "weight": "1"},
        {"seat_id": 1, "key": "fold", "weight": "1"},
        {"seat_id": 2, "key": "check", "weight": "1"},
        {"seat_id": 2, "key": "bet", "weight": "9"},
        {"seat_id": 2, "key": "call", "weight": "1"},
        {"seat_id": 2, "key": "fold", "weight": "1"}])
    status, body = build(hand_server, facts, assumptions)
    assert status == 200 and body["ok"] is True, body
    kinds = [action["kind"] for action in body["amounts"]["root_actions"]]
    assert "fold" in kinds and "call" in kinds and "raise" in kinds, kinds
    call = next(action for action in body["amounts"]["root_actions"]
                if action["kind"] == "call")
    assert call["additional_chips"] == "40"
    assert body["amounts"]["to_call"] == "40"
    raise_action = next(action for action in body["amounts"]["root_actions"]
                        if action["kind"] == "raise")
    assert raise_action["raise_to"] == "80"


@pytest.mark.parametrize("history,accepted", [
    # Stops before Hero is asked: refused.
    ([{"actor": 1, "kind": "check", "target": "0"}], False),
    # Both opponents checked, so Hero is next: accepted.
    ([{"actor": 1, "kind": "check", "target": "0"},
      {"actor": 2, "kind": "check", "target": "0"}], True)])
def test_public_history_must_stop_at_a_hero_decision(hand_server, history, accepted):
    facts = hand_facts(
        history={"value": history, "provenance": "human_confirmed",
                 "candidate": None},
        pot_display={"value": "90", "provenance": "human_confirmed",
                     "candidate": None})
    status, body = build(hand_server, facts)
    assert status == 200, body
    if accepted:
        assert body["ok"] is True, body["reasons"]
        return
    assert body["ok"] is False
    assert any("停在 Hero" in reason for reason in body["reasons"]), body["reasons"]


def test_an_impossible_or_out_of_order_history_is_refused_in_chinese(hand_server):
    # Seat 2 cannot act first: the declared order is 1, 2, 0.
    status, out_of_order = build(hand_server, hand_facts(history={
        "value": [{"actor": 2, "kind": "check", "target": "0"}],
        "provenance": "human_confirmed", "candidate": None}))
    assert status == 200
    assert out_of_order["ok"] is False
    assert any("合法行动里不存在" in reason for reason in out_of_order["reasons"]), \
        out_of_order["reasons"]
    # Raising when nobody has bet is not a legal river action.
    status, illegal_kind = build(hand_server, hand_facts(history={
        "value": [{"actor": 1, "kind": "raise", "target": "40"}],
        "provenance": "human_confirmed", "candidate": None}))
    assert status == 200
    assert illegal_kind["ok"] is False
    assert any("合法行动里不存在" in reason for reason in illegal_kind["reasons"]), \
        illegal_kind["reasons"]


def test_unknown_facts_and_fees_are_refused_without_being_defaulted(hand_server):
    status, blank_hero = build(hand_server, hand_facts(
        hero_seat={"value": None, "provenance": "unknown", "candidate": None}))
    assert status == 400
    assert "Hero 座位号未知" in blank_hero["detail"], blank_hero
    status, unknown_history = build(hand_server, hand_facts(
        history={"value": None, "provenance": "unknown", "candidate": None}))
    assert status == 400
    assert "公开历史未知" in unknown_history["detail"], unknown_history
    status, unknown_fees = build(hand_server, hand_facts(), hand_assumptions(
        other_fees={"value": None, "provenance": "unknown"}))
    assert status == 400
    assert "额外费用未知" in unknown_fees["detail"], unknown_fees
    status, assumed = build(hand_server, hand_facts(), hand_assumptions(
        other_fees={"value": "0", "provenance": "assumed"}))
    assert status == 200 and assumed["ok"] is True, assumed
