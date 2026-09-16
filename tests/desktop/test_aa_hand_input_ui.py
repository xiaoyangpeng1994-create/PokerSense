"""Product-flow interaction test for the hand-entry form (U1).

Runs the shipped `ui/aa-live/hand_input.js` unchanged inside a minimal DOM stub
against the real backend: fill an ended hand, verify it, change it, hit every
refusal, and hand the built document to the existing analysis entry.

The DOM stub is not a browser; it exercises the shipped JavaScript and the real
HTTP contract. Layout and a real click-through stay a separate step.
"""

import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import pytest
import uvicorn

from poker_engine.desktop import aa_server


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "tests" / "js" / "hand_input_dom_stub_test.mjs"
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


@pytest.mark.skipif(
    NODE is None,
    reason="node is unavailable, so the shipped JS cannot be exercised here")
def test_hand_input_flow_drives_the_shipped_javascript(hand_server):
    saved = post_rules(hand_server)
    assert saved["conditional_analysis_ready"] is True, saved
    proc = subprocess.run([NODE, str(HARNESS), hand_server], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=240)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    checks = [json.loads(line) for line in lines]
    failures = [item for item in checks if not item.get("ok")]
    assert proc.returncode == 0, (
        f"failed checks: {failures}\nstderr: {proc.stderr[-2000:]}")
    verdict = checks[-1]
    assert verdict["name"] == "verdict" and verdict["ok"] is True
    assert verdict["failed"] == 0 and verdict["passed"] >= 10
    names = {item["name"] for item in checks}
    for required in ("form_wiring_ran", "verified_input_is_accepted",
                     "capacity_is_measured", "identity_hashes_are_shown",
                     "document_is_the_kernel_shape",
                     "changing_the_input_invalidates_the_verified_input",
                     "pot_conflict_is_refused_in_chinese",
                     "duplicate_card_is_refused",
                     "history_without_probability_is_refused",
                     "wrong_action_order_is_refused",
                     "non_three_way_is_refused",
                     "compute_hands_the_document_to_the_existing_chain",
                     "no_innerhtml_used", "only_hand_input_and_analysis_endpoints"):
        assert required in names, f"missing check: {required}"


def test_page_serves_the_form_and_keeps_safe_text_nodes():
    source = (REPO_ROOT / "ui" / "aa-live" / "hand_input.js").read_text(
        encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source and "handElement" in source
    page = (REPO_ROOT / "ui" / "aa-live" / "index.html").read_text(
        encoding="utf-8")
    assert "/hand_input.js" in page
    for anchor in ('id="hand-input-panel"', 'id="hand-seats"', 'id="hand-ranges"',
                   'id="hand-build"', 'id="hand-compute"', 'id="hand-gaps"'):
        assert anchor in page, anchor


def test_built_input_reaching_the_kernel_uses_the_saved_table_rules(hand_server):
    """The same request the form makes, then the real kernel chain."""
    import urllib.request

    from poker_engine.strategy.threeway_river_v1 import analyze_threeway_river
    from tools.analyze_threeway_river import scenario_from_dict

    current = json.loads(urllib.request.urlopen(
        hand_server + "/api/rules", timeout=10).read().decode("utf-8"))
    facts = {
        "source": None, "source_kind": "manual_form",
        "ended_hand_confirmed": {"value": True, "provenance": "human_confirmed",
                                 "candidate": None},
        "hero_seat": {"value": 0, "provenance": "human_confirmed", "candidate": None},
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
    assumptions = {
        "range_source": "manual_unvalidated",
        "ranges": [
            {"seat_id": 1, "combos": [{"combo": "JhJd", "weight": "1"},
                                      {"combo": "TcTd", "weight": "3"}]},
            {"seat_id": 2, "combos": [{"combo": "7c7s", "weight": "1"},
                                      {"combo": "KhTh", "weight": "3"}]}],
        "models": [
            {"seat_id": 1, "weights": {"check": "1", "bet": "2", "call": "9",
                                       "fold": "1"}},
            {"seat_id": 2, "weights": {"check": "1", "call": "9", "fold": "1"}}],
        "aggression_targets": ["40", "80"], "max_aggressions": 2,
        "other_fees": "0",
    }
    request = urllib.request.Request(
        hand_server + "/api/hand-input/build",
        data=json.dumps({"facts": facts, "assumptions": assumptions,
                         "rules_source": "table",
                         "rules_revision": current["revision"]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-AA-Live": "1"})
    body = json.loads(urllib.request.urlopen(request, timeout=20).read().decode(
        "utf-8"))
    assert body["ok"] is True, body["reasons"]
    assert body["document"]["rules"]["big_blind"] == "4"
    assert body["document"]["rules"]["ante"] == "4"
    assert body["capacity"]["to_call"] == "40"
    result = analyze_threeway_river(scenario_from_dict(body["document"]),
                                    max_joint_assignments=128, max_nodes=20000)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    assert result.joint_assignments == 4
    assert result.strategy_eligible is False and result.advice_emitted is False
    assert body["hashes"]["implementation_version"] == "aa-hand-input-v1"
