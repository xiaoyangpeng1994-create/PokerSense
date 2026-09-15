import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from tools.aa_confirmation_ledger import Ledger, digest
from tools.aa_replay_viewer import REPLAY_SHA, create_app, present


def source(frame=0):
    return {"source_frame": frame, "source": "a" * 64, "opencv_pos_msec": frame,
            "scene": "AA_TABLE_CANDIDATE", "gap_reset": False,
            "incomplete_fields": ["pot", "seat_presence", "full_actions",
                                  "special_modes", "participation", "acceptance"],
            "hero": None, "board_slots": [None] * 5,
            "hero_participation": "UNKNOWN",
            "stacks": {str(i): None for i in range(9)}, "actions": {},
            "strategy_eligible": False}


def book(tmp_path, zones=("development", "reserved", "unknown", "exploration")):
    rows = [source(i) for i in range(len(zones))]
    policy = {"session_id": "synthetic-test", "replay_sha256": REPLAY_SHA,
              "zones": [{"first": i, "last": i, "zone": zone}
                        for i, zone in enumerate(zones)]}
    return Ledger(tmp_path / "test.jsonl", rows, REPLAY_SHA, policy)


HEADERS = {"X-AA-Confirmation": "1"}


def test_persisted_confirmation_ui_and_frame_isolation(tmp_path):
    ledger = book(tmp_path)
    client = TestClient(create_app(ledger.rows, ledger))
    response = client.post("/api/confirm", headers=HEADERS, json={
        "frame": 0, "field": "pot", "seat": None, "human_value": "123"})
    assert response.status_code == 200
    e = response.json()
    assert e["model_output"] is None and e["model_reason"] == "incomplete_fields:pot"
    assert e["exposure"] == "never_trained" and e["training_eligible"] is True
    persisted = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert persisted == e
    restarted = Ledger(ledger.path, ledger.rows, REPLAY_SHA, ledger.policy)
    client = TestClient(create_app(ledger.rows, restarted))
    fields = client.get("/api/frame/0").json()["fields"]
    assert next(f for f in fields if f["name"] == "pot")["status"] == "人工确认"
    assert next(f for f in fields if f["name"] == "pot")["value"] == "123"
    assert next(f for f in client.get("/api/frame/1").json()["fields"]
                if f["name"] == "pot")["status"] == "未实现"


def test_zone_exports_and_policy_tamper(tmp_path):
    ledger = book(tmp_path)
    for i in range(4):
        e = ledger.confirm(i, "pot", None, "10")
        assert e["training_eligible"] is (i == 0)
    export = ledger.export("money")
    assert len(export["entries"]) == 1
    assert export["inputs"][0]["value"] == "10"
    assert export["inputs"][0]["consumer"] == "aa8_money_bank_v2.labelled_glyphs"
    changed = copy.deepcopy(ledger.policy)
    changed["zones"][1]["zone"] = "development"
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        Ledger(ledger.path, ledger.rows, REPLAY_SHA, changed)


def test_unknown_default_and_reserved_exposure_rejected(tmp_path):
    ledger = Ledger(tmp_path / "unknown.jsonl", [source()], REPLAY_SHA)
    e = ledger.confirm(0, "pot", None, "0")
    assert not e["training_eligible"] and e["frame_zone"] == "unknown"
    with pytest.raises(ValueError, match="NOT_ELIGIBLE"):
        ledger.mark_trained(e["entry_sha256"])


def test_exposure_is_monotonic_without_reverting_correction_or_hash(tmp_path):
    ledger = book(tmp_path)
    first = ledger.confirm(0, "pot", None, "10")
    second = ledger.confirm(0, "pot", None, "20")
    ledger.mark_trained(first["entry_sha256"])
    exported = ledger.export("all")
    assert exported["entries"] == [second]
    assert exported["trained_frames"] == [0]
    third = ledger.confirm(0, "current_bet", 2, "5")
    assert third["exposure"] == "trained"
    assert ledger.snapshot()[1][0]["human_value"] == "20"
    for e in ledger.export("all")["entries"]:
        payload = {k: v for k, v in e.items() if k != "entry_sha256"}
        assert digest(payload) == e["entry_sha256"]


def test_report_denominators_latest_only_and_read_only(tmp_path):
    ledger = book(tmp_path)
    ledger.rows[0]["actions"] = {"0": "fold", "1": "call"}
    ledger.confirm(0, "action", 0, "fold")
    ledger.confirm(0, "action", 1, "raise")
    ledger.confirm(0, "action", 2, "fold")
    ledger.confirm(0, "action", 2, "call")
    before = ledger.path.read_bytes()
    report = ledger.report()
    assert report["fields"]["action"] == {
        "confirmed_denominator": 3, "comparable_denominator": 2,
        "consistent": 1, "inconsistent": 1, "rejected_or_unknown": 1}
    assert report["confirmation_events"] == 4
    assert report["ledger_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["false_positive_rate"] is None
    assert before == ledger.path.read_bytes()
    assert report == ledger.report()


@pytest.mark.parametrize("field,seat,value", [
    ("pot", None, -1), ("pot", None, "NaN"), ("actor", None, True),
    ("current_bet", True, "2"), ("action", 9, "fold"),
    ("seat_presence", 0, "UNKNOWN"), ("special_mode", None, {}),
    ("acceptance", None, True), ("pot", 0, "2")])
def test_invalid_confirmation_leaves_ledger_unchanged(tmp_path, field, seat, value):
    ledger = book(tmp_path)
    with pytest.raises(ValueError):
        ledger.confirm(0, field, seat, value)
    assert not ledger.path.exists()


def test_api_forbids_cross_origin_and_client_zone_override(tmp_path):
    ledger = book(tmp_path)
    client = TestClient(create_app(ledger.rows, ledger))
    body = {"frame": 1, "seat": None, "field": "pot", "human_value": "100"}
    assert client.post("/api/confirm", json=body).status_code == 403
    assert client.post("/api/confirm", json=body, headers={
        **HEADERS, "Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/confirm", json={**body, "training_eligible": True},
                       headers=HEADERS).status_code == 422
    assert client.post("/api/confirm", json=body, headers=HEADERS).status_code == 200
    assert client.get("/api/export/money").json()["entries"] == []
    assert client.get("/api/report").status_code == 200


def test_corruption_and_truncation_fail_closed(tmp_path):
    ledger = book(tmp_path)
    ledger.confirm(0, "pot", None, "10")
    original = ledger.path.read_bytes()
    ledger.path.write_bytes(original.replace(b'"10"', b'"11"'))
    with pytest.raises(ValueError, match="HASH_CHAIN"):
        ledger.snapshot()
    ledger.path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="TRUNCATED"):
        ledger.confirm(0, "pot", None, "20")


def test_concurrent_instances_preserve_each_confirmation(tmp_path):
    ledger = book(tmp_path)
    other = Ledger(ledger.path, ledger.rows, REPLAY_SHA, ledger.policy)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: (ledger if i % 2 else other).confirm(
            0, "pot", None, str(i)), range(12)))
    assert len(ledger.snapshot()[0]) == 12


def test_strategy_requires_complete_source_and_effective_values(tmp_path):
    ledger = book(tmp_path)
    ledger.confirm(0, "pot", None, "100")
    assert not present(ledger.rows[0], ledger.snapshot()[1])["strategy_eligible"]
    complete = source()
    complete.update(incomplete_fields=[], acceptance=True, hero=["Jc", "9h"],
                    board_slots=["2c", "3c", "4h"], hero_participation="PARTICIPATING",
                    full_actions=[], pot="100", current_actor=0,
                    seat_presence={str(i): "empty" for i in range(9)},
                    special_modes={k: {"enabled": False, "triggered": False}
                                   for k in ("insurance", "critical_hit", "squid")})
    for seat in (0, 2):
        complete["seat_presence"][str(seat)] = "participating"
        complete["stacks"][str(seat)] = "100"
    complete["current_bet"] = {"0": "0", "2": "0"}
    assert present(complete)["strategy_eligible"] is True
    for key, value in (("hero", []), ("special_modes", {"insurance": {}}),
                       ("current_actor", True), ("pot", "bad")):
        broken = {**complete, key: value}
        assert present(broken)["strategy_eligible"] is False
    complete["current_actor"] = None
    assert present(complete)["strategy_eligible"] is False
    ledger.confirm(0, "actor", None, 2)
    assert present(complete, ledger.snapshot()[1])["strategy_eligible"] is True
    complete["gap_reset"] = True
    assert present(complete, ledger.snapshot()[1])["strategy_eligible"] is False


def test_unknown_declaration_is_contract_error():
    row = source()
    row["incomplete_fields"].append("invented")
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        present(row)


def test_visible_glyph_is_not_full_action_history_conflict():
    row = source()
    row["actions"] = {"0": "fold"}
    assert not present(row)["contract_errors"]
    assert next(f for f in present(row)["fields"]
                if f["name"] == "full_actions")["status"] == "未实现"


def test_amount_comparison_normalizes_types_and_preserves_raw(tmp_path):
    ledger = book(tmp_path)
    ledger.rows[0]["pot"] = 100
    e = ledger.confirm(0, "pot", None, "00100")
    assert e["model_output"] == 100 and e["human_value"] == "00100"
    assert ledger.report()["fields"]["pot"]["consistent"] == 1


def test_glyph_export_reference_arguments_and_report_exposure(tmp_path):
    ledger = book(tmp_path)
    e = ledger.confirm(0, "action", 2, "muck")
    ledger.confirm(1, "action", 3, "fold")
    export = ledger.export("glyph")
    assert len(export["inputs"]) == 1
    assert export["inputs"][0]["label"] == "muck"
    assert export["inputs"][0]["slot"] == 2
    ledger.mark_trained(e["entry_sha256"])
    assert ledger.report()["trained_frames"] == [0]
