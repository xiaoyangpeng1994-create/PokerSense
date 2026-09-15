import copy

import pytest
from fastapi.testclient import TestClient

from tools.aa_replay_viewer import create_app, load_replay, present


def row():
    return dict(source_frame=0, opencv_pos_msec=0, source="a" * 64,
                gap_reset=True, scene="AA_TABLE_CANDIDATE", hero=None,
                board_slots=[None] * 5, stacks={str(i): None for i in range(9)},
                actions={str(i): None for i in range(9)}, hero_participation="UNKNOWN",
                strategy_eligible=False, incomplete_fields=[
                    "pot", "seat_presence", "full_actions", "special_modes",
                    "participation", "acceptance"])


def test_declarations_derive_missing_fields_and_never_hide_new_values():
    value = row()
    value.update(pot=100, current_actor=2)
    value["incomplete_fields"].remove("pot")
    result = present(value)
    fields = {f["name"]: f for f in result["fields"]}
    assert fields["pot"]["value"] == 100
    assert fields["pot"]["status"] == "候选值（未验收）"
    assert fields["current_actor"]["value"] == 2
    for key in ("full_actions", "special_modes", "acceptance"):
        assert fields[key]["status"] == "未实现"
        assert fields[key]["value"] is None
    assert "未实现" in result["participation"]
    assert result["strategy_eligible"] is False
    assert "重置" in result["invalidity"][0]
    value["incomplete_fields"].append("pot")
    conflict = present(value)
    assert conflict["contract_errors"] == ["INCOMPLETE_FIELDS_VALUE_CONFLICT:pot"]
    assert next(f for f in conflict["fields"] if f["name"] == "pot")["value"] == 100


def test_frame_change_clears_unknown_not_last_value():
    first = row()
    first["stacks"]["0"] = "100"
    second = row()
    second["source_frame"] = 1
    client = TestClient(create_app([first, second]))
    assert next(f for f in client.get("/api/frame/0").json()["fields"]
                if f["name"] == "stacks[0]")["value"] == "100"
    current = next(f for f in client.get("/api/frame/1").json()["fields"]
                   if f["name"] == "stacks[0]")
    assert current["value"] is None
    assert "未记录" in current["reason"]
    assert "未记录" in current["confidence"]
    assert client.get("/api/frame/-1").status_code == 404
    assert client.get("/api/frame/2").status_code == 404


def test_only_local_viewer_routes_no_capture_or_arbitrary_files():
    client = TestClient(create_app([row()]))
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/capture").status_code == 404
    assert client.get("/api/frame/not-a-number").status_code == 422
    assert client.post("/api/frame/0").status_code == 403


def test_hash_mismatch_rejected_before_parsing(tmp_path):
    path = tmp_path / "observations.jsonl"
    path.write_text("not json")
    with pytest.raises(ValueError, match="REPLAY_SHA256_MISMATCH"):
        load_replay(path)


def test_presentation_does_not_mutate_input():
    value = row()
    before = copy.deepcopy(value)
    present(value)
    assert value == before
