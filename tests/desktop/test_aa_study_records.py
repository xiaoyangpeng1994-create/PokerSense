"""Offline study result view: controlled synthetic example, records, provenance.

The study view must never attach a synthetic study to an observed hand, never
read a path taken from a report, and never render a world that failed its own
reconciliation as a valid comparison.
"""

from copy import deepcopy
from fractions import Fraction
import ast
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from poker_engine.desktop import aa_server
from poker_engine.desktop import aa_study_records as module
from poker_engine.desktop.aa_study_records import (
    AAStudyRecordStore, STUDY_ID, SUPPORTED_EXAMPLES, StudyRecordError,
    default_examples_root,
)


HEADERS = {"X-AA-Live": "1"}
EXAMPLE = "range-sensitivity-synthetic-v1"
FILES = ("report", "protocol", "input", "study_protocol")
ISSUE_LIKE = "20260916T000000-aaaaaaaaaaaa"


class RecordingSession:
    """Records every session attribute the study view touches (shutdown aside)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        if name == "stop":
            return lambda *args, **kwargs: None
        raise AssertionError(f"study view touched the AA session: {name}")


def registry(tmp_path, mutate=None, stale_hash=False):
    """Copy the registered files into a temporary registry, optionally mutated."""
    source = default_examples_root()
    target = tmp_path / "examples"
    target.mkdir()
    entry = dict(SUPPORTED_EXAMPLES[0])
    for key in FILES:
        (target / entry[key]).write_bytes((source / entry[key]).read_bytes())
    if mutate is not None:
        path = target / entry["report"]
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
        if not stale_hash:
            entry["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return target, entry


def store(tmp_path, **kwargs):
    root, entry = registry(tmp_path, **kwargs)
    return AAStudyRecordStore(tmp_path / "records", examples_root=root,
                              examples=(entry,)), root, entry


def tamper_path_row(document):
    row = document["worlds"][0]["cross_book_branch_difference"]["rows"][0]
    row["contribution_chips_difference"] = "999"


def tamper_probability_sum(document):
    document["worlds"][0]["books"]["training_selected"]["trace"][
        "reach_probability_sum"] = "1/2"


def block_a_world(document):
    document["worlds"][1]["status"] = "BLOCKED"
    document["worlds"][1]["reasons"] = ["trace_node_budget_exceeded"]


def unknown_schema(document):
    document["schema_version"] = 2


def extra_key(document):
    document["unexpected"] = True


def empty_identity(document):
    for item in document["frozen_books"]:
        item["policy_book_sha256_before"] = ""
        item["policy_book_sha256_after"] = ""


def empty_pair(document):
    for world in document["worlds"]:
        for book in world["books"].values():
            book["policy_hash_before"] = ""
            book["policy_hash_after"] = ""


def illegal_path(document):
    document["protocol"]["path"] = "../../../etc/passwd"


def baseline_mismatch(document):
    document["baseline_verification"]["matches_original_reading"] = False


def drop_a_path_row(document):
    document["worlds"][0]["books"]["training_selected"]["reconciliations"][0][
        "rows"] = []


def app(tmp_path, **kwargs):
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    return aa_server.create_app(tmp_path / "profile.json", records_dir=records,
                                **kwargs)


def test_example_flow_saves_and_reopens_with_identical_identity(tmp_path):
    client_app = app(tmp_path)
    with TestClient(client_app) as client:
        examples = client.get("/api/study/examples").json()
        assert [item["example_id"] for item in examples["items"]] == [EXAMPLE]
        assert examples["items"][0]["source_type"] == "SYNTHETIC_STUDY_EXAMPLE"
        assert examples["items"][0]["available"] is True
        assert examples["strategy_eligible"] is False

        preview = client.get(f"/api/study/examples/{EXAMPLE}/view").json()
        assert preview["content_status"] == "PREVIEW_NOT_SAVED"
        assert preview["record_id"] is None
        assert preview["identity"]["bound_record_id"] is None
        assert client.get("/api/study/records").json()["items"] == []

        saved = client.post("/api/study/records", json={"example_id": EXAMPLE},
                            headers=HEADERS).json()
        assert STUDY_ID.fullmatch(saved["record_id"])
        assert saved["content_status"] == "CURRENT"
        assert saved["source_type"] == "SYNTHETIC_STUDY_EXAMPLE"

        listed = client.get("/api/study/records").json()["items"]
        assert [item["record_id"] for item in listed] == [saved["record_id"]]

        reopened = client.get(f"/api/study/records/{saved['record_id']}").json()
        assert reopened["view"] == saved["view"]
        assert reopened["identity"] == saved["identity"]
        assert reopened["content_status"] == "CURRENT"


def test_example_numbers_match_the_frozen_report(tmp_path):
    client_app = app(tmp_path)
    with TestClient(client_app) as client:
        view = client.get(f"/api/study/examples/{EXAMPLE}/view").json()["view"]
    factors = {item["factor"]: item for item in view["factors"]}
    assert sorted(factors) == ["0.5", "1", "2"]
    assert factors["1"]["is_baseline"] is True
    assert factors["1"]["deltas"]["vs_manual_reference"] == {
        "exact": "-30444/189457", "display": "-0.1607", "unit": "chips"}
    assert factors["0.5"]["deltas"]["vs_manual_reference"]["exact"] == "353982/17719"
    assert factors["2"]["deltas"]["vs_manual_reference"]["exact"] == "-5755044/284867"
    assert [item["name"] for item in factors["1"]["outcomes"]] == [
        "frozen_policy", "check_fold", "check_call"]
    assert factors["1"]["outcomes"][0]["net_ev_chips"]["exact"] == "737057752/20650813"
    assert factors["1"]["reference_frozen_policy"]["exact"] == "543196/15151"
    assert len(factors["1"]["paths"]) == 5
    assert all(row["difference"]["unit"] == "chips" for row in factors["1"]["paths"])
    assert all(Fraction(row["difference"]["exact"]) != 0
               for row in factors["1"]["paths"])
    assert factors["1"]["failure_gates"] == [
        "selected_minus_manual_reference", "selected_delta_vs_check_call"]
    assert factors["0.5"]["failure_gates"] == []
    assert view["source"]["source_type"] == "SYNTHETIC_STUDY_EXAMPLE"
    assert view["source"]["scope_label"].startswith("合成研究示例")
    expected_decision = {"strategy_eligible": False, "advice_emitted": False,
                         "live_advice": False}
    assert view["decision"] == expected_decision
    assert len(view["missing"]) >= 4 and len(view["claims"]) >= 5
    assert any("不是本桌" in item for item in view["claims"])


@pytest.mark.parametrize("mutate", [tamper_path_row, tamper_probability_sum,
                                    block_a_world, unknown_schema, extra_key,
                                    empty_identity, empty_pair, illegal_path,
                                    baseline_mismatch, drop_a_path_row])
def test_invalid_reports_never_become_a_result(tmp_path, mutate):
    instance, _, _ = store(tmp_path, mutate=mutate)
    with pytest.raises(StudyRecordError):
        instance.view(EXAMPLE)
    with pytest.raises(StudyRecordError):
        instance.save(EXAMPLE)


def test_empty_policy_identity_is_not_evidence_of_an_unchanged_policy(tmp_path):
    """Two equal empty strings must not read as a verified, unchanged policy."""
    instance, _, _ = store(tmp_path, mutate=empty_pair)
    with pytest.raises(StudyRecordError) as failure:
        instance.view(EXAMPLE)
    message = str(failure.value)
    assert "身份" in message or "保持不变" in message


def test_source_hash_mismatch_is_rejected_without_reading_report_paths(tmp_path):
    instance, _, _ = store(tmp_path, mutate=tamper_path_row, stale_hash=True)
    with pytest.raises(StudyRecordError) as failure:
        instance.view(EXAMPLE)
    assert "哈希" in str(failure.value)


def test_report_declared_path_strings_are_compared_but_never_opened(tmp_path):
    instance, root, _ = store(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    document = json.loads((root / "strategy-diag-c-range-experiment-v1.json").read_text(
        encoding="utf-8"))
    document["input"]["path"] = str(outside)
    (root / "strategy-diag-c-range-experiment-v1.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    # The registry hash now matches the tampered report, so validation proceeds
    # to the declared path and rejects it instead of opening it.
    rebuilt = AAStudyRecordStore(
        tmp_path / "records", examples_root=root, examples=({
            **SUPPORTED_EXAMPLES[0],
            "report_sha256": hashlib.sha256(
                (root / "strategy-diag-c-range-experiment-v1.json").read_bytes()
            ).hexdigest()},))
    with pytest.raises(StudyRecordError) as failure:
        rebuilt.view(EXAMPLE)
    assert "路径" in str(failure.value) or "文件" in str(failure.value)
    assert outside.read_text(encoding="utf-8") == "{}"


def test_superseded_source_keeps_the_file_but_never_shows_it_as_current(tmp_path):
    """A replaced source cannot be re-checked, so the record is unverified history."""
    instance, root, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    report = root / "strategy-diag-c-range-experiment-v1.json"
    report.write_bytes(report.read_bytes() + b"\n")
    reopened = instance.get(saved["record_id"])
    assert reopened["content_status"] == "HISTORICAL_UNVERIFIED"
    assert reopened["display_permitted"] is False
    assert reopened["view"] is None
    assert "历史" in reopened["reason"]
    # The file itself is preserved: the saved bytes are still on disk.
    assert saved["view"] == json.loads((tmp_path / "records" / "study-records"
                                        / saved["record_id"] / "record.json")
                                       .read_text(encoding="utf-8"))["view"]


def test_records_are_independent_and_never_join_the_frame_namespace(tmp_path):
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    with pytest.raises(StudyRecordError):
        instance.get(ISSUE_LIKE)
    root = tmp_path / "records" / "study-records"
    assert sorted(path.name for path in root.iterdir()) == [saved["record_id"]]
    stored = json.loads((root / saved["record_id"] / "record.json").read_text(
        encoding="utf-8"))
    assert stored["identity"]["bound_record_id"] is None
    assert stored["identity"]["source_type"] == "SYNTHETIC_STUDY_EXAMPLE"
    assert not list((tmp_path / "records").glob("*.json"))


def test_record_files_must_stay_inside_the_supported_structure(tmp_path):
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    folder = tmp_path / "records" / "study-records" / saved["record_id"]
    document = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    document["unexpected"] = 1
    (folder / "record.json").write_text(json.dumps(document, ensure_ascii=False),
                                        encoding="utf-8", newline="\n")
    opened = instance.get(saved["record_id"])
    assert opened["content_status"] == "INVALID"
    assert opened["view"] is None and opened["display_permitted"] is False
    assert opened["reason"]
    assert instance.recent()["items"][0]["content_status"] == "INVALID"


def test_oversized_and_duplicate_key_records_are_rejected(tmp_path):
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    folder = tmp_path / "records" / "study-records" / saved["record_id"]
    (folder / "record.json").write_text("x" * (module.MAX_REPORT_BYTES + 1),
                                        encoding="utf-8", newline="\n")
    oversized = instance.get(saved["record_id"])
    assert oversized["content_status"] == "INVALID" and oversized["view"] is None
    (folder / "record.json").write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8",
        newline="\n")
    duplicated = instance.get(saved["record_id"])
    assert duplicated["content_status"] == "INVALID"
    assert duplicated["view"] is None


def test_unregistered_example_and_missing_directory_are_rejected(tmp_path):
    instance, _, _ = store(tmp_path)
    with pytest.raises(StudyRecordError):
        instance.view("not-registered")
    with pytest.raises(StudyRecordError):
        instance.save("not-registered")
    bare = AAStudyRecordStore(None, examples_root=tmp_path)
    with pytest.raises(StudyRecordError):
        bare.save(EXAMPLE)


def test_study_api_rejects_unknown_example_and_foreign_record_ids(tmp_path):
    with TestClient(app(tmp_path)) as client:
        assert client.get("/api/study/examples/unknown/view").status_code == 400
        assert client.get(f"/api/study/records/{ISSUE_LIKE}").status_code == 400
        assert client.post("/api/study/records", json={"example_id": "unknown"},
                           headers=HEADERS).status_code == 400
        assert client.post("/api/study/records", json={}, headers=HEADERS
                           ).status_code == 400
        assert client.post("/api/study/records", json={"example_id": EXAMPLE}
                           ).status_code == 403


def test_study_flow_never_touches_the_session(tmp_path):
    """Every study call is served without touching AA capture or recognition."""
    session = RecordingSession()
    client_app = aa_server.create_app(tmp_path / "profile.json", session=session,
                                      records_dir=tmp_path / "records")
    with TestClient(client_app) as client:
        assert client.get("/api/study/examples").status_code == 200
        assert client.get(f"/api/study/examples/{EXAMPLE}/view").status_code == 200
        saved = client.post("/api/study/records", json={"example_id": EXAMPLE},
                            headers=HEADERS)
        assert saved.status_code == 200
        assert client.get("/api/study/records").status_code == 200
        assert client.get(
            f"/api/study/records/{saved.json()['record_id']}").status_code == 200
        assert client.get("/study.js").status_code == 200
    # Only the application shutdown touches the session, never the study flow.
    assert session.calls == ["stop"]


def test_study_module_has_no_vision_capture_or_network_dependency():
    """Structural guard: the view is pure JSON plus the repository files."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    banned = {"cv2", "requests", "urllib", "socket", "subprocess", "httpx",
              "aa_reader", "aa_sources", "aa_session", "aa_review"}
    assert not [name for name in imported if name.split(".")[0] in banned]
    assert not hasattr(module, "AA8Reader")
    assert not hasattr(module, "source_factory")


def test_view_model_is_plain_json_and_display_values_are_read_only(tmp_path):
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    raw = json.dumps(saved, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in raw and "Infinity" not in raw
    factor = saved["view"]["factors"][1]
    assert factor["deltas"]["vs_manual_reference"]["display"] == "-0.1607"
    assert factor["deltas"]["vs_manual_reference"]["exact"] == "-30444/189457"
    expected = Fraction(factor["deltas"]["vs_manual_reference"]["exact"])
    shown = factor["deltas"]["vs_manual_reference"]["display"]
    assert f"{float(expected):.4f}" == shown
    assert deepcopy(saved) == saved


def record_path(tmp_path, record_id):
    return (tmp_path / "records" / "study-records" / record_id / "record.json")


def rewrite_record(tmp_path, record_id, mutate):
    path = record_path(tmp_path, record_id)
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    return document


def view_is_displayable(opened):
    """A record is only usable when the server vouches for a complete view."""
    return (opened.get("content_status") == "CURRENT"
            and opened.get("display_permitted") is True
            and isinstance(opened.get("view"), dict))


def test_reopening_a_saved_record_revalidates_the_saved_content(tmp_path):
    """A saved view must be re-derived, never returned on the record's word."""
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    reopened = instance.get(saved["record_id"])
    assert view_is_displayable(reopened)
    assert reopened["view"] == saved["view"]
    assert reopened["identity"] == saved["identity"]


def tamper_display(document):
    document["view"]["factors"][1]["deltas"]["vs_manual_reference"][
        "display"] = "999999.0000"


def tamper_exact(document):
    document["view"]["factors"][1]["deltas"]["vs_manual_reference"][
        "exact"] = "1/1000"


def tamper_exact_and_display(document):
    document["view"]["factors"][1]["deltas"]["vs_manual_reference"].update(
        {"exact": "1/1000", "display": "0.0010"})


def tamper_path_contribution(document):
    document["view"]["factors"][1]["paths"][0]["difference"]["exact"] = "1/2"


def tamper_identity_binding(document):
    document["identity"]["bound_record_id"] = ISSUE_LIKE
    document["identity"]["source_type"] = "OBSERVED_HAND"
    document["identity"]["policy_book_sha256"] = {}


def tamper_example(document):
    document["identity"]["example_id"] = "not-registered"


def tamper_nested_decision(document):
    document["view"]["decision"]["strategy_eligible"] = True


def tamper_top_level_false(document):
    document["advice_emitted"] = True


def tamper_nested_structure(document):
    document["view"]["factors"][1]["outcomes"] = []


def tamper_factor_count(document):
    document["view"]["factors"] = document["view"]["factors"][:1]


def tamper_source_hash(document):
    document["identity"]["input_sha256"] = "0" * 64


def tamper_world_hash(document):
    document["identity"]["world_scenario_sha256"] = ["0" * 64]


@pytest.mark.parametrize("mutate", [
    tamper_display, tamper_exact, tamper_exact_and_display,
    tamper_path_contribution, tamper_identity_binding, tamper_example,
    tamper_nested_decision, tamper_top_level_false, tamper_nested_structure,
    tamper_factor_count, tamper_source_hash, tamper_world_hash,
])
def test_a_tampered_record_is_never_displayed_as_a_valid_result(tmp_path, mutate):
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    rewrite_record(tmp_path, saved["record_id"], mutate)
    opened = instance.get(saved["record_id"])
    assert not view_is_displayable(opened), opened.get("content_status")
    assert opened["view"] is None
    assert opened["content_status"] in ("INVALID", "HISTORICAL_UNVERIFIED")
    assert opened["reason"]
    listed = next(item for item in instance.recent()["items"]
                  if item["record_id"] == saved["record_id"])
    assert listed["content_status"] in ("INVALID", "HISTORICAL_UNVERIFIED")


def test_untampered_record_round_trips_through_disk(tmp_path):
    """Positive control: the same fixture with no mutation stays displayable."""
    instance, _, _ = store(tmp_path)
    saved = instance.save(EXAMPLE)
    document = json.loads(record_path(tmp_path, saved["record_id"]).read_text(
        encoding="utf-8"))
    assert document["view"] == saved["view"]
    assert document["identity"] == saved["identity"]
    assert view_is_displayable(instance.get(saved["record_id"]))


def test_study_view_carries_the_minimal_experiment_context(tmp_path):
    """The page must show the cards, the history and both fixed root actions."""
    instance, _, _ = store(tmp_path)
    view = instance.view(EXAMPLE)["view"]
    context = view["table_context"]
    assert context["source_type"] == "SYNTHETIC_STUDY_EXAMPLE"
    assert context["hero_cards"] == ["Qs", "Qd"]
    assert context["board_cards"] == ["2c", "4d", "7h", "9s", "Jc"]
    assert [item["kind"] for item in context["history"]] == ["bet", "call"]
    assert context["unit"] == "chips"
    assert context["pot_at_decision"] == {
        "exact": "130", "display": "130.0000", "unit": "chips"}
    assert context["pot_components"]["committed_before_street"] == "90"
    assert context["pot_components"]["street_wagers_from_history"] == "40"
    assert context["to_call"] == {"exact": "20", "display": "20.0000",
                                  "unit": "chips"}
    actions = {item["book"]: item for item in context["root_actions"]}
    assert sorted(actions) == ["manual_reference", "training_selected"]
    assert actions["training_selected"]["action"]["kind"] == "raise"
    assert actions["training_selected"]["action"]["target"] == "80"
    assert actions["training_selected"]["raise_to"] == "80"
    assert actions["training_selected"]["additional_chips"] == "80"
    assert actions["manual_reference"]["action"]["kind"] == "call"
    assert actions["manual_reference"]["additional_chips"] == "20"
    assert all(item["synthetic_assumption"] is True
               for item in context["root_actions"])
    saved = instance.save(EXAMPLE)
    reopened = instance.get(saved["record_id"])
    assert reopened["view"]["table_context"] == context
