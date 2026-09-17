"""U2 regressions for the saved-analysis store.

The store freezes ONE actually-completed analysis and re-derives everything on
every read. These tests use a controllable fake of the analysis facade so the
server-side ownership rules (only the live COMPLETE job, never a client-reported
value) and every integrity refusal can be exercised without a spawned worker.
"""

from copy import deepcopy
import json

import pytest

from poker_engine.desktop import aa_analysis_records as module


RULES = {**json.load(open(
    "configs/strategy/examples/threeway-river-response-manual.json",
    encoding="utf-8"))["rules"], "rake_percent": "0", "rake_cap_bb": "0"}


def facts(**overrides):
    base = base_facts()
    base.update(overrides)
    return base


def base_facts():
    from poker_engine.desktop import aa_hand_input as hand

    base = hand.blank_facts()
    base["ended_hand_confirmed"] = hand.field(True, "human_confirmed")
    base["hero_seat"] = hand.field(0, "human_confirmed")
    base["hero_cards"] = hand.field(["Qs", "Qd"], "human_confirmed")
    base["board_cards"] = hand.field(["2c", "4d", "7h", "9s", "Jc"],
                                     "human_confirmed")
    base["action_order"] = hand.field([1, 2, 0], "human_confirmed")
    base["seats"] = hand.field([
        {"seat_id": 0, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 1, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 2, "stack": "200", "hand_committed": "20", "status": "ACTIVE"},
        {"seat_id": 3, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 4, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
        {"seat_id": 5, "stack": "200", "hand_committed": "10", "status": "FOLDED"},
    ], "human_confirmed")
    base["history"] = hand.field([
        {"actor": 1, "kind": "bet", "target": "20"},
        {"actor": 2, "kind": "call", "target": "0"}], "human_confirmed")
    base["pot_display"] = hand.field("130", "human_confirmed")
    base["table_rules"] = hand.field(dict(RULES), "human_confirmed")
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


def result_document():
    return {
        "status": "COMPLETE_CONDITIONAL_ABSTRACTION",
        "root_actions": [
            {"action": {"actor": 0, "kind": "fold", "target": "0"},
             "ev": {"exact": "0", "decimal": "0"}, "additional_cost": "0"},
            {"action": {"actor": 0, "kind": "call", "target": "0"},
             "ev": {"exact": "515/8", "decimal": "64.375"},
             "additional_cost": "20"},
            {"action": {"actor": 0, "kind": "raise", "target": "80"},
             "ev": {"exact": "3485/32", "decimal": "108.90625"},
             "additional_cost": "80"},
        ],
        "root_posterior": [{"exact": "1", "decimal": "1"}],
        "nodes": 12, "terminal_nodes": 4, "joint_assignments": 4,
    }


class FakeAnalysis:
    """Stands in for ``AAConditionalAnalysis``: the live report is server state."""

    def __init__(self):
        self.report = {"status": "IDLE", "job_id": None, "input_sha256": None,
                       "binding": {}, "result": None, "kind": None}

    def status(self):
        return deepcopy(self.report)

    def complete(self, document, *, job_id="job-1", rules_source="document",
                 revision="r1", effective_rules=None, result=None):
        self.report = {
            "status": "COMPLETE", "job_id": job_id, "kind": "threeway",
            "input_sha256": module.digest(document),
            "binding": {"generation": 1, "table_rules_revision": revision,
                        "rules_source": rules_source,
                        "effective_rules": deepcopy(
                            effective_rules if effective_rules is not None
                            else document["rules"])},
            "result": deepcopy(result if result is not None
                               else result_document()),
            "manual_notlive": True,
        }
        return self.report


def document_for(facts_block=None, assumptions_block=None):
    from poker_engine.desktop import aa_hand_input as hand

    return hand.build_document(facts_block or facts(),
                               assumptions_block or assumptions())[0]


@pytest.fixture
def store(tmp_path):
    analysis = FakeAnalysis()
    current = {"revision": "r1"}
    records = module.AAAnalysisRecordStore(
        tmp_path / "analysis-records", analysis=analysis,
        rules_revision=lambda: current["revision"])
    return StoreFixture(store=records, analysis=analysis, current=current)


class StoreFixture:
    """The store under test plus the two things it reads from the outside."""

    def __init__(self, *, store, analysis, current):
        self.store = store
        self.analysis = analysis
        self.current = current

    def save(self, **overrides):
        values = {"job_id": "job-1", "expected_input_sha256": None,
                  "facts": facts(), "assumptions": assumptions(),
                  "source": None, "label": None}
        values.update(overrides)
        return self.store.save(**values)


def test_a_completed_job_is_frozen_and_re_derived_on_reopen(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save(expected_input_sha256=store.analysis.report["input_sha256"],
                          label="合成输入一")
    assert envelope["display_permitted"] is True
    assert envelope["status"] == module.OK
    assert envelope["label"] == "合成输入一"
    view = envelope["view"]
    assert view["situation"]["hero_cards"] == ["Qs", "Qd"]
    assert [row["kind"] for row in view["actions"]] == ["fold", "call", "raise"]
    assert view["actions"][1]["ev"] == "64.375"
    assert view["actions"][1]["ev_exact"] == "515/8"
    assert view["actions"][2]["additional_cost"] == "80"
    assert view["to_call"] == "20" and view["root_pot"] == "130"
    assert view["rules"]["rules_revision"] == "r1"
    assert view["strategy_eligible"] is False and view["advice_emitted"] is False
    assert isinstance(view["history"], list) and len(view["history"]) == 2
    assert view["rubric"] == module.SOURCE_KIND_MANUAL
    # Reopening is a fresh read, and it really re-derives from disk.
    again = store.store.get(envelope["record_id"])
    assert again["view"] == view


def test_only_the_live_completed_job_can_be_saved(store):
    document = document_for()
    store.analysis.complete(document)
    store.analysis.report["status"] = "RUNNING"
    with pytest.raises(module.AnalysisRecordError, match="只有已完成"):
        store.save()
    for status in ("ERROR", "CANCELLED", "TIMED_OUT"):
        store.analysis.report["status"] = status
        with pytest.raises(module.AnalysisRecordError, match="只有已完成"):
            store.save()


def test_a_replaced_job_cannot_be_saved(store):
    document = document_for()
    store.analysis.complete(document, job_id="job-2")
    with pytest.raises(module.AnalysisRecordError, match="已被新的输入替换"):
        store.save(job_id="job-1")


def test_the_client_cannot_claim_a_different_input(store):
    document = document_for()
    store.analysis.complete(document)
    with pytest.raises(module.AnalysisRecordError, match="不是同一份"):
        store.save(expected_input_sha256="a" * 64)


def test_facts_that_do_not_rebuild_the_analysed_input_are_refused(store):
    """A pot that no longer reconciles cannot reproduce the analysed document."""
    document = document_for()
    store.analysis.complete(document)
    changed = facts(pot_display={"value": "999", "provenance": "human_confirmed",
                                 "candidate": None})
    with pytest.raises(module.AnalysisRecordError, match="保存被拒绝"):
        store.save(facts=changed)
    with pytest.raises(module.AnalysisRecordError, match="保存被拒绝"):
        store.save(facts=facts(hero_cards={"value": ["Qs", "Qd"],
                                           "provenance": "unknown",
                                           "candidate": None}))


def test_a_repeated_save_returns_the_same_record(store):
    document = document_for()
    store.analysis.complete(document)
    first = store.save()
    second = store.save()
    assert second["record_id"] == first["record_id"]
    assert second.get("duplicate_of_existing") is True
    folders = [path for path in store.store.directory.iterdir() if path.is_dir()]
    assert len(folders) == 1


def test_two_sources_with_identical_numbers_stay_separate(store):
    document = document_for()
    store.analysis.complete(document, job_id="job-a")
    first = store.save(job_id="job-a", source={"issue_id": None})
    store.analysis.complete(document, job_id="job-b")
    second = store.save(job_id="job-b", source={"issue_id": None})
    assert second["record_id"] != first["record_id"]
    assert second.get("duplicate_of_existing") is None
    assert len(store.store.recent()) == 2


def test_an_older_rules_revision_is_viewable_as_history(store):
    document = document_for()
    store.analysis.complete(document, revision="r1")
    envelope = store.save()
    assert envelope["display_permitted"] is True
    store.current["revision"] = "r2"
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is True
    assert reopened["status"] == module.HISTORICAL_RULES
    assert reopened["view"]["historical"] is True
    assert reopened["view"]["rules"]["current_rules_revision"] == "r2"
    assert any("历史结果" in note for note in reopened["notes"])


def test_an_older_implementation_version_is_viewable_as_history(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    store.store.implementation_version = "aa-analysis-record-v2"
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is True
    assert reopened["status"] == module.HISTORICAL_IMPLEMENTATION


def _tamper(store, record_id, mutate):
    path = store.store.directory / record_id / "record.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path.read_text(encoding="utf-8")


def test_an_edited_display_value_is_detected(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc["report"]["result"]["root_actions"][1]["ev"].update(
                decimal="99.999"))
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert reopened["view"] is None
    assert reopened["status"] == module.INVALID
    assert any("显示值" in note for note in reopened["notes"])


def test_an_edited_exact_value_is_detected(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc["report"]["result"]["root_actions"][1]["ev"].update(
                exact="999/2"))
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert any("显示值" in note or "exact" in note for note in reopened["notes"])


def test_an_edited_rules_identity_is_detected(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()

    def mutate(doc):
        doc["job"]["binding"]["effective_rules"]["rake_percent"] = "9"

    _tamper(store, envelope["record_id"], mutate)
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert any("规则" in note for note in reopened["notes"])


def test_an_edited_input_or_source_is_detected(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc["input"]["hero_cards"].__setitem__(0, "As"))
    edited_input = store.store.get(envelope["record_id"])
    assert edited_input["display_permitted"] is False
    assert any("规范输入" in note for note in edited_input["notes"])

    store.analysis.complete(document, job_id="job-2")
    other = store.save(job_id="job-2")
    _tamper(store, other["record_id"],
            lambda doc: doc["identity"].update(input_sha256="b" * 64))
    edited_identity = store.store.get(other["record_id"])
    assert edited_identity["display_permitted"] is False


def test_an_inconsistent_action_field_is_detected(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc["report"]["result"]["root_actions"][1]["action"].update(
                target="20"))
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert any("目标额" in note for note in reopened["notes"])


def test_a_corrupt_file_is_refused_and_preserved(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    path = store.store.directory / envelope["record_id"] / "record.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(module.AnalysisRecordError, match="合法 JSON"):
        store.store.get(envelope["record_id"])
    rows = store.store.recent()
    assert rows[0]["display_permitted"] is False
    assert rows[0]["status"] == module.INVALID
    assert path.read_text(encoding="utf-8") == "{not json"


def test_original_candidates_and_confirmed_empty_history_survive(store):
    """A human-filled history keeps its imported candidate; [] stays confirmed."""
    from poker_engine.desktop import aa_hand_input as hand

    imported = [{"actor": 1, "kind": "bet", "target": "20", "reading": "候选"}]
    filled = facts(history=hand.field(
        [{"actor": 1, "kind": "bet", "target": "20"},
         {"actor": 2, "kind": "call", "target": "0"}],
        "human_confirmed", candidate=imported))
    empty = facts(
        history=hand.field([], "human_confirmed"),
        action_order=hand.field([0, 1, 2], "human_confirmed"),
        pot_display=hand.field("90", "human_confirmed"))
    store.analysis.complete(document_for(filled))
    envelope = store.save(facts=filled)
    assert envelope["display_permitted"] is True
    assert envelope["view"]["facts"]["history"]["candidate"] == imported
    assert envelope["view"]["facts"]["history"]["provenance"] == "human_confirmed"

    # With no public action nobody has bet, so there is no call to reconcile.
    store.analysis.complete(
        document_for(empty), job_id="job-empty",
        result={"status": "COMPLETE_CONDITIONAL_ABSTRACTION",
                "root_actions": [
                    {"action": {"actor": 0, "kind": "check", "target": "0"},
                     "ev": {"exact": "0", "decimal": "0"},
                     "additional_cost": "0"},
                    {"action": {"actor": 0, "kind": "bet", "target": "20"},
                     "ev": {"exact": "515/8", "decimal": "64.375"},
                     "additional_cost": "20"}],
                "root_posterior": [{"exact": "1", "decimal": "1"}],
                "nodes": 5, "terminal_nodes": 2, "joint_assignments": 4})
    second = store.save(job_id="job-empty", facts=empty)
    assert second["display_permitted"] is True
    assert second["view"]["facts"]["history"]["value"] == []
    assert second["view"]["facts"]["history"]["provenance"] == "human_confirmed"
    # Nothing turned an unknown into a confirmed value on the way through.
    assert second["view"]["facts"]["hero_cards"]["provenance"] == "human_confirmed"


def test_the_scenario_for_recompute_never_touches_global_rules(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    store.current["revision"] = "r9"
    scenario = store.store.scenario(envelope["record_id"])
    assert scenario["saved_rules_revision"] == "r1"
    assert scenario["current_rules_revision"] == "r9"
    assert scenario["input"]["rules"] == document["rules"]
    assert scenario["facts"]["hero_cards"]["value"] == ["Qs", "Qd"]
    assert scenario["recompute_source"] == "saved_conditions"
    assert store.current["revision"] == "r9"   # the global rules are untouched


def test_a_refused_record_cannot_be_used_as_a_recompute_source(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc["report"]["result"]["root_actions"][1]["ev"].update(
                decimal="0.001"))
    with pytest.raises(module.AnalysisRecordError, match="自洽校验"):
        store.store.scenario(envelope["record_id"])


# ---------------------------------------------------------------------------
# U2-R1 / review C: the stored CONTENT - the server result, its source link and
# every amount - must be sealed and re-verified, not just the input hashes.
# Every case below keeps the identity digests untouched on purpose: they are
# exactly the edits the previous head still reported as CURRENT.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,keyword", [
    # EV exact and decimal rewritten TOGETHER: the pair still agrees with
    # itself, so only the sealed content can tell it is not the server result.
    (lambda doc: doc["report"]["result"]["root_actions"][2]["ev"].update(
        exact="999/2", decimal="499.5"), "封存"),
    # A raised size that is not on the declared grid.
    (lambda doc: doc["report"]["result"]["root_actions"][2]["action"].update(
        target="999"), "动作"),
    # A raise whose additional cost no longer matches the verified amount.
    (lambda doc: doc["report"]["result"]["root_actions"][2].update(
        additional_cost="999"), "追加"),
    # The source link changed while identity.source_issue_id stayed put.
    (lambda doc: doc["source"].update(issue_id="20260101T000000-aaaaaaaaaaaa"),
     "来源"),
    # The job id changed while identity.job_id stayed put.
    (lambda doc: doc["job"].update(job_id="job-somewhere-else"), "任务"),
    # A derived pot the frozen input cannot produce.
    (lambda doc: doc["capacity"].update(implied_pot="999999"), "底池"),
    # The same derived numbers live in three blocks; one block moving is a
    # disagreement between the sealed blocks themselves.
    (lambda doc: doc["support"].update(to_call="999"), "应付"),
    (lambda doc: doc["amounts"].update(current_bet="999"), "最高下注"),
    (lambda doc: doc["support"]["root_actions"][2].update(additional_chips="999"),
     "动作"),
    (lambda doc: doc["report"]["result"]["root_actions"][2]["action"].update(
        actor=3), "行动者"),
])
def test_a_rewritten_content_field_is_refused(store, mutate, keyword):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    assert envelope["display_permitted"] is True
    _tamper(store, envelope["record_id"], mutate)
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert reopened["view"] is None
    assert reopened["status"] == module.INVALID
    assert any(keyword in note for note in reopened["notes"]), reopened["notes"]


def test_a_record_id_that_does_not_match_its_directory_is_refused(store):
    """The folder names the record; the file must not rename itself."""
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    other = "20260101T000000-abcdefabcdef"
    _tamper(store, envelope["record_id"],
            lambda doc: doc.update(record_id=other))
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert reopened["view"] is None
    assert reopened["record_id"] == envelope["record_id"]
    assert any("编号" in note or "目录" in note for note in reopened["notes"])


def test_every_read_path_runs_the_same_verification(store):
    """get / recent / scenario / the idempotent lookup all agree."""
    document = document_for()
    store.analysis.complete(document, job_id="job-a")
    first = store.save(job_id="job-a", source={"issue_id": None})
    store.analysis.complete(document, job_id="job-b")
    second = store.save(job_id="job-b", source={"issue_id": None})
    _tamper(store, first["record_id"],
            lambda doc: doc["capacity"].update(implied_pot="999999"))

    opened = store.store.get(first["record_id"])
    assert opened["display_permitted"] is False
    rows = {row["record_id"]: row for row in store.store.recent()}
    assert rows[first["record_id"]]["display_permitted"] is False
    assert rows[first["record_id"]]["status"] == module.INVALID
    # A broken record must not take its neighbours down with it.
    assert rows[second["record_id"]]["display_permitted"] is True
    assert rows[second["record_id"]]["status"] == module.OK
    with pytest.raises(module.AnalysisRecordError):
        store.store.scenario(first["record_id"])
    # Re-saving the same job must never hand back the broken file as an
    # already-saved duplicate; the lookup runs the same check as a read.
    with pytest.raises(module.AnalysisRecordError):
        store.save(job_id="job-a", source={"issue_id": None})
    assert len([path for path in store.store.directory.iterdir()
                if path.is_dir()]) == 2


def test_a_legacy_record_is_not_silently_accepted(store):
    """An older file without the content seal is named, not silently approved."""
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    _tamper(store, envelope["record_id"],
            lambda doc: doc.pop("content_seal", None))
    reopened = store.store.get(envelope["record_id"])
    assert reopened["display_permitted"] is False
    assert reopened["view"] is None
    assert reopened["status"] == module.UNVERIFIED_FORMAT
    assert any("旧格式" in note or "封存" in note for note in reopened["notes"])
    path = store.store.directory / envelope["record_id"] / "record.json"
    assert "content_seal" not in path.read_text(encoding="utf-8")
    with pytest.raises(module.AnalysisRecordError):
        store.store.scenario(envelope["record_id"])


def test_the_saved_content_is_sealed_with_an_explicit_schema(store):
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    path = store.store.directory / envelope["record_id"] / "record.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    seal = stored["content_seal"]
    assert seal["schema"] == module.CONTENT_SEAL_SCHEMA
    assert module.HEX64.fullmatch(seal["sha256"])
    # The seal covers the whole frozen content, so any field is inside it.
    assert seal["sha256"] == module.digest(
        {key: value for key, value in stored.items() if key != "content_seal"})


def test_the_source_kind_is_explicit_and_defaults_to_unverified(store):
    """No evidence means a manual unverified input, never a claimed real hand."""
    document = document_for()
    store.analysis.complete(document, job_id="job-plain")
    plain = store.save(job_id="job-plain", source=None)
    assert plain["view"]["rubric"] == module.SOURCE_KIND_MANUAL
    assert plain["view"]["source_kind"] == module.SOURCE_KIND_MANUAL

    store.analysis.complete(document, job_id="job-linked")
    linked = store.save(job_id="job-linked", source={"issue_id": "issue-7"})
    assert linked["view"]["source_kind"] == module.SOURCE_KIND_LINKED

    store.analysis.complete(document, job_id="job-child")
    child = store.save(job_id="job-child", source={
        "issue_id": "issue-7", "parent_analysis_record_id": plain["record_id"]})
    assert child["view"]["source_kind"] == module.SOURCE_KIND_RECOMPUTED
    assert child["view"]["source"]["parent_analysis_record_id"] == (
        plain["record_id"])


def test_the_view_makes_the_assumptions_and_rule_values_readable(store):
    """The detail must show what was assumed, not only revision digests."""
    document = document_for()
    store.analysis.complete(document)
    envelope = store.save()
    view = envelope["view"]
    assert view["rubric"] and view["rubric"] != "linked_to_review_record"
    ranges = {item["seat_id"]: item["combos"]
              for item in view["assumptions"]["ranges"]}
    assert ranges[1][0] == {"combo": "JhJd", "weight": "1"}
    models = {(row["seat_id"], row["key"]): row["weight"]
              for row in view["assumptions"]["models"]}
    assert models[(1, "bet")] == "2"
    assert view["assumptions"]["aggression_targets"] == ["20", "40", "80"]
    assert view["rules"]["effective_rules"]["rake_percent"] == "0"
    assert view["rules"]["rules_source"] in ("document", "table")
    assert view["facts"]["hero_cards"]["value"] == ["Qs", "Qd"]


# ---------------------------------------------------------------------------
# U2-R2 / review B: a record whose file is VALID JSON but the wrong SHAPE must
# be refused on its own - the list must never answer with an unhandled
# exception, and the healthy neighbours must stay readable.
# ---------------------------------------------------------------------------


def _write_bad(store, record_id, payload):
    folder = store.store.directory / record_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "record.json").write_text(payload, encoding="utf-8")
    return folder / "record.json"


BAD_SHAPE_ID = "20260101T000000-abcdefabcdef"


@pytest.mark.parametrize("payload", ["[]", "null", '"not-an-object"', "123",
                                     "true", "[]\n"])
def test_a_non_object_record_is_named_and_does_not_break_the_list(store, payload):
    """The pre-fix behaviour was AttributeError straight through `recent`."""
    document = document_for()
    store.analysis.complete(document, job_id="job-good-1")
    healthy = store.save(job_id="job-good-1", source={"issue_id": None})
    store.analysis.complete(document, job_id="job-good-2")
    neighbour = store.save(job_id="job-good-2", source={"issue_id": None})
    path = _write_bad(store, BAD_SHAPE_ID, payload)

    rows = store.store.recent()
    by_id = {row["record_id"]: row for row in rows}
    assert len(rows) == 3, [row["record_id"] for row in rows]
    bad = by_id[BAD_SHAPE_ID]
    assert bad["display_permitted"] is False
    assert bad["status"] == module.INVALID
    assert bad["notes"] and any("对象" in note for note in bad["notes"]), bad["notes"]
    # The healthy records are untouched by the broken neighbour.
    assert by_id[healthy["record_id"]]["display_permitted"] is True
    assert by_id[healthy["record_id"]]["status"] == module.OK
    assert by_id[neighbour["record_id"]]["display_permitted"] is True
    # A single get / scenario is refused as a record error, not an AttributeError.
    for call in (lambda: store.store.get(BAD_SHAPE_ID),
                 lambda: store.store.scenario(BAD_SHAPE_ID)):
        with pytest.raises(module.AnalysisRecordError):
            call()
    assert path.read_text(encoding="utf-8") == payload


@pytest.mark.parametrize("mutate,keyword", [
    (lambda doc: doc.__setitem__("identity", "not-an-object"), "身份"),
    (lambda doc: doc.__setitem__("job", []), "任务"),
    (lambda doc: doc.__setitem__("source", []), "来源"),
    (lambda doc: doc.__setitem__("input", "not-an-object"), "规范输入"),
    (lambda doc: doc.__setitem__("facts", []), "事实"),
    (lambda doc: doc.__setitem__("assumptions", "not-an-object"), "假设"),
    (lambda doc: doc.__setitem__("report", []), "报告"),
    (lambda doc: doc.__setitem__("support", "not-an-object"), "支持"),
    (lambda doc: doc.__setitem__("amounts", []), "金额"),
    (lambda doc: doc.__setitem__("capacity", "not-an-object"), "容量"),
])
def test_a_nested_shape_error_yields_invalid_without_raising(store, mutate, keyword):
    """A wrong nested block must be a named refusal, never a traceback."""
    document = document_for()
    store.analysis.complete(document, job_id="job-nested-1")
    victim = store.save(job_id="job-nested-1", source={"issue_id": None})
    store.analysis.complete(document, job_id="job-nested-2")
    neighbour = store.save(job_id="job-nested-2", source={"issue_id": None})
    _tamper(store, victim["record_id"], mutate)

    rows = {row["record_id"]: row for row in store.store.recent()}
    victim_row = rows[victim["record_id"]]
    assert victim_row["display_permitted"] is False
    assert victim_row["status"] == module.INVALID
    assert any(keyword in note for note in victim_row["notes"]), victim_row["notes"]
    reopened = store.store.get(victim["record_id"])
    assert reopened["view"] is None and reopened["display_permitted"] is False
    assert rows[neighbour["record_id"]]["display_permitted"] is True
    with pytest.raises(module.AnalysisRecordError):
        store.store.scenario(victim["record_id"])


def test_the_list_keeps_every_healthy_record_while_a_bad_file_sits_between(store):
    """A junk file must not be able to hide the records around it."""
    document = document_for()
    ids = []
    for index in range(2):
        store.analysis.complete(document, job_id=f"job-keep-{index}")
        ids.append(store.save(job_id=f"job-keep-{index}",
                              source={"issue_id": None})["record_id"])
    _write_bad(store, "20250101T000000-000000000001", "[]")
    _write_bad(store, "20250101T000000-000000000002", "null")
    (store.store.directory / "20250101T000000-000000000003").mkdir()
    (store.store.directory / "20250101T000000-000000000003" / "record.json").write_text(
        "{not json", encoding="utf-8")

    rows = store.store.recent()
    assert len(rows) == 5, [row["record_id"] for row in rows]
    healthy = [row for row in rows if row["display_permitted"]]
    assert sorted(row["record_id"] for row in healthy) == sorted(ids)
    assert all(row["status"] == module.OK for row in healthy)
    assert all(row["status"] == module.INVALID
               for row in rows if not row["display_permitted"])
    for record_id in ids:
        assert store.store.get(record_id)["display_permitted"] is True
