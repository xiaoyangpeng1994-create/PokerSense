import copy
import json

import pytest

from tools.aa8_acceptance import FIELDS, SCENARIOS, digest, evaluate, template


def bind(tmp_path, manifest, kind, data):
    path = tmp_path / (kind + ".json")
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest["artifacts"][kind] = [{"path": path.name, "sha256": digest(path)}]


def fixture(tmp_path):
    value = template()
    value.update(freeze_id="frozen-test", frozen_before_predictions=True,
                 development_sessions=["dev"], hardware_session="new-hardware")
    value["policy"] = {"hardware_min_seconds": 1000,
                       "require_independent_hardware_session": True}
    value["development_intervals"] = [
        {"session": "dev", "first_frame": 0, "last_frame": 3}]
    value["hands"] = [{"id": "h", "session": "dev", "role": "holdout",
                       "complete": True, "labels_reviewed": True,
                       "tuning_exposed": False, "first_frame": 4,
                       "last_frame": 5, "action_opportunity_frames": [4, 5],
                       "opportunities_reviewed": True}]
    for kind in ("implementation", "model", "parameters"):
        bind(tmp_path, value, kind, {"test_only": True})
    training = tmp_path / "training.json"
    training.write_text("[]", encoding="utf-8")
    entries = [{"kind": kind, **value["artifacts"][kind][0]}
               for kind in ("implementation", "model", "parameters")]
    entries.append({"kind": "training", "path": training.name,
                    "sha256": digest(training)})
    frozen = {"id": "frozen-test", "frozen_at_utc": "2020-01-01T00:00:00Z",
              "files": sorted(entries, key=lambda row: row["path"]),
              "exposures": [{"used_for": "training", "role": "development",
                             "artifact_sha256": digest(training)}]}
    frozen_path = tmp_path / "freeze.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    value["freeze_manifest"] = {"path": frozen_path.name,
                                "sha256": digest(frozen_path)}
    value["prediction_started_at_utc"] = "2020-01-01T00:00:01Z"
    bind(tmp_path, value, "dataset", value["hands"])
    values = {"actor": 0, "street_wagers": dict.fromkeys(map(str, range(8)), 0),
              "stacks": dict.fromkeys(map(str, range(8)), 100), "actions": [],
              "pot": 0, "street": "flop", "hand": "h",
              "hero_cards": ["5d", "6d"], "board_cards": ["5h", "6c", "6s"],
              "participation": dict.fromkeys(map(str, range(8)), "active"),
              "insurance": {"active": False}, "mushroom": {"active": False},
              "bomb": {"active": False}}
    rows = [{"hand": "h", "frame": frame, "scenarios": list(SCENARIOS),
             "decision_opportunity": True,
             "fields": {field: {"status": "KNOWN", "value": values[field],
                                "origin": "automatic"} for field in FIELDS}}
            for frame in (4, 5)]
    for mode in ("insurance", "mushroom", "bomb"):
        rows[0]["fields"][mode]["value"] = {"active": True}
    bind(tmp_path, value, "labels", rows)
    bind(tmp_path, value, "predictions", rows)
    hardware = {"session": "new-hardware", "freeze_id": "frozen-test",
                "duration_seconds": 1800, "unexplained_gaps": 0,
                "undetected_stale_frames": 0, "crashes": 0}
    hardware.update({key: True for key in (
        "user_authorized", "target_device", "end_to_end", "freshness_checked",
        "reconnect_tested", "recovery_verified", "latency_budget_met",
        "no_strategy_or_game_actions")})
    bind(tmp_path, value, "hardware", [hardware])
    return value


def edit_artifact(tmp_path, manifest, kind, edit):
    path = tmp_path / (kind + ".json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    edit(rows)
    bind(tmp_path, manifest, kind, rows)


def test_empty_template_never_passes(tmp_path):
    result = evaluate(template(), tmp_path)
    assert result["status"] == "PARTIAL"
    assert not result["full_visual_acceptance"]
    assert "no_complete_holdout_hands" in result["failures"]


def test_valid_synthetic_contract_not_real_acceptance(tmp_path):
    result = evaluate(fixture(tmp_path), tmp_path)
    assert result["status"] == "PASS"
    assert not result["strategy_eligible"]
    assert result["coverage"]["actor"]["matched"] == 2


@pytest.mark.parametrize("mutation", [
    {"status": "UNKNOWN"}, {"origin": "manual"}, {"value": None},
    {"value": "wrong"}])
def test_unknown_manual_and_mismatch_are_not_passes(tmp_path, mutation):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, "predictions",
                  lambda rows: rows[0]["fields"]["actor"].update(mutation))
    result = evaluate(manifest, tmp_path)
    assert "field_not_complete:actor" in result["failures"]


def test_hash_tampering_fails(tmp_path):
    manifest = fixture(tmp_path)
    (tmp_path / "model.json").write_text("changed")
    assert "artifact_hash:model" in evaluate(manifest, tmp_path)["failures"]


@pytest.mark.parametrize("kind", ["labels", "predictions"])
def test_missing_frame_not_hidden_by_accuracy(tmp_path, kind):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, kind, lambda rows: rows.pop())
    result = evaluate(manifest, tmp_path)
    assert result["status"] == "PARTIAL"
    assert result["coverage"]["actor"]["total"] == 2


def test_duplicate_frame_rejected(tmp_path):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, "predictions",
                  lambda rows: rows.append(copy.deepcopy(rows[0])))
    assert "duplicate_row:predictions" in evaluate(manifest, tmp_path)["failures"]


@pytest.mark.parametrize("change", [
    {"role": "development"}, {"complete": False}, {"tuning_exposed": True},
    {"labels_reviewed": False}, {"first_frame": 3}])
def test_nonindependent_or_incomplete_hand_rejected(tmp_path, change):
    manifest = fixture(tmp_path)
    manifest["hands"][0].update(change)
    bind(tmp_path, manifest, "dataset", manifest["hands"])
    assert evaluate(manifest, tmp_path)["status"] == "PARTIAL"


def test_hand_manifest_must_be_hash_bound(tmp_path):
    manifest = fixture(tmp_path)
    manifest["hands"][0]["session"] = "fake"
    assert "dataset_hand_binding" in evaluate(manifest, tmp_path)["failures"]


@pytest.mark.parametrize("change", [
    {"session": "dev"}, {"duration_seconds": 995.366},
    {"freshness_checked": False}, {"crashes": False},
    {"duration_seconds": float("nan")}, {"unexplained_gaps": 1},
    {"freeze_id": "old"}])
def test_hardware_claims_fail_closed(tmp_path, change):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, "hardware", lambda rows: rows[0].update(change))
    assert evaluate(manifest, tmp_path)["status"] == "PARTIAL"


def test_absent_modes_not_covered_by_normal_frames(tmp_path):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, "labels",
                  lambda rows: [row.update(scenarios=[]) for row in rows])
    result = evaluate(manifest, tmp_path)
    assert "scenario_missing:insurance" in result["failures"]


def test_freeze_must_precede_predictions(tmp_path):
    manifest = fixture(tmp_path)
    manifest["frozen_before_predictions"] = False
    assert evaluate(manifest, tmp_path)["status"] == "PARTIAL"


def test_mode_scenario_tag_cannot_replace_positive_sample(tmp_path):
    manifest = fixture(tmp_path)
    for kind in ("labels", "predictions"):
        edit_artifact(tmp_path, manifest, kind, lambda rows: rows[0]["fields"][
            "insurance"].update(value={"active": False}))
    assert "mode_positive_missing:insurance" in evaluate(manifest, tmp_path)["failures"]


@pytest.mark.parametrize("participation,passes", [
    ("empty", True), ("waiting", True), ("active", False)])
def test_explicit_not_applicable_needs_seat_context(tmp_path, participation, passes):
    manifest = fixture(tmp_path)

    def edit(rows):
        for row in rows:
            row["fields"]["stacks"]["value"]["6"] = {"status": "NOT_APPLICABLE"}
            row["fields"]["participation"]["value"]["6"] = participation
    for kind in ("labels", "predictions"):
        edit_artifact(tmp_path, manifest, kind, edit)
    assert (evaluate(manifest, tmp_path)["status"] == "PASS") is passes


@pytest.mark.parametrize("field", ["hero_cards", "board_cards"])
def test_card_identities_required_not_just_count(tmp_path, field):
    manifest = fixture(tmp_path)
    edit_artifact(tmp_path, manifest, "predictions",
                  lambda rows: rows[0]["fields"].pop(field))
    assert "field_not_complete:" + field in evaluate(manifest, tmp_path)["failures"]


def test_eight_seat_geometry_required(tmp_path):
    manifest = fixture(tmp_path)
    manifest["geometry"]["seat_count"] = 9
    assert "AA8_geometry_binding" in evaluate(manifest, tmp_path)["failures"]


def optional_actor(tmp_path, manifest):
    source = tmp_path / "source-frame.bin"
    source.write_bytes(b"synthetic-test-witness-not-real-image")
    exemption = {"reason": "transition", "source_reviewed": True,
                 "reviewer": "synthetic-reviewer", "source": {
                     "path": source.name, "sha256": digest(source)}}
    manifest["hands"][0]["action_opportunity_frames"] = [4]
    bind(tmp_path, manifest, "dataset", manifest["hands"])

    def edit(rows):
        rows[1]["decision_opportunity"] = False
        rows[1]["fields"]["actor"] = {"status": "UNKNOWN", "value": None,
                                      "required": False, "exemption": exemption}
    edit_artifact(tmp_path, manifest, "labels", edit)
    edit_artifact(tmp_path, manifest, "predictions", edit)


def test_reviewed_transition_abstention_is_not_an_error_or_correct_prediction(tmp_path):
    manifest = fixture(tmp_path)
    optional_actor(tmp_path, manifest)
    result = evaluate(manifest, tmp_path)
    assert result["status"] == "PASS"
    count = result["coverage"]["actor"]
    assert count["total"] == 2
    assert count["required_total"] == count["required_matched"] == 1
    assert count["matched"] == 1
    assert count["exempt_abstained"] == 1


@pytest.mark.parametrize("mutation", [{"source_reviewed": False}, {"reason": "hard"},
                                      {"reviewer": ""}, {"source": {}}])
def test_exemption_requires_hash_bound_source_review(tmp_path, mutation):
    manifest = fixture(tmp_path)
    optional_actor(tmp_path, manifest)
    edit_artifact(tmp_path, manifest, "labels", lambda rows: rows[1]["fields"][
        "actor"]["exemption"].update(mutation))
    assert "invalid_exemption:actor" in evaluate(manifest, tmp_path)["failures"]


def test_known_prediction_on_unknown_optional_gold_is_not_unchecked(tmp_path):
    manifest = fixture(tmp_path)
    optional_actor(tmp_path, manifest)
    edit_artifact(tmp_path, manifest, "predictions", lambda rows: rows[1]["fields"][
        "actor"].update(status="KNOWN", value=3, origin="automatic"))
    assert "known_prediction_unverified_or_wrong:actor" in evaluate(
        manifest, tmp_path)["failures"]


def test_action_moment_cannot_be_exempted(tmp_path):
    manifest = fixture(tmp_path)
    optional_actor(tmp_path, manifest)
    manifest["hands"][0]["action_opportunity_frames"] = [4, 5]
    bind(tmp_path, manifest, "dataset", manifest["hands"])
    edit_artifact(tmp_path, manifest, "labels",
                  lambda rows: rows[1].update(decision_opportunity=True))
    result = evaluate(manifest, tmp_path)
    assert "action_opportunity_exempted:actor" in result["failures"]
    assert result["coverage"]["actor"]["required_total"] == 2


def test_all_optional_cannot_pass_with_zero_required_evidence(tmp_path):
    manifest = fixture(tmp_path)
    optional_actor(tmp_path, manifest)
    manifest["hands"][0]["action_opportunity_frames"] = []
    bind(tmp_path, manifest, "dataset", manifest["hands"])

    def edit(rows):
        rows[0]["fields"]["actor"] = copy.deepcopy(rows[1]["fields"]["actor"])
        rows[0]["decision_opportunity"] = False
    edit_artifact(tmp_path, manifest, "labels", edit)
    edit_artifact(tmp_path, manifest, "predictions", edit)
    result = evaluate(manifest, tmp_path)
    assert "field_not_complete:actor" in result["failures"]
    assert "opportunity_registry_missing_or_invalid:h" in result["failures"]


def test_explicit_actor_cannot_be_hidden_from_action_registry(tmp_path):
    manifest = fixture(tmp_path)
    manifest["hands"][0]["action_opportunity_frames"] = [4]
    bind(tmp_path, manifest, "dataset", manifest["hands"])
    edit_artifact(tmp_path, manifest, "labels",
                  lambda rows: rows[1].update(decision_opportunity=False))
    assert "known_action_moment_excluded" in evaluate(manifest, tmp_path)["failures"]
