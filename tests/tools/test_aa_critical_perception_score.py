"""Synthetic contract tests; no images, labels, solver or real-hand acceptance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from poker_engine.desktop import aa_critical_perception as producer
from tools import aa_critical_perception_score as score


BOARD = ["2c", "3d", "4h", "5s", "9c"]


def raw_row(frame, *, turn=False, folded=False, all_in=False, blocked=False):
    states = {str(s): "active" if s in (1, 2, 4) else "folded" for s in range(8)}
    if folded:
        states["4"] = "folded"
    if all_in:
        states["2"] = "all_in"
    epoch = "synthetic-hand"
    return {
        "source_id": "synthetic-score-contract", "frame": frame,
        "source_frame": frame, "pts_seconds": frame / 30,
        "source_sha256": hashlib.sha256(f"synthetic:{frame}".encode()).hexdigest(),
        "scene_supported": not blocked, "board_count": 4 if turn else 5, "cards": {
            "board_slots": BOARD[:4] + [None] if turn else list(BOARD),
            "hero": ["As", "Kd"]},
        "observed_state_v2": {
            "observed_epoch": epoch, "street_candidate": "turn" if turn else "river",
            "board_candidate": BOARD[:4] if turn else list(BOARD),
            "positive_board_geometry": {"last_frame": frame, "streak": 2},
            "pending_actions": 0,
            "participants": {seat: {"state": state, "epoch": epoch,
                                    "evidence_frame": frame}
                             for seat, state in states.items()}},
        "participation": {"slots": {}}, "current_actor": 1, "dealer_seat": 0,
        "dealer_observation_v2": {"dealer_seat": 0},
        "actor_evidence": {"actor": 1, "timer_suffix_verified": True,
                           "reason": "unique_bright_ring_candidate"},
        "dealer_evidence_v2": {"dealer_seat": 0, "epoch": epoch, "frame": frame},
        "stacks": {str(s): {"value": "0" if all_in and s == 2 else "100"}
                   for s in range(8)},
        "causal_street_wagers_v2": {
            "status": "OBSERVED_STREET_WAGERS_CANDIDATE",
            "title_center_ledger_reconciled": True,
            "wagers": {str(s): "0" for s in range(8)}},
        "observed_actions_v2": [], "glyph_transitions": [],
        "action_history_candidate": [], "glyphs": {},
    }


def golden(frame):
    seats = {str(s): "ACTIVE" if s in (1, 2, 4) else "FOLDED" for s in range(8)}
    if frame >= 4:
        seats["4"] = "FOLDED"
    if frame == 5:
        seats["2"] = "ALL_IN"
    values = dict(board=list(BOARD), hero_participation=seats["4"], participation=seats,
                  all_in_seats=[2] if frame == 5 else [], river_first_actor=1)
    result = {field: {"status": "KNOWN", "value": value,
                      "reason": "explicit synthetic oracle; not a real annotation"}
              for field, value in values.items()}
    for field in score.FIELDS:
        if frame == 6 or frame == 1 and field in ("board", "river_first_actor"):
            result[field] = {"status": "UNREADABLE", "value": None,
                             "reason": "synthetic unavailable river context"}
    return result


def hero_first_row(frame):
    row = raw_row(frame, turn=frame == 1)
    row.update(source_id="synthetic-hero-first-source", current_actor=4,
               dealer_seat=3, dealer_observation_v2={"dealer_seat": 3},
               actor_evidence={"reason": "hero_buttons", "hero_turn": True})
    row["source_sha256"] = hashlib.sha256(f"hero-source:{frame}".encode()).hexdigest()
    epoch = "synthetic-hero-first-hand"
    row["observed_state_v2"]["observed_epoch"] = epoch
    for participant in row["observed_state_v2"]["participants"].values():
        participant["epoch"] = epoch
    row["dealer_evidence_v2"].update(dealer_seat=3, epoch=epoch)
    return row


def review_fields(binding):
    result = golden(binding["frame"])
    if (binding["source_id"] == "synthetic-hero-first-source"
            and binding["frame"] > 1):
        result["river_first_actor"]["value"] = 4
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return {"path": str(path), "sha256": score._digest(path)}


@pytest.fixture
def bundle(tmp_path):
    machine = producer.CriticalPerceptionBoundary()
    rows = []
    for frame in range(1, 7):
        row = raw_row(frame, turn=frame == 1, folded=frame in (4, 5),
                      all_in=frame == 5, blocked=frame == 6)
        row["critical_perception_v1"] = machine.observe(row)
        rows.append(row)
    for frame in range(1, 4):
        row = hero_first_row(frame)
        row["critical_perception_v1"] = machine.observe(row)
        rows.append(row)
    required = {3: list(score.FIELDS), 4: ["hero_participation", "participation"],
                5: ["participation", "all_in_seats"]}
    registry = {"schema_version": 1, "cohort": "synthetic", "exposure": "synthetic",
                "registered_at_utc": "2020-01-01T00:00:01Z",
                "rows": []}
    for row in rows:
        if row["source_id"] == "synthetic-hero-first-source":
            fields = ["river_first_actor"] if row["frame"] == 3 else []
        else:
            fields = required.get(row["frame"], [])
        registry["rows"].append({
            "binding": deepcopy(row["critical_perception_v1"]["binding"]),
            "required_fields": fields})
    sources = write_json(tmp_path / "sources.json", registry)
    predictions = {
        "schema_version": 1, "started_at_utc": "2020-01-01T00:00:03Z",
        "source_manifest_sha256": sources["sha256"],
        "producer_sha256": score._digest(Path(producer.__file__)), "rows": rows}
    reviews = {
        "schema_version": 1, "started_at_utc": "2020-01-01T00:00:03Z",
        "source_manifest_sha256": sources["sha256"], "reviewer_id": "synthetic-oracle",
        "method": "source_only",
        "rows": [{"binding": deepcopy(entry["binding"]),
                  "fields": review_fields(entry["binding"])}
                 for entry in registry["rows"]]}
    repository = Path(score.__file__).resolve().parents[1]
    freeze_files = [{"path": str(repository / path),
                     "sha256": score._digest(repository / path)}
                    for path in score.REQUIRED_IMPLEMENTATION_PATHS] + [sources]
    manifest = {
        "schema_version": 1, "scope": score.SCOPE, "cohort": "synthetic",
        "declared_at_utc": "2020-01-01T00:00:00Z",
        "freeze": {"frozen_at_utc": "2020-01-01T00:00:02Z", "files": freeze_files},
        "sources": sources,
        "predictions": write_json(tmp_path / "predictions.json", predictions),
        "reviews": write_json(tmp_path / "reviews.json", reviews)}
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def edit(bundle, kind, change):
    manifest = score._read(bundle)
    path = Path(manifest[kind]["path"])
    document = score._read(path)
    change(document)
    manifest[kind] = write_json(path, document)
    if kind == "sources":
        for name in ("predictions", "reviews"):
            dependent = Path(manifest[name]["path"])
            value = score._read(dependent)
            value["source_manifest_sha256"] = manifest[kind]["sha256"]
            manifest[name] = write_json(dependent, value)
        manifest["freeze"]["files"][-1] = manifest[kind]
    write_json(bundle, manifest)


def test_synthetic_finite_pass_never_promotes_truth_or_independence(bundle):
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_FINITE_PASS"
    assert result["owned_rows"] == 9
    for flag in ("strategy_eligible", "full_visual_acceptance", "real_hand_confirmed",
                 "independence_verified", "source_media_verified", "media_decoded",
                 "advice_emitted"):
        assert result[flag] is False
    assert result["declared_cohort"] == "synthetic"
    assert result["metrics"]["river_first_actor"]["correct_required"] == 2
    assert result["metrics"]["river_first_actor"]["guarded_unknown"] == 7
    assert result["metrics"]["river_first_actor"]["guarded_unreadable"] == 3
    assert result["coverage"]["river_first_actor"] == ["hero_first", "other_first"]
    assert result["coverage"]["all_in_seats"] == ["empty", "nonempty"]
    actors = [(item["source_id"], item["prediction"]["value"])
              for item in result["comparisons"] if item["field"] == "river_first_actor"
              and item["correct_automatic"]]
    assert actors == [("synthetic-score-contract", 1),
                      ("synthetic-hero-first-source", 4)]


@pytest.mark.parametrize("kind", ["predictions", "reviews", "sources"])
@pytest.mark.parametrize("mutation", ["drop", "duplicate", "reorder"])
def test_no_dropped_duplicated_or_reordered_registered_rows(bundle, kind, mutation):
    def change(document):
        rows = document["rows"]
        if mutation == "drop":
            del rows[1]
        elif mutation == "duplicate":
            rows.insert(1, deepcopy(rows[0]))
        else:
            rows[0], rows[1] = rows[1], rows[0]
    edit(bundle, kind, change)
    with pytest.raises(ValueError):
        score.evaluate_files(bundle)


def test_wrong_known_even_on_nonrequired_context_is_not_ignored(bundle):
    def change(document):
        document["rows"][1]["fields"]["board"]["value"][4] = "Tc"
    edit(bundle, "reviews", change)
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert result["metrics"]["board"]["wrong_known"] == 1


def test_all_unknown_replay_still_cannot_pass(bundle):
    def change(document):
        machine = producer.CriticalPerceptionBoundary()
        for row in document["rows"]:
            row["scene_supported"] = False
            row["critical_perception_v1"] = machine.observe(row)
    edit(bundle, "predictions", change)
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert all(counts["correct_required"] == 0 for counts in result["metrics"].values())
    assert all(counts["unknown_required"] > 0 for counts in result["metrics"].values())


def test_all_context_cannot_erase_required_denominators(bundle):
    def change(document):
        for row in document["rows"]:
            row["required_fields"] = []
    edit(bundle, "sources", change)
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert all(not values for values in result["coverage"].values())


def test_all_in_positive_coverage_cannot_be_satisfied_by_negative_only(bundle):
    edit(bundle, "sources", lambda doc: doc["rows"][4].update(
        required_fields=["participation"]))
    result = score.evaluate_files(bundle)
    assert "coverage_missing:all_in_seats:nonempty" in result["failures"]


@pytest.mark.parametrize("source_id,stratum", [
    ("synthetic-score-contract", "other_first"),
    ("synthetic-hero-first-source", "hero_first")])
def test_each_first_actor_positive_stratum_is_required(bundle, source_id, stratum):
    def change(document):
        for row in document["rows"]:
            if row["binding"]["source_id"] == source_id:
                row["required_fields"] = [field for field in row["required_fields"]
                                          if field != "river_first_actor"]
    edit(bundle, "sources", change)
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert "coverage_missing:river_first_actor:" + stratum in result["failures"]


@pytest.mark.parametrize("replacement", ["KNOWN", "UNREVIEWED"])
def test_first_actor_guard_requires_unreadable_source_review(bundle, replacement):
    def change(document):
        for row in document["rows"]:
            field = row["fields"]["river_first_actor"]
            if field["status"] == "UNREADABLE":
                actor = 4 if row["binding"]["source_id"].endswith("first-source") else 1
                field.update(status=replacement,
                             value=actor if replacement == "KNOWN" else None)
    edit(bundle, "reviews", change)
    result = score.evaluate_files(bundle)
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert result["metrics"]["river_first_actor"]["guarded_unreadable"] == 0
    assert "coverage_missing:river_first_actor:guarded_unreadable" in result["failures"]
    assert result["coverage"]["river_first_actor"] == ["hero_first", "other_first"]


@pytest.mark.parametrize("flag,value", [
    ("candidate_only", False), ("strategy_eligible", True),
    ("advice_emitted", True), ("schema_version", True)])
def test_forged_candidate_flags_are_blocked(bundle, flag, value):
    def change(document):
        document["rows"][2]["critical_perception_v1"][flag] = value
    edit(bundle, "predictions", change)
    with pytest.raises(ValueError):
        score.evaluate_files(bundle)


def test_manually_patched_candidate_is_not_automatic_success(bundle):
    def change(document):
        view = document["rows"][1]["critical_perception_v1"]
        view["fields"]["board"]["value"][4] = "Tc"
    edit(bundle, "predictions", change)
    with pytest.raises(ValueError, match="machine_candidate"):
        score.evaluate_files(bundle)


def test_explicit_manual_correction_metadata_is_rejected(bundle):
    edit(bundle, "predictions", lambda doc: doc["rows"][2].update(
        manual_corrected=True))
    with pytest.raises(ValueError, match="manual_prediction"):
        score.evaluate_files(bundle)


def test_temporal_known_cannot_be_suppressed_in_cached_projection(bundle):
    def change(document):
        view = document["rows"][2]["critical_perception_v1"]
        view["fields"]["river_first_actor"].update(
            status="UNKNOWN", value=None, reasons=["hide required machine result"])
    edit(bundle, "predictions", change)
    with pytest.raises(ValueError, match="projection_replay"):
        score.evaluate_files(bundle)


def test_conflict_retained_as_rejection_not_correct_prediction(bundle):
    def change(document):
        machine = producer.CriticalPerceptionBoundary()
        for row in document["rows"]:
            if row["frame"] == 3 and row["source_id"] == "synthetic-score-contract":
                row["observed_state_v2"]["critical_status_conflicts"] = ["4"]
            row["critical_perception_v1"] = machine.observe(row)
    edit(bundle, "predictions", change)
    result = score.evaluate_files(bundle)
    counts = result["metrics"]["hero_participation"]
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert counts["conflict"] == counts["unknown_required"] == 1
    assert counts["correct_required"] == 1


def test_required_unreadable_review_does_not_remove_required_denominator(bundle):
    edit(bundle, "reviews", lambda doc: doc["rows"][2]["fields"]["board"].update(
        status="UNREADABLE", value=None, reason="cannot adjudicate source"))
    result = score.evaluate_files(bundle)
    counts = result["metrics"]["board"]
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"
    assert counts["required_total"] == counts["wrong_known"] == 1
    assert counts["correct_required"] == 0
    assert "required_review_unavailable:board" in result["failures"]


def test_same_active_count_with_wrong_seat_ownership_is_an_error(bundle):
    def change(document):
        seats = document["rows"][2]["fields"]["participation"]["value"]
        seats["1"], seats["0"] = seats["0"], seats["1"]
    edit(bundle, "reviews", change)
    result = score.evaluate_files(bundle)
    assert result["metrics"]["participation"]["wrong_known"] == 1
    assert result["status"] == "CRITICAL_PERCEPTION_PARTIAL"


def test_prediction_review_leakage_metadata_is_rejected(bundle):
    edit(bundle, "reviews", lambda doc: doc.update(predictions_used_for_labels=True))
    with pytest.raises(ValueError, match="invalid_keys:reviews"):
        score.evaluate_files(bundle)


def test_prediction_assisted_review_method_rejected(bundle):
    edit(bundle, "reviews", lambda doc: doc.update(method="prediction_assisted"))
    with pytest.raises(ValueError, match="source_only"):
        score.evaluate_files(bundle)


@pytest.mark.parametrize("kind", ["predictions", "reviews"])
def test_mismatched_source_manifest_hash(bundle, kind):
    edit(bundle, kind, lambda doc: doc.update(source_manifest_sha256="0" * 64))
    with pytest.raises(ValueError, match="source_manifest_binding"):
        score.evaluate_files(bundle)


def test_mismatched_producer_version(bundle):
    edit(bundle, "predictions", lambda doc: doc.update(producer_sha256="0" * 64))
    with pytest.raises(ValueError, match="producer_version"):
        score.evaluate_files(bundle)


def test_changed_source_identity_cannot_be_rebound_by_only_updating_file_hash(bundle):
    edit(bundle, "reviews", lambda doc: doc["rows"][2]["binding"].update(
        source_sha256="0" * 64))
    with pytest.raises(ValueError, match="review_source_mismatch"):
        score.evaluate_files(bundle)


def test_unreviewed_is_not_guarded_success(bundle):
    edit(bundle, "reviews", lambda doc: doc["rows"][5]["fields"]["board"].update(
        status="UNREVIEWED", value=None, reason="not reviewed"))
    result = score.evaluate_files(bundle)
    assert "unreviewed:board" in result["failures"]
    assert result["metrics"]["board"]["unreviewed"] == 1


def test_inconsistent_seat_and_hero_review_is_rejected(bundle):
    def change(document):
        document["rows"][2]["fields"]["hero_participation"]["value"] = "FOLDED"
    edit(bundle, "reviews", change)
    with pytest.raises(ValueError, match="hero_participation_contradiction"):
        score.evaluate_files(bundle)


def test_freeze_must_contain_scoring_implementation(bundle):
    manifest = score._read(bundle)
    del manifest["freeze"]["files"][0]
    write_json(bundle, manifest)
    with pytest.raises(ValueError, match="not_frozen"):
        score.evaluate_files(bundle)


def test_post_prediction_declaration_is_rejected(bundle):
    manifest = score._read(bundle)
    manifest["declared_at_utc"] = "2020-01-01T00:00:04Z"
    write_json(bundle, manifest)
    with pytest.raises(ValueError, match="chronology"):
        score.evaluate_files(bundle)


def test_development_exposure_cannot_be_declared_verification(bundle):
    edit(bundle, "sources", lambda doc: doc.update(cohort="verification",
                                                   exposure="development_exposed"))
    manifest = score._read(bundle)
    manifest["cohort"] = "verification"
    write_json(bundle, manifest)
    with pytest.raises(ValueError, match="cohort_exposure"):
        score.evaluate_files(bundle)


def test_forged_independent_boolean_is_not_an_authority(bundle):
    edit(bundle, "reviews", lambda doc: doc.update(independent=True))
    with pytest.raises(ValueError, match="invalid_keys"):
        score.evaluate_files(bundle)


def test_file_tampering_without_manifest_update_is_blocked(bundle):
    manifest = score._read(bundle)
    path = Path(manifest["reviews"]["path"])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file_hash_mismatch"):
        score.evaluate_files(bundle)


def test_duplicate_json_keys_are_blocked(bundle):
    bundle.write_text('{"schema_version": 1, "schema_version": 1}',
                      encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_json_key"):
        score.evaluate_files(bundle)


def test_cli_emits_blocked_and_nonzero_for_bad_manifest(bundle, capsys):
    bundle.write_text("{}", encoding="utf-8")
    assert score.main(["--manifest", str(bundle)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "CRITICAL_PERCEPTION_BLOCKED"


def test_cli_pass_still_explicitly_no_strategy_or_truth(bundle, capsys):
    assert score.main(["--manifest", str(bundle)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["real_hand_confirmed"] is result["strategy_eligible"] is False
