import json
from pathlib import Path

import pytest

from tools.aa8_holdout_plan import sha
from tools.aa8_holdout_predict import FIELDS
from tools import aa8_holdout_predict_v2 as harness
from tools.aa8_holdout_predict_v2 import normalize_fields


def test_v2_exact_thirteen_fields_and_unknown_not_filled():
    fields = normalize_fields({"scene_supported": True})
    assert tuple(fields) == FIELDS
    assert len(fields) == 13
    assert all(value["status"] == "UNKNOWN" for value in fields.values())


def test_candidate_epoch_actions_wagers_not_promoted_to_canonical():
    fields = normalize_fields({"scene_supported": True,
                               "observed_state_v2": {
                                   "observed_epoch": "candidate_123",
                                   "authoritative_hand_boundary": False},
                               "observed_actions_v2": [
                                   {"action": "call", "amount": "2"}],
                               "causal_street_wagers_v2": {
                                   "canonical_verified": False,
                                   "wagers": dict.fromkeys(map(str, range(8)), "2")}})
    for field in ("actions", "street_wagers", "hand"):
        assert fields[field]["status"] == "UNKNOWN"


def test_blocking_bomb_positive_retained_but_stale_money_actor_rejected():
    fields = normalize_fields({"scene_supported": False, "current_actor": 4,
                               "pot": {"value": "99"}, "special_modes": {
                                   "block_state_updates": True,
                                   "critical_hit_title": {
                                       "critical_hit_animation": True}}})
    assert fields["bomb"]["value"] == {"active": True}
    assert fields["actor"]["status"] == fields["pot"]["status"] == "UNKNOWN"


def test_no_absence_guesses_for_special_modes():
    fields = normalize_fields({"scene_supported": True, "special_modes": {
        "mushroom_rule": "3BB", "critical_hit_rule": "7BB"}})
    for field in ("insurance", "mushroom", "bomb"):
        assert fields[field]["status"] == "UNKNOWN"


def test_current_waiting_na_not_history_waiting():
    row = {"scene_supported": True,
           "stacks": {str(s): {"value": "100"} for s in range(7)},
           "participation": {"slots": {"7": {"history": "WAITING_CANDIDATE"}}}}
    assert normalize_fields(row)["stacks"]["status"] == "UNKNOWN"
    row["participation"]["slots"]["7"]["current"] = "WAITING_CANDIDATE"
    assert normalize_fields(row)["stacks"]["value"]["7"] == {"status": "NOT_APPLICABLE"}
    row["participation"]["slots"]["7"]["conflict"] = True
    assert normalize_fields(row)["stacks"]["status"] == "UNKNOWN"


def test_current_positive_preflop_can_report_empty_board_not_missing_cards():
    row = {"scene_supported": True, "observed_state_v2": {
        "street_candidate": "preflop", "board_candidate": []}}
    assert normalize_fields(row)["board_cards"]["value"] == []
    assert normalize_fields({"scene_supported": True, "board_count": 0})[
        "board_cards"]["status"] == "UNKNOWN"


def test_insurance_ui_timer_not_betting_actor():
    fields = normalize_fields({"scene_supported": True, "current_actor": 3,
                               "pot": {"value": "100"},
                               "cards": {"hero": ["Ah", "Kd"]},
                               "special_modes": {"insurance": "VISIBLE"},
                               "observed_state_v2": {"street_candidate": "turn"}})
    assert fields["insurance"]["value"] == {"active": True}
    assert fields["pot"]["value"] == "100"
    assert fields["hero_cards"]["value"] == ["Ah", "Kd"]
    for field in ("actor", "street", "actions", "street_wagers", "hand"):
        assert fields[field]["status"] == "UNKNOWN"


def preflight_fixture(tmp_path, *, extra_spec=False):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    entries, spec = [], {"audit": "audit"}
    for name in ("source", "context_source", "late_source", "bomb_pool"):
        path = write(name + "/samples.json", {"test_only": name})
        spec[name] = str(path.parent)
        entries.append({"path": str(path), "sha256": sha(path), "kind": "training"})
    for name in ("bank_path", "profile_path", "heads_path", "reservations"):
        path = write(name + ".json", {"test_only": name})
        spec[name] = str(path)
        entries.append({"path": str(path), "sha256": sha(path), "kind": "model"})
    if extra_spec:
        spec["gold"] = "forbidden"
    spec_path = write("factory.json", spec)
    split = write("split.json", {
        "source_audit_sha256": "audit", "ranges": [{
            "role": "holdout_candidate", "start_inclusive": "300",
            "end_exclusive": "600"}]})
    for path, kind in ((spec_path, "parameters"), (split, "parameters"),
                       (Path(harness.__file__).resolve(), "implementation")):
        entries.append({"path": str(path), "sha256": sha(path), "kind": kind})
    freeze = write("freeze.json", {
        "frozen_at_utc": "2020-01-01T00:00:00Z",
        "files": sorted(entries, key=lambda r: r["path"].replace("\\", "/").casefold()),
        "exposures": [{"role": "development", "used_for": "training",
                       "artifact_sha256": r["sha256"]} for r in entries
                      if r["kind"] == "training"]})
    registry = write("registry.json", {
        "role": "holdout", "audit_sha256": "audit",
        "freeze_sha256": sha(freeze), "candidate_range_seconds": ["300", "600"],
        "hands": [{"id": "h", "first_frame": 1, "last_frame": 2,
                   "preroll_first_frame": 0, "boundaries_verified": True}]})
    manifest = write("target/samples.json", {
        "registry_sha256": sha(registry), "audit_sha256": "audit", "samples": [{
            "global_frame": f, "role": "holdout", "pts_seconds": str(301 + f / 30),
            "sha256": "a" * 64} for f in range(3)]})
    run = write("run.json", {
        "frozen_at_utc": "2020-01-01T00:00:01Z",
        "freeze_sha256": sha(freeze), "registry_sha256": sha(registry),
        "target_manifest_sha256": sha(manifest), "factory_spec_sha256": sha(spec_path),
        "harness_sha256": sha(Path(harness.__file__)),
        "split_path": str(split)})
    return {"freeze_path": freeze, "run_path": run, "run_sha256": sha(run),
            "registry_path": registry, "target": manifest.parent,
            "spec_path": spec_path}


def test_preflight_only_checks_metadata_never_opens_image_or_model(tmp_path):
    result = harness.preflight(**preflight_fixture(tmp_path))
    assert result[4] == {1, 2}


def test_preflight_gold_input_forbidden_even_when_config_is_frozen(tmp_path):
    with pytest.raises(ValueError, match="gold inputs forbidden"):
        harness.preflight(**preflight_fixture(tmp_path, extra_spec=True))


def test_run_manifest_cannot_change_after_external_hash_pin(tmp_path):
    args = preflight_fixture(tmp_path)
    args["run_path"].write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        harness.preflight(**args)


def test_model_change_rejected_before_candidate_construction(tmp_path):
    args = preflight_fixture(tmp_path)
    (tmp_path / "bank_path.json").write_text("changed")
    with pytest.raises(ValueError):
        harness.preflight(**args)
