"""V2 independent prediction only: no gold input, tuning, or capture device."""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

from tools.aa8_holdout_plan import sha, utc, validate_freeze
from tools.aa8_holdout_predict import FIELDS, bound_file, load_holdout, validate_rows


def normalize_fields(row):
    fields = {key: {"status": "UNKNOWN", "value": None, "origin": "automatic"}
              for key in FIELDS}

    def known(key, value):
        fields[key] = {"status": "KNOWN", "value": value, "origin": "automatic",
                       "semantics": "observed_visual_value_not_strategy_authority"}

    special = row.get("special_modes") or {}
    if special.get("insurance") == "VISIBLE":
        known("insurance", {"active": True})
    if (special.get("critical_hit_title") or {}).get("critical_hit_animation") is True:
        known("bomb", {"active": True})
    if not row.get("scene_supported") or special.get("block_state_updates"):
        return fields
    insurance = special.get("insurance") == "VISIBLE"
    if (not insurance and type(row.get("current_actor")) is int
            and 0 <= row["current_actor"] < 8):
        known("actor", row["current_actor"])
    if (row.get("pot") or {}).get("value") is not None:
        known("pot", row["pot"]["value"])
    seats = (row.get("participation") or {}).get("slots", {})
    states, stacks = {}, {}
    mapping = {"EMPTY_CANDIDATE": "empty", "WAITING_CANDIDATE": "waiting",
               "DEALT_IN_CANDIDATE": "active", "FOLDED_CANDIDATE": "folded"}
    for slot in map(str, range(8)):
        cue = seats.get(slot, {})
        current = None if cue.get("conflict") else mapping.get(cue.get("current"))
        if current is not None:
            states[slot] = current
        value = (row.get("stacks", {}).get(slot) or {}).get("value")
        if value is not None:
            stacks[slot] = value
        elif current in ("empty", "waiting"):
            stacks[slot] = {"status": "NOT_APPLICABLE"}
    if len(states) == 8:
        known("participation", states)
    if len(stacks) == 8:
        known("stacks", stacks)
    cards, state = row.get("cards") or {}, row.get("observed_state_v2") or {}
    if cards.get("hero") is not None:
        known("hero_cards", cards["hero"])
    if insurance:
        return fields
    stage, board = state.get("street_candidate"), state.get("board_candidate")
    if stage in ("preflop", "flop", "turn", "river"):
        known("street", stage)
        size = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[stage]
        if isinstance(board, list) and len(board) == size and all(board):
            known("board_cards", board)
    # Causal candidate accounting does not silently become canonical action truth.
    wagers = row.get("causal_street_wagers_v2") or {}
    if (wagers.get("canonical_verified") is True
            and isinstance(wagers.get("wagers"), dict)):
        if set(wagers["wagers"]) == set(map(str, range(8))):
            known("street_wagers", wagers["wagers"])
    actions = row.get("observed_actions_v2")
    if (row.get("actions_complete_and_canonical_verified") is True
            and isinstance(actions, list)
            and all(a.get("canonical_verified") is True for a in actions)):
        known("actions", actions)
    if state.get("authoritative_hand_boundary") is True and state.get("observed_epoch"):
        known("hand", state["observed_epoch"])
    return fields


def preflight(*, freeze_path, run_path, run_sha256, registry_path, target, spec_path):
    started = datetime.now(timezone.utc).isoformat()
    if sha(run_path) != run_sha256:
        raise ValueError("run manifest hash mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    errors = validate_freeze(freeze, freeze_path.parent, started)
    if errors:
        raise ValueError(errors)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest = json.loads((target / "samples.json").read_text(encoding="utf-8"))
    bindings = {"freeze_sha256": sha(freeze_path),
                "registry_sha256": sha(registry_path),
                "target_manifest_sha256": sha(target / "samples.json"),
                "factory_spec_sha256": sha(spec_path),
                "harness_sha256": sha(Path(__file__))}
    if (any(run.get(key) != value for key, value in bindings.items())
            or not utc(freeze["frozen_at_utc"]) < utc(run["frozen_at_utc"])
            < utc(started)
            or registry.get("freeze_sha256") != bindings["freeze_sha256"]
            or manifest.get("registry_sha256") != bindings["registry_sha256"]):
        raise ValueError("V2 run lineage/chronology mismatch")
    for path in (spec_path, Path(__file__)):
        bound_file(path, freeze)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    required = {"source", "context_source", "late_source", "bank_path", "profile_path",
                "heads_path", "audit", "bomb_pool", "reservations"}
    if set(spec) != required or spec["audit"] != manifest.get("audit_sha256"):
        raise ValueError("factory spec keys/audit mismatch; gold inputs forbidden")
    for key in ("source", "context_source", "late_source", "bomb_pool"):
        bound_file(Path(spec[key]) / "samples.json", freeze)
    for key in ("bank_path", "profile_path", "heads_path", "reservations"):
        bound_file(Path(spec[key]), freeze)
    split_path = Path(run["split_path"])
    bound_file(split_path, freeze)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    limits = registry.get("candidate_range_seconds")
    if not any(r["role"] == "holdout_candidate" and limits == [
            r["start_inclusive"], r["end_exclusive"]] for r in split["ranges"]):
        raise ValueError("registry is not fresh frozen holdout reservation")
    if split.get("source_audit_sha256") != manifest.get("audit_sha256"):
        raise ValueError("split audit mismatch")
    samples = manifest["samples"]
    rowmap = {row["global_frame"]: row for row in samples}
    if len(rowmap) != len(samples):
        raise ValueError("duplicate sample frame")
    scored = validate_rows(rowmap, registry, manifest["audit_sha256"])
    return freeze, run, registry, rowmap, scored, started


def predict(*, freeze_path, run_path, run_sha256, registry_path, target,
            spec_path, output):
    freeze, run, registry, rows, scored, started = preflight(
        freeze_path=freeze_path, run_path=run_path, run_sha256=run_sha256,
        registry_path=registry_path, target=target, spec_path=spec_path)
    module_path = Path(importlib.util.find_spec("tools.aa8_candidate_v2").origin)
    bound_file(module_path, freeze)
    module = importlib.import_module("tools.aa8_candidate_v2")
    bound_file(Path(module.__file__), freeze)
    pinned = {path: sha(path) for path in (
        freeze_path, run_path, registry_path, target / "samples.json",
        spec_path, Path(__file__))}
    # Factory consumes frozen development inputs only, never target rows or gold.
    state = module.create_candidate(json.loads(spec_path.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
        for frame, sample in rows.items():
            value = state.read(load_holdout(target, sample), frame, sample)
            if (value.get("frame") != frame
                    or value.get("source_sha256") != sample["sha256"]):
                raise ValueError("candidate returned wrong source identity")
            value["evaluation_fields"] = normalize_fields(value)
            value["scored_whole_hand_frame"] = frame in scored
            value["strategy_eligible"] = False
            stream.write(json.dumps(value) + "\n")
    if (validate_freeze(freeze, freeze_path.parent, started)
            or any(sha(path) != value for path, value in pinned.items())):
        raise ValueError("inputs changed during prediction; retain partial output")
    report = {"schema_version": 2, "state": "PREDICTIONS_LOCKED_NOT_ACCEPTANCE",
              "freeze_sha256": sha(freeze_path), "run_manifest_sha256": run_sha256,
              "registry_sha256": sha(registry_path), "prediction_start_utc": started,
              "prediction_end_utc": datetime.now(timezone.utc).isoformat(),
              "observations_sha256": sha(output / "observations.jsonl"),
              "frames": len(rows), "scored_frames": len(scored),
              "labels_used_for_prediction": False, "full_visual_acceptance": False,
              "strategy_eligible": False, "fields": list(FIELDS)}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("freeze", "run", "registry", "target", "spec", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    args = parser.parse_args()
    result = predict(freeze_path=args.freeze, run_path=args.run,
                     run_sha256=args.run_sha256, registry_path=args.registry,
                     target=args.target, spec_path=args.spec, output=args.output)
    print(json.dumps(result))
