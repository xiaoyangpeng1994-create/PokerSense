import json

import pytest

from tools.aa8_action_transfer import sha
from tools.aa8_candidate_development_score_v2 import run


def test_development_scoring_preserves_unknown_and_does_not_claim_acceptance(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    row = {"frame": 1, "source_sha256": "a" * 64, "board_count": 0,
           "cards": {"hero": None, "board_slots": [None] * 5}, "current_actor": None,
           "pot": {"value": "10"}, "stacks": {}, "participation": {"slots": {}},
           "observed_state_v2": {}, "causal_street_wagers_v2": {
               "status": "WAGERS_UNKNOWN", "reason": "not_observable"}}
    path = predictions / "observations.jsonl"
    path.write_text(json.dumps(row) + "\n")
    report = {"frames": 1, "observations_sha256": sha(path),
              "independent_holdout": False, "actions": []}
    (predictions / "report.json").write_text(json.dumps(report))
    fields = {key: {"status": "UNKNOWN", "value": None} for key in (
        "actor", "hero_cards", "board_cards", "pot", "stacks")}
    fields["actor"] = {"status": "KNOWN", "value": 4}
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"role": "development", "checkpoints": [
        {"frame": 1, "source_sha256": "a" * 64, "fields": fields}]}))
    output = tmp_path / "score"
    run(predictions, gold, [], output)
    value = json.loads((output / "report.json").read_text())
    assert value["metrics"] == {"actor:total": 1, "actor:matched": 0}
    assert value["ledger_coverage_not_accuracy"] == {"WAGERS_UNKNOWN": 1}
    assert value["full_visual_acceptance"] is False
    report["independent_holdout"] = True
    (predictions / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="must not consume holdout"):
        run(predictions, gold, [], tmp_path / "forbidden")
