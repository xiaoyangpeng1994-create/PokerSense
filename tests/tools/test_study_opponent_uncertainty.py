import json
from pathlib import Path

import pytest

from tools.study_opponent_uncertainty import run_study


INPUT = Path("configs/strategy/examples/threeway-river-response-manual.json")


def test_actual_manual_study_retains_losses_and_unsupported_groups(tmp_path):
    out = tmp_path / "study"
    report = run_study(INPUT, out)
    assert len(report["groups"]) == 6
    assert not report["empirical_approval"] and not report["strategy_eligible"]
    assert report["promotion"] == "NO_EMPIRICAL_PROMOTION"
    for row in report["groups"]:
        assert len(row["selection"].evaluations) == 25
        assert len(row["validation"].evaluations) == 2
        if row["group"].endswith("facing_bet"):
            assert row["selection"].selected_id == "manual_reference"
            assert row["validation"].status == "FAIL_DECLARED_WORLD_SCREEN"
        else:
            assert row["selection"].selected_id is None
            assert row["validation"].status == "INSUFFICIENT_EVIDENCE"
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["calibration_worlds_previously_seen_development_regression"]
    assert protocol["challenge_worlds_also_manual_development_not_empirical_holdout"]
    stored = json.loads((out / "policy-books-before-comparison.json").read_text())
    assert sum(len(v) for v in stored.values()) == 30
    before = (out / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_study(INPUT, out)
    assert (out / "report.json").read_bytes() == before


def test_compilation_budget_rejection_never_prints_success(tmp_path):
    out = tmp_path / "budget"
    with pytest.raises(ValueError, match="planning_BLOCKED"):
        run_study(INPUT, out, max_nodes=1)
    assert not (out / "report.json").exists()
    assert (out / "protocol.json").exists()  # Preserve the attempted run.
