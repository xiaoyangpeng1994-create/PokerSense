import json
import subprocess
import sys

import pytest

from tests.strategy.test_opponent_dataset_v1 import documents
from tools.calibrate_opponent_models import analyze_files


def files(tmp_path):
    data, candidates, _ = documents()
    cp, dp = tmp_path / "candidates.json", tmp_path / "dataset.json"
    cp.write_bytes(json.dumps(candidates, indent=2).encode())
    dp.write_text(json.dumps(data))
    return dp, cp


def test_actual_cli_synthetic_flow_and_no_overwrite(tmp_path):
    dp, cp = files(tmp_path)
    output = tmp_path / "result.json"
    command = [sys.executable, "-m", "tools.calibrate_opponent_models",
               "--dataset", str(dp), "--candidates", str(cp), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report["calibration"]["status"] == "NOT_REAL_CALIBRATION"
    assert not report["strategy_eligible"] and report["range_model"] is None
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert output.read_bytes() == raw


def test_missing_decision_cli_keeps_audit_and_exit_two(tmp_path):
    dp, cp = files(tmp_path)
    data = json.loads(dp.read_text())
    data["decisions"].pop()
    dp.write_text(json.dumps(data))
    output = tmp_path / "blocked.json"
    cp_run = subprocess.run([
        sys.executable, "-m", "tools.calibrate_opponent_models", "--dataset", str(dp),
        "--candidates", str(cp), "--output", str(output)], capture_output=True)
    assert cp_run.returncode == 2
    report = json.loads(output.read_text())
    assert report["audit"]["missing_ids"] and report["calibration"] is None


def test_duplicate_raw_json_keys_rejected(tmp_path):
    dp, cp = files(tmp_path)
    dp.write_text(dp.read_text().replace(
        '"schema_version":', '"schema_version":1,"schema_version":', 1))
    with pytest.raises(ValueError, match="duplicate"):
        analyze_files(dp, cp)
