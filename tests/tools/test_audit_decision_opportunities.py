import json
import subprocess
import sys

from tests.strategy.test_decision_opportunities_v1 import document
from tools.audit_decision_opportunities import analyze_file


def test_cli_writes_synthetic_nonreal_report_once(tmp_path):
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "report.json"
    dataset.write_text(json.dumps(document()), encoding="utf-8")
    command = [sys.executable, "-m", "tools.audit_decision_opportunities",
               "--dataset", str(dataset), "--output", str(output)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report["data_readiness"] == "NOT_REAL_DATA"
    assert not report["model_fit_executed"] and not report["strategy_eligible"]
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert output.read_bytes() == raw


def test_blocked_dataset_is_written_for_review_with_exit_two(tmp_path):
    data = document()
    data["opportunities"].pop()
    dataset, output = tmp_path / "dataset.json", tmp_path / "blocked.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    run = subprocess.run([
        sys.executable, "-m", "tools.audit_decision_opportunities",
        "--dataset", str(dataset), "--output", str(output)], capture_output=True)
    assert run.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_readiness"] == "BLOCKED" and report["missing_ids"]


def test_duplicate_raw_json_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    try:
        analyze_file(path)
    except ValueError as exc:
        assert "duplicate_json_key" in str(exc)
    else:
        raise AssertionError("duplicate key accepted")
