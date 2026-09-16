"""U2 regressions for the versioned offline trial launcher.

The launcher must be non-destructive by construction: it uses its own state
directory, never ends another process, never takes a busy port, and only opens the
browser after the service really answers.
"""

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "launch" / "u2-trial" / "trial_launcher.py"
sys.path.insert(0, str(LAUNCHER.parent))

import trial_launcher as launcher  # noqa: E402 - path set up above


def test_self_check_reports_the_version_and_the_isolated_state(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--self-check", "--state",
         str(tmp_path / "state")],
        capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert completed.returncode == 0, completed.stderr[-1500:]
    report = json.loads(completed.stdout)
    assert report["version"] == "aa-trial-u2.0"
    assert report["implementation"] == "aa-analysis-record-v1"
    assert report["missing_resources"] == []
    assert report["capture_enabled"] is False
    assert report["state"]["records"].startswith(str(tmp_path))
    assert report["state"]["root"].startswith(str(tmp_path))
    assert any("REAL_HAND_ACCEPTANCE_PENDING" in item
               for item in report["unverified"])
    assert any("NOT_ASSESSED" in item for item in report["unverified"])


def test_a_busy_port_is_reported_and_the_occupant_is_left_alone():
    with socket.socket() as occupant:
        occupant.bind(("127.0.0.1", 0))
        occupant.listen(1)
        busy = occupant.getsockname()[1]
        port, note = launcher.pick_port(busy)
        assert port != busy
        assert launcher.port_is_free(port)
        assert note and str(busy) in note and "没有结束任何进程" in note
        # The occupant is still listening: nothing was killed or stolen.
        assert occupant.getsockname()[1] == busy
        probe = socket.socket()
        try:
            with pytest.raises(OSError):
                probe.bind(("127.0.0.1", busy))
        finally:
            probe.close()


def test_a_free_port_is_used_as_preferred():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    port, note = launcher.pick_port(free)
    assert port == free and note is None


def test_an_invalid_port_is_never_used():
    assert launcher.port_is_free(80) is False
    assert launcher.port_is_free(70000) is False
    port, note = launcher.pick_port(80)
    assert port != 80 and note


def test_missing_resources_are_reported_before_anything_starts(tmp_path):
    missing = launcher.preflight(tmp_path)
    assert "ui/aa-live/index.html" in missing
    assert "ui/aa-live/analysis_records.js" in missing
    assert launcher.preflight(REPO_ROOT) == []


def test_prepare_state_never_overwrites_existing_files(tmp_path):
    paths = launcher.prepare_state(tmp_path / "state")
    paths["profile"].write_text('{"mine": true}', encoding="utf-8")
    again = launcher.prepare_state(tmp_path / "state")
    assert again["profile"].read_text(encoding="utf-8") == '{"mine": true}'
    assert again["records"].is_dir()


def _wait_ready(path, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(path).is_file():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        time.sleep(0.2)
    raise AssertionError("the launcher never reported readiness")


def test_the_launcher_serves_and_reads_the_same_records_after_a_restart(tmp_path):
    state = tmp_path / "state"
    # A record placed in the trial state before the first start.
    record_id = "20260101T000000-aaaaaaaaaaaa"
    folder = state / "records" / "analysis-records" / record_id
    folder.mkdir(parents=True)
    (folder / "record.json").write_text(json.dumps({
        "schema_version": 1, "record_id": record_id,
        "record_kind": "manual_hypothesis_analysis"}), encoding="utf-8")
    bases = []
    for attempt in (1, 2):
        ready = tmp_path / f"ready{attempt}.json"
        process = subprocess.Popen(
            [sys.executable, str(LAUNCHER), "--no-browser", "--state", str(state),
             "--port", "8791", "--ready-file", str(ready)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8")
        try:
            info = _wait_ready(ready)
            bases.append(info["base"])
            rules = json.loads(urllib.request.urlopen(
                info["base"] + "api/rules", timeout=10).read().decode("utf-8"))
            assert "conditional_analysis_ready" in rules
            listed = json.loads(urllib.request.urlopen(
                info["base"] + "api/analysis/records", timeout=10).read()
                .decode("utf-8"))
            ids = [row["record_id"] for row in listed["items"]]
            assert record_id in ids, listed
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        # The service really stopped before the next attempt.
        with pytest.raises(OSError):
            urllib.request.urlopen(bases[-1] + "api/rules", timeout=3).read()
    assert (state / "records" / "analysis-records" / record_id
            / "record.json").is_file()


def test_the_launcher_only_opens_the_browser_after_readiness():
    source = LAUNCHER.read_text(encoding="utf-8")
    ready = source.index("while not server.started")
    opened = source.index("webbrowser.open(base)")
    assert ready < opened, "the browser must be opened after the readiness wait"
    assert "allow_capture" not in source, "the trial never opens a capture device"
    for forbidden in ("os.kill", "taskkill", "terminate()", "replay_pool="):
        assert forbidden not in source, forbidden
