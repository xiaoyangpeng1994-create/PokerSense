"""Product-flow interaction test for the offline study view.

Runs the shipped `ui/aa-live/study.js` unchanged inside a minimal DOM stub and
drives the real backend: load the SYNTHETIC example, read the comparison and the
path details, save an independent record, list records, reopen the same record,
supersede an in-flight open with a later one, and hit the rejection paths.

The DOM stub is not a browser. It exercises the shipped JavaScript and the real
HTTP contract; layout and visual acceptance stay a separate, explicitly reported
step.
"""

import json
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time

import pytest
import uvicorn

from poker_engine.desktop import aa_server


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "tests" / "js" / "study_flow_dom_stub_test.mjs"
NODE = shutil.which("node")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def study_server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("study-ui")
    records = directory / "records"
    app = aa_server.create_app(directory / "profile.json", records_dir=records)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the local AA server did not start"
    yield {"base": f"http://127.0.0.1:{port}", "records": str(records)}
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.skipif(
    NODE is None,
    reason="node is unavailable, so the shipped JS cannot be exercised here")
def test_study_flow_drives_the_shipped_javascript(study_server):
    proc = subprocess.run([NODE, str(HARNESS), study_server["base"],
                           study_server["records"]], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=240)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    checks = [json.loads(line) for line in lines]
    failures = [item for item in checks if not item.get("ok")]
    assert proc.returncode == 0, (
        f"failed checks: {failures}\nstderr: {proc.stderr[-2000:]}")
    verdict = checks[-1]
    assert verdict["name"] == "verdict" and verdict["ok"] is True
    assert verdict["failed"] == 0 and verdict["passed"] >= 30
    names = {item["name"] for item in checks}
    for required in ("wiring_ran", "preview_is_labeled_synthetic",
                     "preview_shows_readable_and_exact_values",
                     "preview_shows_main_paths", "saved_as_independent_record",
                     "reopen_identical_content", "reopen_identical_identity",
                     "late_response_does_not_overwrite",
                     "unknown_example_is_rejected",
                     "foreign_record_id_is_rejected", "no_innerhtml_used",
                     "no_vision_or_capture_requests",
                     "experiment_context_rendered",
                     "selection_change_invalidates_a_late_open",
                     "panel_close_invalidates_a_late_open",
                     "tampered_record_is_not_rendered_as_valid",
                     "tampered_record_reports_an_invalid_or_history_state",
                     "tampered_record_has_no_view"):
        assert required in names, f"missing check: {required}"


def test_study_script_is_served_and_uses_safe_text_nodes():
    app_client = aa_server.create_app
    assert app_client is not None
    source = (REPO_ROOT / "ui" / "aa-live" / "study.js").read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source
    page = (REPO_ROOT / "ui" / "aa-live" / "index.html").read_text(encoding="utf-8")
    assert "/study.js" in page
    assert 'id="study-tools"' in page and 'id="study-content"' in page
    assert 'id="study-record-select"' in page and 'id="study-record-open"' in page
