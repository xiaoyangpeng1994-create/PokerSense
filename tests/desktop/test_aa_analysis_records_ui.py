"""U2 end-to-end: compute this input -> save -> restart the service -> reopen.

Two real server processes share one records directory, so the restart is a real
restart rather than a re-request: phase one computes and saves, the process is
stopped, a new process starts on the same directory, and phase two reopens both
records and compares the numbers.
"""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FLOW_SERVER = REPO_ROOT / "tests" / "js" / "analysis_flow_server.py"
HARNESS = REPO_ROOT / "tests" / "js" / "hand_records_flow_test.mjs"
NODE = shutil.which("node")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class RunningServer:
    """One real AA server process, started and stopped by the test."""

    def __init__(self, work):
        self.work = Path(work)
        self.port = free_port()
        self.process = subprocess.Popen(
            [sys.executable, str(FLOW_SERVER), "--work", str(self.work),
             "--port", str(self.port)],
            cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        line = self.process.stdout.readline().strip()
        assert line, f"the server printed nothing: {self.process.stderr.read()[-800:]}"
        self.base = json.loads(line)["base"]
        import urllib.request
        deadline = time.time() + 60
        while True:
            try:
                urllib.request.urlopen(self.base + "/api/rules", timeout=5).read()
                break
            except OSError:
                if time.time() >= deadline:
                    raise AssertionError("the server never became reachable")
                time.sleep(0.2)

    @property
    def records(self):
        return self.work / "records"

    def stop(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
        return self.process.returncode


def run_phase(server, phase, handoff):
    env = {**os.environ, "RECORDS_DIR": str(server.records)}
    proc = subprocess.run(
        [NODE, str(HARNESS), server.base, phase, str(handoff)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600, env=env)
    checks = [json.loads(line) for line in proc.stdout.splitlines()
              if line.strip().startswith("{")]
    failures = [item for item in checks if not item.get("ok")]
    assert proc.returncode == 0, (
        f"{phase} failed checks: {failures}\nstderr: {proc.stderr[-2500:]}")
    return checks


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_a_saved_analysis_survives_a_real_service_restart(tmp_path):
    handoff = tmp_path / "handoff.json"
    work = tmp_path / "run"
    first = RunningServer(work)
    try:
        checks = run_phase(first, "run1", handoff)
        names = {item["name"] for item in checks}
        for required in (
                "the_page_scripts_are_loaded",
                "the_table_rules_are_saved_through_controls",
                "input_one_completes_on_the_real_kernel",
                "the_receipt_keeps_the_facts_this_analysis_used",
                "saving_the_current_analysis_creates_one_record",
                "the_saved_record_shows_this_input_not_a_fixed_sample",
                "the_saved_record_labels_its_source_kind",
                "input_two_completes_on_the_real_kernel",
                "the_second_analysis_is_a_second_record",
                "the_second_record_has_the_second_hand_numbers",
                "the_two_records_are_not_the_same_result",
                "a_repeated_save_does_not_create_a_second_record",
                "a_third_analysis_computes_for_the_stale_case",
                "changing_the_input_makes_the_old_job_unsavable",
                "the_fourth_analysis_computes_before_the_rules_change",
                "the_rules_really_changed",
                "changing_the_rules_does_not_touch_the_existing_records",
                "an_older_rules_record_is_still_viewable_as_history",
                "the_historical_record_says_it_is_not_a_current_recomputation",
                "recompute_under_saved_conditions_loads_the_saved_rules",
                "recompute_under_saved_conditions_does_not_change_the_table_rules",
                "recompute_with_current_rules_loads_the_facts_and_asks_for_a_recheck",
                "an_edited_display_value_is_reported_as_invalid",
                "the_ui_refuses_to_show_numbers_for_a_broken_record",
                "restoring_the_original_value_makes_it_readable_again"):
            assert required in names, f"missing phase-1 check: {required}"
        assert handoff.is_file(), "phase one did not hand off its record ids"
    finally:
        first.stop()
    # The service really is gone before the second one starts.
    with pytest.raises(OSError):
        import urllib.request
        urllib.request.urlopen(first.base + "/api/rules", timeout=3).read()
    handoff_data = json.loads(handoff.read_text(encoding="utf-8"))
    assert len(handoff_data["ids"]) == 2

    second = RunningServer(work)
    try:
        checks = run_phase(second, "run2", handoff)
        names = {item["name"] for item in checks}
        for required in (
                "run2_the_records_survive_the_restart",
                "run2_every_record_still_passes_its_self_check",
                "run2_record_1_reopens_with_the_same_numbers",
                "run2_record_2_reopens_with_the_same_numbers",
                "run2_reopen_did_not_run_the_kernel"):
            assert required in names, f"missing phase-2 check: {required}"
    finally:
        second.stop()


def test_the_page_serves_the_records_panel():
    page = (REPO_ROOT / "ui" / "aa-live" / "index.html").read_text(
        encoding="utf-8")
    assert "/analysis_records.js" in page
    for anchor in ('id="records-panel"', 'id="records-save"', 'id="records-select"',
                   'id="records-open"', 'id="records-view"',
                   'id="records-recompute-saved"', 'id="records-recompute-current"',
                   'id="records-refresh"', 'id="records-label"'):
        assert anchor in page, anchor
    source = (REPO_ROOT / "ui" / "aa-live" / "analysis_records.js").read_text(
        encoding="utf-8")
    assert "innerHTML" not in source
    assert "createElement" in source and "textContent" in source
