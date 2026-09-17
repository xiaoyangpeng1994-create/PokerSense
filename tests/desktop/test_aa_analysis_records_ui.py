"""U2-R1 end-to-end: compute -> save -> reopen -> recompute BOTH ways -> save a
new result each time -> restart the service -> reopen everything again.

Three real server processes share one records directory, so every restart is a
real restart rather than a re-request.
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

PHASE_ONE = (
    "the_page_scripts_are_loaded",
    "the_table_rules_are_saved_through_controls",
    "input_one_completes_on_the_real_kernel",
    "the_receipt_keeps_the_facts_this_analysis_used",
    "saving_the_current_analysis_creates_one_record",
    "the_saved_record_shows_this_input_not_a_fixed_sample",
    "the_saved_record_labels_its_source_kind",
    "the_saved_record_states_the_sample_type_in_words",
    "the_saved_record_shows_the_assumptions_and_the_rule_values",
    "input_two_completes_on_the_real_kernel",
    "the_second_analysis_is_a_second_record",
    "the_second_record_has_the_second_hand_numbers",
    "the_two_records_have_different_facts_AND_different_assumptions",
    "a_repeated_save_does_not_create_a_second_record",
    "a_third_analysis_computes_for_the_stale_case",
    "changing_the_input_makes_the_old_job_unsavable",
    "the_fourth_analysis_computes_before_the_rules_change",
    "the_rules_really_changed",
    "changing_the_rules_does_not_touch_the_existing_records",
    "an_older_rules_record_is_still_viewable_as_history",
    "the_historical_record_says_it_is_not_a_current_recomputation",
    # A / P1 (U2-R2): the rule-source control must really switch the mode.
    "the_saved_conditions_load_selects_the_records_own_rules",
    "the_saved_scenario_verifies_as_a_document_rule_input",
    "ticking_the_real_rule_source_control_leaves_the_saved_scenario",
    "the_switch_keeps_the_hand_the_assumptions_and_the_parent_link",
    "the_switched_verify_asks_the_server_for_the_TABLE_rules",
    "the_switched_recompute_completes_on_the_real_kernel",
    "the_switched_recompute_saves_as_a_NEW_record",
    "the_switched_record_uses_THIS_table_rules_and_keeps_its_parent",
    "the_switched_rules_actually_changed_the_numbers",
    "switching_the_rule_source_during_an_in_flight_verify_discards_the_answer",
    # A / P1: the recompute must go back through the verified, invalidating flow.
    "the_recompute_starts_from_an_existing_receipt",
    "recompute_with_current_rules_voids_the_old_receipt_and_result",
    "recompute_with_current_rules_loads_A_facts_not_the_previous_hand",
    "recompute_with_current_rules_loads_A_own_opponent_assumptions",
    "recompute_with_current_rules_is_not_wired_to_an_analysis_record_id",
    "recompute_with_current_rules_does_not_change_the_table_rules",
    "the_current_rules_recompute_completes_on_the_real_kernel",
    "the_current_rules_recompute_saves_as_a_NEW_record",
    "the_new_current_rules_record_names_its_parent_and_its_own_rules",
    "the_two_current_rules_entries_reach_the_same_rules_by_different_controls",
    "the_original_A_record_is_unchanged_by_the_recompute",
    # B / P1: request identity, out-of-order, forged and failed responses.
    "an_out_of_order_open_paints_only_the_current_selection",
    "a_response_for_another_record_is_refused_without_numbers",
    "no_recompute_button_survives_a_refused_open",
    "a_failed_open_leaves_no_numbers_and_no_recompute",
    "a_late_recompute_answer_does_not_touch_a_form_cleared_while_it_flew",
    "a_late_list_refresh_does_not_steal_the_selection",
    # C / P1: the stored content is sealed and re-verified on every read path.
    "every_rewritten_content_field_is_refused_on_every_read_path",
    "a_refused_record_is_never_deleted_or_rewritten",
    "a_legacy_record_without_a_content_seal_is_not_silently_accepted",
    "restoring_the_original_file_makes_it_readable_again",
    "an_edited_display_value_is_reported_as_invalid",
    "the_ui_refuses_to_show_numbers_for_a_broken_record",
    # B / P2 (U2-R2): a wrong-shaped file, through the real routes.
    "a_wrong_shaped_file_is_listed_as_invalid_and_does_not_hide_the_others",
    "get_and_scenario_answer_a_wrong_shaped_file_with_a_refusal_not_a_crash",
    "a_wrong_shaped_file_is_never_deleted_or_rewritten",
    "the_records_panel_still_reads_the_healthy_records_after_a_bad_shape",
    "the_switched_record_equals_a_direct_kernel_call_on_its_own_input",
    "no_innerhtml_used",
)
PHASE_TWO = (
    "run2_record_1_reopens_with_the_same_numbers",
    "run2_record_2_reopens_with_the_same_numbers",
    "run2_record_3_reopens_with_the_same_numbers",
    "run2_record_4_reopens_with_the_same_numbers",
    "run2_the_parent_record_is_readable_before_the_saved_conditions_run",
    "run2_a_fresh_page_starts_without_a_receipt",
    "run2_saved_conditions_recompute_loads_the_frozen_facts_and_assumptions",
    "run2_saved_conditions_recompute_did_not_touch_the_global_rules",
    "run2_the_saved_conditions_input_reverifies_and_computes_on_the_kernel",
    "run2_the_saved_conditions_result_saves_as_a_NEW_record",
    "run2_the_child_record_names_its_parent_and_keeps_the_review_link",
    "run2_the_child_used_the_PARENT_rules_not_the_current_ones",
    "run2_the_parent_record_is_untouched_by_the_recompute",
    "run2_the_saved_conditions_recompute_did_not_change_the_table_rules",
    "run2_the_new_record_reopens_with_its_own_numbers",
)
PHASE_THREE = (
    "run3_record_1_reopens_with_the_same_numbers",
    "run3_record_2_reopens_with_the_same_numbers",
    "run3_record_3_reopens_with_the_same_numbers",
    "run3_record_4_reopens_with_the_same_numbers",
    "run3_record_5_reopens_with_the_same_numbers",
    "run3_every_record_survived_the_second_restart",
    "run3_every_record_still_passes_its_self_check",
    "run3_each_record_keeps_its_identity_and_numbers",
    "run3_every_recomputed_record_is_readable_as_itself",
    "run3_nothing_was_recomputed_on_any_reopen",
)


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
        errors="replace", timeout=900, env=env)
    checks = [json.loads(line) for line in proc.stdout.splitlines()
              if line.strip().startswith("{")]
    failures = [item for item in checks if not item.get("ok")]
    assert proc.returncode == 0, (
        f"{phase} failed checks: {failures}\n"
        f"stdout tail:\n{proc.stdout[-3000:]}\nstderr: {proc.stderr[-2500:]}")
    return checks


def require(checks, names, phase):
    present = {item["name"] for item in checks}
    for required in names:
        assert required in present, f"missing {phase} check: {required}"


@pytest.mark.skipif(NODE is None, reason="node is unavailable")
def test_both_recomputes_save_new_records_and_survive_two_real_restarts(tmp_path):
    handoff = tmp_path / "handoff.json"
    work = tmp_path / "run"
    first = RunningServer(work)
    try:
        checks = run_phase(first, "run1", handoff)
        require(checks, PHASE_ONE, "phase-1")
        assert handoff.is_file(), "phase one did not hand off its record ids"
    finally:
        first.stop()
    # The service really is gone before the second one starts.
    import urllib.request
    with pytest.raises(OSError):
        urllib.request.urlopen(first.base + "/api/rules", timeout=3).read()
    handoff_data = json.loads(handoff.read_text(encoding="utf-8"))
    assert len(handoff_data["ids"]) == 4, handoff_data["ids"]
    assert len(handoff_data["views"]) == 4

    second = RunningServer(work)
    try:
        checks = run_phase(second, "run2", handoff)
        require(checks, PHASE_TWO, "phase-2")
    finally:
        second.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(second.base + "/api/rules", timeout=3).read()
    handoff_data = json.loads(handoff.read_text(encoding="utf-8"))
    assert len(handoff_data["ids"]) == 5, handoff_data["ids"]

    third = RunningServer(work)
    try:
        checks = run_phase(third, "run3", handoff)
        require(checks, PHASE_THREE, "phase-3")
    finally:
        third.stop()


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
