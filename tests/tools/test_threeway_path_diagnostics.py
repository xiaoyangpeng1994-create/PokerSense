#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end regression for the offline terminal-path diagnosis sample.

The committed sample must be the current output of the real tool on the
committed synthetic world, and it must reconcile exactly: every ledger's path
contributions sum to that policy's reported EV, and every comparison's
per-path differences sum to that pair's reported EV delta. Editing the input
world without regenerating the sample fails here.
"""

import json
import os
import re
import subprocess
import sys
from fractions import Fraction

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLES = os.path.join(REPO_ROOT, "configs", "strategy", "examples")
INPUT = os.path.join(EXAMPLES, "threeway-path-diagnostics-v1.json")
SAMPLE = os.path.join(EXAMPLES, "threeway-path-diagnostics-sample-v1.json")
TOOL = os.path.join(REPO_ROOT, "tools", "threeway_path_diagnostics.py")
POLICIES = ("frozen_policy", "check_fold", "check_call")


def run_tool(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + "."
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, TOOL, *[str(arg) for arg in args]], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env)


@pytest.fixture(scope="module")
def sample():
    with open(SAMPLE, encoding="utf-8") as handle:
        return json.load(handle)


def ledger_of(sample, name):
    return next(item for item in sample["ledgers"] if item["policy_name"] == name)


def metric_of(sample, name):
    return next(item for item in sample["evaluation"]["metrics"]
                if item["name"] == name)


def test_check_mode_accepts_the_committed_sample():
    proc = run_tool("--check")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "COMPLETE_CONDITIONAL_FIXED_POLICY" in proc.stdout
    assert "OFFLINE_FIXED_POLICY_TERMINAL_PATH_DIAGNOSIS_NOT_LIVE_ADVICE" in (
        proc.stdout)


def test_fresh_run_reproduces_the_committed_sample(tmp_path, sample):
    out = tmp_path / "diagnostics.json"
    proc = run_tool("--output", out)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(out.read_text(encoding="utf-8")) == sample


def test_sample_ledgers_reconcile_exactly(sample):
    assert sample["quantity_encoding"] == "exact_rational_string_n_over_d"
    assert {item["policy_name"] for item in sample["ledgers"]} == set(POLICIES)
    for name in POLICIES:
        item, metric = ledger_of(sample, name), metric_of(sample, name)
        assert item["completeness"] == "COMPLETE_TERMINAL_PATH_LEDGER"
        reach = sum(Fraction(path["reach_probability"])
                    for path in item["paths"])
        assert reach == 1
        assert Fraction(item["reach_probability_sum"]) == reach
        contributions = sum(Fraction(path["contribution_chips"])
                            for path in item["paths"])
        assert contributions == Fraction(item["contribution_sum_chips"])
        assert contributions == Fraction(metric["net_ev_chips"])
        assert item["terminal_paths"] == len(item["paths"])
        assert item["reachable_paths"] + item["unreachable_paths"] == len(
            item["paths"])
        assert item["policy_book_sha256"] == sample["policy_book"]["book_sha256"]
        assert item["world_scenario_sha256"] == (
            sample["evaluation"]["world_scenario_sha256"])
        for path in item["paths"]:
            reach = Fraction(path["reach_probability"])
            assert path["history_key"] == "|".join(
                f"{a['actor']}:{a['kind']}"
                + (f":{a['target']}" if a["kind"] in ("bet", "raise") else "")
                for a in path["history"])
            if reach == 0:
                assert path["status"] == "UNREACHABLE"
                assert path["conditional_terminal_net_ev_chips"] is None
                assert Fraction(path["contribution_chips"]) == 0
            else:
                assert path["status"] == "REACHED"
                assert Fraction(path["contribution_chips"]) == (
                    reach * Fraction(path["conditional_terminal_net_ev_chips"]))


def test_sample_comparisons_reconcile_on_the_same_path_union(sample):
    assert len(sample["reconciliations"]) == 2
    for reconciliation in sample["reconciliations"]:
        left, right = reconciliation["left_policy"], reconciliation["right_policy"]
        keys = {path["history_key"] for path in ledger_of(sample, left)["paths"]}
        assert {path["history_key"] for path in ledger_of(
            sample, right)["paths"]} == keys
        assert {row["history_key"] for row in reconciliation["rows"]} == keys
        assert reconciliation["completeness"] == (
            "RECONCILED_EXACT_PATH_CONTRIBUTIONS")
        differences = sum(Fraction(row["contribution_chips_difference"])
                          for row in reconciliation["rows"])
        assert differences == Fraction(
            reconciliation["contribution_difference_sum_chips"])
        total = (Fraction(metric_of(sample, left)["net_ev_chips"])
                 - Fraction(metric_of(sample, right)["net_ev_chips"]))
        assert differences == total
        assert differences == Fraction(reconciliation["total_ev_difference_chips"])
        assert reconciliation["reach_status_counts"] == {
            status: sum(1 for row in reconciliation["rows"]
                        if row["reach_status"] == status)
            for status in ("REACHED_BY_BOTH", "REACHED_BY_LEFT_ONLY",
                           "REACHED_BY_RIGHT_ONLY", "UNREACHABLE_BY_BOTH")}
        for row in reconciliation["rows"]:
            if row["contribution_chips_left"] == "0":
                assert Fraction(row["contribution_chips_right"]) == -Fraction(
                    row["contribution_chips_difference"])
            if row["reach_status"] == "UNREACHABLE_BY_BOTH":
                assert row["conditional_terminal_net_ev_chips_left"] is None
                assert row["conditional_terminal_net_ev_chips_right"] is None


def test_sample_is_sanitized_and_never_authorizes_live_advice(sample):
    text = json.dumps(sample, ensure_ascii=False)
    assert not re.search(r"[A-Za-z]:[\\/]", text)
    assert "/Users/" not in text and "/home/" not in text
    assert ".png" not in text and ".mkv" not in text
    assert sample["strategy_eligible"] is False
    assert sample["advice_emitted"] is False
    assert all(not item["fallback_used"] for ledger in sample["ledgers"]
               for item in ledger["paths"])
    assert all(Fraction(item["fallback_reach_sum"]) == 0
               for item in sample["ledgers"])


def test_tool_reports_blocked_budget_and_refuses_to_overwrite(tmp_path):
    blocked = run_tool("--trace-max-nodes", 1)
    assert blocked.returncode == 2
    assert "trace_node_budget_exceeded" in blocked.stdout
    assert '"status": "BLOCKED"' in blocked.stdout
    out = tmp_path / "once.json"
    assert run_tool("--output", out).returncode == 0
    again = run_tool("--output", out)
    assert again.returncode == 2
    assert "refusing to overwrite" in again.stderr
