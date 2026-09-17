#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counting regressions for the REAL production statistics path.

Review STRATEGY-001-R1 (5217785775) pointed out that the earlier research notes
claimed "9 of the 18 negatives are double-counted". That claim was wrong, and the
correction must be defended by tests that cover the code which actually produced
the number -- `tools/validate_threeway_models.py` -- rather than a new isolated
boolean example.

The production rule is a single append inside the (group, world) loop:

    for group in plans:
        for world_name in protocol["worlds"]:
            ...
            if (delta < 0 or candidate.delta_vs_check_fold_chips < 0
                    or candidate.delta_vs_check_call_chips < 0):
                losses.append({...})

so at most ONE record is appended per scenario, no matter how many comparator
columns are negative. These tests re-run that tool on the repository's own input
and protocol and check the resulting report against an independently recomputed
expectation.
"""

import json
import os
import subprocess
import sys
from fractions import Fraction

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(REPO_ROOT, "configs", "strategy", "examples",
                     "threeway-river-response-manual.json")
PROTOCOL = os.path.join(REPO_ROOT, "configs", "strategy", "examples",
                        "threeway-validation-protocol-v1.json")
TOOL = os.path.join(REPO_ROOT, "tools", "validate_threeway_models.py")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """Run the real tool once and return its report."""
    # the tool refuses to write into an existing directory
    out = tmp_path_factory.mktemp("strategy001-counting") / "validate"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + "."
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, TOOL, "--input", INPUT, "--protocol", PROTOCOL,
         "--output", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(os.path.join(str(out), "report.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _is_negative(encoded) -> bool:
    """Strictly negative, evaluated on the exact Fraction string."""
    return Fraction(encoded["exact"]) < 0


def _scenario(key_record) -> tuple:
    return (key_record["group"], key_record["world"])


def _expected_loss_scenarios(report):
    """Independently recompute the loss set from the report's own evaluations."""
    expected = set()
    for record in report["evaluations"]:
        if not record["complete"]:
            continue
        selected = record["evaluations"]["training_selected"]
        delta_ref = Fraction(
            record["selected_minus_manual_reference_chips"]["exact"])
        worse = (
            delta_ref < 0
            or _is_negative(selected["delta_vs_check_fold_chips"])
            or _is_negative(selected["delta_vs_check_call_chips"])
        )
        if worse:
            expected.add(_scenario(record))
    return expected


def test_every_negative_is_one_scenario_record(report):
    """Property 1: a scenario with several negative comparators still yields ONE
    record. Checked by uniqueness of (group, world) in the real report."""
    losses = report["negative_comparisons"]
    keys = [_scenario(rec) for rec in losses]
    assert len(keys) == len(set(keys)), "a scenario produced more than one record"
    assert len(losses) == 18, "historical reading of 18 failing scenarios changed"


def test_scenario_with_multiple_negative_comparators_counts_once(report):
    """At least one real scenario has all three comparators negative; it must
    still be a single record."""
    losses = report["negative_comparisons"]
    multi = [rec for rec in losses
             if _is_negative(rec["delta_vs_reference"])
             and _is_negative(rec["delta_vs_check_fold"])
             and _is_negative(rec["delta_vs_check_call"])]
    assert multi, "expected at least one all-negative scenario in the fixtures"
    keys = [_scenario(rec) for rec in multi]
    assert len(keys) == len(set(keys))


def test_equivalent_comparator_columns_do_not_add_scenarios(report):
    """Property 2: columns whose value equals another column must not inflate the
    scenario count. The rule counts scenarios, not columns."""
    losses = report["negative_comparisons"]
    negative_columns = 0
    for rec in losses:
        for key in ("delta_vs_reference", "delta_vs_check_fold",
                    "delta_vs_check_call"):
            if _is_negative(rec[key]):
                negative_columns += 1
    assert negative_columns > len(losses), (
        "fixtures should contain rows with several negative columns; "
        "negative_columns=%d losses=%d" % (negative_columns, len(losses)))

    equal_ref_and_call = [rec for rec in losses
                          if rec["delta_vs_reference"] == rec["delta_vs_check_call"]]
    assert equal_ref_and_call, "expected ref == check/call rows in the fixtures"
    # Equal column values are recorded once, in a single row.
    keys = [_scenario(rec) for rec in equal_ref_and_call]
    assert len(keys) == len(set(keys))


def test_negative_set_matches_independent_recomputation(report):
    """The recorded loss set must equal the set implied by the report's own
    per-scenario deltas under the documented OR rule."""
    recorded = {_scenario(rec) for rec in report["negative_comparisons"]}
    expected = _expected_loss_scenarios(report)
    assert recorded == expected, (
        "recorded=%d expected=%d difference=%r"
        % (len(recorded), len(expected), sorted(recorded ^ expected)[:6]))


def test_equal_ev_across_different_scenarios_is_not_merged(report):
    """Property 3: identical deltas in different (group, world) pairs stay
    separate records -- equal EV is not evidence of the same information set."""
    losses = report["negative_comparisons"]
    by_delta = {}
    for rec in losses:
        by_delta.setdefault(
            (rec["delta_vs_reference"]["exact"], rec["delta_vs_check_fold"]["exact"],
             rec["delta_vs_check_call"]["exact"]), []).append(_scenario(rec))

    shared = {deltas: keys for deltas, keys in by_delta.items() if len(keys) > 1}
    assert shared, "fixtures should contain equal-delta scenarios in distinct worlds"
    for deltas, keys in shared.items():
        assert len(keys) == len(set(keys)), (
            "equal deltas across worlds must remain distinct records: %r" % (keys,))
        worlds = {world for _group, world in keys}
        groups = {group for group, _world in keys}
        assert len(groups) > 1 or len(worlds) > 1, (
            "a merged group would hide separate information sets: %r" % (keys,))


def test_shared_baselines_and_planning_inputs_unchanged(report):
    """The correction must not touch candidates, worlds, baselines or metrics."""
    assert report["world_cases"] == 30
    assert report["fixed_policy_evaluations"] == 60
    assert report["groups"] == 6
    assert report["screening_status"] == "FAIL_STRESS_SCREEN"
    assert report["promotion_decision"] == "NO_EMPIRICAL_PROMOTION"
    assert report["selected_id"] == "calling_public_v1"
    assert report["fallback_cases"] == []
