"""Pair future offline candidate reports with the immutable BASELINE V1 cases.

This comparator does not create or optimize a candidate policy. A positive
synthetic delta is a diagnostic, never empirical acceptance or live advice.
"""

import argparse
from fractions import Fraction
import json

from tools.strategy_evaluation_v1 import (
    DEFAULT_RESULTS, digest, read_json, stable_json, write_json,
)

FROZEN_RESULT_SHA256 = (
    "c1a725f3f95a76261fb041af200e45bae02b6836202b36ec56578f0afd680aae")


def _stable(report):
    if not isinstance(report, dict):
        raise ValueError("evaluation_report_required")
    stable = {key: value for key, value in report.items()
              if key not in ("runtime", "deterministic_sha256")}
    if report.get("deterministic_sha256") != digest(
            stable_json(stable).encode()):
        raise ValueError("evaluation_report_digest_mismatch")
    if (stable.get("schema_version") != 1
            or stable.get("scope") != "SYNTHETIC_CONDITIONAL_RIVER_POLICY_EVALUATION"
            or stable.get("strategy_eligible") is not False
            or stable.get("advice_emitted") is not False
            or stable.get("real_hand_acceptance_pending") is not True):
        raise ValueError("evaluation_report_scope_or_safety_mismatch")
    cases = stable.get("cases")
    if (not isinstance(cases, list) or len(cases) != stable.get("case_count")
            or len({row["case_id"] for row in cases}) != len(cases)):
        raise ValueError("evaluation_cases_missing_or_duplicate")
    return stable, {row["case_id"]: row for row in cases}


def compare_reports(baseline, candidate):
    base_body, base_cases = _stable(baseline)
    candidate_body, candidate_cases = _stable(candidate)
    if (base_body.get("baseline_id") != "BASELINE_V1"
            or baseline["deterministic_sha256"] != FROZEN_RESULT_SHA256):
        raise ValueError("frozen_baseline_report_required")
    if set(base_cases) != set(candidate_cases):
        raise ValueError("unpaired_case_ids")
    rows = []
    for case_id, base in base_cases.items():
        proposed = candidate_cases[case_id]
        for key in ("declared_world_sha256", "joint_range_product"):
            if proposed.get(key) != base.get(key):
                raise ValueError("changed_evaluation_world:" + case_id)
        if base["status"] != "COMPLETE_CONDITIONAL_FIXED_POLICY":
            raise ValueError("baseline_case_not_complete:" + case_id)
        if proposed["status"] != "COMPLETE_CONDITIONAL_FIXED_POLICY":
            if (proposed.get("legal_joint_assignments") is not None
                    or proposed.get("world_scenario_sha256") is not None):
                raise ValueError("blocked_case_claims_complete_world:" + case_id)
            rows.append({"case_id": case_id, "status": "CANDIDATE_BLOCKED",
                         "reasons": proposed.get("reasons", [])})
            continue
        for key in ("world_scenario_sha256", "legal_joint_assignments"):
            if proposed.get(key) != base.get(key):
                raise ValueError("changed_evaluation_world:" + case_id)
        before = Fraction(base["metrics"]["frozen_policy"]["net_ev_chips"])
        after = Fraction(proposed["metrics"]["frozen_policy"]["net_ev_chips"])
        if (base["policy_book_sha256"] == proposed["policy_book_sha256"]
                and before != after):
            raise ValueError("unchanged_book_changed_EV:" + case_id)
        rows.append({"case_id": case_id, "status": "PAIRED",
                     "baseline_ev_chips": str(before),
                     "candidate_ev_chips": str(after),
                     "delta_chips": str(after - before),
                     "candidate_fallback_probability": proposed[
                         "metrics"]["frozen_policy"]["fallback_probability"]})
    paired = [row for row in rows if row["status"] == "PAIRED"]
    deltas = [Fraction(row["delta_chips"]) for row in paired]
    same_books = all(base_cases[key]["policy_book_sha256"]
                     == candidate_cases[key]["policy_book_sha256"]
                     for key in base_cases if candidate_cases[key]["status"]
                     == "COMPLETE_CONDITIONAL_FIXED_POLICY")
    status = ("INCOMPLETE" if len(paired) != len(base_cases)
              else "SELF_CONTROL" if same_books
              else "COMPARABLE_SYNTHETIC")
    return {
        "schema_version": 1, "status": status,
        "baseline_report_sha256": baseline["deterministic_sha256"],
        "candidate_report_sha256": candidate["deterministic_sha256"],
        "case_count": len(rows), "paired_count": len(paired),
        "candidate_blocked_count": len(rows) - len(paired),
        "positive_delta_count": sum(value > 0 for value in deltas),
        "negative_delta_count": sum(value < 0 for value in deltas),
        "equal_delta_count": sum(value == 0 for value in deltas),
        "cases": rows,
        "qualification": (
            "paired_declared_synthetic_worlds_only",
            "world_model_assumptions_are_not_real_opponent_truth",
            "no_empirical_promotion_or_profitability_or_GTO_claim",
        ),
        "strategy_eligible": False, "advice_emitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=DEFAULT_RESULTS)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = compare_reports(read_json(args.baseline),
                                 read_json(args.candidate))
        if args.output:
            write_json(args.output, report)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"strategy comparison rejected: {exc}\n")
    print(json.dumps({key: report[key] for key in (
        "status", "case_count", "paired_count", "candidate_blocked_count",
        "positive_delta_count", "negative_delta_count")}, indent=2))


if __name__ == "__main__":
    main()
