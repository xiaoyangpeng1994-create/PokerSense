"""Offline terminal-path diagnosis and exact EV reconciliation, not live advice.

Fixed policies are frozen by `compile_policy_book`, then one declared world is
evaluated with `trace=True`. Every terminal public path of each policy is
reported with its exact reach probability, its conditional terminal EV (defined
only when the path is reachable) and its weighted contribution; the sum of the
contributions reproduces the already-reported policy EV exactly, and the
difference of two policies on the same path reproduces the reported EV delta.

The input is an illustrative synthetic world, not an observed AA range set or a
calibrated response model. It is deliberately NOT part of the declared stress
screen worlds/candidates and adds no policy family. Only the offline
conditional expectation of the given world is reported: this is neither a proof
that an action is optimal nor a profitability, GTO or live-play claim.

    $env:PYTHONPATH='src;.'
    python tools/threeway_path_diagnostics.py \
      --input configs/strategy/examples/threeway-path-diagnostics-v1.json
    python tools/threeway_path_diagnostics.py --check
"""

import argparse
from dataclasses import fields, is_dataclass
from decimal import Decimal
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    compare_policy_paths, compile_policy_book, evaluate_policy_book,
)
from tools.analyze_threeway_river import scenario_from_dict
from tools.analyze_terminal_multiway import unique_object


DEFAULT_INPUT = Path("configs/strategy/examples/threeway-path-diagnostics-v1.json")
DEFAULT_SAMPLE = Path(
    "configs/strategy/examples/threeway-path-diagnostics-sample-v1.json")
POLICIES = ("frozen_policy", "check_fold", "check_call")
COMPARISONS = (("frozen_policy", "check_fold"), ("frozen_policy", "check_call"))
SCOPE = "OFFLINE_FIXED_POLICY_TERMINAL_PATH_DIAGNOSIS_NOT_LIVE_ADVICE"
QUANTITY_ENCODING = "exact_rational_string_n_over_d"


def exact(value):
    """Exact rational string; `None` stays `None` for an undefined quantity.

    Chips are reported as exact fractions of one chip, so no rounding or float
    appears anywhere in the report. Decimal readings are derivable but are not
    stored, which keeps the sample small and unambiguous.
    """
    if value is None:
        return None
    if not isinstance(value, Fraction):
        raise ValueError("exact_fraction_required")
    return str(value)


def encode(value):
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    return value


def history_rows(history):
    return [{"actor": a.actor, "kind": a.kind, "target": str(a.target)}
            for a in history]


def ledger_report(ledger):
    return {
        "policy_name": ledger.policy_name,
        "completeness": ledger.completeness,
        "policy_book_sha256": ledger.policy_book_sha256,
        "world_scenario_sha256": ledger.world_scenario_sha256,
        "terminal_paths": len(ledger.paths),
        "reachable_paths": ledger.reachable_paths,
        "unreachable_paths": ledger.unreachable_paths,
        "reach_probability_sum": exact(ledger.reach_probability_sum),
        "contribution_sum_chips": exact(ledger.contribution_sum_chips),
        "contribution_sum_bb": exact(ledger.contribution_sum_bb),
        "fallback_reach_sum": exact(ledger.fallback_reach_sum),
        "nodes_visited": ledger.nodes_visited,
        "trace_node_budget": ledger.trace_node_budget,
        "qualification": list(ledger.qualification),
        "paths": [{
            "history_key": path.history_key,
            "history": history_rows(path.history),
            "status": path.status,
            "fallback_used": path.fallback_used,
            "reach_probability": exact(path.reach_probability),
            "conditional_terminal_net_ev_chips": exact(
                path.conditional_terminal_net_ev_chips),
            "contribution_chips": exact(path.contribution_chips),
        } for path in ledger.paths],
    }


def reconciliation_report(reconciliation):
    counts = {status: reconciliation.reach_status_count(status) for status in (
        "REACHED_BY_BOTH", "REACHED_BY_LEFT_ONLY", "REACHED_BY_RIGHT_ONLY",
        "UNREACHABLE_BY_BOTH")}
    return {
        "left_policy": reconciliation.left_policy,
        "right_policy": reconciliation.right_policy,
        "completeness": reconciliation.completeness,
        "policy_book_sha256": reconciliation.policy_book_sha256,
        "world_scenario_sha256": reconciliation.world_scenario_sha256,
        "total_ev_difference_chips": exact(
            reconciliation.total_ev_difference_chips),
        "contribution_difference_sum_chips": exact(
            reconciliation.contribution_difference_sum_chips),
        "reach_status_counts": counts,
        "rows": [{
            "history_key": row.history_key,
            "history": history_rows(row.history),
            "reach_status": row.reach_status,
            "reach_probability_left": exact(row.reach_probability_left),
            "reach_probability_right": exact(row.reach_probability_right),
            "contribution_chips_left": exact(row.contribution_chips_left),
            "contribution_chips_right": exact(row.contribution_chips_right),
            "contribution_chips_difference": exact(
                row.contribution_chips_difference),
            "conditional_terminal_net_ev_chips_left": exact(
                row.conditional_terminal_net_ev_chips_left),
            "conditional_terminal_net_ev_chips_right": exact(
                row.conditional_terminal_net_ev_chips_right),
        } for row in reconciliation.rows],
    }


def build_report(input_path, *, max_joint_assignments=128, max_nodes=20000,
                 trace_max_nodes=None):
    """Run one offline fixed-policy path diagnosis on a declared sample world."""
    raw = input_path.read_bytes()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    scenario = scenario_from_dict(data)
    book = compile_policy_book(scenario, max_joint_assignments=max_joint_assignments,
                               max_nodes=max_nodes)
    evaluation = evaluate_policy_book(
        book, scenario, max_joint_assignments=max_joint_assignments,
        max_nodes=max_nodes, trace=True, trace_max_nodes=trace_max_nodes)
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "quantity_encoding": QUANTITY_ENCODING,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "policy_book": {
            "policy_id": book.policy_id,
            "book_sha256": book.book_sha256,
            "planning_scenario_sha256": book.planning_scenario_sha256,
            "planning_best_ev": exact(book.planning_best_ev),
            "public_conditions_sha256": hashlib.sha256(
                book.public_conditions_json.encode("utf-8")).hexdigest(),
            "decisions": [{
                "history": history_rows(item.history),
                "action": {"actor": item.action.actor, "kind": item.action.kind,
                           "target": str(item.action.target)},
            } for item in book.decisions],
        },
        "status": evaluation.status,
        "reasons": list(evaluation.reasons),
        "evaluation": {
            "nodes": evaluation.nodes,
            "policy_hash_before": evaluation.policy_hash_before,
            "policy_hash_after": evaluation.policy_hash_after,
            "world_scenario_sha256": evaluation.world_scenario_sha256,
            "response_family": evaluation.response_family,
            "history_likelihood": exact(evaluation.history_likelihood),
            "root_posterior": [exact(v) for v in evaluation.root_posterior],
            "metrics": [{
                "name": metric.name,
                "net_ev_chips": exact(metric.net_ev_chips),
                "conditional_net_ev_bb": exact(metric.conditional_net_ev_bb),
                "probability_of_any_fallback": exact(
                    metric.probability_of_any_fallback),
                "expected_fallback_count": exact(metric.expected_fallback_count),
                "unsupported_histories": len(metric.unsupported_histories),
                "nodes": metric.nodes,
            } for metric in evaluation.metrics],
            "delta_vs_check_fold_chips": exact(
                evaluation.delta_vs_check_fold_chips),
            "delta_vs_check_call_chips": exact(
                evaluation.delta_vs_check_call_chips),
        },
        "trace": {
            "enabled": bool(evaluation.path_ledgers),
            "trace_node_budget": (
                evaluation.path_ledgers[0].trace_node_budget
                if evaluation.path_ledgers else None),
            "ledger_completeness": sorted(
                {ledger.completeness for ledger in evaluation.path_ledgers}),
        },
        "ledgers": [ledger_report(ledger) for ledger in evaluation.path_ledgers],
        "reconciliations": (
            [reconciliation_report(compare_policy_paths(evaluation, *pair))
             for pair in COMPARISONS]
            if evaluation.path_ledgers else []),
        "qualification": [
            "fixed_policies_are_frozen_before_the_world_is_evaluated",
            "the_evaluated_world_never_calls_an_optimizing_entry_point",
            "manual_synthetic_world_not_observed_ranges_or_calibrated_models",
            "conditional_river_chips_not_bb_per_100_and_no_confidence_interval",
            "strategy_eligible_and_advice_emitted_are_false_by_construction",
        ],
        "strategy_eligible": evaluation.strategy_eligible,
        "advice_emitted": evaluation.advice_emitted,
    }
    return report


def summary(report):
    """Small human-readable digest; the full report stays machine readable."""
    evaluation = report["evaluation"]
    return {
        "scope": report["scope"],
        "status": report["status"],
        "reasons": report["reasons"],
        "policy_book_sha256": report["policy_book"]["book_sha256"],
        "world_scenario_sha256": evaluation["world_scenario_sha256"],
        "metrics": {m["name"]: m["net_ev_chips"]
                    for m in evaluation["metrics"]},
        "delta_vs_check_fold_chips": evaluation["delta_vs_check_fold_chips"],
        "delta_vs_check_call_chips": evaluation["delta_vs_check_call_chips"],
        "ledgers": [{
            "policy_name": ledger["policy_name"],
            "terminal_paths": ledger["terminal_paths"],
            "reachable_paths": ledger["reachable_paths"],
            "unreachable_paths": ledger["unreachable_paths"],
            "reach_probability_sum": ledger["reach_probability_sum"],
            "contribution_sum_chips": ledger["contribution_sum_chips"],
            "fallback_reach_sum": ledger["fallback_reach_sum"],
        } for ledger in report["ledgers"]],
        "reconciliations": [{
            "policies": f"{item['left_policy']}-{item['right_policy']}",
            "total_ev_difference_chips": item["total_ev_difference_chips"],
            "contribution_difference_sum_chips": item[
                "contribution_difference_sum_chips"],
            "rows": len(item["rows"]),
            "reach_status_counts": item["reach_status_counts"],
        } for item in report["reconciliations"]],
    }


def render(report):
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-joint-assignments", type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--trace-max-nodes", type=int)
    args = parser.parse_args()
    try:
        report = build_report(args.input,
                              max_joint_assignments=args.max_joint_assignments,
                              max_nodes=args.max_nodes,
                              trace_max_nodes=args.trace_max_nodes)
    except (OSError, TypeError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"threeway path diagnosis rejected: {exc}\n")
    rendered = render(report)
    if args.check:
        try:
            committed = json.loads(args.sample.read_bytes().decode("utf-8"),
                                   object_pairs_hook=unique_object)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"committed sample unreadable: {exc}\n")
        if committed != report:
            parser.exit(2, f"committed sample differs from {args.input}\n")
    if args.output:
        if args.output.exists():
            parser.exit(2, f"refusing to overwrite {args.output}\n")
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    return 0 if report["status"].startswith("COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
