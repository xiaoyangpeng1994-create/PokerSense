"""Development-only manual range/response uncertainty study, never real calibration."""

import argparse
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import json

from poker_engine.strategy.robust_policy_selection_v1 import (
    WorldHypothesis, select_robust_policy, validate_robust_selection,
)
from poker_engine.strategy.threeway_policy_evaluation_v1 import compile_policy_book
from tools.analyze_terminal_multiway import unique_object
from tools.validate_threeway_models import (
    WORLDS, digest, dump, public_candidates, public_scenario, world_scenario,
)


def challenges(plan):
    """New handwritten development challenges, not an untouched opponent sample."""
    ranges = {r.seat_id: r for r in plan.ranges}
    mixed = tuple(replace(ranges[s], combo_weights={a: Decimal(weight), b: Decimal(1)})
                  for s, a, b, weight in (
                      (1, "JhJd", "TcTd", 4), (2, "7c7s", "8c8d", 3)))
    candidates = public_candidates((1, 2))
    return (
        WorldHypothesis("manual-mixed-value-cautious", replace(
            plan, ranges=mixed, models=candidates[0].models)),
        WorldHypothesis("manual-mixed-value-aggressive", replace(
            plan, ranges=mixed, models=candidates[2].models)),
    )


def run_study(input_path, output, *, max_nodes=20000):
    raw = input_path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    plans = {f"n{n}-{root}": public_scenario(data, n, root)
             for n in (6, 7, 8) for root in ("facing_bet", "unopened")}
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "protocol.json", {
        "input_sha256": digest(raw), "max_nodes_per_evaluation": max_nodes,
        "source_kind": "manual_hypothesis_study",
        "development_calibration_worlds": WORLDS,
        "challenge_worlds": [w.world_id for w in challenges(
            next(iter(plans.values())))],
        "candidate_policy_sources": ["manual_reference", "cautious_public_v1",
                                     "calling_public_v1", "aggressive_public_v1",
                                     "value_heavy_plan"],
        "objective": "max_min_delta_vs_stronger_of_check_fold_and_check_call",
        "calibration_worlds_previously_seen_development_regression": True,
        "challenge_worlds_also_manual_development_not_empirical_holdout": True,
        "no_full_hand_bb100_or_statistical_CI_or_profitability_claim": True,
    })
    candidates = public_candidates((1, 2))
    frozen = {}
    for group, plan in plans.items():
        value_world, _ = world_scenario(plan, "value_heavy", candidates[1])
        planning = [("manual_reference", plan)] + [
            (c.candidate_id, replace(plan, models=c.models)) for c in candidates
        ] + [("value_heavy_plan", value_world)]
        frozen[group] = tuple(compile_policy_book(
            p, policy_id=name, max_nodes=max_nodes) for name, p in planning)
    dump(output / "policy-books-before-comparison.json", frozen)
    reports = []
    for group, plan in plans.items():
        worlds = tuple(WorldHypothesis(name, *world_scenario(plan, name, candidates[1]))
                       for name in WORLDS)
        selection = select_robust_policy(frozen[group], worlds, max_nodes=max_nodes)
        dump(output / f"{group}-selection-before-challenges.json", selection)
        validation = validate_robust_selection(selection, challenges(plan))
        dump(output / f"{group}-validation.json", validation)
        reports.append({"group": group, "selection": selection,
                        "validation": validation})
    report = {"source_kind": "manual_hypothesis_study", "groups": reports,
              "strategy_eligible": False, "advice_emitted": False,
              "empirical_approval": False, "promotion": "NO_EMPIRICAL_PROMOTION"}
    dump(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-nodes", type=int, default=20000)
    args = parser.parse_args()
    try:
        report = run_study(args.input, args.output, max_nodes=args.max_nodes)
    except (ValueError, TypeError, ArithmeticError, OSError) as exc:
        parser.exit(2, f"uncertainty study rejected: {exc}\n")
    print(json.dumps([{"group": r["group"],
                       "selected": r["selection"].selected_id,
                       "status": r["validation"].status}
                      for r in report["groups"]], indent=2))
    return 2 if any(r["validation"].status == "INSUFFICIENT_EVIDENCE"
                    for r in report["groups"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
