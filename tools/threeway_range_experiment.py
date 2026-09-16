"""One-factor opponent-range sensitivity for a frozen policy, offline only.

Stage C of STRATEGY-DIAG. The frozen protocol
(`configs/strategy/examples/strategy-diag-c-range-protocol-v1.json`) declares one
original failing stress scenario, one opponent combo subset and the relative
weight factors 0.5 / 1 / 2. Both policy books are compiled ONCE from the
original planning scenario and reused in every world, so no evaluation world
re-plans Hero; every world is evaluated with the B-stage trace and each reading
is exactly reconciled into terminal-path contributions.

Scope: the offline conditional EV of a declared synthetic world under a fixed
policy. It is not an optimal-action proof, not an improvement claim, not
bb/100, not profitability, not real-opponent validation and not live advice.

    $env:PYTHONPATH='src;.'
    python tools/threeway_range_experiment.py --check
    python tools/threeway_range_experiment.py --output <new-file.json>
"""

import argparse
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import time

from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    compare_policy_paths, compile_policy_book, evaluate_policy_book,
    reconcile_path_ledgers,
)
from tools.analyze_terminal_multiway import exact_string, unique_object
from tools.validate_threeway_models import (
    freeze_candidates, public_candidates, public_scenario, select_candidate,
    synthetic_observations, validate_protocol, world_scenario,
)


DEFAULT_PROTOCOL = Path(
    "configs/strategy/examples/strategy-diag-c-range-protocol-v1.json")
DEFAULT_STUDY_PROTOCOL = Path(
    "configs/strategy/examples/threeway-validation-protocol-v1.json")
DEFAULT_INPUT = Path(
    "configs/strategy/examples/threeway-river-response-manual.json")
DEFAULT_SAMPLE = Path(
    "configs/strategy/examples/strategy-diag-c-range-experiment-v1.json")
SCOPE = "OFFLINE_FIXED_POLICY_SINGLE_FACTOR_RANGE_SENSITIVITY_NOT_LIVE_ADVICE"
FROZEN_STATUS = "FROZEN_BEFORE_ANY_PERTURBED_WORLD_WAS_RUN"
QUANTITY_ENCODING = "exact_rational_string_n_over_d"
VOLATILE_KEYS = ("elapsed_ms",)
BOOK_NAMES = ("manual_reference", "training_selected")
FAILURE_GATES = (("selected_minus_manual_reference", "delta_vs_reference"),
                 ("selected_delta_vs_check_fold", "delta_vs_check_fold"),
                 ("selected_delta_vs_check_call", "delta_vs_check_call"))
TRACKED_QUANTITIES = (
    "training_selected_frozen_policy", "manual_reference_frozen_policy",
    "selected_minus_manual_reference", "selected_delta_vs_check_fold",
    "selected_delta_vs_check_call", "history_likelihood",
)


def exact(value):
    """Exact rational string, or None for an undefined quantity."""
    if value is None:
        return None
    if not isinstance(value, Fraction):
        raise ValueError("exact_fraction_required")
    return str(value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


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


def as_path(value):
    """Accept a Path or a string path so the driver is callable in-process."""
    if isinstance(value, (str, os.PathLike)):
        return Path(value)
    raise ValueError("path_like_required")


def group_key(text):
    """Split the declared study group into its table size and root."""
    match = re.fullmatch(r"n(\d+)-(facing_bet|unopened)", text) \
        if isinstance(text, str) else None
    if match is None:
        raise ValueError("declared_study_group_required")
    return int(match.group(1)), match.group(2)


def load_protocol(path):
    """Read and strictly validate the frozen experiment protocol."""
    raw = as_path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or data.get("scope") != SCOPE
            or data.get("status_of_this_file") != FROZEN_STATUS
            or data.get("strategy_eligible") is not False
            or data.get("advice_emitted") is not False
            or re.fullmatch(r"[0-9a-f]{40}", data.get("parent_head", "")) is None):
        raise ValueError("declared_frozen_protocol_required")
    scenario, varied = data["original_failing_scenario"], data["single_varied_object"]
    group_key(scenario["study_group"])
    if (scenario["study_tool"] != "tools/validate_threeway_models.py"
            or varied["kind"] != "relative_weight_of_one_opponent_combo_subset"
            or varied["target_opponent_seat"] != scenario["applicability_check"][
                "target_opponent_seat"]
            or sorted(varied["untouched_combos_of_that_opponent"])
            != sorted(set(scenario["applicability_check"]["target_opponent_combos"])
                      - set(varied["target_combos"]))
            or varied["baseline_factor"] != "1"
            or varied["subset_is_nonempty_and_not_the_whole_range"] is not True):
        raise ValueError("declared_single_varied_opponent_range_required")
    factors = varied["relative_factors"]
    if factors != ["0.5", "1", "2"]:
        raise ValueError("exactly_the_three_declared_factors_required")
    if (not varied["target_combos"] or len(set(varied["target_combos"]))
            != len(varied["target_combos"])
            or len(varied["target_combos"]) == len(
                scenario["applicability_check"]["target_opponent_combos"])):
        raise ValueError("subset_must_be_nonempty_and_not_the_whole_range")
    if data["compute"]["trace"] is not True:
        raise ValueError("traced_evaluation_required")
    return data, raw


def perturb_target_range(scenario, seat_id, combos, factor):
    """Scale one opponent's declared combo subset, leaving everything else.

    Only the weights of the declared subset of the declared opponent change.
    Everything else - the other opponent, every response model, the rules, the
    seats, the public history and the action grids - is carried over untouched,
    so prior normalization, card blocking, history conditioning and the
    posterior are derived quantities rather than additional choices.
    """
    if not isinstance(factor, Decimal) or not factor.is_finite() or factor <= 0:
        raise ValueError("positive_exact_factor_required")
    target = next((item for item in scenario.ranges if item.seat_id == seat_id),
                  None)
    if target is None:
        raise ValueError("target_seat_is_not_an_opponent")
    if (not isinstance(combos, tuple) or not combos
            or len(set(combos)) != len(combos)):
        raise ValueError("nonempty_unique_combo_subset_required")
    if not set(combos) <= set(target.combo_weights):
        raise ValueError("subset_combo_not_in_declared_range")
    if len(combos) == len(target.combo_weights):
        raise ValueError("subset_must_not_be_the_whole_range")
    scaled = {combo: (weight * factor if combo in set(combos) else weight)
              for combo, weight in target.combo_weights.items()}
    return replace(scenario, ranges=tuple(
        replace(item, combo_weights=scaled) if item.seat_id == seat_id else item
        for item in scenario.ranges))


def planning_context(input_path, study_protocol_path, protocol):
    """Rebuild the production planning scenario and the selected candidate."""
    input_path, study_protocol_path = as_path(input_path), as_path(
        study_protocol_path)
    input_raw, study_raw = input_path.read_bytes(), study_protocol_path.read_bytes()
    data = json.loads(input_raw.decode("utf-8"), object_pairs_hook=unique_object)
    study = json.loads(study_raw.decode("utf-8"), object_pairs_hook=unique_object)
    validate_protocol(study)
    n, root = group_key(protocol["original_failing_scenario"]["study_group"])
    candidates = freeze_candidates(public_candidates((1, 2)))
    generator = next(c for c in candidates.candidates
                     if c.candidate_id == study["generating_candidate_id"])
    train = synthetic_observations(
        generator, partition="train", seed=study["training_seed"],
        sessions=study["training_sessions"],
        count=study["decisions_per_session"])
    selection = select_candidate(candidates, train)
    selected = next(c for c in candidates.candidates
                    if c.candidate_id == selection.selected_id)
    if (digest(input_raw) != protocol["original_failing_scenario"]["input_sha256"]
            or digest(study_raw)
            != protocol["original_failing_scenario"]["study_protocol_sha256"]):
        raise ValueError("protocol_declares_different_input_or_study_protocol")
    return public_scenario(data, n, root), selected, selection.selected_id


def freeze_books(plan, selected, protocol, max_nodes):
    """Compile both policy books once, from the original planning scenario."""
    books = {}
    for name, planning in (("manual_reference", plan),
                           ("training_selected",
                            replace(plan, models=selected.models))):
        books[name] = compile_policy_book(
            planning, policy_id=protocol["original_failing_scenario"]["study_group"]
            + "-" + name, max_nodes=max_nodes)
    declared = {item["name"]: item["policy_book_sha256"]
                for item in protocol["frozen_policy_books"]["books"]}
    for name, book in books.items():
        if book.book_sha256 != declared[name]:
            raise ValueError("declared_policy_book_hash_mismatch:" + name)
    return books


def metric_rows(evaluation):
    return [{
        "name": metric.name,
        "net_ev_chips": exact(metric.net_ev_chips),
        "conditional_net_ev_bb": exact(metric.conditional_net_ev_bb),
        "probability_of_any_fallback": exact(metric.probability_of_any_fallback),
        "expected_fallback_count": exact(metric.expected_fallback_count),
        "unsupported_histories": len(metric.unsupported_histories),
        "nodes": metric.nodes,
    } for metric in evaluation.metrics]


def ledger_rows(ledger):
    return {
        "terminal_paths": len(ledger.paths),
        "reachable_paths": ledger.reachable_paths,
        "unreachable_paths": ledger.unreachable_paths,
        "reach_probability_sum": exact(ledger.reach_probability_sum),
        "contribution_sum_chips": exact(ledger.contribution_sum_chips),
        "fallback_reach_sum": exact(ledger.fallback_reach_sum),
        "nodes_visited": ledger.nodes_visited,
    }


def path_rows(ledger):
    return [{
        "history_key": path.history_key,
        "status": path.status,
        "fallback_used": path.fallback_used,
        "reach_probability": exact(path.reach_probability),
        "conditional_terminal_net_ev_chips": exact(
            path.conditional_terminal_net_ev_chips),
        "contribution_chips": exact(path.contribution_chips),
    } for path in ledger.paths]


def reconciliation_rows(reconciliation):
    """Nonzero rows only: zero rows cannot change an exactly summed delta."""
    moved = [row for row in reconciliation.rows
             if row.contribution_chips_difference]
    return {
        "left_policy": reconciliation.left_policy,
        "right_policy": reconciliation.right_policy,
        "left_policy_book_sha256": reconciliation.policy_book_sha256,
        "right_policy_book_sha256": reconciliation.right_policy_book_sha256,
        "distinct_books": reconciliation.distinct_books,
        "world_scenario_sha256": reconciliation.world_scenario_sha256,
        "completeness": reconciliation.completeness,
        "total_ev_difference_chips": exact(
            reconciliation.total_ev_difference_chips),
        "contribution_difference_sum_chips": exact(
            reconciliation.contribution_difference_sum_chips),
        "rows_total": len(reconciliation.rows),
        "rows_with_nonzero_difference": len(moved),
        "zero_difference_rows": len(reconciliation.rows) - len(moved),
        "reach_status_counts": {
            status: reconciliation.reach_status_count(status) for status in (
                "REACHED_BY_BOTH", "REACHED_BY_LEFT_ONLY",
                "REACHED_BY_RIGHT_ONLY", "UNREACHABLE_BY_BOTH")},
        "rows": [{
            "history_key": row.history_key,
            "reach_status": row.reach_status,
            "contribution_chips_left": exact(row.contribution_chips_left),
            "contribution_chips_right": exact(row.contribution_chips_right),
            "contribution_chips_difference": exact(
                row.contribution_chips_difference),
        } for row in moved],
    }


def evaluate_world(book, world, overrides, compute):
    started = time.perf_counter()
    evaluation = evaluate_policy_book(
        book, world, world_overrides=overrides,
        max_joint_assignments=compute["max_joint_assignments"],
        max_nodes=compute["max_nodes"], trace=True,
        trace_max_nodes=compute["trace_max_nodes"])
    return evaluation, (time.perf_counter() - started) * 1000.0


def run_world(world, book, overrides, protocol, seat_id, compute):
    evaluation, elapsed_ms = evaluate_world(book, world, overrides, compute)
    record = {
        "status": evaluation.status,
        "reasons": list(evaluation.reasons),
        "elapsed_ms": round(elapsed_ms, 3),
        "perturbed_range": {
            "seat_id": seat_id,
            "combo_weights": [[combo, str(weight)] for combo, weight in
                              next(r for r in world.ranges
                                   if r.seat_id == seat_id).combo_weights.items()],
        },
        "policy_hash_before": evaluation.policy_hash_before,
        "policy_hash_after": evaluation.policy_hash_after,
        "policy_unchanged_during_world": (
            evaluation.policy_hash_before == evaluation.policy_hash_after),
        "metrics": metric_rows(evaluation),
        "delta_vs_check_fold_chips": exact(evaluation.delta_vs_check_fold_chips),
        "delta_vs_check_call_chips": exact(evaluation.delta_vs_check_call_chips),
        "nodes": evaluation.nodes,
        "trace": (ledger_rows(evaluation.path_ledgers[0])
                  if evaluation.path_ledgers else None),
        "terminal_paths": (path_rows(evaluation.path_ledgers[0])
                           if evaluation.path_ledgers else []),
        "reconciliations": (
            [reconciliation_rows(compare_policy_paths(
                evaluation, "frozen_policy", name))
             for name in ("check_fold", "check_call")]
            if evaluation.path_ledgers else []),
    }
    return record, evaluation


def run_experiment(input_path, protocol_path, study_protocol_path):
    input_path, study_protocol_path = as_path(input_path), as_path(
        study_protocol_path)
    protocol_path = as_path(protocol_path)
    protocol, protocol_raw = load_protocol(protocol_path)
    scenario, varied = protocol["original_failing_scenario"], protocol[
        "single_varied_object"]
    compute = protocol["compute"]
    plan, selected, selected_id = planning_context(
        input_path, study_protocol_path, protocol)
    books = freeze_books(plan, selected, protocol, compute["max_nodes"])
    before = {name: book.book_sha256 for name, book in books.items()}
    base_world, overrides = world_scenario(plan, scenario["study_world"], selected)
    seat_id, combos = varied["target_opponent_seat"], tuple(varied["target_combos"])
    factors = [exact_string(text) for text in varied["relative_factors"]]
    worlds, readings = [], {}
    for factor in factors:
        world = perturb_target_range(base_world, seat_id, combos, factor)
        world_readings = {}
        per_book = {}
        for name in BOOK_NAMES:
            record, evaluation = run_world(
                world, books[name], overrides, protocol, seat_id, compute)
            per_book[name] = record
            world_readings[name] = evaluation
        key = str(factor)
        selected_evaluation = world_readings["training_selected"]
        reference_evaluation = world_readings["manual_reference"]
        if (selected_evaluation.status != "COMPLETE_CONDITIONAL_FIXED_POLICY"
                or reference_evaluation.status
                != "COMPLETE_CONDITIONAL_FIXED_POLICY"):
            # A blocked world reports its reason; no partial or repaired reading.
            readings[key] = {name: None for name in TRACKED_QUANTITIES}
            worlds.append({
                "factor": key,
                "is_declared_baseline_factor": key == varied["baseline_factor"],
                "status": "BLOCKED",
                "reasons": sorted({reason for name in BOOK_NAMES
                                   for reason in per_book[name]["reasons"]}),
                "world_scenario_sha256": selected_evaluation.world_scenario_sha256,
                "history_likelihood": None,
                "joint_assignments": 0,
                "nonzero_posterior_entries": 0,
                "root_posterior": [],
                "books": per_book,
                "selected_minus_manual_reference_chips": None,
                "failure_gate_fired": None,
                "cross_book_branch_difference": None,
            })
            continue
        delta = (selected_evaluation.metrics[0].net_ev_chips
                 - reference_evaluation.metrics[0].net_ev_chips)
        cross = (reconciliation_rows(reconcile_path_ledgers(
            selected_evaluation.path_ledgers[0],
            reference_evaluation.path_ledgers[0], delta,
            allow_distinct_books=True))
            if selected_evaluation.path_ledgers
            and reference_evaluation.path_ledgers else None)
        readings[key] = {
            "training_selected_frozen_policy": exact(
                selected_evaluation.metrics[0].net_ev_chips),
            "manual_reference_frozen_policy": exact(
                reference_evaluation.metrics[0].net_ev_chips),
            "selected_minus_manual_reference": exact(delta),
            "selected_delta_vs_check_fold": exact(
                selected_evaluation.delta_vs_check_fold_chips),
            "selected_delta_vs_check_call": exact(
                selected_evaluation.delta_vs_check_call_chips),
            "history_likelihood": exact(selected_evaluation.history_likelihood),
        }
        worlds.append({
            "factor": key,
            "is_declared_baseline_factor": key == varied["baseline_factor"],
            "status": "COMPLETE_CONDITIONAL_FIXED_POLICY",
            "reasons": [],
            "world_scenario_sha256": selected_evaluation.world_scenario_sha256,
            "history_likelihood": exact(selected_evaluation.history_likelihood),
            "joint_assignments": len(selected_evaluation.root_posterior),
            "nonzero_posterior_entries": sum(
                1 for value in selected_evaluation.root_posterior if value),
            "root_posterior": [exact(v) for v in selected_evaluation.root_posterior],
            "books": per_book,
            "selected_minus_manual_reference_chips": exact(delta),
            "failure_gate_fired": [name for name, _ in FAILURE_GATES
                                   if Fraction(readings[key][name]) < 0],
            "cross_book_branch_difference": cross,
        })
    after = {name: book.book_sha256 for name, book in books.items()}
    baseline = verify_baseline(readings, worlds, protocol)
    report = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "scope": SCOPE,
        "quantity_encoding": QUANTITY_ENCODING,
        "parent_head": protocol["parent_head"],
        "protocol": {"path": protocol_path.as_posix(),
                     "sha256": digest(protocol_raw),
                     "frozen_status": protocol["status_of_this_file"]},
        "input": {"path": input_path.as_posix(),
                  "sha256": digest(input_path.read_bytes())},
        "study_protocol": {"path": study_protocol_path.as_posix(),
                           "sha256": digest(study_protocol_path.read_bytes())},
        "study_group": scenario["study_group"],
        "study_world": scenario["study_world"],
        "selected_id": selected_id,
        "fixed_single_varied_object": {
            "target_opponent_seat": seat_id,
            "target_combos": list(combos),
            "relative_factors": [str(factor) for factor in factors],
            "selection_rationale": varied["selection_rationale"],
        },
        "frozen_books": [{
            "name": name,
            "policy_book_sha256_before": before[name],
            "policy_book_sha256_after": after[name],
            "unchanged_across_worlds": before[name] == after[name],
            "planning_best_ev_chips": exact(books[name].planning_best_ev),
            "decisions": len(books[name].decisions),
        } for name in BOOK_NAMES],
        "baseline_verification": baseline,
        "worlds": worlds,
        "readings_by_factor": readings,
        "descriptive_direction_over_0.5_1_2": descriptive_direction(readings),
        "claims_not_made": list(protocol["claims_not_made"]),
        "strategy_eligible": False,
        "advice_emitted": False,
    }
    return report


def verify_baseline(readings, worlds, protocol):
    """Factor 1 must reproduce the original readings, or nothing is comparable."""
    declared = protocol["expected_baseline_at_factor_1"]
    observed = {key: readings["1"][key] for key in TRACKED_QUANTITIES
                if key in readings["1"]}
    world = next(item for item in worlds if item["is_declared_baseline_factor"])
    observed["world_scenario_sha256"] = world["world_scenario_sha256"]
    expected = {
        "training_selected_frozen_policy": declared[
            "training_selected_frozen_policy_net_ev_chips"],
        "manual_reference_frozen_policy": declared[
            "manual_reference_frozen_policy_net_ev_chips"],
        "selected_minus_manual_reference": declared[
            "training_selected_minus_manual_reference_chips"],
        "selected_delta_vs_check_fold": declared[
            "training_selected_delta_vs_check_fold_chips"],
        "selected_delta_vs_check_call": declared[
            "training_selected_delta_vs_check_call_chips"],
        "history_likelihood": declared["history_likelihood"],
        "world_scenario_sha256": declared["world_scenario_sha256"],
    }
    mismatches = sorted(key for key, value in expected.items()
                        if observed.get(key) != value)
    return {"declared": expected, "observed": observed,
            "matches_original_reading": not mismatches,
            "mismatches": mismatches}


def descriptive_direction(readings):
    """Factual direction of each reading as the factor increases, no claim."""
    order = [str(factor) for factor in
             (Decimal("0.5"), Decimal("1"), Decimal("2"))]
    direction = {}
    for key in TRACKED_QUANTITIES:
        encoded = [readings[factor][key] for factor in order]
        if any(value is None for value in encoded):
            direction[key] = "not_available"
            continue
        values = [Fraction(value) for value in encoded]
        if len(set(values)) == 1:
            direction[key] = "constant"
        elif values == sorted(values):
            direction[key] = "increasing"
        elif values == sorted(values, reverse=True):
            direction[key] = "decreasing"
        else:
            direction[key] = "non_monotone"
    return {"factor_order": order, "direction": direction,
            "qualification": "descriptive_only_no_improvement_claim"}


def stable_view(value):
    """Report with measured, non-deterministic timing fields removed."""
    if isinstance(value, dict):
        return {key: stable_view(item) for key, item in value.items()
                if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [stable_view(item) for item in value]
    return value


def summary(report):
    return {
        "experiment_id": report["experiment_id"],
        "study_group": report["study_group"],
        "study_world": report["study_world"],
        "selected_id": report["selected_id"],
        "fixed_single_varied_object": report["fixed_single_varied_object"],
        "frozen_books": report["frozen_books"],
        "baseline_verification": {
            "matches_original_reading": report["baseline_verification"][
                "matches_original_reading"],
            "mismatches": report["baseline_verification"]["mismatches"],
        },
        "readings_by_factor": report["readings_by_factor"],
        "descriptive_direction_over_0.5_1_2": report[
            "descriptive_direction_over_0.5_1_2"],
        "world_status": {item["factor"]: item["status"]
                         for item in report["worlds"]},
    }


def render(report):
    return json.dumps(encode(report), ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--study-protocol", type=Path,
                        default=DEFAULT_STUDY_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = run_experiment(args.input, args.protocol, args.study_protocol)
    except (OSError, TypeError, ValueError, ArithmeticError) as exc:
        parser.exit(2, f"threeway range experiment rejected: {exc}\n")
    if args.check:
        try:
            committed = json.loads(args.sample.read_bytes().decode("utf-8"),
                                   object_pairs_hook=unique_object)
        except (OSError, ValueError) as exc:
            parser.exit(2, f"committed sample unreadable: {exc}\n")
        if stable_view(committed) != stable_view(encode(report)):
            parser.exit(2, "committed sample differs after parsing "
                           "(timing fields excluded)\n")
    if args.output:
        if args.output.exists():
            parser.exit(2, f"refusing to overwrite {args.output}\n")
        args.output.write_text(render(report), encoding="utf-8", newline="\n")
    print(json.dumps(summary(report), ensure_ascii=False, indent=2))
    complete = (report["baseline_verification"]["matches_original_reading"]
                and all(item["status"] == "COMPLETE_CONDITIONAL_FIXED_POLICY"
                        for item in report["worlds"]))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
