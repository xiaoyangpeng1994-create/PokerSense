"""Synthetic model-selection and frozen-policy stress study, not profit proof."""

import argparse
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from poker_engine.strategy.response_model_calibration_v1 import (
    DecisionObservation, ModelCandidate, freeze_candidates, observation_probabilities,
    select_candidate, validate_selection,
)
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    PolicyEvaluation, WorldResponseOverride, compile_policy_book,
    evaluate_policy_book, policy_book_hash,
)
from poker_engine.strategy.threeway_river_v1 import (
    ResponseModel, RiverAction, _Node, _Tree,
)
from tools.analyze_threeway_river import scenario_from_dict
from tools.analyze_terminal_multiway import encode as scalar_encode
from tools.analyze_terminal_multiway import unique_object


PUBLIC_WEIGHTS = {
    "cautious_public_v1": (5, 8, 1, 1, 1),
    "calling_public_v1": (1, 8, 6, 2, 1),
    "aggressive_public_v1": (1, 2, 2, 6, 5),
}
KINDS = ("fold", "check", "call", "bet", "raise")
ROOTS = ("facing_bet", "unopened")
WORLDS = ("reference_control", "value_heavy", "weak_callers",
          "rank_aware", "check_trap")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    """Serialize immutable source contracts, including MappingProxyType ranges."""
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return scalar_encode(value)


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(encode(value), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def validate_protocol(p):
    keys = {"schema_version", "source_kind", "candidate_ids", "generating_candidate_id",
            "training_seed", "validation_seed", "training_sessions",
            "validation_sessions",
            "decisions_per_session", "table_sizes", "roots", "worlds", "screening_rule",
            "qualification"}
    if not isinstance(p, dict) or set(p) != keys:
        raise ValueError("exact preregistered protocol required")
    if (type(p["schema_version"]) is not int or p["schema_version"] != 1
            or p["source_kind"] != "synthetic_model_stress"
            or p["candidate_ids"] != list(PUBLIC_WEIGHTS)
            or not isinstance(p["generating_candidate_id"], str)
            or p["generating_candidate_id"] not in PUBLIC_WEIGHTS
            or p["table_sizes"] != [6, 7, 8] or p["roots"] != list(ROOTS)
            or p["worlds"] != list(WORLDS)
            or p["screening_rule"] != (
                "report_every_case_and_reject_promotion_"
                "on_any_negative_delta_or_fallback"
            )
            or not isinstance(p["qualification"], str) or not p["qualification"]):
        raise ValueError("only the declared synthetic study is supported")
    for k in ("training_seed", "validation_seed", "training_sessions",
              "validation_sessions", "decisions_per_session"):
        if type(p[k]) is not int or p[k] < 0:
            raise ValueError("explicit integer sampling parameters required")
    if (not 1 <= p["training_sessions"] <= 20 or not 1 <= p["validation_sessions"] <= 20
            or not 10 <= p["decisions_per_session"] <= 1000
            or p["training_seed"] == p["validation_seed"]):
        raise ValueError("bounded separate synthetic sampling streams required")


def public_candidates(opponents):
    return tuple(ModelCandidate(name, tuple(ResponseModel(
        seat, tuple(zip(KINDS, map(Fraction, weights))), (),
        ((Fraction(1, 5), ()), (Fraction(1), (
            ("fold", Fraction(2)), ("call", Fraction(4, 5))))),
    ) for seat in opponents)) for name, weights in PUBLIC_WEIGHTS.items())


def synthetic_observations(candidate, *, partition, seed, sessions, count):
    """Draw all synthetic decision opportunities, never select by showdown."""
    rng = random.Random(seed)
    rows = []
    for session in range(sessions):
        for index in range(count):
            model = candidate.models[index % len(candidate.models)]
            actor = model.seat_id
            if index % 3 == 0:
                menu = (RiverAction(actor, "check"),
                        RiverAction(actor, "bet", Decimal(20)),
                        RiverAction(actor, "bet", Decimal(40)))
                price = Fraction(0)
            else:
                menu = (RiverAction(actor, "fold"), RiverAction(actor, "call"),
                        RiverAction(actor, "raise", Decimal(40)))
                price = Fraction(1, 6) if index % 3 == 1 else Fraction(2, 5)
            sid = f"synthetic-{partition}-session{session}"
            rid = f"{sid}-decision{index}"
            row = DecisionObservation(
                "synthetic", sid, f"{sid}-group{index // 2}", rid,
                digest(rid.encode()), actor, None, menu, price, menu[0])
            probabilities = observation_probabilities(model, row)
            draw, cumulative = Fraction(rng.randrange(2 ** 53), 2 ** 53), Fraction(0)
            chosen = menu[-1]
            for action, p in zip(menu, probabilities):
                cumulative += p
                if draw < cumulative:
                    chosen = action
                    break
            payload = {"row": encode(row), "observed_action": encode(chosen)}
            source = digest(json.dumps(payload, sort_keys=True).encode())
            rows.append(replace(row, source_hash=source, observed_action=chosen))
    return tuple(rows)


def public_scenario(data, n, root):
    if root not in ROOTS:
        raise ValueError("undeclared_study_root")
    value = deepcopy(data)
    value["rules"]["table_size"] = n
    active = [s for s in value["seats"] if s["status"] == "ACTIVE"]
    if (len(active) != 3 or value["hero_seat"] != 0
            or {s["seat_id"] for s in active} != {0, 1, 2}):
        raise ValueError("study template requires Hero0 and active seats0/1/2")
    value["seats"] = active + [
        {"seat_id": i, "stack": "200", "hand_committed": "10", "status": "FOLDED"}
        for i in range(3, n)]
    if root == "unopened":
        value["history"] = []
        value["action_order"] = [0, 1, 2]
    scenario = scenario_from_dict(value)
    tree = _Tree(scenario, 128, 1)
    node = _Node(tree.active, scenario.action_order,
                 tuple(Decimal(0) for _ in scenario.seats),
                 Decimal(0), scenario.rules.big_blind, 0, ())
    for action in scenario.history:
        node = tree.advance(node, action)
    if not tree.legal(node) or node.pending[0] != scenario.hero_seat:
        raise ValueError("study_root_must_be_nonterminal_Hero_decision")
    owed = node.current_bet - node.wagers[tree.index[scenario.hero_seat]]
    if root == "facing_bet" and owed <= 0:
        raise ValueError("facing_bet_root_requires_positive_actual_to_call")
    if root == "unopened" and (owed != 0 or node.history):
        raise ValueError("unopened_root_requires_no_history_and_no_debt")
    return scenario


def world_scenario(plan, world, selected):
    ranges, models, overrides = plan.ranges, plan.models, ()
    by_seat = {r.seat_id: r for r in ranges}
    if world == "value_heavy":
        ranges = tuple(replace(by_seat[s], combo_weights={c: Decimal(1)})
                       for s, c in ((1, "JhJd"), (2, "7c7s")))
    elif world == "weak_callers":
        ranges = tuple(replace(by_seat[s], combo_weights={c: Decimal(1)})
                       for s, c in ((1, "TcTd"), (2, "KhTh")))
        models = selected.models
    elif world == "rank_aware":
        ranges = tuple(replace(by_seat[s], combo_weights={
            first: Decimal(1), second: Decimal(weight)})
            for s, first, second, weight in (
                (1, "KhKd", "TcTd", 1), (2, "7c7s", "8c8d", 2)))
        overrides = tuple(WorldResponseOverride(
            s, 13, (("fold", Fraction(1, 4)),
                    ("call", Fraction(2)), ("raise", Fraction(5))),
            (("bet", Fraction(4)), ("raise", Fraction(3))),
        ) for s in (1, 2))
    elif world == "check_trap":
        # A public-history response absent from the planner's category/price family.
        models = public_candidates((1, 2))[2].models
        overrides = tuple(WorldResponseOverride(
            s, None, (), (("bet", Fraction(15)), ("raise", Fraction(5))),
        ) for s in (1, 2))
    elif world != "reference_control":
        raise ValueError("undeclared evaluation world")
    return replace(plan, ranges=ranges, models=models), overrides


def run_study(input_path, protocol_path, output, *, max_nodes=20000):
    if type(max_nodes) is not int or not 1 <= max_nodes <= 200000:
        raise ValueError("bounded_integer_node_budget_required")
    input_raw, protocol_raw = input_path.read_bytes(), protocol_path.read_bytes()
    data = json.loads(input_raw, object_pairs_hook=unique_object)
    protocol = json.loads(protocol_raw, object_pairs_hook=unique_object)
    validate_protocol(protocol)
    scenario_from_dict(data)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "protocol-frozen.json", protocol)
    candidates = freeze_candidates(public_candidates((1, 2)))
    dump(output / "candidates-frozen.json", candidates)
    generator = next(c for c in candidates.candidates
                     if c.candidate_id == protocol["generating_candidate_id"])
    train = synthetic_observations(
        generator, partition="train", seed=protocol["training_seed"],
        sessions=protocol["training_sessions"],
        count=protocol["decisions_per_session"])
    dump(output / "training.json", train)
    selection = select_candidate(candidates, train)
    dump(output / "selection-before-validation.json", selection)
    validation = synthetic_observations(
        generator, partition="validation", seed=protocol["validation_seed"],
        sessions=protocol["validation_sessions"],
        count=protocol["decisions_per_session"])
    dump(output / "validation.json", validation)
    calibration = validate_selection(candidates, selection, validation)
    dump(output / "calibration-report.json", calibration)
    selected = next(c for c in candidates.candidates
                    if c.candidate_id == selection.selected_id)
    plans, books, planning_errors = {}, {}, {}
    for n in protocol["table_sizes"]:
        for root in protocol["roots"]:
            group = f"n{n}-{root}"
            plan = public_scenario(data, n, root)
            plans[group] = plan
            books[group] = {}
            for name, planning in (
                ("manual_reference", plan),
                ("training_selected", replace(plan, models=selected.models)),
            ):
                try:
                    books[group][name] = compile_policy_book(
                        planning, policy_id=group + "-" + name, max_nodes=max_nodes)
                except ValueError as exc:
                    books[group][name] = None
                    planning_errors[group + ":" + name] = str(exc)
    # No evaluation world has been executed at this point.
    dump(output / "policy-books-before-worlds.json", books)
    dump(output / "planning-errors.json", planning_errors)
    evaluations, losses, fallback_cases = [], [], []
    for group, plan in plans.items():
        for world_name in protocol["worlds"]:
            world, overrides = world_scenario(plan, world_name, selected)
            pair = {name: evaluate_policy_book(
                        book, world, world_overrides=overrides, max_nodes=max_nodes)
                    if book is not None else PolicyEvaluation(
                        "BLOCKED", reasons=(planning_errors[group + ":" + name],))
                    for name, book in books[group].items()}
            complete = all(r.status == "COMPLETE_CONDITIONAL_FIXED_POLICY"
                           for r in pair.values())
            record = {"group": group, "world": world_name,
                      "world_definition": encode({
                          "scenario": world, "overrides": overrides}),
                      "evaluations": encode(pair), "complete": complete}
            if complete:
                candidate = pair["training_selected"]
                reference = pair["manual_reference"]
                chosen = candidate.metrics[0]
                delta = chosen.net_ev_chips - reference.metrics[0].net_ev_chips
                record["selected_minus_manual_reference_chips"] = encode(delta)
                for i in (1, 2):
                    if (candidate.metrics[i].net_ev_chips
                            != reference.metrics[i].net_ev_chips):
                        raise ValueError("shared_public_baseline_changed")
                if (world_name == "reference_control"
                        and reference.metrics[0].net_ev_chips
                        != books[group]["manual_reference"].planning_best_ev):
                    raise ValueError("reference_self_control_does_not_match_plan")
                if (delta < 0 or candidate.delta_vs_check_fold_chips < 0
                        or candidate.delta_vs_check_call_chips < 0):
                    losses.append({"group": group, "world": world_name,
                                   "delta_vs_reference": encode(delta),
                                   "delta_vs_check_fold": encode(
                                       candidate.delta_vs_check_fold_chips),
                                   "delta_vs_check_call": encode(
                                       candidate.delta_vs_check_call_chips)})
                if chosen.probability_of_any_fallback > 0:
                    fallback_cases.append({"group": group, "world": world_name,
                                           "probability": encode(
                                               chosen.probability_of_any_fallback)})
            evaluations.append(record)
    all_complete = all(r["complete"] for r in evaluations)
    if any(policy_book_hash(b) != b.book_sha256
           for group in books.values() for b in group.values() if b is not None):
        raise ValueError("policy changed during validation study")
    report = {
        "schema_version": 1, "source_kind": "synthetic_model_stress",
        "input_sha256": digest(input_raw), "protocol_sha256": digest(protocol_raw),
        "calibration_status": calibration.status, "selected_id": selection.selected_id,
        "groups": len(plans), "world_cases": len(evaluations),
        "fixed_policy_evaluations": len(evaluations) * 2, "all_complete": all_complete,
        "screening_status": "INCOMPLETE" if not all_complete else (
            "FAIL_STRESS_SCREEN" if losses or fallback_cases
            else "PASS_DECLARED_STRESS_SCREEN"),
        "promotion_decision": "NO_EMPIRICAL_PROMOTION",
        "planning_errors": planning_errors,
        "negative_comparisons": losses, "fallback_cases": fallback_cases,
        "evaluations": evaluations,
        "assumptions": [
            "synthetic_candidate_selection_not_real_calibration_or_range_learning",
            "all_policy_books_fixed_before_evaluation_worlds_no_Hero_reoptimization",
            "rank_and_history_worlds_extend_planner_response_family",
            "shared_rules_kernel_not_an_independent_implementation",
            "conditional_river_EV_only_not_full_hand_bb100_or_profitability",
            "exact_given_world_expectations_no_statistical_confidence_interval_claim",
            "handwritten_stress_worlds_not_representative_opponent_population",
            "synthetic_group_ids_are_sampling_groups_not_complete_hand_trajectories",
        ],
        "strategy_eligible": False, "advice_emitted": False,
    }
    dump(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=20000)
    args = parser.parse_args()
    try:
        result = run_study(args.input, args.protocol, args.output,
                           max_nodes=args.max_nodes)
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        parser.exit(2, f"model validation rejected: {exc}\n")
    print(json.dumps({k: result[k] for k in (
        "calibration_status", "selected_id", "groups", "world_cases",
        "fixed_policy_evaluations", "all_complete", "screening_status",
        "promotion_decision",
    )}, indent=2))
    return 0 if result["all_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
