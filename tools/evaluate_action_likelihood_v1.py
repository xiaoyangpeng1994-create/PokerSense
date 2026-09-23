"""Evaluate synthetic likelihood controls, while refusing historical fitting.

The frozen BASELINE V1 evaluator is the only strategy evaluation kernel used.
No result from this synthetic control grants range, strategy or live authority.
"""

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

from poker_engine.strategy.action_likelihood_v1 import (
    fit_action_likelihood, predict_action, score_action_likelihood,
)
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    compile_policy_book, evaluate_policy_book,
)
from poker_engine.strategy.threeway_river_v1 import (
    ResponseModel, analyze_threeway_river,
)
from tools.generate_action_likelihood_synthetic_v1 import (
    LABELS, OUTPUT, PROTOCOL, PROTOCOL_SHA256, generate,
)
from tools.strategy_evaluation_v1 import (
    DEFAULT_INPUT, DEFAULT_PROTOCOL, action_data, book_from_data, digest,
    load_frozen_baseline, planning_scenarios, read_json, stable_json, write_json,
)


AUDIT = Path("configs/strategy/evaluation/action-likelihood-historical-audit-v1.json")
RESULT = Path("configs/strategy/evaluation/action-likelihood-synthetic-results-v1.json")
AUDIT_SHA256 = "d761e42050cb7d67ac9efd3eab1057bf9ed151ffe26ec033197bcd5c46eb3291"
OPPORTUNITIES_SHA256 = (
    "7cbbc108f8a801f5f4a56f57a7dea7325557f9bddd15850b8427515715ecac45")
LABELS_SHA256 = "27d15e98825a1d816b0b966d404e40e7eb9b026f529473322d6f7d39913ca64b"
COMPLETE = "COMPLETE_CONDITIONAL_FIXED_POLICY"


def _root_action(book, history):
    return action_data(next(item.action for item in book.decisions
                            if item.history == history))


def _read_frozen():
    if digest(AUDIT.read_bytes()) != AUDIT_SHA256:
        raise ValueError("historical_audit_drift")
    audit = read_json(AUDIT)
    if (audit["audit_status"] != "NO_EMPIRICAL_MODEL_FIT_PERMITTED"
            or audit["opportunity_corpus"]["eligible_training_count"] != 0
            or audit["opportunity_corpus"]["eligible_validation_count"] != 0
            or audit["opportunity_corpus"]["eligible_opportunities"] != []):
        raise ValueError("historical_fitting_gate_mismatch")
    protocol = read_json(PROTOCOL)
    if (digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA256
            or protocol["historical_audit_sha256"] != AUDIT_SHA256):
        raise ValueError("prospective_protocol_drift")
    expected_rows, expected_labels = generate()
    if (digest(OUTPUT.read_bytes()) != OPPORTUNITIES_SHA256
            or digest(LABELS.read_bytes()) != LABELS_SHA256
            or OUTPUT.read_bytes() != expected_rows
            or LABELS.read_bytes() != expected_labels):
        raise ValueError("frozen_synthetic_opportunity_fixture_drift")
    corpus, sidecar = read_json(OUTPUT), read_json(LABELS)
    rows = corpus["opportunities"]
    if (len(rows) != 384 or corpus["challenge_sha256"] != PROTOCOL_SHA256
            or sidecar["scope"] != "offline_only_not_a_decision_feature"
            or {x["opportunity_id"] for x in sidecar["labels"]}
            != {x["opportunity_id"] for x in rows}
            or not all(x["status"] == "LATER_STRENGTH_NOT_AVAILABLE"
                       and x["later_revealed_strength"] is None
                       for x in sidecar["labels"])):
        raise ValueError("offline_label_or_opportunity_census_mismatch")
    return audit, protocol, rows


def _response_from_model(model, train, opponent_id, seat):
    weights = {}
    for context in ("unopened", "facing_bet"):
        example = next(row for row in train
                       if row["opponent_id"] == opponent_id
                       and (row["public"]["to_call"] == "0")
                       == (context == "unopened"))
        probabilities, fallback = predict_action(
            model, example["public"], opponent_id=opponent_id)
        if fallback:
            raise ValueError("training_context_missing_for_policy_compile")
        for action, probability in probabilities.items():
            if action in weights:
                raise ValueError("overlapping_public_menu_action")
            weights[action] = probability
    return ResponseModel(seat, tuple(sorted(weights.items())))


def _truth_models(protocol, regime):
    truth = protocol["generator_truth_weights_evaluator_only"][regime]
    return tuple(ResponseModel(seat, tuple(sorted(
        (kind, Fraction(value)) for kind, value in truth[opponent].items())))
        for opponent, seat in (("synthetic-A", 1), ("synthetic-B", 2)))


def run_evaluation():
    audit, protocol, rows = _read_frozen()
    partitions = {item["id"]: item for item in protocol["partitions"]}
    if set(partitions) != {"train", "heldout_stable", "heldout_drift"}:
        raise ValueError("undeclared_session_partition")
    by_partition = {}
    for name, item in partitions.items():
        by_partition[name] = [row for row in rows
                              if row["session_id"] in item["session_ids"]]
    if sum(len(value) for value in by_partition.values()) != len(rows):
        raise ValueError("unassigned_synthetic_opportunity")
    train = by_partition["train"]
    models = {scope: fit_action_likelihood(
        train, train_sessions=tuple(partitions["train"]["session_ids"]),
        scope=scope) for scope in ("pooled", "player")}
    calibration = {scope: {name: score_action_likelihood(
        model, by_partition[name],
        heldout_sessions=tuple(partitions[name]["session_ids"]))
        for name in ("heldout_stable", "heldout_drift")}
        for scope, model in models.items()}
    frozen = load_frozen_baseline()
    plans, _, input_hash, protocol_hash = planning_scenarios(
        DEFAULT_INPUT, DEFAULT_PROTOCOL)
    if (input_hash != frozen["input_sha256"]
            or protocol_hash != frozen["protocol_sha256"]
            or protocol["strategy_impact"]["frozen_baseline_groups"]
            != ["n6-facing_bet", "n6-unopened"]):
        raise ValueError("frozen_strategy_public_input_drift")
    cases = []
    for group in protocol["strategy_impact"]["frozen_baseline_groups"]:
        plan = plans[group]
        baseline_book = book_from_data(frozen["books"][group])
        for scope, model in models.items():
            planned = replace(plan, models=tuple(_response_from_model(
                model, train, opponent, seat) for opponent, seat in (
                    ("synthetic-A", 1), ("synthetic-B", 2))))
            candidate_book = compile_policy_book(
                planned, policy_id="action-likelihood-v1/" + group + "/" + scope,
                max_joint_assignments=frozen["max_joint_assignments"],
                max_nodes=frozen["max_nodes"])
            for regime in protocol["strategy_impact"]["worlds"]:
                world = replace(plan, models=_truth_models(protocol, regime))
                before = evaluate_policy_book(
                    baseline_book, world,
                    max_joint_assignments=frozen["max_joint_assignments"],
                    max_nodes=frozen["max_nodes"])
                after = evaluate_policy_book(
                    candidate_book, world,
                    max_joint_assignments=frozen["max_joint_assignments"],
                    max_nodes=frozen["max_nodes"])
                row = {"case_id": group + "/" + scope + "/" + regime,
                       "group": group, "model": model.model_id,
                       "world": regime,
                       "baseline_status": before.status,
                       "candidate_status": after.status,
                       "baseline_reasons": list(before.reasons),
                       "candidate_reasons": list(after.reasons),
                       "baseline_book_sha256": baseline_book.book_sha256,
                       "candidate_book_sha256": candidate_book.book_sha256}
                if before.status == after.status == COMPLETE:
                    if before.world_scenario_sha256 != after.world_scenario_sha256:
                        raise ValueError("strategy_pair_world_mismatch")
                    known = analyze_threeway_river(
                        world, max_joint_assignments=frozen["max_joint_assignments"],
                        max_nodes=frozen["max_nodes"])
                    base_ev = before.metrics[0].net_ev_chips
                    candidate_ev = after.metrics[0].net_ev_chips
                    baseline_root = _root_action(baseline_book, plan.history)
                    candidate_root = _root_action(candidate_book, plan.history)
                    row.update({
                        "world_sha256": before.world_scenario_sha256,
                        "baseline_root_action": baseline_root,
                        "candidate_root_action": candidate_root,
                        "baseline_ev_chips": str(base_ev),
                        "candidate_ev_chips": str(candidate_ev),
                        "delta_ev_chips": str(candidate_ev - base_ev),
                        "baseline_fallback_probability": str(
                            before.metrics[0].probability_of_any_fallback),
                        "candidate_fallback_probability": str(
                            after.metrics[0].probability_of_any_fallback),
                        "world_replan_status": known.status,
                        "baseline_model_knowledge_regret_chips": (
                            str(known.best_ev - base_ev)
                            if known.best_ev is not None else None),
                        "candidate_model_knowledge_regret_chips": (
                            str(known.best_ev - candidate_ev)
                            if known.best_ev is not None else None),
                    })
                cases.append(row)
    if len(cases) != 8:
        raise ValueError("declared_strategy_case_denominator_changed")
    complete = [row for row in cases if row["baseline_status"]
                == row["candidate_status"] == COMPLETE]
    summary = {"case_count": len(cases), "paired_count": len(complete),
               "blocked_count": len(cases) - len(complete),
               "positive_delta_count": sum(Fraction(row["delta_ev_chips"]) > 0
                                           for row in complete),
               "negative_delta_count": sum(Fraction(row["delta_ev_chips"]) < 0
                                           for row in complete),
               "equal_delta_count": sum(Fraction(row["delta_ev_chips"]) == 0
                                        for row in complete),
               "root_disagreement_count": sum(row["baseline_root_action"]
                                              != row["candidate_root_action"]
                                              for row in complete)}
    stable = {
        "schema_version": 1,
        "verdict": "INSUFFICIENT_DATA",
        "historical_audit_status": audit["audit_status"],
        "historical_eligible_training": 0,
        "historical_eligible_validation": 0,
        "empirical_model_fit_executed": False,
        "historical_audit_sha256": AUDIT_SHA256,
        "prospective_challenge_sha256": PROTOCOL_SHA256,
        "challenge_commit": "dc0bd69775eedcd88eed3c3a9fee6cb94bd14f68",
        "synthetic_opportunities_sha256": OPPORTUNITIES_SHA256,
        "synthetic_labels_sha256": LABELS_SHA256,
        "candidate_source_sha256": digest(Path(
            "src/poker_engine/strategy/action_likelihood_v1.py").read_bytes()),
        "model_training_sha256": {scope: model.training_sha256
                                  for scope, model in models.items()},
        "synthetic_calibration": calibration,
        "strategy_summary": summary,
        "strategy_cases": cases,
        "range_update_supported": False,
        "qualification": (
            "synthetic code-path control only; historical A-poker has no verified "
            "action-opportunity denominator, held-out session or revealed-strength "
            "labels; no empirical calibration, profitability or GTO claim"),
        "strategy_eligible": False, "advice_emitted": False,
        "real_hand_acceptance_pending": True,
        "empirical_strategy": "NOT_ASSESSED",
    }
    return {**stable, "deterministic_sha256": digest(stable_json(stable).encode())}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run_evaluation()
    write_json(args.output, report)
    print(json.dumps({"verdict": report["verdict"],
                      "synthetic_calibration": report["synthetic_calibration"],
                      "strategy_summary": report["strategy_summary"],
                      "deterministic_sha256": report["deterministic_sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
