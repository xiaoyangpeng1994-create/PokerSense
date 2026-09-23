"""Run the precommitted synthetic posterior challenge with the frozen V1 evaluator.

Truth regimes are used only to create evaluation worlds. Candidate construction
receives public action observations and its own declared manual likelihoods.
"""

import argparse
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from poker_engine.equity.evaluator import evaluate
from poker_engine.strategy.range_posterior_v1 import (
    ActionLikelihoodProfile, PreActionRangeBinding, PublicTurnObservation,
    posterior_ranges,
)
from poker_engine.strategy.range_tracker import parse_concrete_combo
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    compile_policy_book, evaluate_policy_book,
)
from poker_engine.strategy.threeway_river_v1 import analyze_threeway_river
from tools.strategy_evaluation_v1 import (
    DEFAULT_INPUT, DEFAULT_PROTOCOL, DEFAULT_RESULTS, action_data, book_from_data,
    digest, load_frozen_baseline, planning_scenarios, read_json, run_benchmark,
    stable_json, verify_published_results, write_json,
)


CHALLENGE = Path("configs/strategy/evaluation/range-posterior-v1-challenge.json")
RESULT = Path("configs/strategy/evaluation/range-posterior-v1-results.json")
CHALLENGE_SHA256 = "073676f61d9f230c060ba0de4f147cb6f7a80387a86cb670acf9f12605f99904"
PROFILE = ActionLikelihoodProfile(Fraction(3, 4), Fraction(1, 4),
                                  "manual_unvalidated", hashlib.sha256(
                                      b"range-posterior-v1-untuned-3-to-1").hexdigest())
PRIOR_BINDING = PreActionRangeBinding(
    "before_turn_observation", "manual_synthetic",
    hashlib.sha256(
        b"baseline-manual-numeric-weights-reinterpreted-as-synthetic-pre-turn-prior"
    ).hexdigest())
COMPLETE = "COMPLETE_CONDITIONAL_FIXED_POLICY"


def _probabilities(distribution):
    weights = {k: Fraction(v) for k, v in distribution.combo_weights.items()}
    total = sum(weights.values(), Fraction(0))
    if total <= 0:
        raise ValueError("zero_range_mass")
    return {key: value / total for key, value in weights.items()}


def _truth_ranges(plan, actions, regime):
    """Independent evaluator-only generator; never passed to posterior_ranges."""
    board = plan.board_cards[:4]
    output = []
    for distribution in plan.ranges:
        ranks = {combo: evaluate(parse_concrete_combo(combo) + board)
                 for combo in distribution.combo_weights}
        maximum = max(ranks.values())
        pressure = (Fraction(regime["p_pressure_strong"]),
                    Fraction(regime["p_pressure_weak"]))
        weights = {}
        for combo, original in distribution.combo_weights.items():
            p = pressure[0] if ranks[combo] == maximum else pressure[1]
            likelihood = p if actions[distribution.seat_id] == "pressure" else 1 - p
            weights[combo] = original * Decimal(likelihood.numerator) / Decimal(
                likelihood.denominator)
        output.append(replace(distribution, combo_weights=weights))
    return tuple(output)


def _scoring(prior, posterior, truth, board):
    """Expected proper scores over every latent combo, without cherry-picking a deal."""
    prior, posterior, truth = map(_probabilities, (prior, posterior, truth))
    if set(prior) != set(posterior) or set(prior) != set(truth):
        raise ValueError("posterior_support_changed")
    ranks = {combo: evaluate(parse_concrete_combo(combo) + board)
             for combo in truth}
    strong = {combo for combo, rank in ranks.items() if rank == max(ranks.values())}

    def score(prediction):
        logloss = -sum(float(p) * math.log(float(prediction[combo]))
                       for combo, p in truth.items())
        brier = sum(float(p) * sum((float(prediction[other])
                                    - (1 if other == combo else 0)) ** 2
                                   for other in truth)
                    for combo, p in truth.items())
        predicted_strong = sum(prediction[c] for c in strong)
        true_strong = sum(truth[c] for c in strong)
        return {"log_loss": round(logloss, 9), "brier": round(brier, 9),
                "strong_probability": str(predicted_strong),
                "strong_calibration_absolute_error": round(
                    abs(float(predicted_strong - true_strong)), 9)}

    return {"baseline": score(prior), "candidate": score(posterior),
            "truth_strong_probability": str(sum(truth[c] for c in strong))}


def _root(book, history):
    return action_data(next(item.action for item in book.decisions
                            if item.history == history))


def run_challenge(path=CHALLENGE):
    raw = Path(path).read_bytes()
    if digest(raw) != CHALLENGE_SHA256:
        raise ValueError("prospective_challenge_bytes_drift")
    challenge = read_json(path)
    frozen = load_frozen_baseline()
    verify_published_results(run_benchmark(), read_json(DEFAULT_RESULTS))
    if (challenge["base_main_sha256"] != "af67b4bb3ba56da2d4fdb70e0731583afc7aaf40"
            or challenge["baseline_file_sha256"] != digest(Path(
                "configs/strategy/evaluation/baseline-v1.json").read_bytes())
            or challenge["evaluation"]["paired_cases"] != 54):
        raise ValueError("challenge_baseline_or_case_count_mismatch")
    plans, protocol, input_hash, protocol_hash = planning_scenarios(
        DEFAULT_INPUT, DEFAULT_PROTOCOL)
    if (set(plans) != set(challenge["groups"])
            or input_hash != frozen["input_sha256"]
            or protocol_hash != frozen["protocol_sha256"]):
        raise ValueError("frozen_public_condition_drift")
    rows = []
    for group in challenge["groups"]:
        plan = plans[group]
        base_book = book_from_data(frozen["books"][group])
        for pattern in challenge["observation_patterns"]:
            actions = {1: pattern["seat_1"], 2: pattern["seat_2"]}
            observations = tuple(PublicTurnObservation(
                seat, actions[seat], plan.board_cards[:4], "manual_synthetic",
                hashlib.sha256((group + "/" + pattern["id"] + "/" + str(seat))
                               .encode()).hexdigest(), seat, 3) for seat in (1, 2))
            candidate = posterior_ranges(plan, observations, PROFILE,
                                         prior_binding=PRIOR_BINDING)
            planning = replace(plan, ranges=candidate.ranges)
            candidate_book = compile_policy_book(
                planning, policy_id="range-posterior-v1/" + group + "/" + pattern["id"],
                max_joint_assignments=frozen["max_joint_assignments"],
                max_nodes=frozen["max_nodes"])
            for regime in challenge["truth_regimes"]:
                truth = _truth_ranges(plan, actions, regime)
                world = replace(plan, ranges=truth)
                before = evaluate_policy_book(
                    base_book, world,
                    max_joint_assignments=frozen["max_joint_assignments"],
                    max_nodes=frozen["max_nodes"])
                after = evaluate_policy_book(
                    candidate_book, world,
                    max_joint_assignments=frozen["max_joint_assignments"],
                    max_nodes=frozen["max_nodes"])
                row = {"case_id": group + "/" + pattern["id"] + "/" + regime["id"],
                       "group": group, "pattern": pattern["id"],
                       "truth_regime": regime["id"],
                       "baseline_status": before.status,
                       "candidate_status": after.status,
                       "baseline_book_sha256": base_book.book_sha256,
                       "candidate_book_sha256": candidate_book.book_sha256,
                       "candidate_prior_sha256": candidate.prior_sha256,
                       "candidate_observations_sha256": candidate.observations_sha256,
                       "candidate_profile_sha256": candidate.profile_sha256,
                       "baseline_reasons": list(before.reasons),
                       "candidate_reasons": list(after.reasons)}
                if before.status == after.status == COMPLETE:
                    if before.world_scenario_sha256 != after.world_scenario_sha256:
                        raise ValueError("unpaired_world_identity")
                    base_ev = before.metrics[0].net_ev_chips
                    candidate_ev = after.metrics[0].net_ev_chips
                    known = analyze_threeway_river(
                        world, max_joint_assignments=frozen["max_joint_assignments"],
                        max_nodes=frozen["max_nodes"])
                    row.update({
                        "world_sha256": before.world_scenario_sha256,
                        "baseline_root_action": _root(base_book, plan.history),
                        "candidate_root_action": _root(candidate_book, plan.history),
                        "baseline_ev_chips": str(base_ev),
                        "candidate_ev_chips": str(candidate_ev),
                        "delta_ev_chips": str(candidate_ev - base_ev),
                        "baseline_fallback_probability": str(
                            before.metrics[0].probability_of_any_fallback),
                        "candidate_fallback_probability": str(
                            after.metrics[0].probability_of_any_fallback),
                        "baseline_nodes": before.nodes, "candidate_nodes": after.nodes,
                        "world_replan_status": known.status,
                        "baseline_model_knowledge_regret_chips": (
                            str(known.best_ev - base_ev) if known.best_ev is not None
                            else None),
                        "candidate_model_knowledge_regret_chips": (
                            str(known.best_ev - candidate_ev)
                            if known.best_ev is not None
                            else None),
                        "seat_scores": {str(seat): _scoring(
                            next(r for r in plan.ranges if r.seat_id == seat),
                            next(r for r in candidate.ranges if r.seat_id == seat),
                            next(r for r in truth if r.seat_id == seat),
                            plan.board_cards[:4]) for seat in (1, 2)},
                    })
                rows.append(row)
    if len(rows) != challenge["evaluation"]["paired_cases"]:
        raise ValueError("declared_challenge_denominator_changed")
    complete = [r for r in rows if r["baseline_status"] == r["candidate_status"]
                == COMPLETE]
    deltas = [Fraction(r["delta_ev_chips"]) for r in complete]

    def mean(values):
        return round(sum(values) / len(values), 9) if values else None

    def metrics(selected):
        scores = [r["seat_scores"][seat] for r in selected for seat in ("1", "2")]
        return {"case_count": len(selected),
                "mean_delta_chips": str(sum((Fraction(r["delta_ev_chips"])
                                             for r in selected), Fraction(0))
                                        / len(selected)) if selected else None,
                "baseline_log_loss": mean([s["baseline"]["log_loss"]
                                           for s in scores]),
                "candidate_log_loss": mean([s["candidate"]["log_loss"]
                                            for s in scores]),
                "baseline_brier": mean([s["baseline"]["brier"] for s in scores]),
                "candidate_brier": mean([s["candidate"]["brier"] for s in scores]),
                "baseline_strong_calibration_error": mean([
                    s["baseline"]["strong_calibration_absolute_error"]
                    for s in scores]),
                "candidate_strong_calibration_error": mean([
                    s["candidate"]["strong_calibration_absolute_error"]
                    for s in scores]),
                "baseline_model_knowledge_regret_chips": str(sum((
                    Fraction(r["baseline_model_knowledge_regret_chips"])
                    for r in selected), Fraction(0)) / len(selected))
                if selected and all(r["baseline_model_knowledge_regret_chips"]
                                    is not None for r in selected) else None,
                "candidate_model_knowledge_regret_chips": str(sum((
                    Fraction(r["candidate_model_knowledge_regret_chips"])
                    for r in selected), Fraction(0)) / len(selected))
                if selected and all(r["candidate_model_knowledge_regret_chips"]
                                    is not None for r in selected) else None}

    summary = {"paired_count": len(complete),
               "blocked_count": len(rows) - len(complete),
               "positive_delta_count": sum(d > 0 for d in deltas),
               "negative_delta_count": sum(d < 0 for d in deltas),
               "equal_delta_count": sum(d == 0 for d in deltas),
               "mean_delta_chips": str(sum(deltas, Fraction(0)) / len(deltas))
               if deltas else None,
               "root_disagreement_count": sum(r["baseline_root_action"]
                                              != r["candidate_root_action"]
                                              for r in complete),
               "zero_fallback_cases": sum(
                   r["baseline_fallback_probability"] == "0"
                   and r["candidate_fallback_probability"] == "0"
                   for r in complete),
               "worst_delta_chips": str(min(deltas)) if deltas else None,
               "negative_case_ids": [r["case_id"] for r in complete
                                     if Fraction(r["delta_ev_chips"]) < 0],
               "scores": metrics(complete),
               "by_regime": {name: {"positive": sum(
                   Fraction(r["delta_ev_chips"]) > 0 for r in complete
                   if r["truth_regime"] == name), "negative": sum(
                   Fraction(r["delta_ev_chips"]) < 0 for r in complete
                   if r["truth_regime"] == name), **metrics([
                       r for r in complete if r["truth_regime"] == name])}
                   for name in (
                       "aligned", "uninformative", "inverted")}}
    stable = {"schema_version": 1, "candidate_id": "RANGE_POSTERIOR_V1",
              "candidate_source_sha256": digest(Path(
                  "src/poker_engine/strategy/range_posterior_v1.py").read_bytes()),
              "challenge_sha256": CHALLENGE_SHA256,
              "challenge_commit": "1c23d0b50e008910c02ef1f9f1ab34a576d4ca2d",
              "baseline_file_sha256": challenge["baseline_file_sha256"],
              "baseline_result_sha256": (
                  "c1a725f3f95a76261fb041af200e45bae02b6836202b36ec56578f0afd680aae"),
              "profile": {"p_pressure_strong": str(PROFILE.p_pressure_strong),
                          "p_pressure_weak": str(PROFILE.p_pressure_weak),
                          "source_kind": PROFILE.source_kind,
                          "source_sha256": PROFILE.source_sha256},
              "prior_binding": {"stage": PRIOR_BINDING.stage,
                                "source_kind": PRIOR_BINDING.source_kind,
                                "source_sha256": PRIOR_BINDING.source_sha256},
              "summary": summary, "cases": rows,
              "strategy_eligible": False, "advice_emitted": False,
              "real_hand_acceptance_pending": True,
              "qualification": (
                   "prospective synthetic stress on baseline public conditions and "
                   "combo support; no real calibration or independent opponent "
                   "population")}
    return {**stable, "deterministic_sha256": digest(stable_json(stable).encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    report = run_challenge()
    write_json(args.output, report)
    print(json.dumps({"summary": report["summary"],
                      "deterministic_sha256": report["deterministic_sha256"]},
                     indent=2))


if __name__ == "__main__":
    main()
