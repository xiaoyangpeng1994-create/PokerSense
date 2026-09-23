"""Frozen synthetic challenge replay and real-data refusal status."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from poker_engine.strategy.action_likelihood_v1 import fit_action_likelihood
from poker_engine.strategy.threeway_river_v1 import (
    RiverAction, _Node, _Tree, amount_text,
)

from tools.evaluate_action_likelihood_v1 import (
    AUDIT, AUDIT_SHA256, LABELS, LABELS_SHA256, OPPORTUNITIES_SHA256,
    OUTPUT, PROTOCOL, PROTOCOL_SHA256, RESULT, run_evaluation,
    _response_from_model,
)
from tools.strategy_evaluation_v1 import (
    DEFAULT_INPUT, DEFAULT_PROTOCOL, digest, planning_scenarios, read_json,
)


def test_challenge_was_frozen_and_every_strategy_world_is_paired():
    assert digest(Path(AUDIT).read_bytes()) == AUDIT_SHA256
    assert digest(Path(PROTOCOL).read_bytes()) == PROTOCOL_SHA256
    assert digest(Path(OUTPUT).read_bytes()) == OPPORTUNITIES_SHA256
    assert digest(Path(LABELS).read_bytes()) == LABELS_SHA256
    result = run_evaluation()
    assert result["verdict"] == "INSUFFICIENT_DATA"
    assert result["empirical_model_fit_executed"] is False
    assert result["historical_eligible_training"] == 0
    assert result["historical_eligible_validation"] == 0
    assert result["range_update_supported"] is False
    assert result["strategy_eligible"] is False
    assert result["advice_emitted"] is False
    assert result["strategy_summary"]["case_count"] == 8
    assert result["strategy_summary"]["paired_count"] == 8
    assert len({row["case_id"] for row in result["strategy_cases"]}) == 8
    for group in ("n6-facing_bet", "n6-unopened"):
        for model in ("pooled_dirichlet_v1", "player_dirichlet_v1"):
            pair = [row for row in result["strategy_cases"]
                    if row["group"] == group and row["model"] == model]
            assert len(pair) == 2
            assert len({row["candidate_book_sha256"] for row in pair}) == 1
            assert len({row["baseline_book_sha256"] for row in pair}) == 1
            assert all(row["world_sha256"] for row in pair)
    published = read_json(RESULT)
    assert published["candidate_source_sha256"] == digest(Path(
        "src/poker_engine/strategy/action_likelihood_v1.py").read_bytes())
    assert published["strategy_cases"] == result["strategy_cases"]


def test_bridge_keeps_learned_amounts_distinct_in_frozen_tree():
    rows = read_json(OUTPUT)["opportunities"]
    train = [row for row in rows if row["session_id"].startswith("syn-train")]
    sessions = tuple(sorted({row["session_id"] for row in train}))
    model = fit_action_likelihood(train, train_sessions=sessions, scope="player")
    response = _response_from_model(model, train, "synthetic-A", 1)
    weights = dict(response.weights)
    assert "bet" not in weights and "raise" not in weights
    assert weights["bet:20"] == Fraction(9, 50)
    plans, _, _, _ = planning_scenarios(DEFAULT_INPUT, DEFAULT_PROTOCOL)
    plan = plans["n6-unopened"]
    tree = _Tree(replace(plan, models=(response, plan.models[1])), 128, 20000)
    node = _Node(tree.active, plan.action_order,
                 tuple(Decimal(0) for _ in plan.seats),
                 Decimal(0), plan.rules.big_blind, 0, ())

    def weights_after(action):
        following = tree.advance(node, action)
        legal = tree.legal(following)
        probs = tree.probabilities(1, 0, legal, tree.price_ratio(following))
        return {a.kind if a.kind not in ("bet", "raise")
                else f"{a.kind}:{amount_text(a.target)}": p
                for a, p in zip(legal, probs)}

    checked = weights_after(RiverAction(0, "check"))
    assert checked == {"check": Fraction(41, 50),
                       "bet:10": 0, "bet:20": Fraction(9, 50),
                       "bet:40": 0, "bet:80": 0}
    bet = weights_after(RiverAction(0, "bet", Decimal(20)))
    assert bet == {"fold": Fraction(7, 51), "call": Fraction(42, 51),
                   "raise:40": Fraction(2, 51), "raise:80": 0}
