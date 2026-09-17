"""Freeze a planning policy, then evaluate it against different manual worlds.

Shared-kernel validation: legal transitions, beliefs and settlement reuse V1.
World evaluation never invokes its optimizing value/analyze entry points.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math

from poker_engine.core.errors import InvalidStateError
from .threeway_river_v1 import (
    RiverAction, ThreewayRiverScenario, WeightTable, _Node, _Tree, _weight_table,
    amount_text, analyze_threeway_river,
)


FALLBACK = "check_if_free_else_fold"


def _canonical(value):
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("nonfinite_value_in_policy_identity")
        text = format(value, "f")  # normalize() could round under Decimal context.
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return {"decimal": "0" if value == 0 else text}
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if type(value) is float and math.isfinite(value):
        return {"float": value.hex()}
    if is_dataclass(value):
        return {"type": type(value).__name__, "fields": {
            f.name: _canonical(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("canonical_mapping_requires_string_keys")
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    raise ValueError("unsupported_canonical_policy_value")


def _json(value):
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_conditions_json(scenario):
    if not isinstance(scenario, ThreewayRiverScenario):
        raise ValueError("threeway_scenario_required")
    return _json({f.name: getattr(scenario, f.name) for f in fields(scenario)
                  if f.name not in ("ranges", "models")})


@dataclass(frozen=True)
class FrozenDecision:
    history: tuple[RiverAction, ...]
    action: RiverAction


@dataclass(frozen=True)
class PolicyBook:
    policy_id: str
    public_conditions_json: str
    planning_scenario_sha256: str
    planning_best_ev: Fraction
    decisions: tuple[FrozenDecision, ...]
    book_sha256: str
    fallback: str = FALLBACK
    schema_version: int = 1


def policy_book_hash(book):
    if not isinstance(book, PolicyBook):
        raise ValueError("policy_book_required")
    return _sha(_json({f.name: getattr(book, f.name) for f in fields(book)
                      if f.name != "book_sha256"}))


def compile_policy_book(planning_scenario, *, policy_id="manual-policy-v1",
                        max_joint_assignments=128, max_nodes=20000):
    """Optimization is permitted only HERE, before world evaluation."""
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("nonempty_policy_id_required")
    planned = analyze_threeway_river(
        planning_scenario, max_joint_assignments=max_joint_assignments,
        max_nodes=max_nodes)
    if planned.status != "COMPLETE_CONDITIONAL_ABSTRACTION":
        raise ValueError("planning_BLOCKED:" + ";".join(planned.reasons))
    choices = {}
    for policy in planned.hero_policy:
        if policy.history in choices and choices[policy.history] != policy.best_action:
            raise ValueError("conflicting_planning_information_set")
        choices[policy.history] = policy.best_action
    decisions = tuple(FrozenDecision(h, a) for h, a in sorted(
        choices.items(), key=lambda item: _json(item[0])))
    book = PolicyBook(policy_id, public_conditions_json(planning_scenario),
                      _sha(_json(planning_scenario)), planned.best_ev, decisions, "")
    return PolicyBook(book.policy_id, book.public_conditions_json,
                      book.planning_scenario_sha256, book.planning_best_ev,
                      book.decisions, policy_book_hash(book))


@dataclass(frozen=True)
class UnsupportedHistory:
    history: tuple[RiverAction, ...]
    fallback_action: RiverAction
    reach_probability: Fraction


@dataclass(frozen=True)
class PolicyMetric:
    name: str
    net_ev_chips: Fraction
    conditional_net_ev_bb: Fraction
    probability_of_any_fallback: Fraction
    expected_fallback_count: Fraction
    unsupported_histories: tuple[UnsupportedHistory, ...]
    nodes: int


@dataclass(frozen=True)
class WorldResponseOverride:
    seat_id: int
    strong_pair_min_rank: int | None = None
    strong_pair_multipliers: WeightTable = ()
    after_hero_check_multipliers: WeightTable = ()


@dataclass(frozen=True)
class PolicyEvaluation:
    status: str
    reasons: tuple[str, ...] = ()
    metrics: tuple[PolicyMetric, ...] = ()
    delta_vs_check_fold_chips: Fraction | None = None
    delta_vs_check_call_chips: Fraction | None = None
    delta_vs_check_fold_bb: Fraction | None = None
    delta_vs_check_call_bb: Fraction | None = None
    policy_hash_before: str | None = None
    policy_hash_after: str | None = None
    world_scenario_sha256: str | None = None
    root_posterior: tuple[Fraction, ...] = ()
    history_likelihood: Fraction | None = None
    nodes: int = 0
    world_overrides: tuple[WorldResponseOverride, ...] = ()
    response_family: str = "planning_family_parameters"
    assumptions: tuple[str, ...] = (
        "fixed_planning_policy_not_reoptimized_for_evaluation_world",
        "manual_world_ranges_and_response_models_not_empirical_validation",
        "shared_threeway_V1_transition_belief_and_settlement_kernel",
        "conditional_river_net_EV_not_bb_per_100_or_profit_rate_or_confidence_interval",
        "unseen_public_history_fallback_check_if_free_else_fold_predeclared",
        "world_posterior_used_for_expectation_only_not_Hero_action_selection",
        "rank_and_public_history_world_overrides_are_not_planner_response_features",
    )
    strategy_eligible: bool = False
    advice_emitted: bool = False

    def __post_init__(self):
        if self.strategy_eligible is not False or self.advice_emitted is not False:
            raise ValueError("evaluation_cannot_authorize_live_advice")


def _validate_book(book):
    if (not isinstance(book, PolicyBook) or type(book.schema_version) is not int
            or book.schema_version != 1 or book.fallback != FALLBACK
            or policy_book_hash(book) != book.book_sha256):
        raise ValueError("invalid_or_changed_policy_book")
    if not isinstance(book.decisions, tuple) or not book.decisions:
        raise ValueError("nonempty_frozen_decision_table_required")
    seen = set()
    for item in book.decisions:
        if (not isinstance(item, FrozenDecision) or not isinstance(item.history, tuple)
                or not isinstance(item.action, RiverAction) or item.history in seen
                or not all(isinstance(a, RiverAction) for a in item.history)):
            raise ValueError("invalid_frozen_public_history_table")
        seen.add(item.history)


def _select(choices, history, legal, name):
    """Pure public lookup. No scenario, cards, belief or world model argument."""
    if name == "frozen_policy" and history in choices:
        chosen = choices[history]
        if chosen not in legal:
            raise ValueError("frozen_action_is_illegal_in_world")
        return chosen, False
    kind = "check" if any(a.kind == "check" for a in legal) else (
        "call" if name == "check_call" else "fold")
    chosen = next((a for a in legal if a.kind == kind), None)
    if chosen is None:
        raise ValueError("declared_public_baseline_or_fallback_not_legal")
    return chosen, name == "frozen_policy"


def _validate_overrides(overrides, hero, opponents):
    if not isinstance(overrides, tuple):
        raise ValueError("world_overrides_must_be_tuple")
    seen = set()
    for item in overrides:
        if (not isinstance(item, WorldResponseOverride) or type(item.seat_id) is not int
                or item.seat_id == hero or item.seat_id not in opponents
                or item.seat_id in seen):
            raise ValueError("unique_opponent_only_world_override_required")
        seen.add(item.seat_id)
        rank = item.strong_pair_min_rank
        if rank is not None and (type(rank) is not int or not 2 <= rank <= 14):
            raise ValueError("invalid_strong_pair_rank_threshold")
        if rank is None and item.strong_pair_multipliers:
            raise ValueError("pair_multipliers_require_explicit_rank_threshold")
        _weight_table(item.strong_pair_multipliers, allow_empty=True)
        _weight_table(item.after_hero_check_multipliers, allow_empty=True)


def _override_probabilities(override, own_strength, public_history, hero_seat,
                            legal, base_probabilities):
    """Own hand strength plus public history only; no other player's cards."""
    factors = []
    if (override.strong_pair_min_rank is not None and own_strength[0] == 1
            and own_strength[1] >= override.strong_pair_min_rank):
        factors.append(dict(override.strong_pair_multipliers))
    last_hero = next((a for a in reversed(public_history)
                      if a.actor == hero_seat), None)
    if last_hero is not None and last_hero.kind == "check":
        factors.append(dict(override.after_hero_check_multipliers))
    weights = []
    for action, probability in zip(legal, base_probabilities):
        key = (f"{action.kind}:{amount_text(action.target)}"
               if action.kind in ("bet", "raise") else action.kind)
        weight = probability
        for table in factors:
            weight *= table.get(key, table.get(action.kind, Fraction(1)))
        weights.append(weight)
    total = sum(weights, Fraction(0))
    if not total:
        raise ValueError("zero_legal_world_override_mass")
    return tuple(w / total for w in weights)


def _world_branches(tree, node, belief, overrides):
    actor = node.pending[0]
    if actor not in overrides:
        return tree.branches(node, belief)
    legal = tree.legal(node)
    price = tree.price_ratio(node)
    probabilities = []
    for weight, strengths in zip(belief, tree.strengths):
        if not weight:
            probabilities.append(tuple(Fraction(0) for _ in legal))
            continue
        own = strengths[actor]
        base = tree.probabilities(actor, own[0], legal, price)
        probabilities.append(_override_probabilities(
            overrides[actor], own, node.history, tree.s.hero_seat, legal, base))
    branches = []
    for index, action in enumerate(legal):
        weights = tuple(w * p[index] for w, p in zip(belief, probabilities))
        mass = sum(weights, Fraction(0))
        if mass:
            branches.append((action, mass, tuple(w / mass for w in weights)))
    return branches


class _Evaluation:
    def __init__(self, tree, choices, max_nodes, overrides):
        self.tree, self.choices, self.max_nodes = tree, choices, max_nodes
        self.nodes = 0
        self.unsupported = []
        self.overrides = overrides

    def walk(self, node, belief, name, reach=Fraction(1)):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError("global_evaluation_node_budget_exceeded")
        legal = self.tree.legal(node)
        if not legal:
            return self.tree.terminal(node, belief), Fraction(0), Fraction(0)
        if node.pending[0] == self.tree.s.hero_seat:
            action, fallback = _select(self.choices, node.history, legal, name)
            if fallback:
                self.unsupported.append(UnsupportedHistory(node.history, action, reach))
            ev, any_fallback, count = self.walk(
                self.tree.advance(node, action), belief, name, reach)
            # Event UNION: once fallback occurred, this trajectory has probability 1,
            # while expected count still adds each occurrence separately.
            return ev, Fraction(1) if fallback else any_fallback, count + int(fallback)
        ev = any_fallback = count = Fraction(0)
        for action, mass, posterior in _world_branches(
                self.tree, node, belief, self.overrides):
            child_ev, child_any, child_count = self.walk(
                self.tree.advance(node, action), posterior, name, reach * mass)
            ev += mass * child_ev
            any_fallback += mass * child_any
            count += mass * child_count
        return ev, any_fallback, count


def evaluate_policy_book(book, world_scenario, *, world_overrides=(),
                         max_joint_assignments=128, max_nodes=20000):
    evaluator = None
    try:
        _validate_book(book)
        before = policy_book_hash(book)
        if public_conditions_json(world_scenario) != book.public_conditions_json:
            raise ValueError("public_conditions_or_action_grid_mismatch")
        tree = _Tree(world_scenario, max_joint_assignments, max_nodes)
        s = world_scenario
        _validate_overrides(world_overrides, s.hero_seat, set(tree.models))
        overrides = {item.seat_id: item for item in world_overrides}
        node = _Node(tree.active, s.action_order, tuple(Decimal(0) for _ in s.seats),
                     Decimal(0), s.rules.big_blind, 0, ())
        belief, likelihood = tree.prior, Fraction(1)
        for action in s.history:
            following = tree.advance(node, action)
            if action.actor != s.hero_seat:
                matches = [(mass, posterior) for a, mass, posterior in _world_branches(
                    tree, node, belief, overrides) if a == action]
                if not matches:
                    raise ValueError("observed_history_has_zero_world_probability")
                mass, belief = matches[0]
                likelihood *= mass
            node = following
        if not tree.legal(node) or node.pending[0] != s.hero_seat:
            raise ValueError("world_history_must_end_at_Hero_decision")
        tree.root_wager = node.wagers[tree.index[s.hero_seat]]
        choices = {item.history: item.action for item in book.decisions}
        evaluator = _Evaluation(tree, choices, max_nodes, overrides)
        metrics = []
        for name in ("frozen_policy", "check_fold", "check_call"):
            start = evaluator.nodes
            evaluator.unsupported = []
            ev, probability, count = evaluator.walk(node, belief, name)
            metrics.append(PolicyMetric(
                name, ev, ev / Fraction(s.rules.big_blind), probability, count,
                tuple(evaluator.unsupported), evaluator.nodes - start))
        after = policy_book_hash(book)
        if before != after or after != book.book_sha256:
            raise ValueError("policy_changed_during_evaluation")
        delta_fold = metrics[0].net_ev_chips - metrics[1].net_ev_chips
        delta_call = metrics[0].net_ev_chips - metrics[2].net_ev_chips
        return PolicyEvaluation(
            "COMPLETE_CONDITIONAL_FIXED_POLICY", metrics=tuple(metrics),
            delta_vs_check_fold_chips=delta_fold, delta_vs_check_call_chips=delta_call,
            delta_vs_check_fold_bb=delta_fold / Fraction(s.rules.big_blind),
            delta_vs_check_call_bb=delta_call / Fraction(s.rules.big_blind),
            policy_hash_before=before, policy_hash_after=after,
            world_scenario_sha256=_sha(_json({
                "scenario": s, "overrides": world_overrides})),
            root_posterior=belief,
            history_likelihood=likelihood, nodes=evaluator.nodes,
            world_overrides=world_overrides,
            response_family="rank_history_extended" if world_overrides else (
                "planning_family_parameters"),
        )
    except (TypeError, ValueError, ArithmeticError, InvalidStateError) as exc:
        return PolicyEvaluation("BLOCKED", reasons=(str(exc),),
                                nodes=evaluator.nodes if evaluator else 0)
