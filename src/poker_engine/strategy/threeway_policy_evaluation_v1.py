"""Freeze a planning policy, then evaluate it against different manual worlds.

Shared-kernel validation: legal transitions, beliefs and settlement reuse V1.
World evaluation never invokes its optimizing value/analyze entry points.

Optional path tracing decomposes each fixed policy's conditional EV into exact
per-terminal-path contributions. It re-uses the same kernel, is off by default,
and changes nothing except the populated `path_ledgers` field.
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

    path_ledgers: tuple["PolicyPathLedger", ...] = ()

    def __post_init__(self):
        if self.strategy_eligible is not False or self.advice_emitted is not False:
            raise ValueError("evaluation_cannot_authorize_live_advice")


REACHED = "REACHED"
UNREACHABLE = "UNREACHABLE"
LEDGER_COMPLETE = "COMPLETE_TERMINAL_PATH_LEDGER"
PATH_RECONCILED = "RECONCILED_EXACT_PATH_CONTRIBUTIONS"
TRACE_QUALIFICATION = (
    "terminal_paths_are_mutually_exclusive_and_exhaustive_under_one_policy",
    "reach_probability_is_exact_under_the_declared_world_not_an_estimate",
    "conditional_terminal_EV_is_undefined_for_an_unreachable_path",
    "weighted_contributions_sum_to_the_policy_conditional_net_EV_chips",
    "only_terminal_paths_are_summed_so_ancestors_are_never_added_again",
    "offline_conditional_expectation_only_no_best_action_or_profitability_claim",
)


def path_key(history):
    """Canonical text key of one public action history."""
    if (not isinstance(history, tuple)
            or not all(isinstance(a, RiverAction) for a in history)):
        raise ValueError("river_action_tuple_required")
    return "|".join(
        f"{action.actor}:{action.kind}"
        + (f":{amount_text(action.target)}" if action.kind in ("bet", "raise") else "")
        for action in history) or "root"


@dataclass(frozen=True)
class TerminalPath:
    """One complete public terminal path below the evaluated Hero decision.

    `conditional_terminal_net_ev_chips` is the settlement value conditional on
    the path belief, and is `None` exactly when the path is unreachable: a zero
    reach probability carries no conditional information. `contribution_chips`
    is `reach_probability * conditional EV`, so summing contributions over the
    mutually exclusive terminal paths reproduces the policy EV exactly.
    """

    history: tuple[RiverAction, ...]
    history_key: str
    reach_probability: Fraction
    fallback_used: bool
    conditional_terminal_net_ev_chips: Fraction | None
    contribution_chips: Fraction
    status: str

    def __post_init__(self):
        if (self.history_key != path_key(self.history)
                or not isinstance(self.reach_probability, Fraction)
                or not 0 <= self.reach_probability <= 1
                or type(self.fallback_used) is not bool
                or not isinstance(self.contribution_chips, Fraction)):
            raise ValueError("invalid_terminal_path")
        defined = self.conditional_terminal_net_ev_chips
        if self.reach_probability:
            if (self.status != REACHED or not isinstance(defined, Fraction)
                    or self.contribution_chips
                    != self.reach_probability * defined):
                raise ValueError("reachable_terminal_path_requires_conditional_EV")
        elif (self.status != UNREACHABLE or defined is not None
              or self.contribution_chips != 0):
            raise ValueError("unreachable_terminal_path_has_undefined_conditional_EV")


@dataclass(frozen=True)
class PolicyPathLedger:
    """Exact terminal-path decomposition of one policy's conditional EV.

    Every sum is recomputed from `paths`, and `__post_init__` requires it to
    equal the independently computed policy metric, so a ledger cannot exist in
    a partially reconciled state. A policy that fails the trace reconciliation
    yields `PolicyEvaluation("BLOCKED", ...)` instead of a short ledger.
    """

    policy_name: str
    policy_book_sha256: str
    world_scenario_sha256: str
    big_blind: Fraction
    paths: tuple[TerminalPath, ...]
    nodes_visited: int
    trace_node_budget: int
    reconciled_policy_net_ev_chips: Fraction
    reconciled_fallback_probability: Fraction
    completeness: str = LEDGER_COMPLETE
    qualification: tuple[str, ...] = TRACE_QUALIFICATION

    @property
    def reach_probability_sum(self):
        return sum((p.reach_probability for p in self.paths), Fraction(0))

    @property
    def contribution_sum_chips(self):
        return sum((p.contribution_chips for p in self.paths), Fraction(0))

    @property
    def contribution_sum_bb(self):
        return self.contribution_sum_chips / self.big_blind

    @property
    def fallback_reach_sum(self):
        return sum((p.reach_probability for p in self.paths if p.fallback_used),
                   Fraction(0))

    @property
    def reachable_paths(self):
        return sum(1 for p in self.paths if p.reach_probability)

    @property
    def unreachable_paths(self):
        return sum(1 for p in self.paths if not p.reach_probability)

    def __post_init__(self):
        if (not isinstance(self.policy_name, str) or not self.policy_name
                or not isinstance(self.policy_book_sha256, str)
                or not isinstance(self.world_scenario_sha256, str)
                or not isinstance(self.big_blind, Fraction) or self.big_blind <= 0
                or not isinstance(self.paths, tuple) or not self.paths
                or not all(isinstance(p, TerminalPath) for p in self.paths)
                or type(self.nodes_visited) is not int
                or not 1 <= self.nodes_visited <= self.trace_node_budget
                or self.completeness != LEDGER_COMPLETE):
            raise ValueError("invalid_policy_path_ledger")
        if len({p.history_key for p in self.paths}) != len(self.paths):
            raise ValueError("duplicate_terminal_path_in_ledger")
        if self.reach_probability_sum != 1:
            raise ValueError("terminal_path_reach_does_not_sum_to_one")
        if self.contribution_sum_chips != self.reconciled_policy_net_ev_chips:
            raise ValueError("terminal_path_contributions_do_not_match_policy_EV")
        if self.fallback_reach_sum != self.reconciled_fallback_probability:
            raise ValueError("terminal_path_fallback_reach_does_not_match_metric")


@dataclass(frozen=True)
class PathContributionRow:
    """One terminal path of the union, with each policy's own contribution."""

    history: tuple[RiverAction, ...]
    history_key: str
    reach_probability_left: Fraction
    reach_probability_right: Fraction
    contribution_chips_left: Fraction
    contribution_chips_right: Fraction
    conditional_terminal_net_ev_chips_left: Fraction | None
    conditional_terminal_net_ev_chips_right: Fraction | None

    @property
    def contribution_chips_difference(self):
        return self.contribution_chips_left - self.contribution_chips_right

    @property
    def reach_status(self):
        left = bool(self.reach_probability_left)
        right = bool(self.reach_probability_right)
        if left and right:
            return "REACHED_BY_BOTH"
        if left:
            return "REACHED_BY_LEFT_ONLY"
        if right:
            return "REACHED_BY_RIGHT_ONLY"
        return "UNREACHABLE_BY_BOTH"


@dataclass(frozen=True)
class PathReconciliation:
    """Subtract two traced policies on the union of their terminal paths."""

    left_policy: str
    right_policy: str
    policy_book_sha256: str
    world_scenario_sha256: str
    rows: tuple[PathContributionRow, ...]
    total_ev_difference_chips: Fraction
    completeness: str = PATH_RECONCILED

    @property
    def contribution_difference_sum_chips(self):
        return sum((row.contribution_chips_difference for row in self.rows),
                   Fraction(0))

    def reach_status_count(self, status):
        return sum(1 for row in self.rows if row.reach_status == status)

    def __post_init__(self):
        if (not isinstance(self.rows, tuple) or not self.rows
                or not all(isinstance(r, PathContributionRow) for r in self.rows)
                or len({row.history_key for row in self.rows}) != len(self.rows)
                or self.completeness != PATH_RECONCILED
                or not isinstance(self.total_ev_difference_chips, Fraction)):
            raise ValueError("invalid_path_reconciliation")
        if (self.contribution_difference_sum_chips
                != self.total_ev_difference_chips):
            raise ValueError("path_contribution_differences_do_not_match_EV_delta")


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


class _PathTrace:
    """Enumerate every terminal public path of one fixed policy exactly once.

    The transition kernel and the declared world response branches are the only
    inputs: the optimizing entry points are never called, so tracing a world
    cannot re-plan Hero. Hero nodes are expanded over all legal actions, with
    the policy's own action carrying the full mass, which makes the enumerated
    path set identical for every policy and therefore comparable as a union.
    """

    def __init__(self, tree, choices, name, overrides, max_nodes):
        self.tree, self.choices, self.name = tree, choices, name
        self.overrides, self.max_nodes = overrides, max_nodes
        self.nodes = 0
        self.paths = []

    def walk_all(self, node, belief):
        self.walk(node, belief, Fraction(1))
        return self

    def walk(self, node, belief, reach, fallback_used=False):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError("trace_node_budget_exceeded")
        legal = self.tree.legal(node)
        if not legal:
            self.record(node, belief, reach, fallback_used)
            return
        if node.pending[0] == self.tree.s.hero_seat:
            chosen, fallback = _select(self.choices, node.history, legal, self.name)
            for action in legal:
                taken = action == chosen
                self.walk(self.tree.advance(node, action), belief,
                          reach if taken else Fraction(0),
                          fallback_used or (taken and fallback))
            return
        for action, mass, posterior in _world_branches(
                self.tree, node, belief, self.overrides):
            self.walk(self.tree.advance(node, action), posterior, reach * mass,
                      fallback_used)

    def record(self, node, belief, reach, fallback_used):
        # Only terminal paths are recorded, so no ancestor is summed again.
        conditional = self.tree.terminal(node, belief) if reach else None
        self.paths.append(TerminalPath(
            tuple(node.history), path_key(node.history), reach, fallback_used,
            conditional, reach * conditional if reach else Fraction(0),
            REACHED if reach else UNREACHABLE))


def _build_ledger(traced, metric, book_sha256, world_sha256, big_blind):
    return PolicyPathLedger(
        metric.name, book_sha256, world_sha256, big_blind,
        tuple(sorted(traced.paths, key=lambda p: p.history_key)), traced.nodes,
        traced.max_nodes, metric.net_ev_chips, metric.probability_of_any_fallback)


def compare_policy_paths(evaluation, left="frozen_policy", right="check_fold"):
    """Reconcile two traced policies path by path on their union.

    Contributions are subtracted for the same path; the sum of those
    differences equals the difference of the two original policy EVs. This is
    the offline conditional expectation of one declared world: it is not a
    proof that any action is optimal, and no live advice is produced.
    """
    if not isinstance(evaluation, PolicyEvaluation):
        raise ValueError("policy_evaluation_required")
    if evaluation.status != "COMPLETE_CONDITIONAL_FIXED_POLICY":
        raise ValueError("path_diagnosis_requires_complete_evaluation")
    if (not isinstance(left, str) or not isinstance(right, str) or left == right):
        raise ValueError("two_distinct_policy_names_required")
    ledgers = {item.policy_name: item for item in evaluation.path_ledgers}
    metrics = {item.name: item for item in evaluation.metrics}
    if not {left, right} <= set(ledgers) or not {left, right} <= set(metrics):
        raise ValueError("path_trace_not_collected_for_named_policy")
    if (ledgers[left].policy_book_sha256 != ledgers[right].policy_book_sha256
            or ledgers[left].world_scenario_sha256
            != ledgers[right].world_scenario_sha256
            or ledgers[left].big_blind != ledgers[right].big_blind):
        raise ValueError("path_ledgers_belong_to_different_book_or_world")
    maps = [{p.history_key: p for p in ledgers[name].paths}
            for name in (left, right)]
    rows = []
    for key in sorted(set(maps[0]) | set(maps[1])):
        first, second = maps[0].get(key), maps[1].get(key)
        present = first if first is not None else second
        rows.append(PathContributionRow(
            present.history, key,
            first.reach_probability if first else Fraction(0),
            second.reach_probability if second else Fraction(0),
            first.contribution_chips if first else Fraction(0),
            second.contribution_chips if second else Fraction(0),
            first.conditional_terminal_net_ev_chips if first else None,
            second.conditional_terminal_net_ev_chips if second else None))
    rows.sort(key=lambda row: (
        -row.contribution_chips_left, -row.contribution_chips_right,
        row.history_key))
    return PathReconciliation(
        left, right, ledgers[left].policy_book_sha256,
        ledgers[left].world_scenario_sha256, tuple(rows),
        metrics[left].net_ev_chips - metrics[right].net_ev_chips)


def evaluate_policy_book(book, world_scenario, *, world_overrides=(),
                         max_joint_assignments=128, max_nodes=20000, trace=False,
                         trace_max_nodes=None):
    evaluator = None
    try:
        _validate_book(book)
        if not isinstance(trace, bool):
            raise ValueError("trace_switch_must_be_boolean")
        budget = max_nodes if trace_max_nodes is None else trace_max_nodes
        if type(budget) is not int or not 1 <= budget <= 200000:
            raise ValueError("invalid_explicit_trace_budget")
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
        world_sha256 = _sha(_json({"scenario": s, "overrides": world_overrides}))
        ledgers = ()
        if trace:
            # Second, independent traversal: the ledger must reproduce the walk.
            ledgers = tuple(
                _build_ledger(
                    _PathTrace(tree, choices, name, overrides, budget).walk_all(
                        node, belief),
                    metric, after, world_sha256, Fraction(s.rules.big_blind))
                for name, metric in zip(
                    ("frozen_policy", "check_fold", "check_call"), metrics))
        delta_fold = metrics[0].net_ev_chips - metrics[1].net_ev_chips
        delta_call = metrics[0].net_ev_chips - metrics[2].net_ev_chips
        return PolicyEvaluation(
            "COMPLETE_CONDITIONAL_FIXED_POLICY", metrics=tuple(metrics),
            delta_vs_check_fold_chips=delta_fold, delta_vs_check_call_chips=delta_call,
            delta_vs_check_fold_bb=delta_fold / Fraction(s.rules.big_blind),
            delta_vs_check_call_bb=delta_call / Fraction(s.rules.big_blind),
            policy_hash_before=before, policy_hash_after=after,
            world_scenario_sha256=world_sha256,
            root_posterior=belief,
            history_likelihood=likelihood, nodes=evaluator.nodes,
            world_overrides=world_overrides,
            response_family="rank_history_extended" if world_overrides else (
                "planning_family_parameters"),
            path_ledgers=ledgers,
        )
    except (TypeError, ValueError, ArithmeticError, InvalidStateError) as exc:
        return PolicyEvaluation("BLOCKED", reasons=(str(exc),),
                                nodes=evaluator.nodes if evaluator else 0)
