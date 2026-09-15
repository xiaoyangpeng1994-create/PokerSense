"""Finite three-player river study with public-history Bayesian beliefs.

Opponent policies see their own hand category and public legal actions only.
Hero maximizes expected value at an information set, never per hidden deal.
No all-ins, side pots, live advice, equilibrium or empirical policy is claimed.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from math import prod
import re

from poker_engine.core.enums import PlayerStatus
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity.evaluator import evaluate
from .aa_rules_v2 import AARuleProfileV2
from .contracts import DecisionSeat, RangeDistribution
from .range_tracker import enumerate_joint_assignments, parse_concrete_combo
from .state import calculate_side_pots
from .terminal_multiway_v1 import _rake


KINDS = ("check", "fold", "call", "bet", "raise")
WeightTable = tuple[tuple[str, Fraction], ...]


def amount_text(value):
    return format(value.normalize(), "f")


@dataclass(frozen=True)
class RiverAction:
    actor: int
    kind: str
    target: Decimal = Decimal(0)


@dataclass(frozen=True)
class ResponseModel:
    seat_id: int
    weights: WeightTable
    category_weights: tuple[tuple[int, WeightTable], ...] = ()
    price_multipliers: tuple[tuple[Fraction, WeightTable], ...] = ()


@dataclass(frozen=True)
class ThreewayRiverScenario:
    seats: tuple[DecisionSeat, ...]
    hero_seat: int
    hero_cards: tuple[Card, Card]
    board_cards: tuple[Card, ...]
    ranges: tuple[RangeDistribution, ...]
    rules: AARuleProfileV2
    action_order: tuple[int, ...]
    models: tuple[ResponseModel, ...]
    aggression_targets: tuple[Decimal, ...]
    history: tuple[RiverAction, ...] = ()
    max_aggressions: int = 3
    other_fees: Decimal | None = Decimal(0)
    range_start: str = "river_start"


@dataclass(frozen=True)
class ActionValue:
    action: RiverAction
    ev: Fraction
    additional_cost: Decimal = Decimal(0)


@dataclass(frozen=True)
class PublicRootContext:
    root_pot: Decimal
    current_bet: Decimal
    hero_street_committed: Decimal
    to_call: Decimal
    pending: tuple[int, ...]
    action_order: tuple[int, ...]
    price_ratio: Fraction = Fraction(0)


@dataclass(frozen=True)
class HeroPolicy:
    history: tuple[RiverAction, ...]
    belief: tuple[Fraction, ...]
    action_values: tuple[ActionValue, ...]
    best_action: RiverAction


@dataclass(frozen=True)
class ThreewayRiverResult:
    status: str
    reasons: tuple[str, ...] = ()
    root_actions: tuple[ActionValue, ...] = ()
    best_action: RiverAction | None = None
    best_ev: Fraction | None = None
    root_posterior: tuple[Fraction, ...] = ()
    root_context: PublicRootContext | None = None
    joint_hypotheses: tuple[tuple[tuple[int, str], ...], ...] = ()
    history_likelihood: Fraction | None = None
    hero_policy: tuple[HeroPolicy, ...] = ()
    nodes: int = 0
    terminal_nodes: int = 0
    joint_assignments: int = 0
    rule_fingerprint: str | None = None
    assumptions: tuple[str, ...] = (
        "manual_river_start_ranges_and_response_weights_not_observed_or_GTO",
        "independent_prior_ranges_conditioned_on_collisions_and_public_actions",
        "finite_declared_bet_raise_abstraction_not_full_no_limit_action_space",
        "response_weights_are_renormalized_at_each_legal_public_node",
        "price_ratio_is_public_to_call_divided_by_current_pot_plus_to_call",
        "fractional_equal_splits_single_board_cash_chip_EV_no_ICM",
        "rake_on_settled_pots_not_uncalled_returns_no_extra_fees",
        "Hero_actions_do_not_reveal_unknown_opponent_cards",
        "clockwise_action_order_is_supplied_not_inferred_from_position_labels",
    )
    strategy_eligible: bool = False
    advice_emitted: bool = False

    def __post_init__(self):
        if self.strategy_eligible is not False or self.advice_emitted is not False:
            raise ValueError("offline study cannot authorize strategy/advice")


@dataclass(frozen=True)
class _Node:
    alive: tuple[int, ...]
    pending: tuple[int, ...]
    wagers: tuple[Decimal, ...]  # Aligned with scenario.seats.
    current_bet: Decimal
    last_increment: Decimal
    aggressions: int
    history: tuple[RiverAction, ...]


def _weight_table(table, *, allow_empty=False):
    if not isinstance(table, tuple) or not table and not allow_empty:
        raise ValueError("explicit_response_weights_required")
    seen = set()
    for pair in table:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("response_weights_must_be_key_weight_pairs")
        key, weight = pair
        if not isinstance(key, str) or key in seen:
            raise ValueError("duplicate_or_invalid_response_key")
        seen.add(key)
        if key not in KINDS:
            prefix, separator, text = key.partition(":")
            if prefix not in ("bet", "raise") or separator != ":":
                raise ValueError("invalid_response_action_key")
            number = Decimal(text)
            if not number.is_finite() or number <= 0 or amount_text(number) != text:
                raise ValueError("noncanonical_response_target_key")
        if not isinstance(weight, Fraction) or weight < 0:
            raise ValueError("nonnegative_Fraction_response_weight_required")


def _validate(s, max_joint_assignments, max_nodes):
    if not isinstance(s, ThreewayRiverScenario):
        raise ValueError("threeway_scenario_required")
    if (type(max_joint_assignments) is not int or not 1 <= max_joint_assignments <= 4096
            or type(max_nodes) is not int or not 1 <= max_nodes <= 200000):
        raise ValueError("invalid_explicit_computation_budget")
    if (not isinstance(s.rules, AARuleProfileV2)
            or s.rules.verification_status != "simulation"
            or not isinstance(s.other_fees, Decimal) or s.other_fees != 0
            or s.range_start != "river_start"):
        raise ValueError("manual_river_start_and_known_zero_extra_fees_required")
    if (not isinstance(s.seats, tuple) or len(s.seats) != s.rules.table_size
            or not all(isinstance(p, DecisionSeat) for p in s.seats)
            or len({p.seat_id for p in s.seats}) != len(s.seats)
            or any(not p.occupied or p.status not in (
                PlayerStatus.ACTIVE, PlayerStatus.FOLDED) for p in s.seats)):
        raise ValueError("six_to_eight_unique_dealt_seats_required")
    active = tuple(p.seat_id for p in s.seats if p.status is PlayerStatus.ACTIVE)
    if (type(s.hero_seat) is not int or len(active) != 3 or s.hero_seat not in active
            or [p.seat_id for p in s.seats if p.is_hero] != [s.hero_seat]
            or not isinstance(s.action_order, tuple) or len(s.action_order) != 3
            or any(type(v) is not int for v in s.action_order)
            or set(s.action_order) != set(active)):
        raise ValueError("three_active_and_explicit_unique_action_order_required")
    active_levels = {p.hand_committed.value for p in s.seats if p.seat_id in active}
    if any(p.street_committed.value != 0 for p in s.seats) or len(active_levels) != 1:
        raise ValueError("single_pot_river_start_equal_active_contributions_required")
    level = next(p.hand_committed.value for p in s.seats if p.seat_id in active)
    if any(p.hand_committed.value > level for p in s.seats):
        raise ValueError("existing_sidepot_or_uncalled_money_unsupported")
    if (not isinstance(s.aggression_targets, tuple) or not s.aggression_targets
            or any(not isinstance(v, Decimal) or not v.is_finite() or v <= 0
                   or v % s.rules.minimum_chip != 0 for v in s.aggression_targets)
            or tuple(sorted(set(s.aggression_targets))) != s.aggression_targets
            or type(s.max_aggressions) is not int or not 1 <= s.max_aggressions <= 3):
        raise ValueError("ordered_chip_aligned_targets_and_declared_raise_cap_required")
    if any(p.stack.value <= s.aggression_targets[-1]
           for p in s.seats if p.seat_id in active):
        raise ValueError("allin_or_sidepot_boundary_not_supported")
    if any(v % s.rules.minimum_chip != 0 for p in s.seats
           for v in (p.stack.value, p.hand_committed.value)):
        raise ValueError("seat_amounts_not_chip_aligned")
    if (not isinstance(s.hero_cards, tuple) or len(s.hero_cards) != 2
            or not isinstance(s.board_cards, tuple) or len(s.board_cards) != 5
            or not all(isinstance(c, Card) for c in s.hero_cards + s.board_cards)
            or len(set(s.hero_cards + s.board_cards)) != 7):
        raise ValueError("distinct_hero_and_five_river_cards_required")
    opponents = set(active) - {s.hero_seat}
    if (not isinstance(s.ranges, tuple) or len(s.ranges) != 2
            or not all(isinstance(r, RangeDistribution) for r in s.ranges)
            or {r.seat_id for r in s.ranges} != opponents):
        raise ValueError("two_river_start_opponent_ranges_required")
    for r in s.ranges:
        if (r.source not in ("manual_terminal_assumption", "manual_river_assumption",
                             "manual_river_start_assumption")
                or re.fullmatch(r"manual:[a-zA-Z0-9_-]+", r.source_version) is None
                or any(t in r.source_version.lower() for t in ("gto", "solver", "live"))
                or r.effective_sample_size != 0 or r.confidence != 0):
            raise ValueError("manual_unvalidated_range_source_required")
        holdings = [frozenset(parse_concrete_combo(c)) for c in r.combo_weights]
        if not holdings or len(set(holdings)) != len(holdings):
            raise ValueError("empty_or_duplicate_concrete_range")
    if prod(len(r.combo_weights) for r in s.ranges) > max_joint_assignments:
        raise ValueError("joint_assignment_budget_exceeded")
    if (not isinstance(s.models, tuple) or len(s.models) != 2
            or not all(isinstance(m, ResponseModel) for m in s.models)
            or any(type(m.seat_id) is not int for m in s.models)
            or {m.seat_id for m in s.models} != opponents):
        raise ValueError("two_own_hand_response_models_required")
    for model in s.models:
        _weight_table(model.weights)
        if not isinstance(model.category_weights, tuple):
            raise ValueError("category_tables_must_be_tuple")
        seen = set()
        for category, table in model.category_weights:
            if type(category) is not int or not 0 <= category <= 8 or category in seen:
                raise ValueError("invalid_or_duplicate_hand_category")
            seen.add(category)
            _weight_table(table)
        if not isinstance(model.price_multipliers, tuple):
            raise ValueError("price_bands_must_be_tuple")
        previous = Fraction(0)
        for bound, table in model.price_multipliers:
            if not isinstance(bound, Fraction) or not previous < bound <= 1:
                raise ValueError("price_bounds_must_increase_in_zero_one")
            _weight_table(table, allow_empty=True)
            previous = bound
        if model.price_multipliers and previous != 1:
            raise ValueError("price_bands_must_end_at_one")
    if not isinstance(s.history, tuple) or len(s.history) > 20:
        raise ValueError("bounded_public_history_required")
    return active


class _Tree:
    def __init__(self, scenario, max_joint_assignments, max_nodes):
        self.s = scenario
        self.active = _validate(scenario, max_joint_assignments, max_nodes)
        self.index = {p.seat_id: i for i, p in enumerate(scenario.seats)}
        self.max_nodes = max_nodes
        self.nodes = self.terminals = 0
        self.policies = []
        self.models = {m.seat_id: m for m in scenario.models}
        self.assignments = enumerate_joint_assignments(
            scenario.ranges, scenario.hero_cards + scenario.board_cards,
            max_combinations=max_joint_assignments)
        weights = {
            r.seat_id: {frozenset(parse_concrete_combo(c)): Fraction(w)
                        for c, w in r.combo_weights.items()} for r in scenario.ranges
        }
        raw = tuple(prod(weights[seat][frozenset(holding)]
                         for seat, holding in a.holdings.items())
                    for a in self.assignments)
        total = sum(raw, Fraction(0))
        self.prior = tuple(w / total for w in raw)
        self.strengths = tuple({
            scenario.hero_seat: evaluate(scenario.hero_cards + scenario.board_cards),
            **{seat: evaluate(holding + scenario.board_cards)
               for seat, holding in a.holdings.items()},
        } for a in self.assignments)
        self.root_wager = Decimal(0)

    def legal(self, node):
        if len(node.alive) == 1 or not node.pending:
            return ()
        actor = node.pending[0]
        owed = node.current_bet - node.wagers[self.index[actor]]
        values = ([RiverAction(actor, "fold"), RiverAction(actor, "call")]
                  if owed > 0 else [RiverAction(actor, "check")])
        if node.aggressions < self.s.max_aggressions:
            minimum = (self.s.rules.big_blind if node.current_bet == 0
                       else node.current_bet + node.last_increment)
            kind = "bet" if node.current_bet == 0 else "raise"
            values += [RiverAction(actor, kind, t)
                       for t in self.s.aggression_targets if t >= minimum]
        return tuple(values)

    def advance(self, node, action):
        if (not isinstance(action, RiverAction) or type(action.actor) is not int
                or not isinstance(action.target, Decimal)
                or not action.target.is_finite()
                or action not in self.legal(node)):
            raise ValueError("illegal_public_history_or_action")
        actor = action.actor
        alive = (tuple(p for p in node.alive if p != actor)
                 if action.kind == "fold" else node.alive)
        wagers = list(node.wagers)
        bet, increment, count = node.current_bet, node.last_increment, node.aggressions
        if action.kind in ("bet", "raise"):
            wagers[self.index[actor]] = action.target
            increment, bet, count = action.target - bet, action.target, count + 1
            order = self.s.action_order
            start = order.index(actor)
            pending = tuple(order[(start + i) % 3] for i in (1, 2)
                            if order[(start + i) % 3] in alive)
        else:
            if action.kind == "call":
                wagers[self.index[actor]] = bet
            pending = tuple(p for p in node.pending[1:] if p in alive)
        return _Node(alive, pending, tuple(wagers), bet, increment, count,
                     (*node.history, action))

    def price_ratio(self, node):
        owed = node.current_bet - node.wagers[self.index[node.pending[0]]]
        pot = sum((p.hand_committed.value for p in self.s.seats), Decimal(0))
        pot += sum(node.wagers, Decimal(0))
        return Fraction(owed) / Fraction(pot + owed) if owed else Fraction(0)

    def probabilities(self, actor, category, legal, price_ratio=Fraction(0)):
        # Only own category, public legal actions and public price enter policy.
        if not isinstance(price_ratio, Fraction) or not 0 <= price_ratio <= 1:
            raise ValueError("public_price_ratio_must_be_fraction_in_zero_one")
        model = self.models[actor]
        table = dict(dict(model.category_weights).get(category, model.weights))
        multiplier = next((dict(weights) for bound, weights in model.price_multipliers
                           if price_ratio <= bound), {})

        def weight(action):
            key = f"{action.kind}:{amount_text(action.target)}"
            if action.kind not in ("bet", "raise"):
                key = action.kind
            return (table.get(key, table.get(action.kind, 0))
                    * multiplier.get(key, multiplier.get(action.kind, 1)))

        weights = tuple(weight(action) for action in legal)
        total = sum(weights, Fraction(0))
        if not total:
            raise ValueError("zero_legal_response_mass")
        return tuple(w / total for w in weights)

    def branches(self, node, belief):
        legal = self.legal(node)
        actor = node.pending[0]
        price = self.price_ratio(node)
        probabilities = [self.probabilities(actor, strength[actor][0], legal, price)
                         if prior
                         else tuple(Fraction(0) for _ in legal)
                         for prior, strength in zip(belief, self.strengths)]
        result = []
        for column, action in enumerate(legal):
            weights = tuple(w * p[column] for w, p in zip(belief, probabilities))
            mass = sum(weights, Fraction(0))
            if mass:
                result.append((action, mass, tuple(w / mass for w in weights)))
        return result

    def terminal(self, node, belief):
        self.terminals += 1
        seats = tuple(replace(
            p, stack=ChipAmount(p.stack.value - node.wagers[i]),
            street_committed=ChipAmount(node.wagers[i]),
            hand_committed=ChipAmount(p.hand_committed.value + node.wagers[i]),
            status=(PlayerStatus.ACTIVE if p.seat_id in node.alive
                    else PlayerStatus.FOLDED),
        ) for i, p in enumerate(self.s.seats))
        settled = calculate_side_pots(seats, settle_uncalled=True)
        if len(settled.pots) > 1:
            raise ValueError("unexpected_sidepot_outside_declared_game")
        expected = sum((p.hand_committed.value for p in seats), Decimal(0))
        conserved = sum((p.amount.value for p in settled.pots), Decimal(0)) + sum(
            (v.value for v in settled.uncalled_returns.values()), Decimal(0))
        if conserved != expected:
            raise ValueError("terminal_money_conservation_failed")
        rake = _rake(settled.pots, self.s.rules)
        reward = Fraction(settled.uncalled_returns.get(
            self.s.hero_seat, ChipAmount("0")).value)
        for pot in settled.pots:
            if self.s.hero_seat not in pot.eligible_seats:
                continue
            share = Fraction(0)
            for weight, strengths in zip(belief, self.strengths):
                best = max(strengths[p] for p in pot.eligible_seats)
                winners = [p for p in pot.eligible_seats if strengths[p] == best]
                if self.s.hero_seat in winners:
                    share += weight / len(winners)
            reward += (Fraction(pot.amount.value) - rake[pot.pot_id]) * share
        cost = node.wagers[self.index[self.s.hero_seat]] - self.root_wager
        return reward - Fraction(cost)

    def value(self, node, belief):
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise ValueError("global_node_budget_exceeded_no_partial_EV")
        legal = self.legal(node)
        if not legal:
            return self.terminal(node, belief)
        if node.pending[0] == self.s.hero_seat:
            # A single action is chosen for the WHOLE public posterior belief.
            own = node.wagers[self.index[self.s.hero_seat]]
            values = tuple(ActionValue(
                a, self.value(self.advance(node, a), belief),
                node.current_bet - own if a.kind == "call" else (
                    a.target - own if a.kind in ("bet", "raise") else Decimal(0)),
            ) for a in legal)
            best = max(values, key=lambda v: v.ev)
            self.policies.append(HeroPolicy(node.history, belief, values, best.action))
            return best.ev
        return sum((mass * self.value(self.advance(node, action), posterior)
                    for action, mass, posterior in self.branches(node, belief)),
                   Fraction(0))


def analyze_threeway_river(scenario, *, max_joint_assignments=128, max_nodes=20000):
    tree = None
    try:
        tree = _Tree(scenario, max_joint_assignments, max_nodes)
        s = scenario
        node = _Node(tree.active, s.action_order, tuple(Decimal(0) for _ in s.seats),
                     Decimal(0), s.rules.big_blind, 0, ())
        belief, likelihood = tree.prior, Fraction(1)
        for action in s.history:
            following = tree.advance(node, action)
            if action.actor != s.hero_seat:
                matches = [(mass, posterior) for a, mass, posterior in tree.branches(
                    node, belief) if a == action]
                if not matches:
                    raise ValueError("observed_history_has_zero_response_probability")
                mass, belief = matches[0]
                likelihood *= mass
            node = following
        if not tree.legal(node) or node.pending[0] != s.hero_seat:
            raise ValueError("history_must_end_at_nonterminal_Hero_decision")
        tree.root_wager = node.wagers[tree.index[s.hero_seat]]
        best_ev = tree.value(node, belief)
        root = next(p for p in tree.policies if p.history == node.history)
        return ThreewayRiverResult(
            "COMPLETE_CONDITIONAL_ABSTRACTION", root_actions=root.action_values,
            best_action=root.best_action, best_ev=best_ev, root_posterior=belief,
            root_context=PublicRootContext(
                sum((p.hand_committed.value for p in s.seats), Decimal(0))
                + sum(node.wagers, Decimal(0)), node.current_bet, tree.root_wager,
                node.current_bet - tree.root_wager, node.pending, s.action_order,
                tree.price_ratio(node)),
            joint_hypotheses=tuple(tuple(sorted(
                (seat, "".join(map(str, holding)))
                for seat, holding in a.holdings.items())) for a in tree.assignments),
            history_likelihood=likelihood, hero_policy=tuple(tree.policies),
            nodes=tree.nodes, terminal_nodes=tree.terminals,
            joint_assignments=len(tree.assignments),
            rule_fingerprint=s.rules.fingerprint,
        )
    except (TypeError, ValueError, ArithmeticError, InvalidStateError) as exc:
        return ThreewayRiverResult("BLOCKED", reasons=(str(exc),),
                                   nodes=tree.nodes if tree else 0)
