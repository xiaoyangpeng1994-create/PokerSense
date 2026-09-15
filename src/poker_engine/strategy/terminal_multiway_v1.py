"""Exact river call/fold study when every opponent is already all-in.

Manual hypotheses only. No provider, live state, advice or asset approval is
created. Fractional equal splits are an explicit economic model assumption.
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
from .aa_rules_v2 import AARuleProfileV2, estimate_rake
from .contracts import DecisionSeat, RangeDistribution
from .range_tracker import enumerate_joint_assignments, parse_concrete_combo
from .state import calculate_side_pots


@dataclass(frozen=True)
class TerminalScenario:
    seats: tuple[DecisionSeat, ...]
    hero_seat: int
    actor_seat: int
    hero_cards: tuple[Card, Card]
    board_cards: tuple[Card, ...]
    current_bet: ChipAmount
    pot_before: ChipAmount
    ranges: tuple[RangeDistribution, ...]
    rules: AARuleProfileV2
    other_fees: Decimal | None = Decimal("0")
    split_policy: str = "fractional_equal_split"


@dataclass(frozen=True)
class TerminalPotAudit:
    pot_id: str
    amount: Decimal
    eligible_seats: tuple[int, ...]
    hero_share: Fraction
    configured_rake: Fraction
    hero_gross_return: Fraction
    hero_net_return: Fraction


@dataclass(frozen=True)
class TerminalAnalysis:
    status: str
    reasons: tuple[str, ...] = ()
    recommendation: str | None = None
    call_cost: Decimal | None = None
    call_gross_ev: Fraction | None = None
    call_net_ev: Fraction | None = None
    fold_ev: Fraction | None = None
    net_call_minus_fold: Fraction | None = None
    call_pots: tuple[TerminalPotAudit, ...] = ()
    call_refunds: tuple[tuple[int, Decimal], ...] = ()
    fold_refunds: tuple[tuple[int, Decimal], ...] = ()
    fold_pots: tuple[tuple[str, Decimal, tuple[int, ...]], ...] = ()
    seat_commitments_before: tuple[tuple[int, Decimal], ...] = ()
    seat_commitments_after_call: tuple[tuple[int, Decimal], ...] = ()
    joint_assignments: int = 0
    rule_fingerprint: str | None = None
    range_sources: tuple[tuple[int, str, str], ...] = ()
    assumptions: tuple[str, ...] = (
        "manual_ranges_and_rules_not_platform_observations_or_GTO",
        "independent_range_weights_conditioned_on_card_collision_constraints",
        "fractional_equal_split_no_odd_chip_allocation",
        "rake_on_settled_contestable_pots_not_uncalled_returns",
        "river_no_future_cards_or_opponent_actions",
        "chip_EV_linear_utility_not_ICM_or_bankroll_utility",
        "complete_betting_history_reachability_not_verified",
        "EV_relative_to_current_decision_sunk_contributions_not_subtracted_again",
    )
    strategy_eligible: bool = False
    advice_emitted: bool = False

    def __post_init__(self):
        if self.strategy_eligible is not False or self.advice_emitted is not False:
            raise ValueError("offline study cannot authorize live strategy or advice")


def _validate(scenario, max_joint_assignments):
    if not isinstance(scenario, TerminalScenario):
        raise ValueError("terminal_scenario_required")
    s = scenario
    if (type(max_joint_assignments) is not int
            or not 1 <= max_joint_assignments <= 10000):
        raise ValueError("invalid_bounded_joint_budget")
    if not isinstance(s.rules, AARuleProfileV2) or s.rules.verification_status != (
            "simulation"):
        raise ValueError("explicit_hypothetical_rules_required")
    if (s.split_policy != "fractional_equal_split"
            or not isinstance(s.other_fees, Decimal)
            or not s.other_fees.is_finite() or s.other_fees != 0):
        raise ValueError("unsupported_split_or_unknown_additional_fees")
    if (not isinstance(s.seats, tuple) or len(s.seats) != s.rules.table_size
            or not all(isinstance(p, DecisionSeat) for p in s.seats)
            or len({p.seat_id for p in s.seats}) != len(s.seats)
            or any(not p.occupied or p.status not in (
                PlayerStatus.ACTIVE, PlayerStatus.ALL_IN, PlayerStatus.FOLDED)
                   for p in s.seats)):
        raise ValueError("complete_unique_six_to_eight_dealt_seats_required")
    if (type(s.hero_seat) is not int or type(s.actor_seat) is not int
            or s.hero_seat != s.actor_seat
            or [p.seat_id for p in s.seats if p.is_hero] != [s.hero_seat]):
        raise ValueError("hero_must_be_unique_current_actor")
    hero = next(p for p in s.seats if p.is_hero)
    active = [p.seat_id for p in s.seats if p.status is PlayerStatus.ACTIVE]
    opponents = [p for p in s.seats if p.status is PlayerStatus.ALL_IN]
    if (active != [s.hero_seat] or hero.stack.value <= 0 or not 2 <= len(opponents) <= 7
            or any(p.stack.value != 0 or p.hand_committed.value <= 0
                   for p in opponents)):
        raise ValueError("terminal_requires_only_hero_active_and_all_opponents_all_in")
    if (not isinstance(s.hero_cards, tuple) or len(s.hero_cards) != 2
            or not isinstance(s.board_cards, tuple) or len(s.board_cards) != 5
            or not all(isinstance(c, Card) for c in s.hero_cards + s.board_cards)
            or len(set(s.hero_cards + s.board_cards)) != 7):
        raise ValueError("distinct_known_hero_and_river_cards_required")
    if (not isinstance(s.current_bet, ChipAmount)
            or not isinstance(s.pot_before, ChipAmount)):
        raise ValueError("exact_current_bet_and_pot_required")
    if (s.current_bet.value != max(p.street_committed.value for p in s.seats)
            or s.current_bet.value != max(p.street_committed.value for p in opponents)
            or s.pot_before.value != sum(p.hand_committed.value for p in s.seats)):
        raise ValueError("pot_or_current_bet_does_not_reconcile")
    previous = sorted(p.hand_committed.value - p.street_committed.value
                      for p in s.seats)
    if previous[-1] > previous[-2]:
        raise ValueError("unsettled_prior_street_refund_unsupported")
    if any(value % s.rules.minimum_chip for p in s.seats for value in (
            p.stack.value, p.street_committed.value, p.hand_committed.value)):
        raise ValueError("chip_amounts_not_aligned")
    cost = min(hero.stack.value, s.current_bet.value - hero.street_committed.value)
    if cost <= 0:
        raise ValueError("positive_terminal_call_required")
    if (not isinstance(s.ranges, tuple) or not all(
            isinstance(r, RangeDistribution) for r in s.ranges)
            or len(s.ranges) != len(opponents)
            or {r.seat_id for r in s.ranges} != {p.seat_id for p in opponents}):
        raise ValueError("one_range_per_all_in_opponent_required")
    for r in s.ranges:
        if (r.source != "manual_terminal_assumption"
                or re.fullmatch(r"manual:[a-zA-Z0-9_-]+", r.source_version) is None
                or any(t in r.source_version.lower() for t in (
                    "gto", "solver", "live", "approved"))
                or r.effective_sample_size != 0 or r.confidence != 0):
            raise ValueError("range_must_disclose_manual_unvalidated_hypothesis")
        concrete = [frozenset(parse_concrete_combo(c)) for c in r.combo_weights]
        if not concrete or len(set(concrete)) != len(concrete):
            raise ValueError("empty_or_duplicate_concrete_range")
    if prod(len(r.combo_weights) for r in s.ranges) > max_joint_assignments:
        raise ValueError("joint_assignment_budget_exceeded")
    return hero, cost


def _rake(pots, rules):
    total = sum((p.amount.value for p in pots), Decimal(0))
    estimate = estimate_rake(rules, str(total), saw_flop=True)
    if (estimate.amount is None or rules.rake_distribution not in (
            "proportional_all_pots", "main_pot_first")
            or not 0 <= estimate.amount <= total):
        raise ValueError("rake_policy_unknown_or_invalid")
    remaining = Fraction(estimate.amount)
    values = {}
    for pot in pots:
        value = (Fraction(estimate.amount) * Fraction(pot.amount.value)
                 / Fraction(total)
                 if rules.rake_distribution == "proportional_all_pots" and total
                 else min(remaining, Fraction(pot.amount.value)))
        values[pot.pot_id] = value
        remaining -= value
    return values


def analyze_terminal_multiway(scenario, *, max_joint_assignments=4096):
    """Return conditional exact action EVs or BLOCKED; never a live Advice."""
    try:
        hero, cost = _validate(scenario, max_joint_assignments)
        s = scenario
        called = replace(hero, stack=ChipAmount(hero.stack.value - cost),
                         street_committed=ChipAmount(
                             hero.street_committed.value + cost),
                         hand_committed=ChipAmount(hero.hand_committed.value + cost),
                         status=PlayerStatus.ALL_IN if cost == hero.stack.value else (
                             PlayerStatus.ACTIVE))
        after_call = tuple(called if p.is_hero else p for p in s.seats)
        after_fold = tuple(replace(p, status=PlayerStatus.FOLDED) if p.is_hero else p
                           for p in s.seats)
        call = calculate_side_pots(after_call, settle_uncalled=True)
        fold = calculate_side_pots(after_fold, settle_uncalled=True)
        for calculated, expected in ((call, s.pot_before.value + cost),
                                     (fold, s.pot_before.value)):
            if sum((p.amount.value for p in calculated.pots), Decimal(0)) + sum(
                    (v.value for v in calculated.uncalled_returns.values()), Decimal(0)
            ) != expected:
                raise ValueError("settled_pot_refund_conservation_failed")
        rake = _rake(call.pots, s.rules)
        _rake(fold.pots, s.rules)
        assignments = enumerate_joint_assignments(
            s.ranges, s.hero_cards + s.board_cards,
            max_combinations=max_joint_assignments)
        # Recover exact ORIGINAL product weights, avoiding Decimal 1/3 rounding
        # before an indifference comparison. Enumeration still owns blockers.
        weights = {
            r.seat_id: {frozenset(parse_concrete_combo(c)): Fraction(w)
                        for c, w in r.combo_weights.items()} for r in s.ranges
        }
        raw = [prod(weights[seat][frozenset(holding)]
                    for seat, holding in a.holdings.items()) for a in assignments]
        weight_total = sum(raw, Fraction(0))
        shares = {p.pot_id: Fraction(0) for p in call.pots}
        for assignment, weight in zip(assignments, raw):
            strength = {s.hero_seat: evaluate(s.hero_cards + s.board_cards)}
            strength.update({seat: evaluate(holding + s.board_cards)
                             for seat, holding in assignment.holdings.items()})
            for pot in call.pots:
                best = max(strength[seat] for seat in pot.eligible_seats)
                winners = [seat for seat in pot.eligible_seats
                           if strength[seat] == best]
                if s.hero_seat in winners:
                    shares[pot.pot_id] += weight / weight_total / len(winners)
        audits = tuple(TerminalPotAudit(
            p.pot_id, p.amount.value, p.eligible_seats,
            shares[p.pot_id], rake[p.pot_id],
            Fraction(p.amount.value) * shares[p.pot_id],
            (Fraction(p.amount.value) - rake[p.pot_id]) * shares[p.pot_id],
        ) for p in call.pots)
        refund = Fraction(call.uncalled_returns.get(s.hero_seat, ChipAmount("0")).value)
        fold_ev = Fraction(fold.uncalled_returns.get(
            s.hero_seat, ChipAmount("0")).value)
        gross = (sum((p.hero_gross_return for p in audits), Fraction(0))
                 + refund - Fraction(cost))
        net = (sum((p.hero_net_return for p in audits), Fraction(0))
               + refund - Fraction(cost))
        gap = net - fold_ev
        return TerminalAnalysis(
            "COMPLETE_CONDITIONAL", recommendation=(
                "CALL" if gap > 0 else "FOLD" if gap < 0 else "INDIFFERENT"),
            call_cost=cost, call_gross_ev=gross, call_net_ev=net, fold_ev=fold_ev,
            net_call_minus_fold=gap, call_pots=audits,
            call_refunds=tuple(sorted(
                (k, v.value) for k, v in call.uncalled_returns.items())),
            fold_refunds=tuple(sorted(
                (k, v.value) for k, v in fold.uncalled_returns.items())),
            fold_pots=tuple((p.pot_id, p.amount.value, p.eligible_seats)
                            for p in fold.pots),
            seat_commitments_before=tuple((p.seat_id, p.hand_committed.value)
                                          for p in s.seats),
            seat_commitments_after_call=tuple((p.seat_id, p.hand_committed.value)
                                              for p in after_call),
            joint_assignments=len(assignments), rule_fingerprint=s.rules.fingerprint,
            range_sources=tuple((r.seat_id, r.source, r.source_version)
                                for r in s.ranges),
        )
    except (TypeError, ValueError, ArithmeticError, InvalidStateError) as exc:
        return TerminalAnalysis("BLOCKED", reasons=(str(exc),))
