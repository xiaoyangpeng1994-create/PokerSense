"""Offline terminal-river differential oracle, independent of strategy math.

PokerKit 0.7.5 supplies showdown ordering. This module separately enumerates
manual card hypotheses and derives pot layers, refunds, rake, and payouts. It
never consumes private media or emits live advice. It cannot certify opponent
ranges, game rules, betting-history reachability, or a full-hand strategy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

from poker_engine.core.enums import PlayerStatus
from poker_engine.strategy.terminal_multiway_v1 import (
    TerminalScenario, analyze_terminal_multiway,
)


PINNED_POKERKIT = "0.7.5"


@dataclass(frozen=True)
class OraclePot:
    amount: Fraction
    eligible_seats: tuple[int, ...]


@dataclass(frozen=True)
class OracleSettlement:
    pots: tuple[OraclePot, ...]
    refunds: tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True)
class OracleTerminal:
    call: OracleSettlement
    fold: OracleSettlement
    call_cost: Fraction
    call_rake: tuple[Fraction, ...]
    hero_shares: tuple[Fraction, ...]
    call_gross_ev: Fraction
    call_net_ev: Fraction
    fold_ev: Fraction
    joint_assignments: int
    # The policy-relative value is computed separately from card/rule truth.
    oracle_kind: str = "POKERKIT_SHOWDOWN_PLUS_INDEPENDENT_EXACT_SETTLEMENT"


def _pokerkit_hand():
    try:
        from importlib.metadata import version
        from pokerkit import StandardHighHand
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("pokerkit_0_7_5_not_installed") from exc
    if version("pokerkit") != PINNED_POKERKIT:
        raise RuntimeError("pokerkit_version_mismatch")
    return StandardHighHand


def _settle(commitments, folded):
    """Derive terminal pot layers from each player's total chips at risk.

    This intentionally does not call PokerSense's calculate_side_pots. A lone
    contributor's top layer is refunded; folded contributions remain dead money.
    Adjacent layers with identical eligible players form one pot.
    """
    levels = sorted({amount for amount in commitments.values() if amount > 0})
    prior = Fraction(0)
    layers = []
    refunds = {}
    for ceiling in levels:
        contributors = tuple(sorted(
            seat for seat, amount in commitments.items() if amount >= ceiling
        ))
        chips = (ceiling - prior) * len(contributors)
        prior = ceiling
        if len(contributors) == 1:
            seat = contributors[0]
            refunds[seat] = refunds.get(seat, Fraction(0)) + chips
            continue
        eligible = tuple(seat for seat in contributors if seat not in folded)
        if not eligible:
            raise ValueError("orphan_pot_layer")
        if layers and layers[-1].eligible_seats == eligible:
            layers[-1] = OraclePot(layers[-1].amount + chips, eligible)
        else:
            layers.append(OraclePot(chips, eligible))
    if sum((p.amount for p in layers), Fraction(0)) + sum(
            refunds.values(), Fraction(0)) != sum(commitments.values(), Fraction(0)):
        raise ValueError("independent_chip_conservation_failed")
    return OracleSettlement(tuple(layers), tuple(sorted(refunds.items())))


def _rake(pots, rules):
    if rules.rake_application not in ("all_pots", "postflop_only"):
        raise ValueError("unverified_rake_application")
    if rules.rake_rounding not in (
            "exact", "floor_to_chip", "ceil_to_chip"):
        raise ValueError("unverified_rake_rounding")
    if rules.rake_distribution not in (
            "proportional_all_pots", "main_pot_first"):
        raise ValueError("unverified_rake_distribution")
    total = sum((p.amount for p in pots), Fraction(0))
    cap = Fraction(rules.rake_cap_bb * rules.big_blind)
    charge = min(total * Fraction(rules.rake_percent), cap)
    if rules.rake_rounding != "exact":
        unit = Fraction(rules.minimum_chip)
        units = charge / unit
        rounded = units.numerator // units.denominator
        if rules.rake_rounding == "ceil_to_chip" and units.denominator != 1:
            rounded += 1
        charge = min(rounded * unit, cap)
    if rules.rake_distribution == "proportional_all_pots":
        return tuple(charge * p.amount / total for p in pots)
    remaining = charge
    values = []
    for pot in pots:
        assigned = min(pot.amount, remaining)
        values.append(assigned)
        remaining -= assigned
    return tuple(values)


def _parse_combo(text):
    if len(text) != 4 or text[:2] == text[2:]:
        raise ValueError("invalid_concrete_combo")
    ranks, suits = "23456789TJQKA", "cdhs"
    if (text[0] not in ranks or text[2] not in ranks
            or text[1] not in suits or text[3] not in suits):
        raise ValueError("invalid_concrete_combo")
    return text[:2], text[2:]


def _assignments(scenario, budget):
    options = []
    for distribution in sorted(scenario.ranges, key=lambda r: r.seat_id):
        choices = tuple((distribution.seat_id, _parse_combo(combo),
                         Fraction(weight))
                        for combo, weight in sorted(
                            distribution.combo_weights.items()) if weight > 0)
        options.append(choices)
    count = 1
    for choices in options:
        count *= len(choices)
    if not 0 < count <= budget:
        raise ValueError("oracle_joint_assignment_budget_exceeded")
    known = {str(card) for card in scenario.hero_cards + scenario.board_cards}
    valid = []
    for selected in product(*options):
        seen = set(known)
        weight = Fraction(1)
        holdings = {}
        for seat, cards, chance in selected:
            if set(cards) & seen:
                break
            seen.update(cards)
            holdings[seat] = cards
            weight *= chance
        else:
            valid.append((holdings, weight))
    total = sum((weight for _, weight in valid), Fraction(0))
    if total <= 0:
        raise ValueError("oracle_no_collision_free_joint_assignment")
    return tuple((holdings, weight / total) for holdings, weight in valid)


def reference_terminal(scenario: TerminalScenario, *,
                       max_joint_assignments=256) -> OracleTerminal:
    """Compute terminal counterfactuals with PokerKit ranks and exact chips."""
    if not isinstance(scenario, TerminalScenario):
        raise TypeError("terminal_scenario_required")
    if scenario.rules.verification_status != "simulation":
        raise ValueError("reference_accepts_simulation_rules_only")
    if scenario.split_policy != "fractional_equal_split":
        raise ValueError("reference_requires_fractional_equal_split")
    if scenario.other_fees != Decimal(0):
        raise ValueError("reference_requires_declared_zero_other_fees")
    hand_class = _pokerkit_hand()
    seats = {p.seat_id: p for p in scenario.seats}
    hero = seats[scenario.hero_seat]
    cost = min(hero.stack.value,
               scenario.current_bet.value - hero.street_committed.value)
    if cost <= 0:
        raise ValueError("reference_requires_positive_call")
    prior = {seat: Fraction(p.hand_committed.value)
             for seat, p in seats.items()}
    called = dict(prior)
    called[scenario.hero_seat] += Fraction(cost)
    folded = {seat for seat, p in seats.items()
              if p.status is PlayerStatus.FOLDED}
    call = _settle(called, folded)
    fold = _settle(prior, folded | {scenario.hero_seat})
    rake = _rake(call.pots, scenario.rules)
    assignments = _assignments(scenario, max_joint_assignments)
    board = "".join(map(str, scenario.board_cards))
    hero_cards = "".join(map(str, scenario.hero_cards))
    hero_share = [Fraction(0) for _ in call.pots]
    for holdings, probability in assignments:
        ranks = {scenario.hero_seat: hand_class.from_game(hero_cards, board)}
        ranks.update({seat: hand_class.from_game("".join(cards), board)
                      for seat, cards in holdings.items()})
        for index, pot in enumerate(call.pots):
            best = max(ranks[seat] for seat in pot.eligible_seats)
            winners = [seat for seat in pot.eligible_seats
                       if ranks[seat] == best]
            if scenario.hero_seat in winners:
                hero_share[index] += probability / len(winners)
    hero_refund = dict(call.refunds).get(scenario.hero_seat, Fraction(0))
    gross = hero_refund - Fraction(cost) + sum(
        (p.amount * share for p, share in zip(call.pots, hero_share)),
        Fraction(0))
    net = hero_refund - Fraction(cost) + sum(
        ((p.amount - fee) * share
         for p, fee, share in zip(call.pots, rake, hero_share)), Fraction(0))
    return OracleTerminal(call, fold, Fraction(cost), rake, tuple(hero_share),
                          gross, net,
                          dict(fold.refunds).get(scenario.hero_seat, Fraction(0)),
                          len(assignments))


def compare_terminal(scenario: TerminalScenario, *, max_joint_assignments=256):
    """Fail closed on a blocked baseline, unavailable reference, or mismatch."""
    baseline = analyze_terminal_multiway(
        scenario, max_joint_assignments=max_joint_assignments)
    if baseline.status != "COMPLETE_CONDITIONAL":
        return {"status": "BASELINE_BLOCKED", "reasons": baseline.reasons}
    try:
        oracle = reference_terminal(
            scenario, max_joint_assignments=max_joint_assignments)
    except RuntimeError as exc:
        return {"status": "REFERENCE_UNAVAILABLE", "reasons": (str(exc),)}
    except (TypeError, ValueError, ArithmeticError) as exc:
        return {"status": "ORACLE_BLOCKED", "reasons": (str(exc),)}
    expected = {
        "call_pots": tuple((p.amount, p.eligible_seats) for p in oracle.call.pots),
        "call_refunds": oracle.call.refunds,
        "fold_pots": tuple((p.amount, p.eligible_seats) for p in oracle.fold.pots),
        "fold_refunds": oracle.fold.refunds,
        "call_cost": oracle.call_cost,
        "rake_by_pot": oracle.call_rake,
        "hero_share_by_pot": oracle.hero_shares,
        "hero_gross_return_by_pot": tuple(
            p.amount * share for p, share in zip(
                oracle.call.pots, oracle.hero_shares)),
        "hero_net_return_by_pot": tuple(
            (p.amount - rake) * share
            for p, rake, share in zip(
                oracle.call.pots, oracle.call_rake, oracle.hero_shares)),
        "call_gross_ev": oracle.call_gross_ev,
        "call_net_ev": oracle.call_net_ev,
        "fold_ev": oracle.fold_ev,
        "net_call_minus_fold": oracle.call_net_ev - oracle.fold_ev,
        "recommendation": (
            "CALL" if oracle.call_net_ev > oracle.fold_ev else
            "FOLD" if oracle.call_net_ev < oracle.fold_ev else "INDIFFERENT"),
        "joint_assignments": oracle.joint_assignments,
    }
    actual = {
        "call_pots": tuple((Fraction(p.amount), p.eligible_seats)
                           for p in baseline.call_pots),
        "call_refunds": tuple((seat, Fraction(amount))
                              for seat, amount in baseline.call_refunds),
        "fold_pots": tuple((Fraction(amount), eligible)
                           for _, amount, eligible in baseline.fold_pots),
        "fold_refunds": tuple((seat, Fraction(amount))
                              for seat, amount in baseline.fold_refunds),
        "call_cost": Fraction(baseline.call_cost),
        "rake_by_pot": tuple(p.configured_rake for p in baseline.call_pots),
        "hero_share_by_pot": tuple(p.hero_share for p in baseline.call_pots),
        "hero_gross_return_by_pot": tuple(
            p.hero_gross_return for p in baseline.call_pots),
        "hero_net_return_by_pot": tuple(
            p.hero_net_return for p in baseline.call_pots),
        "call_gross_ev": baseline.call_gross_ev,
        "call_net_ev": baseline.call_net_ev,
        "fold_ev": baseline.fold_ev,
        "net_call_minus_fold": baseline.net_call_minus_fold,
        "recommendation": baseline.recommendation,
        "joint_assignments": baseline.joint_assignments,
    }
    mismatches = tuple(name for name in expected if actual[name] != expected[name])
    return {"status": "MATCH" if not mismatches else "MISMATCH",
            "oracle_kind": oracle.oracle_kind, "pokerkit_version": PINNED_POKERKIT,
            "mismatches": mismatches, "baseline": actual, "reference": expected,
            "qualification": (
                "manual_conditional_river_inputs_only",
                "pokerkit_independent_showdown_ordering",
                "pot_refund_rake_math_independently_implemented_not_pokerkit_state",
                "does_not_validate_ranges_rules_history_or_profitability",
                "never_live_advice",
            )}


def _jsonable(value):
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-joint-assignments", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from tools.analyze_terminal_multiway import scenario_from_dict, unique_object
    raw = args.input.read_bytes()
    scenario = scenario_from_dict(json.loads(raw, object_pairs_hook=unique_object))
    report = compare_terminal(
        scenario, max_joint_assignments=args.max_joint_assignments)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    rendered = json.dumps(_jsonable(report), ensure_ascii=False,
                          sort_keys=True, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
