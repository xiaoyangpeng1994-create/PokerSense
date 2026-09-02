"""Exact weighted multi-player pot-share equity for main and side pots."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from math import comb
import math
import random
import time
from typing import Callable

from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity._deck import remaining_deck
from poker_engine.equity.evaluator import evaluate

from .contracts import PotState
from .contracts import RangeDistribution
from .range_tracker import JointRangeAssignment, parse_concrete_combo


@dataclass(frozen=True)
class PotEquity:
    pot_id: str
    amount: ChipAmount
    win_probability: Decimal
    tie_probability: Decimal
    loss_probability: Decimal
    expected_share: Decimal
    expected_chips: ChipAmount

    def __post_init__(self) -> None:
        if not isinstance(self.pot_id, str) or not self.pot_id:
            raise ValueError("pot_id must be a non-empty str")
        if not isinstance(self.amount, ChipAmount):
            raise TypeError("amount must be a ChipAmount")
        for name in (
            "win_probability",
            "tie_probability",
            "loss_probability",
            "expected_share",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TypeError(f"{name} must be a finite Decimal")
            if not Decimal("0") <= value <= Decimal("1"):
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.win_probability
            + self.tie_probability
            + self.loss_probability
            != Decimal("1")
        ):
            raise ValueError("win/tie/loss probabilities must sum to 1")
        if not isinstance(self.expected_chips, ChipAmount):
            raise TypeError("expected_chips must be a ChipAmount")


@dataclass(frozen=True)
class MultiwayEquityResult:
    hero_seat: int
    pots: tuple[PotEquity, ...]
    total_pot: ChipAmount
    expected_chips: ChipAmount
    pot_equity: Decimal
    assignment_count: int
    samples: int

    def __post_init__(self) -> None:
        if not isinstance(self.hero_seat, int) or isinstance(self.hero_seat, bool):
            raise TypeError("hero_seat must be an int")
        pots = tuple(self.pots)
        if not pots or not all(isinstance(pot, PotEquity) for pot in pots):
            raise TypeError("pots must contain PotEquity values")
        object.__setattr__(self, "pots", pots)
        if not isinstance(self.total_pot, ChipAmount):
            raise TypeError("total_pot must be a ChipAmount")
        if not isinstance(self.expected_chips, ChipAmount):
            raise TypeError("expected_chips must be a ChipAmount")
        if not isinstance(self.pot_equity, Decimal) or not self.pot_equity.is_finite():
            raise TypeError("pot_equity must be a finite Decimal")
        if not Decimal("0") <= self.pot_equity <= Decimal("1"):
            raise ValueError("pot_equity must be in [0, 1]")
        for name in ("assignment_count", "samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class MonteCarloMultiwayResult:
    result: MultiwayEquityResult
    standard_error: Decimal
    confidence_low: Decimal
    confidence_high: Decimal
    confidence_level: Decimal
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.result, MultiwayEquityResult):
            raise TypeError("result must be a MultiwayEquityResult")
        for name in (
            "standard_error",
            "confidence_low",
            "confidence_high",
            "confidence_level",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TypeError(f"{name} must be a finite Decimal")
        if self.standard_error < 0:
            raise ValueError("standard_error must be >= 0")
        if not Decimal("0") <= self.confidence_low <= self.confidence_high <= 1:
            raise ValueError("confidence interval must be in [0, 1]")
        if not Decimal("0") < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an int")


def exact_multiway_pot_share(
    hero_seat: int,
    hero_cards: tuple[Card, Card],
    joint_assignments: tuple[JointRangeAssignment, ...],
    board_cards: tuple[Card, ...],
    pots: tuple[PotState, ...],
    *,
    max_outcomes: int = 1_000_000,
) -> MultiwayEquityResult:
    """Enumerate board runouts and allocate every pot only among its eligibles."""
    if not isinstance(hero_seat, int) or isinstance(hero_seat, bool):
        raise TypeError("hero_seat must be an int")
    hero = tuple(hero_cards)
    board = tuple(board_cards)
    assignments = tuple(joint_assignments)
    pots = tuple(pots)
    if len(hero) != 2 or not all(isinstance(card, Card) for card in hero):
        raise TypeError("hero_cards must contain exactly two Cards")
    if len(board) not in (0, 3, 4, 5):
        raise ValueError("board_cards must contain 0, 3, 4, or 5 cards")
    if not all(isinstance(card, Card) for card in board):
        raise TypeError("board_cards must contain Card values")
    if len(set(hero + board)) != len(hero + board):
        raise ValueError("hero and board cards must be distinct")
    if not assignments or not all(
        isinstance(item, JointRangeAssignment) for item in assignments
    ):
        raise TypeError("joint_assignments must contain values")
    if not pots or not all(isinstance(pot, PotState) for pot in pots):
        raise TypeError("pots must contain PotState values")
    if len({pot.pot_id for pot in pots}) != len(pots):
        raise ValueError("pots must have unique pot IDs")
    if not isinstance(max_outcomes, int) or isinstance(max_outcomes, bool):
        raise TypeError("max_outcomes must be an int")
    if max_outcomes <= 0:
        raise ValueError("max_outcomes must be > 0")

    weight_total = sum((item.weight for item in assignments), Decimal("0"))
    if weight_total <= 0:
        raise ValueError("joint assignment weight must be positive")
    board_needed = 5 - len(board)
    accumulators = {
        pot.pot_id: {
            "win": Decimal("0"),
            "tie": Decimal("0"),
            "share": Decimal("0"),
        }
        for pot in pots
    }
    samples = 0
    for assignment in assignments:
        holding_seats = set(assignment.holdings)
        required = {
            seat
            for pot in pots
            for seat in pot.eligible_seats
            if seat != hero_seat
        }
        if not required <= holding_seats:
            raise ValueError("assignment lacks a pot-eligible player's holding")
        opponent_cards = tuple(
            card
            for holding in assignment.holdings.values()
            for card in holding
        )
        known = hero + board + opponent_cards
        if len(known) != len(set(known)):
            raise ValueError("assignment collides with hero or board cards")
        available = remaining_deck(known)
        runout_count = comb(len(available), board_needed)
        if samples + runout_count > max_outcomes:
            raise ValueError("equity enumeration exceeds max_outcomes")
        outcome_weight = assignment.weight / weight_total / runout_count
        for runout in combinations(available, board_needed):
            full_board = board + runout
            strengths = {
                hero_seat: evaluate(hero + full_board),
                **{
                    seat: evaluate(holding + full_board)
                    for seat, holding in assignment.holdings.items()
                },
            }
            for pot in pots:
                eligible = pot.eligible_seats
                best = max(strengths[seat] for seat in eligible)
                winners = tuple(
                    seat for seat in eligible if strengths[seat] == best
                )
                values = accumulators[pot.pot_id]
                if hero_seat in winners:
                    share = Decimal("1") / len(winners)
                    values["share"] += outcome_weight * share
                    values["win" if len(winners) == 1 else "tie"] += outcome_weight
            samples += 1

    results = []
    for pot in pots:
        values = accumulators[pot.pot_id]
        win = values["win"]
        tie = values["tie"]
        loss = Decimal("1") - win - tie
        share = values["share"]
        results.append(PotEquity(
            pot_id=pot.pot_id,
            amount=pot.amount,
            win_probability=win,
            tie_probability=tie,
            loss_probability=loss,
            expected_share=share,
            expected_chips=ChipAmount(pot.amount.value * share),
        ))
    total = sum((pot.amount.value for pot in pots), Decimal("0"))
    expected = sum(
        (pot.expected_chips.value for pot in results), Decimal("0")
    )
    return MultiwayEquityResult(
        hero_seat=hero_seat,
        pots=tuple(results),
        total_pot=ChipAmount(total),
        expected_chips=ChipAmount(expected),
        pot_equity=expected / total if total > 0 else Decimal("0"),
        assignment_count=len(assignments),
        samples=samples,
    )


def monte_carlo_multiway_pot_share(
    hero_seat: int,
    hero_cards: tuple[Card, Card],
    joint_assignments: tuple[JointRangeAssignment, ...],
    board_cards: tuple[Card, ...],
    pots: tuple[PotState, ...],
    *,
    trials: int,
    seed: int,
    deadline_at: float | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    deadline_check_interval: int = 16,
) -> MonteCarloMultiwayResult:
    """Seeded weighted sampling over joint ranges and board runouts."""
    if not isinstance(trials, int) or isinstance(trials, bool):
        raise TypeError("trials must be an int")
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    hero = tuple(hero_cards)
    board = tuple(board_cards)
    assignments = tuple(joint_assignments)
    pots = tuple(pots)
    if len(hero) != 2 or not all(isinstance(card, Card) for card in hero):
        raise TypeError("hero_cards must contain exactly two Cards")
    if len(board) not in (0, 3, 4, 5) or not all(
        isinstance(card, Card) for card in board
    ):
        raise ValueError("board_cards must contain 0, 3, 4, or 5 Cards")
    if len(set(hero + board)) != len(hero + board):
        raise ValueError("hero and board cards must be distinct")
    if not assignments or not all(
        isinstance(item, JointRangeAssignment) for item in assignments
    ):
        raise TypeError("joint_assignments must contain values")
    if not pots or not all(isinstance(pot, PotState) for pot in pots):
        raise TypeError("pots must contain values")
    required = {
        seat
        for pot in pots
        for seat in pot.eligible_seats
        if seat != hero_seat
    }
    prepared = []
    for assignment in assignments:
        if not required <= set(assignment.holdings):
            raise ValueError("assignment lacks a pot-eligible player's holding")
        opponent_cards = tuple(
            card for holding in assignment.holdings.values() for card in holding
        )
        known = hero + board + opponent_cards
        if len(known) != len(set(known)):
            raise ValueError("assignment collides with hero or board cards")
        prepared.append((assignment, remaining_deck(known)))
    weights = [float(item[0].weight) for item in prepared]

    def sample_assignment(rng):
        return rng.choices(prepared, weights=weights, k=1)[0]

    return _run_monte_carlo(
        hero_seat,
        hero,
        board,
        pots,
        trials=trials,
        seed=seed,
        confidence_z=Decimal("1.959963984540054"),
        assignment_count=len(assignments),
        sample_assignment=sample_assignment,
        deadline_at=deadline_at,
        monotonic_clock=monotonic_clock,
        deadline_check_interval=deadline_check_interval,
    )


def monte_carlo_multiway_ranges(
    hero_seat: int,
    hero_cards: tuple[Card, Card],
    villain_ranges: tuple[RangeDistribution, ...],
    board_cards: tuple[Card, ...],
    pots: tuple[PotState, ...],
    *,
    trials: int,
    seed: int,
    max_assignment_attempts: int = 1_000,
    deadline_at: float | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
    deadline_check_interval: int = 16,
) -> MonteCarloMultiwayResult:
    """Sample large concrete ranges without materializing their Cartesian product."""
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise ValueError("trials must be an int > 0")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    if not isinstance(max_assignment_attempts, int) or isinstance(
        max_assignment_attempts, bool
    ):
        raise TypeError("max_assignment_attempts must be an int")
    if max_assignment_attempts <= 0:
        raise ValueError("max_assignment_attempts must be > 0")
    hero = tuple(hero_cards)
    board = tuple(board_cards)
    ranges = tuple(villain_ranges)
    pots = tuple(pots)
    if len(hero) != 2 or not all(isinstance(card, Card) for card in hero):
        raise TypeError("hero_cards must contain exactly two Cards")
    if len(board) not in (0, 3, 4, 5) or not all(
        isinstance(card, Card) for card in board
    ):
        raise ValueError("board_cards must contain 0, 3, 4, or 5 Cards")
    known = hero + board
    if len(known) != len(set(known)):
        raise ValueError("hero and board cards must be distinct")
    if not ranges or not all(isinstance(item, RangeDistribution) for item in ranges):
        raise TypeError("villain_ranges must contain values")
    if len({item.seat_id for item in ranges}) != len(ranges):
        raise ValueError("villain_ranges must have unique seat IDs")
    if not pots or not all(isinstance(pot, PotState) for pot in pots):
        raise TypeError("pots must contain values")
    required = {
        seat
        for pot in pots
        for seat in pot.eligible_seats
        if seat != hero_seat
    }
    if not required <= {item.seat_id for item in ranges}:
        raise ValueError("ranges lack a pot-eligible player")
    blocked = set(known)
    prepared_ranges = []
    assignment_count = 1
    for distribution in sorted(ranges, key=lambda item: item.seat_id):
        choices = []
        for combo, weight in distribution.combo_weights.items():
            if weight <= 0:
                continue
            holding = parse_concrete_combo(combo)
            if not blocked.intersection(holding):
                choices.append((holding, float(weight)))
        choices = tuple(choices)
        if not choices:
            raise ValueError("range has no combo compatible with hero/board")
        prepared_ranges.append((distribution.seat_id, choices))
        assignment_count *= len(choices)

    def sample_assignment(rng):
        for _ in range(max_assignment_attempts):
            holdings = {
                seat: rng.choices(
                    [choice[0] for choice in choices],
                    weights=[choice[1] for choice in choices],
                    k=1,
                )[0]
                for seat, choices in prepared_ranges
            }
            cards = tuple(card for holding in holdings.values() for card in holding)
            if len(cards) != len(set(cards)):
                continue
            assignment = JointRangeAssignment(holdings, Decimal("1"))
            return assignment, remaining_deck(known + cards)
        raise ValueError("unable to sample a collision-free joint assignment")

    return _run_monte_carlo(
        hero_seat,
        hero,
        board,
        pots,
        trials=trials,
        seed=seed,
        confidence_z=Decimal("1.959963984540054"),
        assignment_count=assignment_count,
        sample_assignment=sample_assignment,
        deadline_at=deadline_at,
        monotonic_clock=monotonic_clock,
        deadline_check_interval=deadline_check_interval,
    )


def _run_monte_carlo(
    hero_seat,
    hero,
    board,
    pots,
    *,
    trials,
    seed,
    confidence_z,
    assignment_count,
    sample_assignment,
    deadline_at,
    monotonic_clock,
    deadline_check_interval,
):
    if deadline_at is not None and not isinstance(deadline_at, (int, float)):
        raise TypeError("deadline_at must be a monotonic number or None")
    if not callable(monotonic_clock):
        raise TypeError("monotonic_clock must be callable")
    if not isinstance(deadline_check_interval, int) or isinstance(
        deadline_check_interval, bool
    ):
        raise TypeError("deadline_check_interval must be an int")
    if deadline_check_interval <= 0:
        raise ValueError("deadline_check_interval must be > 0")
    total_pot = sum((pot.amount.value for pot in pots), Decimal("0"))
    if total_pot <= 0:
        raise ValueError("total pot must be > 0")
    rng = random.Random(seed)
    wins = {pot.pot_id: 0 for pot in pots}
    ties = {pot.pot_id: 0 for pot in pots}
    shares = {pot.pot_id: Decimal("0") for pot in pots}
    total_share_samples: list[float] = []
    board_needed = 5 - len(board)
    completed = 0
    for trial_index in range(trials):
        if (
            completed > 0
            and trial_index % deadline_check_interval == 0
            and deadline_at is not None
            and monotonic_clock() >= deadline_at
        ):
            break
        assignment, deck = sample_assignment(rng)
        runout = tuple(rng.sample(deck, board_needed))
        full_board = board + runout
        strengths = {
            hero_seat: evaluate(hero + full_board),
            **{
                seat: evaluate(holding + full_board)
                for seat, holding in assignment.holdings.items()
            },
        }
        trial_chips = Decimal("0")
        for pot in pots:
            best = max(strengths[seat] for seat in pot.eligible_seats)
            winners = tuple(
                seat for seat in pot.eligible_seats if strengths[seat] == best
            )
            if hero_seat not in winners:
                continue
            share = Decimal("1") / len(winners)
            shares[pot.pot_id] += share
            trial_chips += pot.amount.value * share
            if len(winners) == 1:
                wins[pot.pot_id] += 1
            else:
                ties[pot.pot_id] += 1
        total_share_samples.append(float(trial_chips / total_pot))
        completed += 1
    denominator = Decimal(completed)
    pot_results = []
    for pot in pots:
        win = Decimal(wins[pot.pot_id]) / denominator
        tie = Decimal(ties[pot.pot_id]) / denominator
        share = shares[pot.pot_id] / denominator
        pot_results.append(PotEquity(
            pot_id=pot.pot_id,
            amount=pot.amount,
            win_probability=win,
            tie_probability=tie,
            loss_probability=Decimal("1") - win - tie,
            expected_share=share,
            expected_chips=ChipAmount(pot.amount.value * share),
        ))
    expected = sum(
        (pot.expected_chips.value for pot in pot_results), Decimal("0")
    )
    result = MultiwayEquityResult(
        hero_seat=hero_seat,
        pots=tuple(pot_results),
        total_pot=ChipAmount(total_pot),
        expected_chips=ChipAmount(expected),
        pot_equity=expected / total_pot,
        assignment_count=assignment_count,
        samples=completed,
    )
    if completed < 2:
        standard_error = Decimal("0.5")
        low = Decimal("0")
        high = Decimal("1")
    else:
        mean = sum(total_share_samples) / completed
        variance = sum(
            (value - mean) ** 2 for value in total_share_samples
        ) / (completed - 1)
        standard_error = Decimal(str(math.sqrt(variance / completed)))
        margin = confidence_z * standard_error
        low = max(Decimal("0"), result.pot_equity - margin)
        high = min(Decimal("1"), result.pot_equity + margin)
    return MonteCarloMultiwayResult(
        result=result,
        standard_error=standard_error,
        confidence_low=low,
        confidence_high=high,
        confidence_level=Decimal("0.95"),
        seed=seed,
    )


__all__ = [
    "MonteCarloMultiwayResult",
    "MultiwayEquityResult",
    "PotEquity",
    "exact_multiway_pot_share",
    "monte_carlo_multiway_pot_share",
    "monte_carlo_multiway_ranges",
]
