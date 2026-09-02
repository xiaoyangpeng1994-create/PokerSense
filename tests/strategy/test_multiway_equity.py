"""Exact weighted multi-player main/side-pot equity tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.contracts import PotState, RangeDistribution
from poker_engine.strategy.multiway_equity import (
    exact_multiway_pot_share,
    monte_carlo_multiway_pot_share,
    monte_carlo_multiway_ranges,
)
from poker_engine.strategy.range_tracker import JointRangeAssignment

from .helpers import card


def _assignment(
    holdings: dict[int, tuple[str, str]],
    weight: str = "1",
) -> JointRangeAssignment:
    return JointRangeAssignment(
        {
            seat: (card(cards[0]), card(cards[1]))
            for seat, cards in holdings.items()
        },
        Decimal(weight),
    )


def test_three_way_tie_allocates_only_among_actual_pot_winners():
    result = exact_multiway_pot_share(
        hero_seat=2,
        hero_cards=(card("2d"), card("3c")),
        joint_assignments=(
            _assignment({0: ("2h", "3h"), 1: ("2s", "4c")}),
        ),
        board_cards=(
            card("As"), card("Ah"), card("Kd"), card("2c"), card("3d"),
        ),
        pots=(PotState("main", ChipAmount("60"), (0, 1, 2)),),
    )

    pot = result.pots[0]
    assert pot.win_probability == Decimal("0")
    assert pot.tie_probability == Decimal("1")
    assert pot.loss_probability == Decimal("0")
    assert pot.expected_share == Decimal("0.5")
    assert pot.expected_chips == ChipAmount("30.0")
    assert result.pot_equity == Decimal("0.5")


def test_side_pot_is_allocated_only_to_its_eligible_players():
    result = exact_multiway_pot_share(
        hero_seat=2,
        hero_cards=(card("2d"), card("3c")),
        joint_assignments=(
            _assignment({0: ("2h", "3h"), 1: ("2s", "4c")}),
        ),
        board_cards=(
            card("As"), card("Ah"), card("Kd"), card("2c"), card("3d"),
        ),
        pots=(
            PotState("main", ChipAmount("60"), (0, 1, 2)),
            PotState("side-1", ChipAmount("60"), (0, 1)),
        ),
    )

    main, side = result.pots
    assert main.expected_chips == ChipAmount("30.0")
    assert side.loss_probability == Decimal("1")
    assert side.expected_share == Decimal("0")
    assert side.expected_chips == ChipAmount("0")
    assert result.expected_chips == ChipAmount("30.0")
    assert result.pot_equity == Decimal("0.25")


def test_weighted_joint_assignments_drive_expected_pot_share():
    result = exact_multiway_pot_share(
        hero_seat=1,
        hero_cards=(card("As"), card("Ad")),
        joint_assignments=(
            _assignment({0: ("Ks", "Kd")}, "0.75"),
            _assignment({0: ("5s", "6s")}, "0.25"),
        ),
        board_cards=(
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        pots=(PotState("main", ChipAmount("100"), (0, 1)),),
    )

    pot = result.pots[0]
    assert pot.win_probability == Decimal("0.75")
    assert pot.tie_probability == Decimal("0")
    assert pot.loss_probability == Decimal("0.25")
    assert pot.expected_share == Decimal("0.75")
    assert result.expected_chips == ChipAmount("75.00")
    assert result.assignment_count == 2
    assert result.samples == 2


def test_turn_enumeration_is_deterministic_and_probabilities_sum_exactly():
    result = exact_multiway_pot_share(
        hero_seat=1,
        hero_cards=(card("As"), card("Ad")),
        joint_assignments=(_assignment({0: ("Ks", "Kd")}),),
        board_cards=(card("2c"), card("3d"), card("4h"), card("9s")),
        pots=(PotState("main", ChipAmount("20"), (0, 1)),),
    )

    pot = result.pots[0]
    assert result.samples == 44
    assert (
        pot.win_probability + pot.tie_probability + pot.loss_probability
        == Decimal("1")
    )
    assert pot.expected_chips.value == pot.amount.value * pot.expected_share


def test_multiway_equity_rejects_missing_eligible_holding():
    with pytest.raises(ValueError, match="eligible"):
        exact_multiway_pot_share(
            hero_seat=2,
            hero_cards=(card("As"), card("Ad")),
            joint_assignments=(_assignment({0: ("Ks", "Kd")}),),
            board_cards=(
                card("2c"),
                card("3d"),
                card("4h"),
                card("9s"),
                card("Tc"),
            ),
            pots=(PotState("main", ChipAmount("30"), (0, 1, 2)),),
        )


def test_multiway_equity_enforces_exact_enumeration_budget():
    with pytest.raises(ValueError, match="max_outcomes"):
        exact_multiway_pot_share(
            hero_seat=1,
            hero_cards=(card("As"), card("Ad")),
            joint_assignments=(_assignment({0: ("Ks", "Kd")}),),
            board_cards=(card("2c"), card("3d"), card("4h")),
            pots=(PotState("main", ChipAmount("20"), (0, 1)),),
            max_outcomes=100,
        )


def test_multiway_equity_rejects_card_collision_with_board():
    with pytest.raises(ValueError, match="collides"):
        exact_multiway_pot_share(
            hero_seat=1,
            hero_cards=(card("As"), card("Ad")),
            joint_assignments=(_assignment({0: ("Ks", "2c")}),),
            board_cards=(
                card("2c"),
                card("3d"),
                card("4h"),
                card("9s"),
                card("Tc"),
            ),
            pots=(PotState("main", ChipAmount("20"), (0, 1)),),
        )


def test_seeded_multiway_monte_carlo_is_reproducible_and_close_to_exact():
    assignments = (
        _assignment({0: ("Ks", "Kd")}, "0.75"),
        _assignment({0: ("5s", "6s")}, "0.25"),
    )
    kwargs = {
        "hero_seat": 1,
        "hero_cards": (card("As"), card("Ad")),
        "joint_assignments": assignments,
        "board_cards": (
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        "pots": (PotState("main", ChipAmount("100"), (0, 1)),),
        "trials": 5000,
        "seed": 17,
    }

    first = monte_carlo_multiway_pot_share(**kwargs)
    second = monte_carlo_multiway_pot_share(**kwargs)

    assert first == second
    assert abs(first.result.pot_equity - Decimal("0.75")) < Decimal("0.03")
    assert first.confidence_low <= Decimal("0.75") <= first.confidence_high
    assert first.result.samples == 5000


def test_single_trial_monte_carlo_reports_maximum_uncertainty():
    result = monte_carlo_multiway_pot_share(
        hero_seat=1,
        hero_cards=(card("As"), card("Ad")),
        joint_assignments=(_assignment({0: ("Ks", "Kd")}),),
        board_cards=(
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        pots=(PotState("main", ChipAmount("100"), (0, 1)),),
        trials=1,
        seed=7,
    )

    assert result.confidence_low == Decimal("0")
    assert result.confidence_high == Decimal("1")
    assert result.standard_error == Decimal("0.5")


def test_direct_range_monte_carlo_rejects_collisions_without_cartesian_product():
    ranges = (
        RangeDistribution(
            0,
            {"KsKd": Decimal("0.5"), "QsQd": Decimal("0.5")},
            "test",
            "v1",
        ),
        RangeDistribution(
            1,
            {"KsKh": Decimal("0.5"), "JsJh": Decimal("0.5")},
            "test",
            "v1",
        ),
    )
    kwargs = {
        "hero_seat": 2,
        "hero_cards": (card("As"), card("Ad")),
        "villain_ranges": ranges,
        "board_cards": (
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        "pots": (PotState("main", ChipAmount("90"), (0, 1, 2)),),
        "trials": 500,
        "seed": 23,
    }

    first = monte_carlo_multiway_ranges(**kwargs)
    second = monte_carlo_multiway_ranges(**kwargs)

    assert first == second
    assert first.result.samples == 500
    assert first.result.assignment_count == 4
    assert Decimal("0") <= first.result.pot_equity <= Decimal("1")


def test_direct_range_monte_carlo_refuses_ranges_with_no_legal_joint_deal():
    ranges = (
        RangeDistribution(0, {"KsKd": Decimal("1")}, "test", "v1"),
        RangeDistribution(1, {"KsKh": Decimal("1")}, "test", "v1"),
    )

    with pytest.raises(ValueError, match="collision-free"):
        monte_carlo_multiway_ranges(
            hero_seat=2,
            hero_cards=(card("As"), card("Ad")),
            villain_ranges=ranges,
            board_cards=(
                card("2c"),
                card("3d"),
                card("4h"),
                card("9s"),
                card("Tc"),
            ),
            pots=(PotState("main", ChipAmount("90"), (0, 1, 2)),),
            trials=1,
            seed=3,
            max_assignment_attempts=10,
        )


def test_monte_carlo_stops_at_wall_deadline_and_returns_partial_samples():
    result = monte_carlo_multiway_pot_share(
        hero_seat=1,
        hero_cards=(card("As"), card("Ad")),
        joint_assignments=(_assignment({0: ("Ks", "Kd")}),),
        board_cards=(
            card("2c"), card("3d"), card("4h"), card("9s"), card("Tc"),
        ),
        pots=(PotState("main", ChipAmount("100"), (0, 1)),),
        trials=100,
        seed=7,
        deadline_at=1.0,
        monotonic_clock=lambda: 2.0,
        deadline_check_interval=16,
    )

    assert result.result.samples == 16
