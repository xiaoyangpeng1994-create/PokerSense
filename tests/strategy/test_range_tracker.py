"""Range blocker, Bayesian update, shrinkage, and multiway tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from poker_engine.core.errors import InvalidStateError
from poker_engine.strategy.contracts import RangeDistribution
from poker_engine.strategy.range_tracker import (
    bayesian_action_update,
    enumerate_joint_assignments,
    filter_blocked_combos,
    parse_concrete_combo,
    shrink_action_likelihoods,
)

from .helpers import card


def _range(
    seat: int,
    weights: dict[str, str],
    *,
    confidence: float = 0.8,
) -> RangeDistribution:
    return RangeDistribution(
        seat_id=seat,
        combo_weights={combo: Decimal(weight) for combo, weight in weights.items()},
        source="test-prior",
        source_version="v1",
        confidence=confidence,
    )


def test_parse_concrete_combo_accepts_exact_cards_only():
    assert tuple(str(value) for value in parse_concrete_combo("AsQc")) == (
        "As",
        "Qc",
    )

    with pytest.raises(ValueError, match="concrete"):
        parse_concrete_combo("AKs")
    with pytest.raises(ValueError, match="distinct"):
        parse_concrete_combo("AsAs")


def test_blocker_filter_removes_hero_and_board_collisions_and_normalizes():
    prior = _range(0, {"AsQc": "0.5", "QsQh": "0.3", "KdJd": "0.2"})

    result = filter_blocked_combos(prior, (card("As"), card("Kd")))

    assert result.combo_weights == {"QsQh": Decimal("1")}
    assert result.source_version == "v1:blockers"
    assert result.confidence == prior.confidence
    assert result.entropy == Decimal("0.0")


def test_blocker_filter_refuses_when_all_combos_collide():
    prior = _range(0, {"AsQc": "0.5", "KdJd": "0.5"})

    with pytest.raises(InvalidStateError, match="range_card_collision"):
        filter_blocked_combos(prior, (card("As"), card("Kd")))


def test_bayesian_update_matches_hand_calculation():
    prior = _range(0, {"AsQc": "0.5", "KhKc": "0.5"})

    result = bayesian_action_update(
        prior,
        {"AsQc": Decimal("0.8"), "KhKc": Decimal("0.2")},
        source_version="action-v1",
    )

    assert result.distribution.combo_weights == {
        "AsQc": Decimal("0.8"),
        "KhKc": Decimal("0.2"),
    }
    assert result.applied
    assert result.likelihood_coverage == Decimal("1.0")
    assert result.missing_likelihood_combos == ()
    assert result.distribution.confidence == prior.confidence
    assert result.distribution.effective_sample_size == 1


def test_bayesian_update_preserves_missing_likelihood_and_lowers_quality():
    prior = _range(0, {"AsQc": "0.5", "KhKc": "0.5"})

    result = bayesian_action_update(
        prior,
        {"AsQc": Decimal("0.2")},
        source_version="partial-action-v1",
    )

    assert result.distribution.combo_weights == {
        "AsQc": Decimal("1") / Decimal("6"),
        "KhKc": Decimal("5") / Decimal("6"),
    }
    assert result.likelihood_coverage == Decimal("0.5")
    assert result.missing_likelihood_combos == ("KhKc",)
    assert result.distribution.confidence == pytest.approx(0.4)


def test_bayesian_update_with_no_likelihood_keeps_prior_and_marks_unapplied():
    prior = _range(0, {"AsQc": "0.25", "KhKc": "0.75"})

    result = bayesian_action_update(
        prior,
        {},
        source_version="missing-action-v1",
    )

    assert result.distribution.combo_weights == prior.combo_weights
    assert result.distribution.confidence == 0.0
    assert not result.applied
    assert result.likelihood_coverage == Decimal("0")
    assert result.missing_likelihood_combos == ("AsQc", "KhKc")


def test_bayesian_update_rejects_impossible_or_unknown_likelihoods():
    prior = _range(0, {"AsQc": "0.5", "KhKc": "0.5"})

    with pytest.raises(InvalidStateError, match="entire range"):
        bayesian_action_update(
            prior,
            {"AsQc": Decimal("0"), "KhKc": Decimal("0")},
            source_version="action-v1",
        )
    with pytest.raises(ValueError, match="absent"):
        bayesian_action_update(
            prior,
            {"QdQs": Decimal("0.5")},
            source_version="action-v1",
        )


def test_small_sample_shrinkage_moves_from_population_toward_observation():
    population = {"AsQc": Decimal("0.2"), "KhKc": Decimal("0.8")}
    observed = {"AsQc": Decimal("0.9"), "KhKc": Decimal("0.1")}

    no_sample = shrink_action_likelihoods(
        population,
        observed,
        sample_size=0,
        prior_strength=Decimal("20"),
    )
    small_sample = shrink_action_likelihoods(
        population,
        observed,
        sample_size=5,
        prior_strength=Decimal("20"),
    )
    large_sample = shrink_action_likelihoods(
        population,
        observed,
        sample_size=2000,
        prior_strength=Decimal("20"),
    )

    assert no_sample == population
    assert population["AsQc"] < small_sample["AsQc"] < observed["AsQc"]
    assert abs(large_sample["AsQc"] - observed["AsQc"]) < Decimal("0.01")
    with pytest.raises(TypeError):
        no_sample["AsQc"] = Decimal("1")


def test_joint_assignments_exclude_cross_player_card_collisions():
    ranges = (
        _range(0, {"AsQc": "0.5", "KhKc": "0.5"}),
        _range(1, {"AsJh": "0.5", "QdQs": "0.5"}),
    )

    assignments = enumerate_joint_assignments(ranges)

    assert len(assignments) == 3
    assert sum((item.weight for item in assignments), Decimal("0")) == Decimal("1")
    for assignment in assignments:
        cards = [card for holding in assignment.holdings.values() for card in holding]
        assert len(cards) == len(set(cards))


def test_joint_assignments_apply_known_card_blockers_and_are_immutable():
    ranges = (
        _range(0, {"AsQc": "0.5", "KhKc": "0.5"}),
        _range(1, {"AsJh": "0.5", "QdQs": "0.5"}),
    )

    assignments = enumerate_joint_assignments(ranges, (card("As"),))

    assert len(assignments) == 1
    assert assignments[0].weight == Decimal("1")
    assert {
        seat: tuple(str(card) for card in holding)
        for seat, holding in assignments[0].holdings.items()
    } == {0: ("Kh", "Kc"), 1: ("Qd", "Qs")}
    with pytest.raises(TypeError):
        assignments[0].holdings[2] = parse_concrete_combo("2c3c")


def test_joint_assignments_refuse_collision_only_or_excessive_product():
    with pytest.raises(InvalidStateError, match="no joint assignments"):
        enumerate_joint_assignments((
            _range(0, {"AsQc": "1"}),
            _range(1, {"AsJh": "1"}),
        ))

    with pytest.raises(ValueError, match="max_combinations"):
        enumerate_joint_assignments(
            (
                _range(0, {"AsQc": "0.5", "KhKc": "0.5"}),
                _range(1, {"AsJh": "0.5", "QdQs": "0.5"}),
            ),
            max_combinations=3,
        )
