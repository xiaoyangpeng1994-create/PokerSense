"""Deterministic range filtering, updating, and joint compatibility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from types import MappingProxyType
from typing import Mapping

from poker_engine.core._freeze import freeze_mapping
from poker_engine.core.enums import Rank, Suit
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.value_objects import Card

from .contracts import RangeDistribution


@dataclass(frozen=True)
class RangeUpdate:
    distribution: RangeDistribution
    applied: bool
    likelihood_coverage: Decimal
    missing_likelihood_combos: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, RangeDistribution):
            raise TypeError("distribution must be a RangeDistribution")
        if not isinstance(self.applied, bool):
            raise TypeError("applied must be a bool")
        _require_unit_decimal(self.likelihood_coverage, "likelihood_coverage")
        missing = tuple(self.missing_likelihood_combos)
        if not all(isinstance(combo, str) and combo for combo in missing):
            raise TypeError("missing_likelihood_combos must contain strings")
        object.__setattr__(self, "missing_likelihood_combos", missing)


@dataclass(frozen=True)
class JointRangeAssignment:
    holdings: Mapping[int, tuple[Card, Card]]
    weight: Decimal

    def __post_init__(self) -> None:
        holdings = dict(self.holdings)
        if not holdings:
            raise ValueError("holdings cannot be empty")
        if not all(
            isinstance(seat, int) and not isinstance(seat, bool) and seat >= 0
            for seat in holdings
        ):
            raise TypeError("holding keys must be non-negative seats")
        for holding in holdings.values():
            if (
                not isinstance(holding, tuple)
                or len(holding) != 2
                or not all(isinstance(card, Card) for card in holding)
                or holding[0] == holding[1]
            ):
                raise TypeError("each holding must contain two distinct Cards")
        if not isinstance(self.weight, Decimal) or not self.weight.is_finite():
            raise TypeError("weight must be a finite Decimal")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")
        object.__setattr__(self, "holdings", freeze_mapping(holdings))


def parse_concrete_combo(combo: str) -> tuple[Card, Card]:
    """Parse one concrete combo such as ``AsQc``; abstractions are rejected."""
    if not isinstance(combo, str):
        raise TypeError("combo must be a str")
    if len(combo) != 4:
        raise ValueError(f"combo must be two concrete cards, got {combo!r}")
    try:
        cards = (
            Card(Rank(combo[0]), Suit(combo[1])),
            Card(Rank(combo[2]), Suit(combo[3])),
        )
    except ValueError as exc:
        raise ValueError(f"invalid concrete combo {combo!r}") from exc
    if cards[0] == cards[1]:
        raise ValueError("combo cards must be distinct")
    return cards


def filter_blocked_combos(
    distribution: RangeDistribution,
    known_cards: tuple[Card, ...],
    *,
    source_version: str | None = None,
) -> RangeDistribution:
    """Remove combos colliding with Hero/board cards and renormalize exactly."""
    _require_range(distribution)
    known = tuple(known_cards)
    if not all(isinstance(card, Card) for card in known):
        raise TypeError("known_cards must contain Card values")
    if len(known) != len(set(known)):
        raise ValueError("known_cards must be distinct")
    blocked = set(known)
    retained = {
        combo: weight
        for combo, weight in distribution.combo_weights.items()
        if not blocked.intersection(parse_concrete_combo(combo))
    }
    if not retained:
        raise InvalidStateError("range_card_collision: no compatible combos")
    weights = _normalize(retained)
    return _range_from_weights(
        distribution,
        weights,
        source_version or f"{distribution.source_version}:blockers",
        confidence=distribution.confidence,
    )


def bayesian_action_update(
    prior: RangeDistribution,
    action_likelihoods: Mapping[str, Decimal],
    *,
    source_version: str,
) -> RangeUpdate:
    """Apply P(action|combo) to a prior, preserving missing likelihoods."""
    _require_range(prior)
    if not isinstance(source_version, str) or not source_version:
        raise ValueError("source_version must be a non-empty str")
    likelihoods = dict(action_likelihoods)
    unknown = set(likelihoods) - set(prior.combo_weights)
    if unknown:
        raise ValueError("likelihoods contain combos absent from prior")
    for combo, likelihood in likelihoods.items():
        _require_unit_decimal(likelihood, f"likelihood[{combo}]")
    normalized_prior = _normalize(prior.combo_weights)
    missing = tuple(sorted(set(normalized_prior) - set(likelihoods)))
    coverage = sum(
        (normalized_prior[combo] for combo in likelihoods), Decimal("0")
    )
    weighted = {
        combo: prior_weight * likelihoods.get(combo, Decimal("1"))
        for combo, prior_weight in normalized_prior.items()
    }
    if sum(weighted.values(), Decimal("0")) <= 0:
        raise InvalidStateError("action likelihood removes the entire range")
    posterior = _normalize(weighted)
    confidence = prior.confidence * float(coverage)
    distribution = _range_from_weights(
        prior,
        posterior,
        source_version,
        confidence=confidence,
        effective_sample_size=(
            prior.effective_sample_size + 1 if likelihoods else 0
        ),
    )
    return RangeUpdate(
        distribution=distribution,
        applied=bool(likelihoods),
        likelihood_coverage=coverage,
        missing_likelihood_combos=missing,
    )


def shrink_action_likelihoods(
    population: Mapping[str, Decimal],
    observed: Mapping[str, Decimal],
    *,
    sample_size: int,
    prior_strength: Decimal,
) -> Mapping[str, Decimal]:
    """Shrink observed per-combo likelihoods toward a population prior."""
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an int")
    if sample_size < 0:
        raise ValueError("sample_size must be >= 0")
    if not isinstance(prior_strength, Decimal):
        raise TypeError("prior_strength must be a Decimal")
    if not prior_strength.is_finite() or prior_strength <= 0:
        raise ValueError("prior_strength must be finite and > 0")
    population_values = dict(population)
    observed_values = dict(observed)
    if not population_values or set(population_values) != set(observed_values):
        raise ValueError("population and observed must have identical combo keys")
    for label, values in (
        ("population", population_values),
        ("observed", observed_values),
    ):
        for combo, value in values.items():
            if not isinstance(combo, str) or not combo:
                raise TypeError(f"{label} keys must be non-empty strings")
            _require_unit_decimal(value, f"{label}[{combo}]")
    count = Decimal(sample_size)
    denominator = prior_strength + count
    result = {
        combo: (
            prior_strength * population_values[combo]
            + count * observed_values[combo]
        ) / denominator
        for combo in sorted(population_values)
    }
    return MappingProxyType(result)


def enumerate_joint_assignments(
    ranges: tuple[RangeDistribution, ...],
    known_cards: tuple[Card, ...] = (),
    *,
    max_combinations: int = 100_000,
) -> tuple[JointRangeAssignment, ...]:
    """Enumerate and normalize collision-free multi-player assignments."""
    ranges = tuple(ranges)
    if not ranges or not all(isinstance(item, RangeDistribution) for item in ranges):
        raise TypeError("ranges must contain RangeDistribution values")
    seat_ids = [item.seat_id for item in ranges]
    if len(seat_ids) != len(set(seat_ids)):
        raise ValueError("ranges must have unique seat IDs")
    known = tuple(known_cards)
    if not all(isinstance(card, Card) for card in known):
        raise TypeError("known_cards must contain Card values")
    if len(known) != len(set(known)):
        raise ValueError("known_cards must be distinct")
    if not isinstance(max_combinations, int) or isinstance(max_combinations, bool):
        raise TypeError("max_combinations must be an int")
    if max_combinations <= 0:
        raise ValueError("max_combinations must be > 0")
    concrete = []
    total_combinations = 1
    for distribution in ranges:
        _require_range(distribution)
        holdings = tuple(
            (combo, parse_concrete_combo(combo), weight)
            for combo, weight in distribution.combo_weights.items()
        )
        concrete.append(holdings)
        total_combinations *= len(holdings)
        if total_combinations > max_combinations:
            raise ValueError("joint range exceeds max_combinations")
    raw = []
    blocked = set(known)
    for assignment in product(*concrete):
        cards = tuple(card for _, holding, _ in assignment for card in holding)
        if blocked.intersection(cards) or len(cards) != len(set(cards)):
            continue
        weight = math.prod(item[2] for item in assignment)
        if weight <= 0:
            continue
        raw.append((assignment, weight))
    if not raw:
        raise InvalidStateError("range_card_collision: no joint assignments")
    normalized = _normalize({str(index): item[1] for index, item in enumerate(raw)})
    return tuple(
        JointRangeAssignment(
            holdings={
                distribution.seat_id: assignment[index][1]
                for index, distribution in enumerate(ranges)
            },
            weight=normalized[str(result_index)],
        )
        for result_index, (assignment, _) in enumerate(raw)
    )


def _range_from_weights(
    original: RangeDistribution,
    weights: Mapping[str, Decimal],
    source_version: str,
    *,
    confidence: float,
    effective_sample_size: int | None = None,
) -> RangeDistribution:
    return RangeDistribution(
        seat_id=original.seat_id,
        combo_weights=weights,
        source=original.source,
        source_version=source_version,
        entropy=_entropy(weights),
        effective_sample_size=(
            original.effective_sample_size
            if effective_sample_size is None else effective_sample_size
        ),
        confidence=confidence,
    )


def _normalize(weights: Mapping[str, Decimal]) -> dict[str, Decimal]:
    values = dict(weights)
    if not values:
        raise ValueError("weights cannot be empty")
    for combo, weight in values.items():
        if not isinstance(combo, str) or not combo:
            raise TypeError("weight keys must be non-empty strings")
        if not isinstance(weight, Decimal):
            raise TypeError("weights must be Decimal values")
        if not weight.is_finite() or weight < 0:
            raise ValueError("weights must be finite and >= 0")
    positive = [(combo, weight) for combo, weight in values.items() if weight > 0]
    total = sum((weight for _, weight in positive), Decimal("0"))
    if total <= 0:
        raise ValueError("weights must contain positive mass")
    result: dict[str, Decimal] = {}
    running = Decimal("0")
    for combo, weight in positive[:-1]:
        normalized = weight / total
        result[combo] = normalized
        running += normalized
    last_combo, _ = positive[-1]
    result[last_combo] = Decimal("1") - running
    return result


def _entropy(weights: Mapping[str, Decimal]) -> Decimal:
    value = -sum(
        float(weight) * math.log(float(weight))
        for weight in weights.values()
        if weight > 0
    )
    return Decimal(str(value))


def _require_range(value: RangeDistribution) -> None:
    if not isinstance(value, RangeDistribution):
        raise TypeError("distribution must be a RangeDistribution")
    if not value.combo_weights:
        raise ValueError("range must contain combos")


def _require_unit_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite() or not Decimal("0") <= value <= Decimal("1"):
        raise ValueError(f"{name} must be finite and in [0, 1]")


__all__ = [
    "JointRangeAssignment",
    "RangeUpdate",
    "bayesian_action_update",
    "enumerate_joint_assignments",
    "filter_blocked_combos",
    "parse_concrete_combo",
    "shrink_action_likelihoods",
]
