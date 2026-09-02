"""Strategy Provider capability and result contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Mapping, Protocol, runtime_checkable

from poker_engine.core._freeze import _require_aware_dt, freeze_mapping
from poker_engine.core.enums import ActionType, Position, Street
from poker_engine.core.events import EventType
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from .contracts import DecisionContext, GameType


class MatchKind(str, Enum):
    EXACT = "exact"
    INTERPOLATED = "interpolated"
    HEURISTIC = "heuristic"
    EQUITY_ONLY = "equity_only"


class LookupState(str, Enum):
    NOT_CHECKED = "NOT_CHECKED"
    HIT_EXACT = "HIT_EXACT"
    HIT_APPROXIMATE = "HIT_APPROXIMATE"
    NOT_FOUND = "NOT_FOUND"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REJECTED = "REJECTED"
    NO_STRATEGY = "NO_STRATEGY"


@dataclass(frozen=True)
class MatchDimension:
    """One transparent abstraction difference used by a Provider match."""

    name: str
    requested: str
    matched: str
    distance: Decimal
    maximum_distance: Decimal

    def __post_init__(self) -> None:
        for field_name in ("name", "requested", "matched"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty str")
        for field_name in ("distance", "maximum_distance"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be a Decimal")
            if not value.is_finite() or value < 0:
                raise ValueError(f"{field_name} must be finite and >= 0")
        if self.distance > self.maximum_distance:
            raise ValueError("distance cannot exceed maximum_distance")

    @property
    def score(self) -> float:
        if self.maximum_distance == 0:
            return 1.0 if self.distance == 0 else 0.0
        return float(Decimal("1") - self.distance / self.maximum_distance)


@dataclass(frozen=True)
class CapabilityMatch:
    applicable: bool
    match_kind: MatchKind | None
    score: float
    differences: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    dimensions: tuple[MatchDimension, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.applicable, bool):
            raise TypeError("applicable must be a bool")
        if self.match_kind is not None and not isinstance(
            self.match_kind, MatchKind
        ):
            raise TypeError("match_kind must be a MatchKind or None")
        if not isinstance(self.score, (int, float)) or isinstance(
            self.score, bool
        ):
            raise TypeError("score must be a float")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        for name in ("differences", "reasons"):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        dimensions = tuple(self.dimensions)
        if not all(isinstance(value, MatchDimension) for value in dimensions):
            raise TypeError("dimensions must contain MatchDimension values")
        names = [value.name for value in dimensions]
        if len(names) != len(set(names)):
            raise ValueError("match dimension names must be unique")
        object.__setattr__(self, "dimensions", dimensions)
        if self.applicable and self.match_kind is None:
            raise ValueError("applicable match requires match_kind")
        if not self.applicable and self.match_kind is not None:
            raise ValueError("non-applicable match cannot have match_kind")


def _validated_buckets(values, name: str) -> tuple[Decimal, ...]:
    result = tuple(values)
    if any(
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value < 0
        for value in result
    ) or tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be sorted unique non-negative Decimals")
    return result


@dataclass(frozen=True)
class ProviderCapability:
    player_counts: frozenset[int]
    streets: frozenset[Street]
    game_types: frozenset[GameType]
    stack_buckets_bb: tuple[Decimal, ...]
    ante_values: tuple[ChipAmount, ...]
    rake_percent_values: tuple[Decimal, ...]
    action_lines: frozenset[str]
    base_match_kind: MatchKind = MatchKind.EXACT
    allow_stack_interpolation: bool = False
    max_stack_distance_bb: Decimal = Decimal("0")
    priority: int = 100
    ante_values_are_bb: bool = False
    stack_ante_pairs_bb: tuple[tuple[Decimal, Decimal], ...] = ()
    hero_positions: frozenset[Position] = frozenset()
    pot_buckets_bb: tuple[Decimal, ...] = ()
    allow_pot_interpolation: bool = False
    max_pot_distance_bb: Decimal = Decimal("0")
    aggressive_size_buckets_bb: tuple[Decimal, ...] = ()
    allow_aggressive_size_interpolation: bool = False
    max_aggressive_size_distance_bb: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        counts = frozenset(self.player_counts)
        if not counts or not all(
            isinstance(count, int) and not isinstance(count, bool)
            and 2 <= count <= 9 for count in counts
        ):
            raise ValueError("player_counts must contain values in [2, 9]")
        object.__setattr__(self, "player_counts", counts)
        streets = frozenset(self.streets)
        if not streets or not all(isinstance(street, Street) for street in streets):
            raise ValueError("streets must contain Street values")
        object.__setattr__(self, "streets", streets)
        game_types = frozenset(self.game_types)
        if not game_types or not all(
            isinstance(game_type, GameType) for game_type in game_types
        ):
            raise ValueError("game_types must contain GameType values")
        object.__setattr__(self, "game_types", game_types)
        stacks = tuple(self.stack_buckets_bb)
        if not stacks or not all(isinstance(value, Decimal) for value in stacks):
            raise ValueError("stack_buckets_bb must contain Decimal values")
        if any(not value.is_finite() or value < 0 for value in stacks):
            raise ValueError("stack buckets must be finite and non-negative")
        if tuple(sorted(set(stacks))) != stacks:
            raise ValueError("stack_buckets_bb must be sorted and unique")
        object.__setattr__(self, "stack_buckets_bb", stacks)
        antes = tuple(self.ante_values)
        if not antes or not all(isinstance(value, ChipAmount) for value in antes):
            raise ValueError("ante_values must contain ChipAmount values")
        object.__setattr__(self, "ante_values", antes)
        rake = tuple(self.rake_percent_values)
        if not rake or not all(isinstance(value, Decimal) for value in rake):
            raise ValueError("rake_percent_values must contain Decimal values")
        if any(not value.is_finite() or not Decimal("0") <= value <= Decimal("1")
               for value in rake):
            raise ValueError("rake values must be finite and in [0, 1]")
        object.__setattr__(self, "rake_percent_values", rake)
        action_lines = frozenset(self.action_lines)
        if not action_lines or not all(
            isinstance(value, str) and value for value in action_lines
        ):
            raise ValueError("action_lines must contain non-empty strings")
        object.__setattr__(self, "action_lines", action_lines)
        if not isinstance(self.base_match_kind, MatchKind):
            raise TypeError("base_match_kind must be a MatchKind")
        if self.base_match_kind is MatchKind.EQUITY_ONLY:
            raise ValueError("strategy Provider cannot declare equity_only")
        if not isinstance(self.allow_stack_interpolation, bool):
            raise TypeError("allow_stack_interpolation must be a bool")
        if not isinstance(self.max_stack_distance_bb, Decimal):
            raise TypeError("max_stack_distance_bb must be a Decimal")
        if (not self.max_stack_distance_bb.is_finite()
                or self.max_stack_distance_bb < 0):
            raise ValueError("max_stack_distance_bb must be finite and >= 0")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError("priority must be an int")
        if not isinstance(self.ante_values_are_bb, bool):
            raise TypeError("ante_values_are_bb must be a bool")
        pairs = tuple(self.stack_ante_pairs_bb)
        if any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, Decimal) for value in pair)
            or any(not value.is_finite() or value < 0 for value in pair)
            for pair in pairs
        ):
            raise ValueError(
                "stack_ante_pairs_bb must contain finite non-negative "
                "Decimal pairs"
            )
        if tuple(sorted(set(pairs))) != pairs:
            raise ValueError("stack_ante_pairs_bb must be sorted and unique")
        if any(stack not in stacks for stack, _ in pairs):
            raise ValueError("stack/ante pair stack must exist in stack buckets")
        object.__setattr__(self, "stack_ante_pairs_bb", pairs)
        positions = frozenset(self.hero_positions)
        if not positions or not all(
            isinstance(value, Position) for value in positions
        ):
            raise ValueError("hero_positions must contain supported positions")
        if Position.UNKNOWN in positions:
            raise ValueError("hero_positions cannot contain UNKNOWN")
        object.__setattr__(self, "hero_positions", positions)
        pot_buckets = tuple(self.pot_buckets_bb)
        if any(
            not isinstance(value, Decimal)
            or not value.is_finite()
            or value < 0
            for value in pot_buckets
        ):
            raise ValueError(
                "pot_buckets_bb must be sorted unique non-negative Decimals"
            )
        if tuple(sorted(set(pot_buckets))) != pot_buckets:
            raise ValueError(
                "pot_buckets_bb must be sorted unique non-negative Decimals"
            )
        object.__setattr__(self, "pot_buckets_bb", pot_buckets)
        if not isinstance(self.allow_pot_interpolation, bool):
            raise TypeError("allow_pot_interpolation must be a bool")
        if not isinstance(self.max_pot_distance_bb, Decimal):
            raise TypeError("max_pot_distance_bb must be a Decimal")
        if (
            not self.max_pot_distance_bb.is_finite()
            or self.max_pot_distance_bb < 0
        ):
            raise ValueError("max_pot_distance_bb must be finite and >= 0")
        if self.allow_pot_interpolation and not pot_buckets:
            raise ValueError("pot interpolation requires pot_buckets_bb")
        aggressive = _validated_buckets(
            self.aggressive_size_buckets_bb,
            "aggressive_size_buckets_bb",
        )
        object.__setattr__(self, "aggressive_size_buckets_bb", aggressive)
        if not isinstance(self.allow_aggressive_size_interpolation, bool):
            raise TypeError("allow_aggressive_size_interpolation must be a bool")
        if not isinstance(self.max_aggressive_size_distance_bb, Decimal):
            raise TypeError("max_aggressive_size_distance_bb must be a Decimal")
        if (
            not self.max_aggressive_size_distance_bb.is_finite()
            or self.max_aggressive_size_distance_bb < 0
        ):
            raise ValueError(
                "max_aggressive_size_distance_bb must be finite and >= 0"
            )
        if self.allow_aggressive_size_interpolation and not aggressive:
            raise ValueError(
                "aggressive size interpolation requires size buckets"
            )

    def match(self, context: DecisionContext) -> CapabilityMatch:
        reasons = []
        if context.strategy_player_count not in self.player_counts:
            reasons.append("unsupported_player_count")
        if context.street not in self.streets:
            reasons.append("unsupported_street")
        if context.game_config.game_type not in self.game_types:
            reasons.append("unsupported_game_type")
        ante_value = context.game_config.ante
        if self.ante_values_are_bb:
            ante_value = ChipAmount(
                context.game_config.ante.value
                / context.game_config.big_blind.value
            )
        if ante_value not in self.ante_values:
            reasons.append("unsupported_ante")
        if context.game_config.rake_percent not in self.rake_percent_values:
            reasons.append("unsupported_rake")
        if context.action_line is None:
            reasons.append("missing_action_line")
        elif context.action_line not in self.action_lines:
            reasons.append("unsupported_action_line")
        hero_position = next(
            seat.position for seat in context.seats if seat.seat_id == context.hero_seat
        )
        if self.hero_positions and hero_position not in self.hero_positions:
            reasons.append("unsupported_hero_position")
        if reasons:
            return CapabilityMatch(False, None, 0.0, reasons=tuple(reasons))
        if context.effective_stack_bb is None:
            return CapabilityMatch(
                False, None, 0.0, reasons=("missing_effective_stack",)
            )
        stack = context.effective_stack_bb
        stack_buckets = self.stack_buckets_bb
        if self.stack_ante_pairs_bb:
            ante_bb = (
                context.game_config.ante.value
                / context.game_config.big_blind.value
            )
            stack_buckets = tuple(
                pair_stack for pair_stack, pair_ante in self.stack_ante_pairs_bb
                if pair_ante == ante_bb
            )
            if not stack_buckets:
                return CapabilityMatch(
                    False, None, 0.0,
                    reasons=("unsupported_stack_ante_combination",),
                )
            if stack in self.stack_buckets_bb and stack not in stack_buckets:
                return CapabilityMatch(
                    False, None, 0.0,
                    reasons=("unsupported_stack_ante_combination",),
                )
        dimensions = []
        if stack not in stack_buckets:
            if not self.allow_stack_interpolation:
                return CapabilityMatch(
                    False, None, 0.0, reasons=("unsupported_stack",)
                )
            matched_stack = min(
                stack_buckets, key=lambda value: (abs(stack - value), value)
            )
            distance = abs(stack - matched_stack)
            if distance > self.max_stack_distance_bb:
                return CapabilityMatch(
                    False, None, 0.0, reasons=("unsupported_stack",)
                )
            dimensions.append(MatchDimension(
                "effective_stack_bb",
                str(stack),
                str(matched_stack),
                distance,
                self.max_stack_distance_bb,
            ))
        if self.pot_buckets_bb:
            requested_pot = sum(
                (pot.amount.value for pot in context.pots), Decimal("0")
            ) / context.game_config.big_blind.value
            if requested_pot not in self.pot_buckets_bb:
                if not self.allow_pot_interpolation:
                    return CapabilityMatch(
                        False, None, 0.0, reasons=("unsupported_pot",)
                    )
                matched_pot = min(
                    self.pot_buckets_bb,
                    key=lambda value: (abs(requested_pot - value), value),
                )
                distance = abs(requested_pot - matched_pot)
                if distance > self.max_pot_distance_bb:
                    return CapabilityMatch(
                        False, None, 0.0, reasons=("unsupported_pot",)
                    )
                dimensions.append(MatchDimension(
                    "pot_bb",
                    str(requested_pot),
                    str(matched_pot),
                    distance,
                    self.max_pot_distance_bb,
                ))
        if self.aggressive_size_buckets_bb:
            aggressive_event = next((
                event for event in reversed(context.action_history)
                if event.event_type in (
                    EventType.BET, EventType.RAISE, EventType.ALL_IN,
                )
            ), None)
            raw_amount = (
                aggressive_event.payload.get("amount_total_street")
                if aggressive_event is not None else None
            )
            try:
                requested_size = (
                    Decimal(str(raw_amount))
                    / context.game_config.big_blind.value
                )
            except (ArithmeticError, TypeError, ValueError):
                return CapabilityMatch(
                    False, None, 0.0,
                    reasons=("missing_aggressive_size",),
                )
            if requested_size not in self.aggressive_size_buckets_bb:
                if not self.allow_aggressive_size_interpolation:
                    return CapabilityMatch(
                        False, None, 0.0,
                        reasons=("unsupported_aggressive_size",),
                    )
                matched_size = min(
                    self.aggressive_size_buckets_bb,
                    key=lambda value: (abs(requested_size - value), value),
                )
                distance = abs(requested_size - matched_size)
                if distance > self.max_aggressive_size_distance_bb:
                    return CapabilityMatch(
                        False, None, 0.0,
                        reasons=("unsupported_aggressive_size",),
                    )
                dimensions.append(MatchDimension(
                    "last_aggressive_total_bb",
                    str(requested_size),
                    str(matched_size),
                    distance,
                    self.max_aggressive_size_distance_bb,
                ))
        score = min((item.score for item in dimensions), default=1.0)
        kind = self.base_match_kind
        if dimensions and kind is MatchKind.EXACT:
            kind = MatchKind.INTERPOLATED
        return CapabilityMatch(
            True,
            kind,
            score,
            differences=tuple(
                f"{item.name}:{item.requested}->{item.matched}"
                for item in dimensions
            ),
            dimensions=tuple(dimensions),
        )


@dataclass(frozen=True)
class ActionOption:
    """One canonical action/size branch and its strategy probability."""

    action: ActionType
    probability: Decimal
    amount: ChipAmount | None = None
    source_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionType):
            raise TypeError("action must be an ActionType")
        if not isinstance(self.probability, Decimal):
            raise TypeError("probability must be a Decimal")
        if not self.probability.is_finite() or not (
            Decimal("0") <= self.probability <= Decimal("1")
        ):
            raise ValueError("probability must be in [0, 1]")
        if self.amount is not None and not isinstance(self.amount, ChipAmount):
            raise TypeError("amount must be a ChipAmount or None")
        if self.amount is not None and self.action not in (
            ActionType.BET, ActionType.RAISE, ActionType.ALL_IN,
        ):
            raise ValueError("only bet/raise/all-in options can carry amount")
        if self.source_label is not None and (
            not isinstance(self.source_label, str) or not self.source_label
        ):
            raise ValueError("source_label must be a non-empty str or None")


@dataclass(frozen=True)
class StrategyCandidate:
    hand_id: str
    state_version: int
    request_id: str
    provider_id: str
    provider_version: str
    match_kind: MatchKind
    state_match_score: float
    action_probabilities: Mapping[ActionType, Decimal]
    match_dimensions: tuple[MatchDimension, ...] = ()
    recommended_sizes: Mapping[ActionType, tuple[ChipAmount, ...]] = field(
        default_factory=dict
    )
    action_options: tuple[ActionOption, ...] = ()
    action_ev: Mapping[ActionType, ChipDelta] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    produced_at: datetime | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("hand_id", "request_id", "provider_id", "provider_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")
        if not isinstance(self.match_kind, MatchKind):
            raise TypeError("match_kind must be a MatchKind")
        if self.match_kind is MatchKind.EQUITY_ONLY:
            raise ValueError("StrategyCandidate cannot be equity_only")
        if not isinstance(self.state_match_score, (int, float)) or isinstance(
            self.state_match_score, bool
        ):
            raise TypeError("state_match_score must be a float")
        if not math.isfinite(self.state_match_score) or not (
            0.0 <= self.state_match_score <= 1.0
        ):
            raise ValueError("state_match_score must be in [0, 1]")
        dimensions = tuple(self.match_dimensions)
        if not all(isinstance(value, MatchDimension) for value in dimensions):
            raise TypeError("match_dimensions must contain MatchDimension values")
        names = [value.name for value in dimensions]
        if len(names) != len(set(names)):
            raise ValueError("match dimension names must be unique")
        object.__setattr__(self, "match_dimensions", dimensions)
        if self.match_kind is MatchKind.INTERPOLATED and not dimensions:
            raise ValueError(
                "interpolated candidate requires match_dimensions"
            )
        if dimensions and self.state_match_score > min(
            value.score for value in dimensions
        ):
            raise ValueError(
                "state_match_score cannot exceed match dimension score"
            )
        probabilities = dict(self.action_probabilities)
        if not probabilities:
            raise ValueError("action_probabilities cannot be empty")
        for action, probability in probabilities.items():
            if not isinstance(action, ActionType):
                raise TypeError("action probability keys must be ActionType")
            if not isinstance(probability, Decimal):
                raise TypeError("action probabilities must be Decimal")
            if not probability.is_finite() or not (
                Decimal("0") <= probability <= Decimal("1")
            ):
                raise ValueError("action probabilities must be in [0, 1]")
        if sum(probabilities.values(), Decimal("0")) != Decimal("1"):
            raise ValueError("action probabilities must sum exactly to 1")
        object.__setattr__(self, "action_probabilities", freeze_mapping(probabilities))
        sizes = dict(self.recommended_sizes)
        if not all(isinstance(action, ActionType) for action in sizes):
            raise TypeError("recommended_sizes keys must be ActionType")
        normalized_sizes = {}
        for action, values in sizes.items():
            values = tuple(values)
            if not all(isinstance(size, ChipAmount) for size in values):
                raise TypeError("recommended_sizes must contain ChipAmount values")
            if action not in probabilities:
                raise ValueError("recommended size action must have probability")
            normalized_sizes[action] = values
        object.__setattr__(
            self, "recommended_sizes", freeze_mapping(normalized_sizes)
        )
        options = tuple(self.action_options)
        if not all(isinstance(option, ActionOption) for option in options):
            raise TypeError("action_options must contain ActionOption values")
        if options:
            if sum(
                (option.probability for option in options), Decimal("0")
            ) != Decimal("1"):
                raise ValueError("action option probabilities must sum to 1")
            option_totals: dict[ActionType, Decimal] = {}
            for option in options:
                option_totals[option.action] = (
                    option_totals.get(option.action, Decimal("0"))
                    + option.probability
                )
            if option_totals != probabilities:
                raise ValueError(
                    "action options must aggregate to action_probabilities"
                )
        object.__setattr__(self, "action_options", options)
        action_ev = dict(self.action_ev)
        if not all(isinstance(action, ActionType) for action in action_ev):
            raise TypeError("action_ev keys must be ActionType")
        if not all(isinstance(value, ChipDelta) for value in action_ev.values()):
            raise TypeError("action_ev values must be ChipDelta")
        object.__setattr__(self, "action_ev", freeze_mapping(action_ev))
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("confidence must be a float")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        for name in ("evidence", "assumptions"):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        for name in ("produced_at", "expires_at"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, datetime):
                    raise TypeError(f"{name} must be a datetime or None")
                _require_aware_dt(value)


@dataclass(frozen=True)
class ProviderResult:
    state: LookupState
    provider_id: str
    candidate: StrategyCandidate | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, LookupState):
            raise TypeError("state must be a LookupState")
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("provider_id must be a non-empty str")
        hit = self.state in (LookupState.HIT_EXACT, LookupState.HIT_APPROXIMATE)
        if hit != (self.candidate is not None):
            raise ValueError("hit state and candidate presence must agree")
        if self.candidate is not None and not isinstance(
            self.candidate, StrategyCandidate
        ):
            raise TypeError("candidate must be a StrategyCandidate or None")
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise TypeError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)


@runtime_checkable
class StrategyProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def source_version(self) -> str: ...

    @property
    def capability(self) -> ProviderCapability: ...

    def query(self, context: DecisionContext) -> ProviderResult: ...


class FakeProvider:
    """Deterministic Provider for contract/router/integration tests."""

    def __init__(
        self,
        provider_id: str,
        source_version: str,
        capability: ProviderCapability,
        result: ProviderResult | Callable[[DecisionContext], ProviderResult],
    ) -> None:
        if not isinstance(provider_id, str) or not provider_id:
            raise ValueError("provider_id must be a non-empty str")
        if not isinstance(source_version, str) or not source_version:
            raise ValueError("source_version must be a non-empty str")
        if not isinstance(capability, ProviderCapability):
            raise TypeError("capability must be a ProviderCapability")
        if not isinstance(result, ProviderResult) and not callable(result):
            raise TypeError("result must be ProviderResult or callable")
        self._provider_id = provider_id
        self._source_version = source_version
        self._capability = capability
        self._result = result

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def source_version(self) -> str:
        return self._source_version

    @property
    def capability(self) -> ProviderCapability:
        return self._capability

    def query(self, context: DecisionContext) -> ProviderResult:
        if callable(self._result):
            return self._result(context)
        return self._result


__all__ = [
    "ActionOption",
    "CapabilityMatch",
    "FakeProvider",
    "LookupState",
    "MatchDimension",
    "MatchKind",
    "ProviderCapability",
    "ProviderResult",
    "StrategyCandidate",
    "StrategyProvider",
]
