"""Immutable multi-player input contracts for the strategy layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

from poker_engine.core._freeze import (
    _require_aware_dt,
    freeze_mapping,
)
from poker_engine.core.enums import ActionType, PlayerStatus, Position, Street
from poker_engine.core.events import StateEvent
from poker_engine.core.request_context import RequestContext
from poker_engine.core.value_objects import Card, ChipAmount


class GameType(str, Enum):
    CASH = "cash"
    TOURNAMENT = "tournament"


class InputSource(str, Enum):
    VISION = "vision"
    MANUAL = "manual"
    CONFIG = "config"
    DERIVED = "derived"
    INFERRED = "inferred"


class QualityStatus(str, Enum):
    VALID = "VALID"
    UNKNOWN = "UNKNOWN"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CONFLICT = "CONFLICT"


class ActionAmountSemantics(str, Enum):
    NONE = "none"
    ADDITIONAL = "additional"
    TOTAL_STREET = "total_street"


def _require_probability(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a float")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _require_decimal(value: Decimal, name: str, *, non_negative=True) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class GameConfig:
    variant: str
    game_type: GameType
    max_seats: int
    dealt_player_count: int
    small_blind: ChipAmount
    big_blind: ChipAmount
    ante: ChipAmount = field(default_factory=ChipAmount.zero)
    rake_percent: Decimal = Decimal("0")
    rake_cap: ChipAmount = field(default_factory=ChipAmount.zero)
    minimum_chip: ChipAmount = field(default_factory=lambda: ChipAmount("1"))

    def __post_init__(self) -> None:
        if not isinstance(self.variant, str) or not self.variant:
            raise ValueError("variant must be a non-empty str")
        if not isinstance(self.game_type, GameType):
            raise TypeError("game_type must be a GameType")
        for name in ("max_seats", "dealt_player_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if not 2 <= value <= 9:
                raise ValueError(f"{name} must be in [2, 9]")
        if self.dealt_player_count > self.max_seats:
            raise ValueError("dealt_player_count cannot exceed max_seats")
        for name in (
            "small_blind", "big_blind", "ante", "rake_cap", "minimum_chip",
        ):
            if not isinstance(getattr(self, name), ChipAmount):
                raise TypeError(f"{name} must be a ChipAmount")
        if self.small_blind.value <= 0 or self.big_blind.value <= 0:
            raise ValueError("blinds must be > 0")
        if self.small_blind >= self.big_blind:
            raise ValueError("small_blind must be less than big_blind")
        if self.minimum_chip.value <= 0:
            raise ValueError("minimum_chip must be > 0")
        _require_decimal(self.rake_percent, "rake_percent")
        if self.rake_percent > Decimal("1"):
            raise ValueError("rake_percent must be <= 1")


@dataclass(frozen=True)
class DecisionSeat:
    seat_id: int
    player_id: str | None
    position: Position
    stack: ChipAmount
    street_committed: ChipAmount
    hand_committed: ChipAmount
    status: PlayerStatus
    occupied: bool = True
    is_hero: bool = False
    is_dealer: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seat_id, int) or isinstance(self.seat_id, bool):
            raise TypeError("seat_id must be an int")
        if self.seat_id < 0:
            raise ValueError("seat_id must be >= 0")
        if self.player_id is not None and (
            not isinstance(self.player_id, str) or not self.player_id
        ):
            raise ValueError("player_id must be a non-empty str or None")
        if not isinstance(self.position, Position):
            raise TypeError("position must be a Position")
        for name in ("stack", "street_committed", "hand_committed"):
            if not isinstance(getattr(self, name), ChipAmount):
                raise TypeError(f"{name} must be a ChipAmount")
        if self.street_committed > self.hand_committed:
            raise ValueError("street_committed cannot exceed hand_committed")
        if not isinstance(self.status, PlayerStatus):
            raise TypeError("status must be a PlayerStatus")
        for name in ("occupied", "is_hero", "is_dealer"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True)
class PotState:
    pot_id: str
    amount: ChipAmount
    eligible_seats: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pot_id, str) or not self.pot_id:
            raise ValueError("pot_id must be a non-empty str")
        if not isinstance(self.amount, ChipAmount):
            raise TypeError("amount must be a ChipAmount")
        seats = tuple(self.eligible_seats)
        if not seats or len(seats) != len(set(seats)):
            raise ValueError("eligible_seats must be non-empty and unique")
        if not all(isinstance(seat, int) and not isinstance(seat, bool)
                   and seat >= 0 for seat in seats):
            raise TypeError("eligible_seats must contain non-negative ints")
        object.__setattr__(self, "eligible_seats", seats)


@dataclass(frozen=True)
class LegalAction:
    action: ActionType
    min_amount: ChipAmount
    max_amount: ChipAmount
    amount_semantics: ActionAmountSemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ActionType):
            raise TypeError("action must be an ActionType")
        if not isinstance(self.min_amount, ChipAmount):
            raise TypeError("min_amount must be a ChipAmount")
        if not isinstance(self.max_amount, ChipAmount):
            raise TypeError("max_amount must be a ChipAmount")
        if self.min_amount > self.max_amount:
            raise ValueError("min_amount cannot exceed max_amount")
        semantics = self.amount_semantics
        if semantics is None:
            semantics = {
                ActionType.FOLD: ActionAmountSemantics.NONE,
                ActionType.CHECK: ActionAmountSemantics.NONE,
                ActionType.CALL: ActionAmountSemantics.ADDITIONAL,
                ActionType.ALL_IN: ActionAmountSemantics.ADDITIONAL,
                ActionType.BET: ActionAmountSemantics.TOTAL_STREET,
                ActionType.RAISE: ActionAmountSemantics.TOTAL_STREET,
                ActionType.POST_SB: ActionAmountSemantics.ADDITIONAL,
                ActionType.POST_BB: ActionAmountSemantics.ADDITIONAL,
                ActionType.POST_ANTE: ActionAmountSemantics.ADDITIONAL,
            }[self.action]
            object.__setattr__(self, "amount_semantics", semantics)
        if not isinstance(semantics, ActionAmountSemantics):
            raise TypeError("amount_semantics must be ActionAmountSemantics")
        if self.action in (ActionType.FOLD, ActionType.CHECK):
            if self.min_amount.value != 0 or self.max_amount.value != 0:
                raise ValueError("fold/check amounts must be zero")
            if semantics is not ActionAmountSemantics.NONE:
                raise ValueError("fold/check amount semantics must be none")


@dataclass(frozen=True)
class EffectiveStack:
    opponent_seat: int
    amount: ChipAmount

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_seat, int) or isinstance(
            self.opponent_seat, bool
        ):
            raise TypeError("opponent_seat must be an int")
        if self.opponent_seat < 0:
            raise ValueError("opponent_seat must be >= 0")
        if not isinstance(self.amount, ChipAmount):
            raise TypeError("amount must be a ChipAmount")


@dataclass(frozen=True)
class RangeDistribution:
    seat_id: int
    combo_weights: Mapping[str, Decimal]
    source: str
    source_version: str
    entropy: Decimal | None = None
    effective_sample_size: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.seat_id, int) or isinstance(self.seat_id, bool):
            raise TypeError("seat_id must be an int")
        if self.seat_id < 0:
            raise ValueError("seat_id must be >= 0")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty str")
        if not isinstance(self.source_version, str) or not self.source_version:
            raise ValueError("source_version must be a non-empty str")
        weights = dict(self.combo_weights)
        if not all(isinstance(combo, str) and combo for combo in weights):
            raise TypeError("combo keys must be non-empty strings")
        for weight in weights.values():
            _require_decimal(weight, "combo weight")
        if weights and sum(weights.values(), Decimal("0")) <= 0:
            raise ValueError("non-empty range must have positive total weight")
        object.__setattr__(self, "combo_weights", freeze_mapping(weights))
        if self.entropy is not None:
            _require_decimal(self.entropy, "entropy")
        if not isinstance(self.effective_sample_size, int) or isinstance(
            self.effective_sample_size, bool
        ):
            raise TypeError("effective_sample_size must be an int")
        if self.effective_sample_size < 0:
            raise ValueError("effective_sample_size must be >= 0")
        _require_probability(self.confidence, "confidence")


@dataclass(frozen=True)
class InputProvenance:
    field_name: str
    source: InputSource
    status: QualityStatus
    confidence: float
    evidence_ref: str
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name:
            raise ValueError("field_name must be a non-empty str")
        if not isinstance(self.source, InputSource):
            raise TypeError("source must be an InputSource")
        if not isinstance(self.status, QualityStatus):
            raise TypeError("status must be a QualityStatus")
        _require_probability(self.confidence, "confidence")
        if not isinstance(self.evidence_ref, str) or not self.evidence_ref:
            raise ValueError("evidence_ref must be a non-empty str")
        if self.observed_at is not None:
            if not isinstance(self.observed_at, datetime):
                raise TypeError("observed_at must be a datetime or None")
            _require_aware_dt(self.observed_at)


@dataclass(frozen=True)
class ContextQuality:
    overall_confidence: float
    field_confidences: Mapping[str, float] = field(default_factory=dict)
    hard_failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_probability(self.overall_confidence, "overall_confidence")
        values = dict(self.field_confidences)
        if not all(isinstance(name, str) and name for name in values):
            raise TypeError("field confidence keys must be non-empty strings")
        for name, confidence in values.items():
            _require_probability(confidence, f"field_confidences[{name}]")
        object.__setattr__(self, "field_confidences", freeze_mapping(values))
        failures = tuple(self.hard_failures)
        if not all(isinstance(reason, str) and reason for reason in failures):
            raise TypeError("hard_failures must contain non-empty strings")
        object.__setattr__(self, "hard_failures", failures)

    @property
    def is_decision_ready(self) -> bool:
        return not self.hard_failures


@dataclass(frozen=True)
class DecisionContext:
    request: RequestContext
    game_config: GameConfig
    seats: tuple[DecisionSeat, ...]
    hero_seat: int
    actor_seat: int | None
    active_seats: tuple[int, ...]
    hero_cards: tuple[Card, ...]
    board_cards: tuple[Card, ...]
    street: Street
    pots: tuple[PotState, ...]
    legal_actions: tuple[LegalAction, ...]
    action_history: tuple[StateEvent, ...]
    effective_stacks: tuple[EffectiveStack, ...]
    hero_range: RangeDistribution | None
    villain_ranges: tuple[RangeDistribution, ...]
    input_quality: ContextQuality
    input_provenance: tuple[InputProvenance, ...]
    missing_fields: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    action_line: str | None = None
    effective_stack_bb: Decimal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, RequestContext):
            raise TypeError("request must be a RequestContext")
        if not isinstance(self.game_config, GameConfig):
            raise TypeError("game_config must be a GameConfig")
        seats = tuple(self.seats)
        if not seats or not all(isinstance(seat, DecisionSeat) for seat in seats):
            raise TypeError("seats must contain DecisionSeat values")
        seat_ids = [seat.seat_id for seat in seats]
        if len(seat_ids) != len(set(seat_ids)):
            raise ValueError("seats must have unique seat_id values")
        if len(seats) > self.game_config.max_seats:
            raise ValueError("seat count cannot exceed max_seats")
        object.__setattr__(self, "seats", seats)
        hero_seats = [seat.seat_id for seat in seats if seat.is_hero]
        if hero_seats != [self.hero_seat]:
            raise ValueError("exactly one seat must match hero_seat")
        active = tuple(self.active_seats)
        if len(active) < 2 or len(active) != len(set(active)):
            raise ValueError("active_seats must contain at least two unique seats")
        if not set(active) <= set(seat_ids):
            raise ValueError("active_seats must reference known seats")
        object.__setattr__(self, "active_seats", active)
        if self.actor_seat is not None and self.actor_seat not in active:
            raise ValueError("actor_seat must be active or None")
        hero_cards = tuple(self.hero_cards)
        board_cards = tuple(self.board_cards)
        if len(hero_cards) not in (0, 2):
            raise ValueError("hero_cards must contain zero or two cards")
        expected_board = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5,
        }.get(self.street)
        if expected_board is not None and len(board_cards) != expected_board:
            raise ValueError("board card count does not match street")
        all_cards = hero_cards + board_cards
        if not all(isinstance(card, Card) for card in all_cards):
            raise TypeError("hero_cards and board_cards must contain Card values")
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("known cards must be distinct")
        object.__setattr__(self, "hero_cards", hero_cards)
        object.__setattr__(self, "board_cards", board_cards)
        if not isinstance(self.street, Street):
            raise TypeError("street must be a Street")
        for name, expected in (
            ("pots", PotState),
            ("legal_actions", LegalAction),
            ("action_history", StateEvent),
            ("effective_stacks", EffectiveStack),
            ("villain_ranges", RangeDistribution),
            ("input_provenance", InputProvenance),
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, expected) for value in values):
                raise TypeError(f"{name} must contain {expected.__name__} values")
            object.__setattr__(self, name, values)
        if self.hero_range is not None and not isinstance(
            self.hero_range, RangeDistribution
        ):
            raise TypeError("hero_range must be a RangeDistribution or None")
        if not isinstance(self.input_quality, ContextQuality):
            raise TypeError("input_quality must be a ContextQuality")
        for name in ("missing_fields", "assumptions"):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        fields = [item.field_name for item in self.input_provenance]
        if len(fields) != len(set(fields)):
            raise ValueError("input_provenance fields must be unique")
        if self.action_line is not None and (
            not isinstance(self.action_line, str) or not self.action_line
        ):
            raise ValueError("action_line must be a non-empty str or None")
        if self.effective_stack_bb is not None:
            _require_decimal(self.effective_stack_bb, "effective_stack_bb")

    @property
    def hand_id(self) -> str:
        return self.request.hand_id

    @property
    def state_version(self) -> int:
        return self.request.state_version

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def strategy_player_count(self) -> int:
        if self.street is Street.PREFLOP:
            return self.game_config.dealt_player_count
        return len(self.active_seats)

    @property
    def legal_action_types(self) -> frozenset[ActionType]:
        return frozenset(action.action for action in self.legal_actions)

    @property
    def is_decision_ready(self) -> bool:
        return (
            not self.missing_fields
            and self.input_quality.is_decision_ready
            and self.actor_seat == self.hero_seat
        )


__all__ = [
    "ActionAmountSemantics",
    "ContextQuality",
    "DecisionContext",
    "DecisionSeat",
    "EffectiveStack",
    "GameConfig",
    "GameType",
    "InputProvenance",
    "InputSource",
    "LegalAction",
    "PotState",
    "QualityStatus",
    "RangeDistribution",
]
