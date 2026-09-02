"""Authoritative state contracts: PokerState, StateContext, ValidationResult.

PokerState is the system's authoritative, immutable game state. Its ONLY
responsibility is to protect structural correctness — it does NOT infer or
reason about what happened (that is the State Engine's job in a later task).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._freeze import freeze_mapping
from .enums import Street
from .events import StateEvent
from .opponents import PlayerState
from .value_objects import Card, ChipAmount


# ---------------------------------------------------------------------------
# PokerState
# ---------------------------------------------------------------------------

def _validate_cards_distinct(cards: tuple[Card, ...]) -> None:
    if not all(isinstance(c, Card) for c in cards):
        raise TypeError("every card must be a Card instance")
    if len(set(cards)) != len(cards):
        raise ValueError("cards must not contain duplicates")


@dataclass(frozen=True)
class PokerState:
    """Authoritative, immutable poker game state.

    Structural invariants (checked in __post_init__):
    - state_version >= 0
    - hand_id non-empty
    - hero_cards length 0 or 2
    - board_cards length 0 / 3 / 4 / 5
    - no duplicate cards; hero/board disjoint
    - players: no duplicate seat, no duplicate player_id, at most one is_hero
    - actor (if not None) must reference an existing seat
    - money fields are ChipAmount (non-negative by construction)
    """

    state_version: int
    hand_id: str
    street: Street
    hero_cards: tuple[Card, ...]
    board_cards: tuple[Card, ...]
    players: tuple[PlayerState, ...]
    pot: ChipAmount
    current_bet: ChipAmount
    to_call: ChipAmount
    actor: int | None = None

    def __post_init__(self) -> None:
        # state_version
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")

        # hand_id
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")

        # street
        if not isinstance(self.street, Street):
            raise TypeError("street must be a Street enum")

        # cards
        hero = tuple(self.hero_cards)
        board = tuple(self.board_cards)
        object.__setattr__(self, "hero_cards", hero)
        object.__setattr__(self, "board_cards", board)

        if len(hero) not in (0, 2):
            raise ValueError(f"hero_cards must have 0 or 2 cards, got {len(hero)}")
        if len(board) not in (0, 3, 4, 5):
            raise ValueError(
                f"board_cards must have 0/3/4/5 cards, got {len(board)}"
            )
        _validate_cards_distinct(hero)
        _validate_cards_distinct(board)
        if set(hero) & set(board):
            raise ValueError("hero_cards and board_cards overlap")

        # players
        players = tuple(self.players)
        object.__setattr__(self, "players", players)
        if not all(isinstance(p, PlayerState) for p in players):
            raise TypeError("players must be PlayerState instances")

        seats = [p.seat for p in players]
        if len(set(seats)) != len(seats):
            raise ValueError("players must not have duplicate seats")

        ids = [p.player_id for p in players]
        if len(set(ids)) != len(ids):
            raise ValueError("players must not have duplicate player_id")

        hero_count = sum(1 for p in players if p.is_hero)
        if hero_count > 1:
            raise ValueError("at most one player may have is_hero=True")

        # money
        for name in ("pot", "current_bet", "to_call"):
            if not isinstance(getattr(self, name), ChipAmount):
                raise TypeError(f"{name} must be a ChipAmount")

        # actor
        if self.actor is not None:
            if not isinstance(self.actor, int) or isinstance(self.actor, bool):
                raise TypeError("actor must be an int or None")
            if self.actor not in seats:
                raise ValueError("actor must reference an existing seat")


# ---------------------------------------------------------------------------
# StateContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateContext:
    """Read-only context prepared by the Orchestrator for the State Engine.

    The State Engine does NOT query a database. This object carries the
    read-only inputs it needs. All containers are deep-immutable.
    """

    previous_state: PokerState | None = None
    platform_rules: Mapping[str, Any] = field(default_factory=dict)
    confidence_thresholds: Mapping[str, float] = field(default_factory=dict)
    recent_events: tuple[StateEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.previous_state is not None and not isinstance(
            self.previous_state, PokerState
        ):
            raise TypeError("previous_state must be a PokerState or None")
        object.__setattr__(
            self, "platform_rules", freeze_mapping(self.platform_rules)
        )
        object.__setattr__(
            self,
            "confidence_thresholds",
            freeze_mapping(self.confidence_thresholds),
        )
        events = tuple(self.recent_events)
        if not all(isinstance(e, StateEvent) for e in events):
            raise TypeError("recent_events must be StateEvent instances")
        object.__setattr__(self, "recent_events", events)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a candidate PokerState.

    A plain result object; does NOT implement a validation engine.
    """

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.is_valid, bool):
            raise TypeError("is_valid must be a bool")
        errors = tuple(self.errors)
        warnings = tuple(self.warnings)
        if not all(isinstance(e, str) for e in errors):
            raise TypeError("errors items must be str")
        if not all(isinstance(w, str) for w in warnings):
            raise TypeError("warnings items must be str")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)


__all__ = ["PokerState", "StateContext", "ValidationResult"]
