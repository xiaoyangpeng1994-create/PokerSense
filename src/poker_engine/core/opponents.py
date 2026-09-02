"""Player contracts: PlayerState, OpponentProfile.

``PlayerState`` describes a single player's state within a single hand.
``OpponentProfile`` describes a player's aggregate behavior across hands.

CRITICAL: this module MUST NOT import from ``.state`` (which defines
PokerState) — ``opponents.py`` must not depend on ``PokerState``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ._freeze import _require_aware_dt
from .enums import PlayerStatus, Position
from .value_objects import ChipAmount


@dataclass(frozen=True)
class PlayerState:
    """State of one player in the current hand.

    - ``seat``: stable seat number (non-negative int), distinct from
      ``player_id`` (a stable identity across hands).
    - ``position``: position relative to the button (Position enum).
    - ``stack`` / ``committed_*``: ChipAmount (strictly non-negative).
    - ``last_action`` / ``last_action_amount`` are intentionally ABSENT:
      they are expressed by the StateEvent stream, not stored here.

    Invariants (checked in __post_init__):
    - seat is an int >= 0
    - committed_this_street <= committed_this_hand
    """

    player_id: str
    seat: int
    position: Position
    stack: ChipAmount
    committed_this_street: ChipAmount
    committed_this_hand: ChipAmount
    status: PlayerStatus
    has_cards: bool
    is_hero: bool
    is_dealer: bool

    def __post_init__(self) -> None:
        if not isinstance(self.player_id, str) or not self.player_id:
            raise ValueError("player_id must be a non-empty str")
        if not isinstance(self.seat, int) or isinstance(self.seat, bool):
            raise TypeError("seat must be an int")
        if self.seat < 0:
            raise ValueError(f"seat must be >= 0, got {self.seat}")
        if not isinstance(self.position, Position):
            raise TypeError("position must be a Position enum")
        if not isinstance(self.stack, ChipAmount):
            raise TypeError("stack must be a ChipAmount")
        if not isinstance(self.committed_this_street, ChipAmount):
            raise TypeError("committed_this_street must be a ChipAmount")
        if not isinstance(self.committed_this_hand, ChipAmount):
            raise TypeError("committed_this_hand must be a ChipAmount")
        if self.committed_this_street > self.committed_this_hand:
            raise ValueError(
                "committed_this_street cannot exceed committed_this_hand"
            )
        if not isinstance(self.status, PlayerStatus):
            raise TypeError("status must be a PlayerStatus enum")
        if not isinstance(self.has_cards, bool):
            raise TypeError("has_cards must be a bool")
        if not isinstance(self.is_hero, bool):
            raise TypeError("is_hero must be a bool")
        if not isinstance(self.is_dealer, bool):
            raise TypeError("is_dealer must be a bool")


@dataclass(frozen=True)
class OpponentProfile:
    """Aggregate behavioral statistics for a player across hands.

    All frequency fields are ratios (float in [0, 1]), NOT monetary amounts.
    """

    player_id: str
    vpip: float
    pfr: float
    af: float
    cbet_freq: float
    threebet_freq: float
    bluff_freq: float
    sample_size: int
    last_updated: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.player_id, str) or not self.player_id:
            raise ValueError("player_id must be a non-empty str")
        for name in (
            "vpip", "pfr", "af", "cbet_freq", "threebet_freq", "bluff_freq",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"{name} must be a float")
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            # af (aggression factor) is a ratio without a strict upper bound;
            # other frequencies are bounded in [0, 1].
            if name == "af":
                if val < 0:
                    raise ValueError(f"{name} must be >= 0, got {val}")
            else:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        if not isinstance(self.sample_size, int) or isinstance(
            self.sample_size, bool
        ):
            raise TypeError("sample_size must be an int")
        if self.sample_size < 0:
            raise ValueError("sample_size must be >= 0")
        if not isinstance(self.last_updated, datetime):
            raise TypeError("last_updated must be a datetime")
        _require_aware_dt(self.last_updated)


__all__ = ["PlayerState", "OpponentProfile"]
