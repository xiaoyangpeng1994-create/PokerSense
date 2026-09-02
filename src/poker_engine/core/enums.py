"""Core enumerations for Poker Intelligence Engine.

All enums are fixed domain vocabulary. They are immutable by nature and
shared across the entire domain layer.

Task 1B scope: Rank, Suit, Street, ActionType, PlayerStatus, Position.
"""

from __future__ import annotations

from enum import Enum


class Rank(str, Enum):
    """Card rank, 2 through Ace."""

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    @property
    def value_int(self) -> int:
        """Numeric rank value (2..14) for range/hand-strength math."""
        return _RANK_VALUE[self]


_RANK_VALUE: dict[Rank, int] = {
    Rank.TWO: 2,
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 7,
    Rank.EIGHT: 8,
    Rank.NINE: 9,
    Rank.TEN: 10,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
}


class Suit(str, Enum):
    """Card suit."""

    CLUBS = "c"      # ♣
    DIAMONDS = "d"   # ♦
    HEARTS = "h"     # ♥
    SPADES = "s"     # ♠


class Street(str, Enum):
    """Betting street / stage of a hand."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(str, Enum):
    """Player action types."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    # Blind / forced actions
    POST_SB = "post_sb"
    POST_BB = "post_bb"
    POST_ANTE = "post_ante"


class PlayerStatus(str, Enum):
    """Player status in the current hand."""

    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    SITTING_OUT = "sitting_out"
    UNKNOWN = "unknown"


class Position(str, Enum):
    """Seat position relative to the dealer button.

    Covers both 6-max and 9-max without hardcoding table size:
    - 6-max uses SB/BB/UTG/HJ/CO/BTN.
    - 9-max uses the full set including UTG1, UTG2, LJ.
    """

    SB = "SB"
    BB = "BB"
    UTG = "UTG"
    UTG1 = "UTG1"
    UTG2 = "UTG2"
    LJ = "LJ"
    HJ = "HJ"
    CO = "CO"
    BTN = "BTN"
    UNKNOWN = "UNKNOWN"
