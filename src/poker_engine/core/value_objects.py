"""Core value objects for Poker Intelligence Engine.

Task 1B scope: Card, ChipAmount (strictly non-negative), ChipDelta (signed).

Design constraints (from architecture v0.2.1 FROZEN):
- Deep immutability via @dataclass(frozen=True).
- Money/stack amounts MUST NOT use float — decimal.Decimal only.
- ChipAmount is STRICTLY non-negative; ChipDelta may be negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .enums import Rank, Suit
from .errors import InvalidStateError


@dataclass(frozen=True)
class Card:
    """A single playing card, e.g. Card(Rank.ACE, Suit.SPADES) == "As"."""

    rank: Rank
    suit: Suit

    def __post_init__(self) -> None:
        # Enforce enum types defensively (also aids typed construction).
        if not isinstance(self.rank, Rank):
            raise TypeError("rank must be a Rank")
        if not isinstance(self.suit, Suit):
            raise TypeError("suit must be a Suit")

    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"

    @property
    def rank_value(self) -> int:
        return self.rank.value_int

    # Ordering: first by rank, then by suit (for deterministic sorting).
    def __lt__(self, other: "Card") -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return (self.rank_value, self.suit.value) < (other.rank_value, other.suit.value)

    def to_dict(self) -> dict:
        return {"rank": self.rank.value, "suit": self.suit.value}

    @classmethod
    def from_dict(cls, data: dict) -> "Card":
        return cls(rank=Rank(data["rank"]), suit=Suit(data["suit"]))


_ALLOWED_MONEY_TYPES = (Decimal, str, int)


def _coerce_decimal(value: Decimal | str | int) -> Decimal:
    """Coerce a construction argument to an exact, finite Decimal.

    Allowed input: Decimal, str, int.
    Rejected input: float, Decimal('NaN'/'Infinity'/'-Infinity'), bool.

    The Core layer performs NO rounding/quantization: the input value is
    preserved exactly. Platform minimum chip unit is validated later by
    Platform Config / State Validation, never silently applied here.
    """
    if isinstance(value, bool):
        raise TypeError("money value must not be a bool")
    if isinstance(value, float):
        raise TypeError(
            "money value must not be a float (use str/Decimal/int)"
        )
    if not isinstance(value, _ALLOWED_MONEY_TYPES):
        raise TypeError(
            f"money value must be Decimal, str, or int; "
            f"got {type(value).__name__}"
        )
    d = value if isinstance(value, Decimal) else Decimal(value)
    if not d.is_finite():
        raise ValueError(
            f"money value must be a finite Decimal; got {d!r}"
        )
    return d


@dataclass(frozen=True, slots=True)
class _MoneyBase:
    """Shared decimal-backed money value object.

    Immutability: ``frozen=True`` + ``slots=True`` makes ``_value`` read-only
    via ``FrozenInstanceError`` on any assignment attempt. Construction is
    deferred to subclasses which validate sign semantics.
    """

    _value: Decimal

    def __post_init__(self) -> None:
        # Safety net: the stored value must be a finite Decimal (already
        # coerced by subclasses). No rounding/quantization is applied here.
        if not isinstance(self._value, Decimal) or not self._value.is_finite():
            raise ValueError("money value must be a finite Decimal")

    @property
    def value(self) -> Decimal:
        return self._value

    # --- equality / hash: same-type only ---
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value == other._value  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self), self._value))

    # --- total ordering within the same type ---
    def __lt__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value < other._value  # type: ignore[attr-defined]

    def __le__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value <= other._value  # type: ignore[attr-defined]

    def __gt__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value > other._value  # type: ignore[attr-defined]

    def __ge__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._value >= other._value  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"

    def to_dict(self) -> dict:
        """Serialize as a string to avoid float entirely."""
        return {"value": str(self._value)}


@dataclass(frozen=True, slots=True)
class ChipAmount(_MoneyBase):
    """Strictly non-negative money value object.

    Represents a stack, pot, bet, or any amount that must be >= 0.
    Constructing with a negative value raises InvalidStateError.
    """

    _value: Decimal

    def __init__(
        self, value: Decimal | str | int
    ) -> None:
        d = _coerce_decimal(value)
        if d < 0:
            raise InvalidStateError(
                f"ChipAmount must be non-negative, got {d}"
            )
        object.__setattr__(self, "_value", d)

    # --- arithmetic ---
    # amount + amount -> amount
    # amount + delta  -> amount (raises InvalidStateError if result < 0)
    # amount - amount -> delta
    def __add__(self, other: "ChipAmount | ChipDelta") -> "ChipAmount":
        if type(other) is ChipAmount:
            return ChipAmount(self._value + other._value)
        if type(other) is ChipDelta:
            return self._add_delta_checked(other)
        return NotImplemented

    def __sub__(self, other: "ChipAmount") -> "ChipDelta":
        if type(other) is not ChipAmount:
            return NotImplemented
        return ChipDelta(self._value - other._value)

    # No __radd__: ``ChipDelta + ChipAmount`` is intentionally undefined.
    # ``ChipAmount + ChipDelta`` is handled by __add__ above; without
    # __radd__, the reflected ``delta + amount`` raises TypeError.

    def __neg__(self) -> "ChipDelta":
        return ChipDelta(-self._value)

    def _add_delta_checked(self, delta: "ChipDelta") -> "ChipAmount":
        """Shared implementation for amount + delta with non-negativity guard."""
        result = self._value + delta._value
        if result < 0:
            raise InvalidStateError(
                f"ChipAmount + ChipDelta would be negative: "
                f"{self._value} + {delta._value} = {result}"
            )
        return ChipAmount(result)

    def add_delta(self, delta: "ChipDelta") -> "ChipAmount":
        """Explicit API; semantics identical to ``ChipAmount + ChipDelta``.

        Raises InvalidStateError if the result would be negative.
        Raises TypeError if ``delta`` is not a ChipDelta.
        """
        if type(delta) is not ChipDelta:
            raise TypeError(
                f"add_delta requires a ChipDelta, got {type(delta).__name__}"
            )
        return self._add_delta_checked(delta)

    @classmethod
    def from_dict(cls, data: dict) -> "ChipAmount":
        return cls(data["value"])

    @classmethod
    def zero(cls) -> "ChipAmount":
        return cls("0")


@dataclass(frozen=True, slots=True)
class ChipDelta(_MoneyBase):
    """Signed money value object (may be negative).

    Represents a change/net result: net_result, profit_loss, chip_change,
    or the difference between two ChipAmount values.
    """

    _value: Decimal

    def __init__(self, value: Decimal | str | int) -> None:
        object.__setattr__(self, "_value", _coerce_decimal(value))

    def __add__(self, other: "ChipDelta") -> "ChipDelta":
        if type(other) is not ChipDelta:
            return NotImplemented
        return ChipDelta(self._value + other._value)

    def __sub__(self, other: "ChipDelta") -> "ChipDelta":
        if type(other) is not ChipDelta:
            return NotImplemented
        return ChipDelta(self._value - other._value)

    # No __radd__: ``ChipDelta + ChipAmount`` is intentionally undefined.

    def __neg__(self) -> "ChipDelta":
        return ChipDelta(-self._value)

    @classmethod
    def from_dict(cls, data: dict) -> "ChipDelta":
        return cls(data["value"])


__all__ = [
    "Card",
    "ChipAmount",
    "ChipDelta",
]
