"""Tests for core value objects (Task 1B): Card, ChipAmount, ChipDelta."""

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.value_objects import Card, ChipAmount, ChipDelta


# ---------- Card (unchanged) ----------

def test_card_str():
    assert str(Card(Rank.ACE, Suit.SPADES)) == "As"
    assert str(Card(Rank.TEN, Suit.DIAMONDS)) == "Td"


def test_card_hashable_and_comparable():
    a = Card(Rank.ACE, Suit.SPADES)
    b = Card(Rank.ACE, Suit.SPADES)
    c = Card(Rank.KING, Suit.SPADES)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert a > c  # ace > king


def test_card_frozen():
    a = Card(Rank.ACE, Suit.SPADES)
    with pytest.raises(FrozenInstanceError):
        a.rank = Rank.KING  # type: ignore[misc]


def test_card_roundtrip():
    a = Card(Rank.QUEEN, Suit.HEARTS)
    d = a.to_dict()
    assert Card.from_dict(d) == a


# ---------- ChipAmount ----------

def test_amount_strictly_non_negative():
    with pytest.raises(InvalidStateError):
        ChipAmount("-1")
    with pytest.raises(InvalidStateError):
        ChipAmount(Decimal("-0.01"))


def test_amount_zero_ok():
    assert ChipAmount("0").value == Decimal("0")


def test_amount_comparison():
    assert ChipAmount("10") > ChipAmount("5")
    assert ChipAmount("5") == ChipAmount("5.00")


# === (A) instance immutability ===

def test_amount_instance_immutable():
    x = ChipAmount("100")
    with pytest.raises(FrozenInstanceError):
        x._value = Decimal("200")  # type: ignore[misc]


# === (B) delta instance immutability ===

def test_delta_instance_immutable():
    d = ChipDelta("100")
    with pytest.raises(FrozenInstanceError):
        d._value = Decimal("200")  # type: ignore[misc]


# === (C) cross-type inequality ===

def test_amount_not_equal_delta():
    assert ChipAmount("10") != ChipDelta("10")


# === (D) equal objects have equal hash ===

def test_equal_objects_equal_hash():
    a = ChipAmount("10")
    b = ChipAmount("10.00")
    assert a == b
    assert hash(a) == hash(b)

    d1 = ChipDelta("-5")
    d2 = ChipDelta("-5.00")
    assert d1 == d2
    assert hash(d1) == hash(d2)


# === (E) float construction rejected ===

def test_float_construction_rejected():
    with pytest.raises(TypeError):
        ChipAmount(0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ChipDelta(0.1)  # type: ignore[arg-type]


# === (F) NaN / Infinity rejected ===

def test_nan_infinity_rejected():
    with pytest.raises(ValueError):
        ChipAmount(Decimal("NaN"))
    with pytest.raises(ValueError):
        ChipAmount(Decimal("Infinity"))
    with pytest.raises(ValueError):
        ChipDelta(Decimal("-Infinity"))


# === (G) amount - amount == delta ===

def test_amount_sub_returns_delta():
    assert ChipAmount("100") - ChipAmount("120") == ChipDelta("-20")


# === (H) amount.add_delta(negative) == amount ===

def test_amount_add_delta_ok():
    assert ChipAmount("100").add_delta(ChipDelta("-20")) == ChipAmount("80")


# === (I) amount.add_delta underflow raises InvalidStateError ===

def test_amount_add_delta_negative_raises():
    with pytest.raises(InvalidStateError):
        ChipAmount("100").add_delta(ChipDelta("-120"))


# === (J) undefined cross-type arithmetic raises TypeError ===
# Note: ChipAmount + ChipDelta is a DEFINED operation (returns ChipAmount),
# so it is NOT a cross-type error. Only truly undefined ops raise TypeError.

def test_undefined_cross_type_raises_typeerror():
    # amount - delta is undefined (only amount - amount is defined)
    with pytest.raises(TypeError):
        ChipAmount("10") - ChipDelta("5")  # type: ignore[operator]
    # delta + amount is undefined (only delta + delta is defined)
    with pytest.raises(TypeError):
        ChipDelta("10") + ChipAmount("5")  # type: ignore[operator]
    # delta - amount is undefined
    with pytest.raises(TypeError):
        ChipDelta("10") - ChipAmount("5")  # type: ignore[operator]
    # add_delta with a non-ChipDelta argument
    with pytest.raises(TypeError):
        ChipAmount("10").add_delta(ChipAmount("5"))  # type: ignore[arg-type]


# ---------- arithmetic ----------

def test_amount_add_amount():
    assert ChipAmount("10") + ChipAmount("5") == ChipAmount("15")


def test_amount_add_delta_via_operator():
    # amount + delta -> amount (defined contract)
    assert ChipAmount("100") + ChipDelta("-20") == ChipAmount("80")
    assert ChipAmount("100") + ChipDelta("20") == ChipAmount("120")


def test_amount_add_delta_operator_underflow_raises():
    with pytest.raises(InvalidStateError):
        ChipAmount("100") + ChipDelta("-120")


def test_amount_neg_returns_delta():
    result = -ChipAmount("5")
    assert isinstance(result, ChipDelta)
    assert result.value == Decimal("-5")


def test_delta_can_be_negative():
    d = ChipDelta("-7")
    assert d.value == Decimal("-7")


def test_delta_arithmetic():
    assert ChipDelta("3") + ChipDelta("-1") == ChipDelta("2")
    assert ChipDelta("3") - ChipDelta("5") == ChipDelta("-2")
    assert -ChipDelta("4") == ChipDelta("-4")


# ---------- Exactness (no silent quantization) ----------

def test_exact_value_preserved():
    assert ChipAmount("2.345").value == Decimal("2.345")
    assert ChipDelta("2.345").value == Decimal("2.345")


def test_float_precision_trap_avoided():
    # 0.1 + 0.2 != 0.3 in float; must be exact in ChipAmount.
    a = ChipAmount("0.1")
    b = ChipAmount("0.2")
    assert (a + b) == ChipAmount("0.3")


def test_many_small_amounts_reconcile_exactly():
    total = ChipAmount("0")
    for _ in range(1000):
        total = total + ChipAmount("0.01")
    assert total == ChipAmount("10.00")


# ---------- Serialization ----------

def test_amount_roundtrip():
    a = ChipAmount("123.45")
    assert ChipAmount.from_dict(a.to_dict()) == a


def test_amount_exact_roundtrip():
    a = ChipAmount("2.345")
    assert ChipAmount.from_dict(a.to_dict()) == a


def test_delta_roundtrip():
    d = ChipDelta("-42.00")
    assert ChipDelta.from_dict(d.to_dict()) == d


def test_serialization_is_not_float():
    d = ChipAmount("0.1").to_dict()
    assert d == {"value": "0.1"}
    assert isinstance(d["value"], str)
