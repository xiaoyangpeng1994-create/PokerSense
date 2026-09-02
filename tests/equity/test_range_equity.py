"""Tests for opponent-range equity (Task 9 extension).

The key self-check: a SINGLE-holding range must equal the known-card estimator
for that exact opponent. Monte Carlo range estimation must converge (within a
small tolerance) to the exact range enumeration.
"""

from __future__ import annotations

import pytest

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.equity import (
    EnumerationEquity,
    MonteCarloEquity,
    enumeration_range_equity,
)


def C(s: str) -> Card:
    rank = Rank(s[0])
    suit = Suit(s[1].lower())
    return Card(rank=rank, suit=suit)


# ---------------------------------------------------------------------------
# degenerate singleton range == known-card enumerator (the anchor property)
# ---------------------------------------------------------------------------

def test_singleton_range_equals_exact_opponent():
    hero = (C("Ah"), C("Kd"))
    opp = (C("Qc"), C("Qs"))
    board = (C("Jh"), C("7d"), C("2c"))

    exact = EnumerationEquity().estimate(hero, (opp,), board)
    via_range = enumeration_range_equity(hero, ((opp,),), board)

    assert via_range.win == pytest.approx(exact.win, abs=1e-12)
    assert via_range.tie == pytest.approx(exact.tie, abs=1e-12)
    assert via_range.loss == pytest.approx(exact.loss, abs=1e-12)
    assert via_range.equity == pytest.approx(exact.equity, abs=1e-12)


def test_range_with_duplicated_holding_weights_it_twice():
    # A range where the same holding appears twice weights it double; a range
    # with one strong + one weak holding averages them.
    hero = (C("As"), C("Ad"))
    strong = (C("Kh"), C("Kd"))
    weak = (C("7s"), C("2h"))
    board = (C("Js"), C("7d"), C("2c"), C("9h"), C("3c"))

    # hero (AA) vs strong (KK): hero wins. vs weak (72): hero wins too on this
    # board (72 hits two pair 7s&2s... actually verify via exact enum instead).
    r = enumeration_range_equity(
        hero,
        (
            (
                strong,
                weak,
                strong,  # strong appears twice -> weighted 2:1 over weak
            ),
        ),
        board,
    )

    e_strong = EnumerationEquity().estimate(hero, (strong,), board)
    e_weak = EnumerationEquity().estimate(hero, (weak,), board)

    # weighted average: (2*strong + 1*weak) / 3
    expected_equity = (2 * e_strong.equity + e_weak.equity) / 3
    assert r.equity == pytest.approx(expected_equity, abs=1e-12)


# ---------------------------------------------------------------------------
# multi-opponent ranges
# ---------------------------------------------------------------------------

def test_two_opponent_ranges():
    hero = (C("Ah"), C("Ad"))
    r1 = ((C("Ks"), C("Kh")), (C("Qs"), C("Qh")))
    r2 = ((C("Js"), C("Jh")),)
    board = (C("2c"), C("7d"), C("Jc"))

    r = enumeration_range_equity(hero, (r1, r2), board)
    # sanity: probabilities sum to 1
    assert r.win + r.tie + r.loss == pytest.approx(1.0, abs=1e-9)
    assert r.samples > 0


# ---------------------------------------------------------------------------
# Monte Carlo range converges to exact range enumeration
# ---------------------------------------------------------------------------

def test_monte_carlo_range_matches_enumeration_within_tolerance():
    hero = (C("Ah"), C("Kd"))
    opp_range = ((C("Qc"), C("Qs")), (C("Jc"), C("Js")))
    board = (C("Th"), C("7d"), C("2c"))

    exact = enumeration_range_equity(hero, (opp_range,), board)
    mc = MonteCarloEquity(trials=20000, seed=7).estimate_range(
        hero, (opp_range,), board
    )

    assert mc.equity == pytest.approx(exact.equity, abs=0.01)


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------

def test_empty_range_rejected():
    with pytest.raises(ValueError):
        enumeration_range_equity((C("Ah"), C("Kd")), ((),), ())


def test_holding_with_same_card_rejected():
    with pytest.raises(ValueError):
        enumeration_range_equity(
            (C("Ah"), C("Kd")), (((C("As"), C("As")),),), ()
        )


def test_range_clashing_with_hero_board_is_skipped_not_error():
    # A range whose only holding clashes with hero's card cannot form a legal
    # deal; enumeration skips it (no crash), yielding samples=0.
    hero = (C("Ah"), C("Kd"))
    clashing = ((C("Ah"), C("Qs")),)  # Ah is also hero's card
    r = enumeration_range_equity(hero, (clashing,), ())
    assert r.samples == 0
    assert r.equity == 0.0
