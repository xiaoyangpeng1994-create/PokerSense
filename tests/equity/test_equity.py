"""Tests for the Equity Engine (Task 9) — evaluator, enumeration, Monte Carlo,
and pot odds."""

from __future__ import annotations

import pytest

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity import (
    EnumerationEquity,
    MonteCarloEquity,
    equity_call_is_profitable,
    evaluate,
    pot_odds,
)


def C(s: str) -> Card:
    """Build a Card from a 2-char string like 'As', 'Td', '2c'."""
    rank = Rank(s[0])
    suit = Suit(s[1].lower())
    return Card(rank=rank, suit=suit)


# ---------------------------------------------------------------------------
# Hand evaluator
# ---------------------------------------------------------------------------

def test_hand_ranking_categories():
    # straight flush > four of a kind > full house > flush > straight
    straight_flush = [C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts")]
    four_kind = [C("As"), C("Ah"), C("Ad"), C("Ac"), C("Kd")]
    full_house = [C("As"), C("Ah"), C("Ad"), C("Kc"), C("Kd")]
    flush = [C("Ah"), C("Kh"), C("Qh"), C("Jh"), C("9h")]
    straight = [C("9s"), C("8h"), C("7d"), C("6c"), C("5s")]
    high_card = [C("As"), C("Kh"), C("Qd"), C("Jc"), C("9s")]

    strengths = [
        evaluate(h)
        for h in (straight_flush, four_kind, full_house, flush, straight,
                  high_card)
    ]
    # strictly decreasing by hand strength
    for a, b in zip(strengths, strengths[1:]):
        assert a > b


def test_wheel_straight():
    # A-2-3-4-5 is a straight with high card 5
    wheel = [C("As"), C("2h"), C("3d"), C("4c"), C("5s")]
    # wheel ties another wheel of different suits
    wheel2 = [C("Ad"), C("2c"), C("3s"), C("4h"), C("5d")]
    assert evaluate(wheel) == evaluate(wheel2)


def test_evaluate_7_cards_picks_best_5():
    # best 5 of 7 is a flush even though also possibility of a pair etc.
    hand = [C("Ah"), C("Kh"), C("2c"), C("5h"), C("9h"), C("Jh"), C("3d")]
    s = evaluate(hand)
    # flush category is 5
    assert s[0] == 5


def test_evaluate_rejects_wrong_card_count():
    with pytest.raises(ValueError):
        evaluate([C("As"), C("Kh")])  # too few


# ---------------------------------------------------------------------------
# Enumeration (exact) equity
# ---------------------------------------------------------------------------

def test_enumeration_heads_up_identical_hands_tie_on_river():
    # AA (same suits) vs AA (other suits) with a full board already dealt:
    # hero and opponent hold identical ranks, so they MUST tie on every board.
    hero = (C("As"), C("Ah"))
    opp = (C("Ad"), C("Ac"))
    board = (C("2h"), C("7c"), C("9d"), C("Js"), C("Qh"))
    res = EnumerationEquity().estimate(hero=hero, opponents=(opp,), board=board)
    assert res.win == 0.0
    assert res.tie == 1.0
    assert res.equity == pytest.approx(0.5, abs=1e-9)


def test_enumeration_nutted_hero_wins_on_river():
    # Hero has a royal flush already on the river; opponent cannot beat it.
    hero = (C("As"), C("Ks"))
    opp = (C("2d"), C("2c"))
    board = (C("Qs"), C("Js"), C("Ts"), C("5h"), C("3d"))
    res = EnumerationEquity().estimate(hero=hero, opponents=(opp,), board=board)
    assert res.win == 1.0
    assert res.loss == 0.0
    assert res.equity == 1.0


# ---------------------------------------------------------------------------
# Monte Carlo vs enumeration (must converge)
# ---------------------------------------------------------------------------

def test_monte_carlo_matches_enumeration():
    hero = (C("As"), C("Kh"))
    opp = (C("2d"), C("2c"))
    # flop board (3 cards) -> enumeration is fast and exact
    board = (C("7s"), C("8h"), C("9d"))
    exact = EnumerationEquity().estimate(hero=hero, opponents=(opp,), board=board)
    mc = MonteCarloEquity(trials=50000, seed=1).estimate(
        hero=hero, opponents=(opp,), board=board
    )
    assert mc.equity == pytest.approx(exact.equity, abs=0.01)


# ---------------------------------------------------------------------------
# Pot odds
# ---------------------------------------------------------------------------

def test_pot_odds_basic():
    # pot 100, call 50 -> required equity = 50 / 150 = 1/3
    odds = pot_odds(pot=ChipAmount("100"), call=ChipAmount("50"))
    assert odds.required_equity == pytest.approx(1 / 3, abs=1e-9)


def test_pot_odds_free_check():
    odds = pot_odds(pot=ChipAmount("100"), call=ChipAmount("0"))
    assert odds.required_equity == 0.0


def test_equity_call_profitable():
    odds = pot_odds(pot=ChipAmount("100"), call=ChipAmount("50"))
    assert equity_call_is_profitable(0.4, odds) is True    # 40% > 33%
    assert equity_call_is_profitable(0.3, odds) is False   # 30% < 33%


# ---------------------------------------------------------------------------
# multi-way tie splitting (2-9 players)
# ---------------------------------------------------------------------------

def test_three_way_tie_split_is_half_not_third():
    # On a fixed 5-card board, hero ties with ONE opponent (split 1/2) while the
    # third opponent loses. The hero's equity must be 1/2, NOT the buggy
    # tie/n_players = 1/3 (which assumed a tie splits across the whole table).
    board = (C("As"), C("Ah"), C("Kd"), C("2c"), C("3d"))
    hero = (C("2d"), C("3c"))
    opp_tie = (C("2h"), C("3h"))
    opp_lose = (C("2s"), C("4c"))

    # sanity: hero ties opp_tie, beats opp_lose
    from poker_engine.equity.evaluator import compare
    assert compare(list(hero) + list(board), list(opp_tie) + list(board)) == 0
    assert compare(list(hero) + list(board), list(opp_lose) + list(board)) == 1

    res = EnumerationEquity().estimate(hero, (opp_tie, opp_lose), board)
    # only one possible outcome (board complete): a tie between hero & opp_tie.
    assert res.tie == 1.0
    assert res.win == 0.0
    assert res.loss == 0.0
    assert res.equity == pytest.approx(0.5, abs=1e-12)


def test_monte_carlo_matches_enumeration_multiway_tie():
    board = (C("As"), C("Ah"), C("Kd"), C("2c"), C("3d"))
    hero = (C("2d"), C("3c"))
    opp_tie = (C("2h"), C("3h"))
    opp_lose = (C("2s"), C("4c"))
    exact = EnumerationEquity().estimate(hero, (opp_tie, opp_lose), board)
    mc = MonteCarloEquity(trials=10000, seed=1).estimate(
        hero, (opp_tie, opp_lose), board
    )
    assert mc.equity == pytest.approx(exact.equity, abs=0.01)
