"""Independent exact-reference parity for the realtime evaluator."""

import random

import pytest

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card
from poker_engine.equity._deck import full_deck
from poker_engine.equity.evaluator import evaluate
from poker_engine.equity.fast_evaluator import evaluate_fast


@pytest.mark.parametrize("count", (5, 6, 7))
def test_random_hands_match_exhaustive_five_card_subsets(count):
    rng = random.Random(608 + count)
    deck = full_deck()
    for _ in range(1500):
        cards = rng.sample(deck, count)
        assert evaluate_fast(cards) == evaluate(cards)


@pytest.mark.parametrize("codes", (
    "As 2s 3s 4s 5s Kd Qh",  # wheel straight flush
    "As Ah Ad Ks Kh Kd 2c",  # two trips: ace full of kings
    "As Ah Ks Kh Qs Qh Jd",  # three pairs: queen kicker
    "As Ah Ad Ac Ks Kh Qc",  # quads with paired kicker
    "As Js 9s 7s 5s 3s 2s",  # seven-card flush
    "As Kd Qh Jc Ts 9d 8c",  # competing straights
    "As 2d 3h 4c 5s 6d Kh",  # six-high outranks wheel
))
def test_adversarial_rank_patterns_match_reference(codes):
    cards = tuple(Card(Rank(code[0]), Suit(code[1])) for code in codes.split())
    assert evaluate_fast(cards) == evaluate(cards)


def test_duplicate_cards_are_rejected():
    cards = full_deck()[:4]
    with pytest.raises(ValueError, match="distinct"):
        evaluate_fast(cards + (cards[0],))
