"""Optional independent OSS parity; base installations keep Python fallback."""

import random

import pytest

from poker_engine.equity._deck import full_deck
from poker_engine.equity.backend import select_evaluator
from poker_engine.equity.evaluator import evaluate


def test_phevaluator_orders_random_hands_like_reference():
    pytest.importorskip("phevaluator")
    fast, name = select_evaluator("phevaluator")
    assert name == "phevaluator"
    rng = random.Random(6809)
    deck = full_deck()
    for count in (5, 6, 7):
        for _ in range(1000):
            left, right = rng.sample(deck, count), rng.sample(deck, count)
            a, b = evaluate(left), evaluate(right)
            assert (fast(left) > fast(right)) == (a > b)
            assert (fast(left) == fast(right)) == (a == b)


def test_python_backend_remains_available_without_external_libraries():
    evaluator, name = select_evaluator("python")
    assert name == "python-direct"
    cards = full_deck()[:7]
    assert evaluator(cards) == evaluate(cards)
