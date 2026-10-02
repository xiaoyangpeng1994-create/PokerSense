"""Small mechanism checks; the complete native audit is a separate one-shot job."""
from copy import deepcopy
from fractions import Fraction
import time

import pytest

from tools.aa_turn_river_game import (
    B2, B4, BOARD, C, DEALS, HOLDINGS, STANDARD_DECK, audit_native, compose_deck,
    infoset_id, joint_rivers, leaf_ledger, make_root, public_tree, research_key,
)


@pytest.mark.parametrize("deal", DEALS)
def test_complete_conditional_river_support_and_two_full_decks(deal):
    support = joint_rivers(deal)
    blocked = set(BOARD + HOLDINGS[1][deal[0]] + HOLDINGS[2][deal[1]])
    assert len(support) == len(set(support)) == 44
    assert set(support) == set(STANDARD_DECK) - blocked
    assert sum((Fraction(1, 176) for _ in support), Fraction(0)) == Fraction(1, 4)
    for card in support:
        first, second = (compose_deck(deal, card, mode) for mode in (0, 1))
        assert len(first) == len(second) == 52
        assert set(first) == set(second) == set(STANDARD_DECK)
        assert first != second and first[19] == second[19] == card
        assert first[2:6] == second[2:6]
        assert first[13:16] == second[13:16] == list(BOARD[:3])
        assert first[17] == second[17] == BOARD[3]
        assert first[:2] != second[:2] and first[12] != second[12]


def test_native_hiding_memory_and_cross_street_menu():
    deal = DEALS[0]
    card = joint_rivers(deal)[0]
    first, second = (make_root(deal, card, mode) for mode in (0, 1))
    assert first.observe(1) == second.observe(1)
    assert research_key(first.observe(1)) == research_key(second.observe(1))
    assert tuple(a.id for a in first.legal_actions()) == (C, B2, B4)
    for arena in (first, second):
        arena.step(B2).step(C)
    assert first.street == second.street == "river"
    assert tuple(a.id for a in first.legal_actions()) == (C, B2)
    assert first.observe(1) == second.observe(1)
    assert infoset_id(first.observe(1)) == infoset_id(second.observe(1))
    assert research_key(first.observe(1)) == research_key(second.observe(1))


def test_future_river_is_not_turn_information():
    deal = DEALS[0]
    first = make_root(deal, joint_rivers(deal)[0])
    second = make_root(deal, joint_rivers(deal)[-1])
    assert first.observe(1) == second.observe(1)
    assert research_key(first.observe(1)) == research_key(second.observe(1))
    for arena in (first, second):
        arena.step(C).step(C)
    assert first.observe(1)["board"] != second.observe(1)["board"]
    assert research_key(first.observe(1)) != research_key(second.observe(1))


def test_bet_three_is_native_legal_but_rejected_by_frozen_abstraction():
    arena = make_root(DEALS[0], joint_rivers(DEALS[0])[0])
    before = deepcopy(arena.observe(arena.actor))
    native = arena.arena.clone()
    native.step("raise_to:3")
    assert native.observe(native.actor)["to_call"] == "3"
    with pytest.raises(ValueError, match="outside_frozen_action_abstraction"):
        arena.step("raise_to:3")
    assert arena.observe(arena.actor) == before


@pytest.mark.parametrize("mutation", ("future", "memory", "range", "board"))
def test_visible_key_rejects_invalid_scope_and_memory(mutation):
    arena = make_root(DEALS[0], joint_rivers(DEALS[0])[0])
    obs = arena.observe(arena.actor)
    if mutation == "future":
        obs["next_board"] = "Ac"
    elif mutation == "memory":
        obs["own_memory"].pop()
    elif mutation == "range":
        obs["own_hole"] = ["4c", "4d"]
    else:
        obs["board"][0] = "2c"
    with pytest.raises(ValueError):
        research_key(obs)


def test_independent_fold_ledger_and_expired_audit_retains_denominator():
    raised = dict(public_tree().children)[B4]
    folded = dict(raised.children)["fold"]
    ledger = leaf_ledger(DEALS[0], folded)
    assert ledger["settled_pot"] == 21 and ledger["rake"] == 0
    assert ledger["refunds"][1] == 4
    assert ledger["returns"][1] == 15 and ledger["returns"][2] == -6
    progress = {}
    with pytest.raises(TimeoutError):
        audit_native(deadline=time.monotonic() - 1, progress=progress)
    assert progress["status"] == "STOP_ERROR_OR_BUDGET"
    assert progress["terminal_not_run"] == 9552
    assert progress["physical_terminal_checks"] == 0
