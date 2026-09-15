from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy import threeway_policy_evaluation_v1 as module
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    WorldResponseOverride, _Evaluation, _override_probabilities, _world_branches,
    compile_policy_book, evaluate_policy_book, policy_book_hash, public_conditions_json,
)
from poker_engine.strategy.threeway_river_v1 import RiverAction, _Node, _Tree
from .helpers import card
from .test_terminal_multiway_v1 import ranges, rules
from .test_threeway_river_v1 import fixture, model


def facing_bet():
    s = fixture()
    return replace(s, ranges=ranges({1: "ThTd", 2: "3c3d"}),
                   models=(model(1, bet=1, fold=1), model(2, check=1, fold=1)),
                   history=(RiverAction(0, "check"),
                            RiverAction(1, "bet", Decimal(100)),
                            RiverAction(2, "fold")))


def metric(result, name="frozen_policy"):
    assert result.status == "COMPLETE_CONDITIONAL_FIXED_POLICY", result.reasons
    return next(m for m in result.metrics if m.name == name)


def test_misspecified_world_keeps_old_call_without_reoptimizing(monkeypatch):
    s = facing_bet()
    book = compile_policy_book(s)
    assert book.decisions[0].action.kind == "call"
    assert book.planning_best_ev == 160
    before = policy_book_hash(book)
    world = replace(s, ranges=ranges({1: "JhJd", 2: "3c3d"}))

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation attempted to optimize Hero")

    monkeypatch.setattr(module, "analyze_threeway_river", forbidden)
    monkeypatch.setattr(_Tree, "value", forbidden)
    result = evaluate_policy_book(book, world)
    assert metric(result).net_ev_chips == -100  # Reoptimization would fold for 0.
    assert metric(result).conditional_net_ev_bb == -50
    assert metric(result, "check_fold").net_ev_chips == 0
    assert result.delta_vs_check_fold_chips == -100
    assert metric(result).probability_of_any_fallback == 0
    assert result.policy_hash_before == result.policy_hash_after == before
    assert policy_book_hash(book) == before


def test_unseen_public_history_uses_predeclared_fallback_with_path_probability():
    s = replace(fixture(), ranges=ranges({1: "ThTd", 2: "3c3d"}))
    book = compile_policy_book(s)
    # Planner saw only checks; world now bets half the time after Hero's check.
    world = replace(s, models=(model(1, check=1, bet=1, fold=1), s.models[1]))
    result = evaluate_policy_book(book, world)
    outcome = metric(result)
    assert outcome.net_ev_chips == 30  # Half win 60, half fallback fold 0.
    assert outcome.probability_of_any_fallback == Fraction(1, 2)
    assert outcome.expected_fallback_count == Fraction(1, 2)
    assert len(outcome.unsupported_histories) == 1
    missing = outcome.unsupported_histories[0]
    assert missing.fallback_action.kind == "fold"
    assert missing.reach_probability == Fraction(1, 2)
    assert tuple(a.kind for a in missing.history) == ("check", "bet", "fold")


@pytest.mark.parametrize("n", [6, 7, 8])
def test_shared_kernel_returns_and_rake_keep_conditional_units(n):
    s = replace(fixture(n), hero_cards=(card("5d"), card("6d")),
                ranges=ranges({1: "2d3d", 2: "4c5c"}),
                board_cards=tuple(map(card, ("As", "Ks", "Qs", "Js", "Ts"))),
                rules=rules(n, rake_percent="0.1"))
    book = compile_policy_book(s)
    result = evaluate_policy_book(book, s)
    assert metric(result).net_ev_chips == 54  # Bet uncalled100 returns; rake on pot60.
    assert metric(result, "check_call").net_ev_chips == 18
    assert metric(result).conditional_net_ev_bb == 27
    assert result.delta_vs_check_call_bb == 18
    assert result.strategy_eligible is result.advice_emitted is False


@pytest.mark.parametrize("mutation", [
    "cards", "grid", "order", "rules", "history", "allin", "sidepot", "book_hash",
    "budget",
])
def test_public_changes_and_bad_books_never_translate_or_partial_return(mutation):
    s = fixture()
    book = compile_policy_book(s)
    kwargs = {}
    if mutation == "cards":
        s = replace(s, hero_cards=(card("5c"), card("6c")))
    elif mutation == "grid":
        s = replace(s, aggression_targets=(Decimal(20),))
    elif mutation == "order":
        s = replace(s, action_order=(1, 2, 0))
    elif mutation == "rules":
        s = replace(s, rules=rules(rake_percent="0.1"))
    elif mutation == "history":
        s = replace(s, history=(RiverAction(0, "check"),))
    elif mutation in ("allin", "sidepot"):
        changes = ({"stack": ChipAmount("100")} if mutation == "allin"
                   else {"hand_committed": ChipAmount("40")})
        p = replace(s.seats[0], **changes)
        s = replace(s, seats=(p, *s.seats[1:]))
    elif mutation == "book_hash":
        book = replace(book, planning_best_ev=Fraction(999))
    else:
        kwargs["max_nodes"] = 1
    result = evaluate_policy_book(book, s, **kwargs)
    assert result.status == "BLOCKED"
    assert result.metrics == () and result.delta_vs_check_call_chips is None


def test_compile_rejects_scope_failure_and_world_card_collisions():
    s = fixture()
    with pytest.raises(ValueError, match="planning_BLOCKED"):
        compile_policy_book(replace(s, range_start="post_history"))
    book = compile_policy_book(s)
    world = replace(s, ranges=ranges({1: "QcKd", 2: "3c3d"}))
    result = evaluate_policy_book(book, world)
    assert result.status == "BLOCKED"


def test_rank_and_public_history_extensions_distinguish_same_pair_category():
    override = WorldResponseOverride(1, 12, (("call", Fraction(3)),),
                                     (("fold", Fraction(2)),))
    legal = (RiverAction(1, "fold"), RiverAction(1, "call"))
    base = (Fraction(1, 2), Fraction(1, 2))
    checked = (RiverAction(0, "check"), RiverAction(2, "bet", Decimal(100)))
    kk = _override_probabilities(override, (1, 13, 11, 9, 7), checked, 0, legal, base)
    tt = _override_probabilities(override, (1, 10, 11, 9, 7), checked, 0, legal, base)
    assert kk == (Fraction(2, 5), Fraction(3, 5))
    assert tt == (Fraction(2, 3), Fraction(1, 3))
    hero_bet = (RiverAction(0, "bet", Decimal(100)),)
    assert _override_probabilities(override, (1, 13), hero_bet, 0, legal, base) == (
        Fraction(1, 4), Fraction(3, 4))


def test_world_override_uses_own_cards_and_public_history_not_hero_cards():
    s = replace(fixture(), models=(
        model(1, fold=1, call=1, check=1), fixture().models[1]))
    trees = [_Tree(s, 128, 20000), _Tree(replace(
        s, hero_cards=(card("5c"), card("6c"))), 128, 20000)]
    override = WorldResponseOverride(1, 12, (("call", Fraction(3)),))
    outcomes = []
    for tree in trees:
        node = _Node(tree.active, s.action_order, tuple(Decimal(0) for _ in s.seats),
                     Decimal(0), Decimal(2), 0, ())
        node = tree.advance(node, RiverAction(0, "bet", Decimal(100)))
        outcomes.append(_world_branches(tree, node, tree.prior, {1: override}))
    assert outcomes[0] == outcomes[1]


def test_extended_world_preserves_book_hash_and_updates_family_evidence():
    s = replace(fixture(), ranges=ranges({1: "ThTd", 2: "3c3d"}),
                models=(model(1, check=1, bet=1, fold=1), fixture().models[1]))
    book = compile_policy_book(s)
    override = WorldResponseOverride(
        1, after_hero_check_multipliers=(("bet", Fraction(3)),))
    ordinary = evaluate_policy_book(book, s)
    extended = evaluate_policy_book(book, s, world_overrides=(override,))
    assert metric(extended)
    assert extended.response_family == "rank_history_extended"
    assert ordinary.world_scenario_sha256 != extended.world_scenario_sha256
    assert extended.policy_hash_before == extended.policy_hash_after == book.book_sha256


@pytest.mark.parametrize("override", [
    WorldResponseOverride(0), WorldResponseOverride(1, True),
    WorldResponseOverride(1, 15),
    WorldResponseOverride(1, None, (("call", Fraction(1)),)),
    WorldResponseOverride(1, 12, (("call", Fraction(-1)),)),
])
def test_invalid_or_hero_world_override_is_blocked(override):
    s = fixture()
    result = evaluate_policy_book(
        compile_policy_book(s), s, world_overrides=(override,))
    assert result.status == "BLOCKED"


def test_zero_legal_override_mass_is_not_uniform_fallback():
    s = facing_bet()
    book = compile_policy_book(s)
    override = WorldResponseOverride(1, after_hero_check_multipliers=(
        ("check", Fraction(0)), ("bet", Fraction(0))))
    result = evaluate_policy_book(book, s, world_overrides=(override,))
    assert result.status == "BLOCKED"
    assert result.reasons == ("zero_legal_world_override_mass",)


def test_any_fallback_is_path_union_not_sum_of_fallback_visits():
    s = replace(fixture(), models=(model(1, bet=1, fold=1), fixture().models[1]))
    tree = _Tree(s, 128, 20000)
    root = _Node(tree.active, s.action_order, tuple(Decimal(0) for _ in s.seats),
                 Decimal(0), Decimal(2), 0, ())
    # Deliberately no public entries: fallback check, then fallback fold on SAME path.
    evaluator = _Evaluation(tree, {}, 20000, {})
    ev, probability, count = evaluator.walk(root, tree.prior, "frozen_policy")
    assert ev == 0
    assert probability == 1 and count == 2
    assert sum(v.reach_probability for v in evaluator.unsupported) == 2


def test_rank_override_updates_history_belief_but_keeps_planned_call():
    s = facing_bet()
    r = replace(s.ranges[0], combo_weights={"KhKd": Decimal(1), "ThTd": Decimal(1)})
    s = replace(s, ranges=(r, s.ranges[1]),
                models=(model(1, check=1, bet=1, fold=1), s.models[1]))
    book = compile_policy_book(s)
    assert book.decisions[0].action.kind == "call"
    override = WorldResponseOverride(1, 12, (("bet", Fraction(9)),))
    result = evaluate_policy_book(book, s, world_overrides=(override,))
    assert result.history_likelihood == Fraction(7, 10)
    assert sorted(result.root_posterior) == [Fraction(5, 14), Fraction(9, 14)]
    assert metric(result).net_ev_chips == Fraction(-50, 7)
    assert metric(result, "check_fold").net_ev_chips == 0


def test_node_budget_is_global_across_policy_and_both_baselines():
    s = facing_bet()
    result = evaluate_policy_book(compile_policy_book(s), s, max_nodes=4)
    assert result.status == "BLOCKED"
    assert result.metrics == ()
    assert result.nodes == 5


def test_public_numeric_identity_never_rounds_under_decimal_context():
    s = fixture()
    a = replace(s, seats=(replace(s.seats[0], stack=ChipAmount(
        "10000000000000000000000000001")), *s.seats[1:]))
    b = replace(s, seats=(replace(s.seats[0], stack=ChipAmount(
        "10000000000000000000000000002")), *s.seats[1:]))
    assert public_conditions_json(a) != public_conditions_json(b)
