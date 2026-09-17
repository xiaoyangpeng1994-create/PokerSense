from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy import threeway_policy_evaluation_v1 as module
from poker_engine.strategy.threeway_policy_evaluation_v1 import (
    REACHED, UNREACHABLE, PolicyPathLedger, TerminalPath, WorldResponseOverride,
    _Evaluation, _override_probabilities, _world_branches, compare_policy_paths,
    compile_policy_book, evaluate_policy_book, path_key, policy_book_hash,
    public_conditions_json, reconcile_path_ledgers,
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


def hand_example():
    """Three-handed river where every terminal EV is an integer.

    Pot is 60 (three active seats committed 20). Hero QQ always wins the
    showdown against 3c3d and 5h6h, so each terminal value is the pot Hero
    collects minus the chips Hero adds after the decision root.
    """
    return replace(fixture(), ranges=ranges({1: "3c3d", 2: "5h6h"}),
                   models=(model(1, check=1, bet=1, fold=1, call=1),
                           model(2, check=1, fold=1, call=1)))


# (reach probability, conditional terminal EV, contribution) per public history.
# Derived by hand: Hero bet 100 -> 1/2 fold / 1/2 call each (pot 160 if one
# opponent continues, 260 if both do, 60 when both fold), and the checked line
# wins 60 unless Seat 1 bets, where the baselines fold (0) or call (160/260).
HAND_PATHS = {
    "frozen_policy": {
        "0:bet:100|1:call|2:call": (Fraction(1, 4), Fraction(260), Fraction(65)),
        "0:bet:100|1:call|2:fold": (Fraction(1, 4), Fraction(160), Fraction(40)),
        "0:bet:100|1:fold|2:call": (Fraction(1, 4), Fraction(160), Fraction(40)),
        "0:bet:100|1:fold|2:fold": (Fraction(1, 4), Fraction(60), Fraction(15)),
    },
    "check_fold": {
        "0:check|1:bet:100|2:call|0:fold": (
            Fraction(1, 4), Fraction(0), Fraction(0)),
        "0:check|1:bet:100|2:fold|0:fold": (
            Fraction(1, 4), Fraction(0), Fraction(0)),
        "0:check|1:check|2:check": (Fraction(1, 2), Fraction(60), Fraction(30)),
    },
    "check_call": {
        "0:check|1:bet:100|2:call|0:call": (
            Fraction(1, 4), Fraction(260), Fraction(65)),
        "0:check|1:bet:100|2:fold|0:call": (
            Fraction(1, 4), Fraction(160), Fraction(40)),
        "0:check|1:check|2:check": (Fraction(1, 2), Fraction(60), Fraction(30)),
    },
}
HAND_EV = {"frozen_policy": Fraction(160), "check_fold": Fraction(30),
           "check_call": Fraction(135)}


def ledger(result, name):
    assert result.status == "COMPLETE_CONDITIONAL_FIXED_POLICY", result.reasons
    return next(item for item in result.path_ledgers if item.policy_name == name)


def test_traced_terminal_paths_match_the_hand_computed_example():
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    assert {m.name: m.net_ev_chips for m in result.metrics} == HAND_EV
    assert (result.delta_vs_check_fold_chips, result.delta_vs_check_call_chips) == (
        130, 25)
    for name, expected in HAND_PATHS.items():
        item = ledger(result, name)
        reachable = {p.history_key: (p.reach_probability,
                                     p.conditional_terminal_net_ev_chips,
                                     p.contribution_chips)
                     for p in item.paths if p.status == REACHED}
        assert reachable == expected
        # Reachable and unreachable paths partition the same exhaustive union.
        assert len(item.paths) == len(next(
            ledger(result, other).paths for other in HAND_PATHS))
        assert item.reach_probability_sum == 1
        assert item.contribution_sum_chips == item.reconciled_policy_net_ev_chips
        assert item.contribution_sum_chips == HAND_EV[name]
        assert item.contribution_sum_bb == HAND_EV[name] / 2
        assert item.nodes_visited == item.trace_node_budget or (
            item.nodes_visited < item.trace_node_budget)
        assert (item.reachable_paths, item.unreachable_paths) == (
            len(expected), len(item.paths) - len(expected))


def test_unreachable_paths_report_undefined_conditional_ev_and_zero_weight():
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    for item in result.path_ledgers:
        for path in item.paths:
            if path.status == REACHED:
                assert path.reach_probability > 0
                assert isinstance(path.conditional_terminal_net_ev_chips, Fraction)
                assert path.contribution_chips == (
                    path.reach_probability * path.conditional_terminal_net_ev_chips)
            else:
                assert path.status == UNREACHABLE
                assert path.reach_probability == 0
                assert path.conditional_terminal_net_ev_chips is None
                assert path.contribution_chips == 0
    # The zero-mass Hero branches are the policy difference, not lost mass.
    assert {p.history_key for p in ledger(result, "check_fold").paths} == {
        p.history_key for p in ledger(result, "frozen_policy").paths}


def test_path_comparison_subtracts_the_union_without_adding_ancestors_twice():
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    keys = {p.history_key for p in ledger(result, "frozen_policy").paths}
    # Every recorded row is a terminal path, so no row is an ancestor of another.
    assert [key for key in keys if any(
        other != key and key.startswith(other + "|") for other in keys)] == []
    for left, right, expected in (("frozen_policy", "check_fold", Fraction(130)),
                                  ("frozen_policy", "check_call", Fraction(25))):
        reconciliation = compare_policy_paths(result, left, right)
        assert {row.history_key for row in reconciliation.rows} == keys
        assert reconciliation.contribution_difference_sum_chips == expected
        assert reconciliation.total_ev_difference_chips == expected
        moved = [row for row in reconciliation.rows
                 if row.contribution_chips_difference]
        assert moved and all(row.history_key in HAND_PATHS[left]
                             or row.history_key in HAND_PATHS[right]
                             for row in moved)
        assert reconciliation.reach_status_count("REACHED_BY_BOTH") == 0


def test_trace_switch_changes_only_the_populated_ledgers():
    s = hand_example()
    book = compile_policy_book(s)
    plain = evaluate_policy_book(book, s)
    traced = evaluate_policy_book(book, s, trace=True)
    assert plain.path_ledgers == ()
    assert traced.path_ledgers and all(
        item.completeness == "COMPLETE_TERMINAL_PATH_LEDGER"
        for item in traced.path_ledgers)
    assert replace(traced, path_ledgers=()) == plain
    assert policy_book_hash(book) == book.book_sha256
    assert traced.policy_hash_before == traced.policy_hash_after == book.book_sha256


def test_traced_evaluation_never_calls_an_optimizing_entry_point(monkeypatch):
    s = hand_example()
    book = compile_policy_book(s)
    before = policy_book_hash(book)

    def forbidden(*args, **kwargs):
        raise AssertionError("traced evaluation attempted to optimize Hero")

    monkeypatch.setattr(module, "analyze_threeway_river", forbidden)
    monkeypatch.setattr(_Tree, "value", forbidden)
    result = evaluate_policy_book(book, s, trace=True)
    assert ledger(result, "frozen_policy").contribution_sum_chips == 160
    assert result.policy_hash_before == result.policy_hash_after == before
    assert policy_book_hash(book) == before


def test_fallback_paths_are_flagged_and_their_reach_matches_the_metric():
    s = replace(fixture(), ranges=ranges({1: "ThTd", 2: "3c3d"}))
    book = compile_policy_book(s)
    world = replace(s, models=(model(1, check=1, bet=1, fold=1), s.models[1]))
    result = evaluate_policy_book(book, world, trace=True)
    outcome = metric(result)
    item = ledger(result, "frozen_policy")
    flagged = [p for p in item.paths if p.fallback_used]
    assert flagged
    assert sum((p.reach_probability for p in flagged), Fraction(0)) == (
        outcome.probability_of_any_fallback)
    assert item.fallback_reach_sum == outcome.probability_of_any_fallback
    assert outcome.probability_of_any_fallback == Fraction(1, 2)


def test_trace_budget_failure_is_blocked_without_a_partial_ledger():
    s = hand_example()
    book = compile_policy_book(s)
    result = evaluate_policy_book(book, s, trace=True, trace_max_nodes=1)
    assert result.status == "BLOCKED"
    assert result.reasons == ("trace_node_budget_exceeded",)
    assert result.metrics == () and result.path_ledgers == ()
    for bad in (0, -1, 200001, True):
        blocked = evaluate_policy_book(book, s, trace=True, trace_max_nodes=bad)
        assert blocked.status == "BLOCKED"
        assert blocked.reasons == ("invalid_explicit_trace_budget",)
    assert evaluate_policy_book(book, s, trace="yes").status == "BLOCKED"


@pytest.mark.parametrize("mutation", ["reach", "sum", "unreachable_ev",
                                      "reachable_ev", "duplicate", "bad_status"])
def test_ledger_refuses_inconsistent_paths_instead_of_returning_them(mutation):
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    item = ledger(result, "frozen_policy")
    zero = next(p for p in item.paths if p.status == UNREACHABLE)
    live = next(p for p in item.paths if p.status == REACHED)
    corrupted = {
        "reach": lambda: replace(zero, reach_probability=Fraction(1, 2)),
        "sum": lambda: replace(item, paths=tuple(
            p for p in item.paths if p is not live)),
        "unreachable_ev": lambda: replace(
            zero, conditional_terminal_net_ev_chips=Fraction(0)),
        "reachable_ev": lambda: replace(live, contribution_chips=Fraction(999)),
        "duplicate": lambda: replace(item, paths=(live, live)),
        "bad_status": lambda: replace(zero, status=REACHED),
    }[mutation]
    with pytest.raises(ValueError):
        corrupted()


def test_path_comparison_rejects_untraced_or_unknown_policies():
    s = hand_example()
    book = compile_policy_book(s)
    untraced = evaluate_policy_book(book, s)
    with pytest.raises(ValueError, match="path_trace_not_collected"):
        compare_policy_paths(untraced)
    traced = evaluate_policy_book(book, s, trace=True)
    with pytest.raises(ValueError, match="two_distinct_policy_names"):
        compare_policy_paths(traced, "frozen_policy", "frozen_policy")
    with pytest.raises(ValueError, match="path_trace_not_collected"):
        compare_policy_paths(traced, "frozen_policy", "unnamed_policy")
    blocked = evaluate_policy_book(book, s, trace=True, trace_max_nodes=1)
    with pytest.raises(ValueError, match="requires_complete_evaluation"):
        compare_policy_paths(blocked)


def test_path_key_is_canonical_for_sized_and_unsized_actions():
    assert path_key(()) == "root"
    assert path_key((RiverAction(0, "check"),
                     RiverAction(1, "bet", Decimal(100)))) == "0:check|1:bet:100"
    with pytest.raises(ValueError, match="river_action_tuple"):
        path_key([RiverAction(0, "check")])
    with pytest.raises(ValueError, match="river_action_tuple"):
        path_key((object(),))


def test_ledger_and_path_dataclasses_are_self_verifying():
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    item = ledger(result, "frozen_policy")
    with pytest.raises(ValueError):
        replace(item, reconciled_policy_net_ev_chips=Fraction(1))
    with pytest.raises(ValueError):
        replace(item, reconciled_fallback_probability=Fraction(1, 3))
    with pytest.raises(ValueError):
        replace(item, completeness="PARTIAL")
    with pytest.raises(ValueError):
        replace(item, big_blind=Fraction(0))
    with pytest.raises(ValueError):
        TerminalPath((RiverAction(0, "check"),), "0:check", Fraction(0), False,
                     None, Fraction(0), REACHED)
    assert isinstance(item, PolicyPathLedger)


def test_cross_book_reconciliation_requires_an_explicit_opt_in():
    s = hand_example()
    book = compile_policy_book(s)
    other = compile_policy_book(s, policy_id="second-book-v1")
    assert other.book_sha256 != book.book_sha256
    left = ledger(evaluate_policy_book(book, s, trace=True), "frozen_policy")
    right = ledger(evaluate_policy_book(other, s, trace=True), "frozen_policy")
    total = (left.reconciled_policy_net_ev_chips
             - right.reconciled_policy_net_ev_chips)
    with pytest.raises(ValueError, match="different_books"):
        reconcile_path_ledgers(left, right, total)
    reconciliation = reconcile_path_ledgers(left, right, total,
                                            allow_distinct_books=True)
    assert reconciliation.distinct_books is True
    assert reconciliation.right_policy_book_sha256 == other.book_sha256
    assert reconciliation.policy_book_sha256 == book.book_sha256
    assert reconciliation.total_ev_difference_chips == 0
    assert reconciliation.contribution_difference_sum_chips == 0
    for row in reconciliation.rows:
        assert row.contribution_chips_difference == 0
    with pytest.raises(ValueError, match="path_ledgers_describe_different_worlds"):
        reconcile_path_ledgers(
            left, replace(right, world_scenario_sha256="0" * 64),
            total, allow_distinct_books=True)
    with pytest.raises(ValueError, match="explicit_exact_ev_difference"):
        reconcile_path_ledgers(left, right, 0.0, allow_distinct_books=True)
    with pytest.raises(ValueError):
        reconcile_path_ledgers(left, right, Fraction(1),
                               allow_distinct_books=True)


def test_reconciliation_dataclass_rejects_a_misstated_book_relation():
    s = hand_example()
    result = evaluate_policy_book(compile_policy_book(s), s, trace=True)
    reconciliation = compare_policy_paths(result)
    assert reconciliation.distinct_books is False
    assert reconciliation.right_policy_book_sha256 is None
    with pytest.raises(ValueError):
        replace(reconciliation, distinct_books=True)
    with pytest.raises(ValueError):
        replace(reconciliation, right_policy_book_sha256="f" * 64)
