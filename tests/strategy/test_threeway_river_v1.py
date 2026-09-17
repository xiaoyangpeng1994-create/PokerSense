from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from poker_engine.core.enums import PlayerStatus
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.threeway_river_v1 import (
    ResponseModel, RiverAction, ThreewayRiverScenario, _Node, _Tree,
    analyze_threeway_river,
)
from .helpers import card
from .test_terminal_multiway_v1 import ranges, rules, seat


def model(i, **weights):
    return ResponseModel(i, tuple((k, Fraction(v)) for k, v in weights.items()))


def fixture(n=6):
    return ThreewayRiverScenario(
        seats=tuple(seat(i, 20 if i < 3 else 0, street=0, stack="200",
                         status=PlayerStatus.ACTIVE if i < 3 else PlayerStatus.FOLDED)
                    for i in range(n)),
        hero_seat=0, hero_cards=(card("Qc"), card("Qd")),
        board_cards=tuple(map(card, ("2c", "4d", "7h", "9s", "Jc"))),
        ranges=ranges({1: "KhKd", 2: "3c3d"}), rules=rules(n),
        action_order=(0, 1, 2),
        models=(model(1, check=1, fold=1), model(2, check=1, fold=1)),
        aggression_targets=(Decimal(100),), max_aggressions=1,
    )


def value(result, kind, target=0):
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION", result.reasons
    return next(v.ev for v in result.root_actions
                if v.action.kind == kind and v.action.target == target)


def test_information_set_max_not_hidden_assignment_strategy_fusion():
    s = fixture()
    r = replace(s.ranges[0], combo_weights={"JhJd": Decimal(3), "ThTd": Decimal(1)})
    s = replace(s, ranges=(r, s.ranges[1]),
                models=(model(1, bet=1, fold=1), model(2, fold=1, check=1)))
    result = analyze_threeway_river(s)
    assert value(result, "check") == 0  # Illegal per-deal max would give 40.
    later = next(p for p in result.hero_policy if len(p.history) == 3)
    assert later.best_action.kind == "fold"
    assert next(v.ev for v in later.action_values if v.action.kind == "call") == -35


def test_public_bet_signal_changes_belief_before_hero_continuation():
    s = fixture()
    r = replace(s.ranges[0], combo_weights={"JhJd": Decimal(1), "ThTd": Decimal(1)})
    response = ResponseModel(1, (("fold", Fraction(1)), ("check", Fraction(1))), (
        (3, (("bet:100", Fraction(1)), ("fold", Fraction(1)))),
        (1, (("bet:100", Fraction(1)), ("check", Fraction(2)), ("fold", Fraction(1)))),
    ))
    result = analyze_threeway_river(replace(
        s, ranges=(r, s.ranges[1]), models=(response, s.models[1])))
    assert value(result, "check") == 20
    later = next(p for p in result.hero_policy
                 if tuple(a.kind for a in p.history) == ("check", "bet", "fold"))
    assert sorted(later.belief) == [Fraction(1, 4), Fraction(3, 4)]


def test_fold_likelihood_and_joint_card_blockers_remain_in_posterior():
    s = fixture()
    r1 = replace(s.ranges[0], combo_weights={"JhJd": Decimal(1), "ThTd": Decimal(1)})
    r2 = replace(s.ranges[1], combo_weights={"Jh3d": Decimal(1), "Th3d": Decimal(1)})
    m1 = model(1, bet=1, fold=1)
    m2 = ResponseModel(2, (("check", Fraction(1)), ("fold", Fraction(1))), (
        (1, (("check", Fraction(1)), ("fold", Fraction(1)))),
        (0, (("check", Fraction(1)), ("call", Fraction(1)))),
    ))
    result = analyze_threeway_river(replace(s, ranges=(r1, r2), models=(m1, m2)))
    assert result.joint_assignments == 2
    assert value(result, "check") == 80


@pytest.mark.parametrize("n", [6, 7, 8])
def test_three_current_players_and_six_to_eight_dealt_seats(n):
    s = replace(fixture(n), hero_cards=(card("5d"), card("6d")),
                ranges=ranges({1: "2d3d", 2: "4c5c"}),
                board_cards=tuple(map(card, ("As", "Ks", "Qs", "Js", "Ts"))))
    result = analyze_threeway_river(s)
    assert value(result, "check") == 20  # Starting 20 is sunk, not subtracted again.
    assert value(result, "bet", 100) == 60  # Uncalled 100 is returned.
    assert result.strategy_eligible is result.advice_emitted is False
    result = analyze_threeway_river(replace(s, rules=rules(n, rake_percent="0.1")))
    assert value(result, "check") == 18
    assert value(result, "bet", 100) == 54  # Rake on 60, not 160 including refund.


def test_history_conditions_river_start_belief_and_sunk_hero_cost():
    s = fixture()
    r = replace(s.ranges[0], combo_weights={"JhJd": Decimal(1), "ThTd": Decimal(1)})
    response = ResponseModel(1, (("fold", Fraction(1)),), (
        (3, (("bet", Fraction(1)),)),
        (1, (("bet", Fraction(1)), ("check", Fraction(2)), ("fold", Fraction(1)))),
    ))
    history = (RiverAction(0, "check"), RiverAction(1, "bet", Decimal(100)),
               RiverAction(2, "fold"))
    result = analyze_threeway_river(replace(
        s, ranges=(r, s.ranges[1]), models=(response, s.models[1]), history=history))
    assert result.history_likelihood == Fraction(2, 3)
    assert sorted(result.root_posterior) == [Fraction(1, 4), Fraction(3, 4)]
    assert value(result, "call") == -35
    assert result.root_context.root_pot == 160
    assert result.root_context.to_call == 100
    assert len(result.joint_hypotheses) == len(result.root_posterior) == 2
    call = next(v for v in result.root_actions if v.action.kind == "call")
    assert call.additional_cost == 100


def test_re_raise_branch_and_later_root_only_charge_new_hero_chips():
    s = replace(fixture(), aggression_targets=(Decimal(10), Decimal(30)),
                max_aggressions=2,
                models=(model(1, **{"check": 1, "raise": 1, "fold": 1}),
                        model(2, check=1, fold=1)))
    result = analyze_threeway_river(s)
    assert value(result, "bet", 10) == 25  # Half win 60, half lose the initial 10.
    history = (RiverAction(0, "bet", Decimal(10)), RiverAction(1, "raise", Decimal(30)),
               RiverAction(2, "fold"))
    result = analyze_threeway_river(replace(s, history=history))
    assert value(result, "fold") == 0
    assert value(result, "call") == -20
    assert result.root_context.hero_street_committed == 10
    assert result.root_context.root_pot == 100
    assert next(v.additional_cost for v in result.root_actions
                if v.action.kind == "call") == 20


def test_semantically_equal_numeric_target_uses_canonical_policy_key():
    response = ResponseModel(1, (("bet:100", Fraction(1)), ("fold", Fraction(1))))
    s = replace(fixture(), aggression_targets=(Decimal("100.0"),),
                models=(response, model(2, check=1, fold=1)))
    result = analyze_threeway_river(s)
    assert result.status == "COMPLETE_CONDITIONAL_ABSTRACTION"


def tree():
    s = replace(fixture(), aggression_targets=tuple(map(Decimal, (10, 30, 40, 50, 60))),
                max_aggressions=3)
    t = _Tree(s, 128, 20000)
    node = _Node(t.active, s.action_order, tuple(Decimal(0) for _ in s.seats),
                 Decimal(0), Decimal(2), 0, ())
    return t, node


def test_second_check_does_not_close_before_third_player_bets():
    t, node = tree()
    node = t.advance(node, RiverAction(0, "check"))
    node = t.advance(node, RiverAction(1, "check"))
    assert node.pending == (2,)
    node = t.advance(node, RiverAction(2, "bet", Decimal(10)))
    assert node.pending == (0, 1)
    assert {a.kind for a in t.legal(node)} == {"fold", "call", "raise"}


def test_raise_reopens_action_and_minimum_uses_last_increment():
    t, node = tree()
    node = t.advance(node, RiverAction(0, "bet", Decimal(10)))
    node = t.advance(node, RiverAction(1, "raise", Decimal(30)))
    assert node.pending == (2, 0)
    assert [a.target for a in t.legal(node) if a.kind == "raise"] == [50, 60]
    node = t.advance(node, RiverAction(2, "fold"))
    assert node.pending == (0,)
    node = t.advance(node, RiverAction(0, "call"))
    assert t.legal(node) == ()


def test_hero_fold_still_expands_remaining_opponent_response():
    s = fixture()
    history = (RiverAction(0, "check"), RiverAction(1, "bet", Decimal(100)),
               RiverAction(2, "raise", Decimal(200)))
    # Separate public prefix where Hero folds but opponent one still owes call.
    s = replace(s, aggression_targets=(Decimal(100), Decimal(200)), max_aggressions=2,
                seats=tuple(replace(p, stack=ChipAmount("300")) for p in s.seats),
                models=(model(1, bet=1, call=1, fold=1),
                        model(2, **{"raise": 1, "check": 1})),
                history=history)
    result = analyze_threeway_river(s)
    assert value(result, "fold") == 0
    assert result.terminal_nodes >= 3  # Two opponent replies after fold plus Hero call.


@pytest.mark.parametrize("mutation", [
    "call_out_of_turn", "fold_free", "after_closed", "noncanonical_key", "zero_mass",
    "allin_edge", "sidepot", "street_not_zero", "wrong_order", "unknown_fee",
    "posterior_range", "future_board", "zero_history", "node_budget", "joint_budget",
])
def test_reject_invalid_or_incomplete_game_without_leaf_equity_fallback(mutation):
    s = fixture()
    budgets = {}
    if mutation == "call_out_of_turn":
        s = replace(s, history=(RiverAction(1, "check"),))
    elif mutation == "fold_free":
        s = replace(s, history=(RiverAction(0, "fold"),))
    elif mutation == "after_closed":
        s = replace(s, history=tuple(RiverAction(i, "check") for i in (0, 1, 2, 0)))
    elif mutation == "noncanonical_key":
        s = replace(s, models=(
            ResponseModel(1, (("bet:10.0", Fraction(1)),)), s.models[1]))
    elif mutation == "zero_mass":
        s = replace(s, models=(model(1, **{"raise": 1}), s.models[1]))
    elif mutation in ("allin_edge", "sidepot", "street_not_zero"):
        p = s.seats[0]
        changes = {"stack": ChipAmount("100")} if mutation == "allin_edge" else (
            {"hand_committed": ChipAmount("21")} if mutation == "sidepot" else
            {"street_committed": ChipAmount("1")})
        s = replace(s, seats=(replace(p, **changes), *s.seats[1:]))
    elif mutation == "wrong_order":
        s = replace(s, action_order=(0, 1, 1))
    elif mutation == "unknown_fee":
        s = replace(s, other_fees=None)
    elif mutation == "posterior_range":
        s = replace(s, range_start="decision_point")
    elif mutation == "future_board":
        s = replace(s, board_cards=s.board_cards[:4])
    elif mutation == "zero_history":
        s = replace(s, history=(
            RiverAction(0, "check"), RiverAction(1, "bet", Decimal(100))))
    elif mutation == "node_budget":
        budgets["max_nodes"] = 1
    else:
        r = replace(s.ranges[0], combo_weights={"KhKd": Decimal(1), "ThTd": Decimal(1)})
        s = replace(s, ranges=(r, s.ranges[1]))
        budgets["max_joint_assignments"] = 1
    result = analyze_threeway_river(s, **budgets)
    assert result.status == "BLOCKED"
    assert result.best_ev is None and result.root_actions == ()


def priced_fixture():
    response = ResponseModel(1, (("check", Fraction(1)), ("fold", Fraction(1)),
                                 ("call", Fraction(1))), price_multipliers=(
        (Fraction(1, 4), (("call", Fraction(3)),)),
        (Fraction(1), (("fold", Fraction(3)),)),
    ))
    return replace(fixture(), aggression_targets=(Decimal(20), Decimal(80)),
                   models=(response, model(2, check=1, fold=1)))


def priced_node(t, target):
    node = _Node(t.active, t.s.action_order, tuple(Decimal(0) for _ in t.s.seats),
                 Decimal(0), Decimal(2), 0, ())
    return t.advance(node, RiverAction(0, "bet", Decimal(target)))


def test_same_holding_responds_differently_to_public_twenty_and_eighty_price():
    t = _Tree(priced_fixture(), 128, 20000)
    small, large = priced_node(t, 20), priced_node(t, 80)
    assert t.price_ratio(small) == Fraction(1, 5)
    assert t.price_ratio(large) == Fraction(4, 11)
    small_prob = t.probabilities(1, 1, t.legal(small), t.price_ratio(small))
    large_prob = t.probabilities(1, 1, t.legal(large), t.price_ratio(large))
    assert [a.kind for a in t.legal(small)] == ["fold", "call"]
    assert small_prob == (Fraction(1, 4), Fraction(3, 4))
    assert large_prob == (Fraction(3, 4), Fraction(1, 4))


def test_price_policy_does_not_receive_or_depend_on_hero_private_cards():
    a = _Tree(priced_fixture(), 128, 20000)
    b = _Tree(replace(priced_fixture(), hero_cards=(card("5c"), card("6c"))),
              128, 20000)
    na, nb = priced_node(a, 80), priced_node(b, 80)
    assert a.probabilities(1, 1, a.legal(na), a.price_ratio(na)) == (
        b.probabilities(1, 1, b.legal(nb), b.price_ratio(nb)))


@pytest.mark.parametrize("bands", [
    ((Fraction(0), (("call", Fraction(1)),)),),
    ((Fraction(1, 2), ()),),
    ((Fraction(1), ()), (Fraction(1, 2), ())),
    ((Fraction(1, 2), ()), (Fraction(1, 2), ()), (Fraction(1), ())),
    ((Fraction(2), ()),),
    ((Fraction(1), (("call", Fraction(-1)),)),),
    ((Fraction(1), (("bet:10.0", Fraction(1)),)),),
])
def test_invalid_price_tables_are_rejected(bands):
    s = priced_fixture()
    result = analyze_threeway_river(replace(
        s, models=(replace(s.models[0], price_multipliers=bands), s.models[1])))
    assert result.status == "BLOCKED"


def test_zero_price_adjusted_legal_mass_never_becomes_uniform():
    s = priced_fixture()
    bands = ((Fraction(1), (("fold", Fraction(0)), ("call", Fraction(0)))),)
    result = analyze_threeway_river(replace(
        s, models=(replace(s.models[0], price_multipliers=bands), s.models[1])))
    assert result.status == "BLOCKED"
    assert result.reasons == ("zero_legal_response_mass",)


def test_price_multiplier_cannot_invent_an_action_absent_from_base_table():
    s = priced_fixture()
    response = replace(s.models[0], weights=(
        ("check", Fraction(1)), ("fold", Fraction(1))))
    t = _Tree(replace(s, models=(response, s.models[1])), 128, 20000)
    node = priced_node(t, 20)
    assert t.probabilities(1, 1, t.legal(node), t.price_ratio(node)) == (1, 0)
