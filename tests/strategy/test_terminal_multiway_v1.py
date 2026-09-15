from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from poker_engine.core.enums import PlayerStatus, Position
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from poker_engine.strategy.contracts import DecisionSeat, RangeDistribution
from poker_engine.strategy.terminal_multiway_v1 import (
    TerminalAnalysis, TerminalScenario, analyze_terminal_multiway,
)
from .helpers import card


def rules(n=6, **changes):
    spec = dict(schema_version=2, table_size=n, small_blind="1", big_blind="2",
                ante="0", ante_mode="none", straddle_mode="none", straddle_amount="0",
                rake_percent="0", rake_cap_bb="100", rake_application="all_pots",
                rake_rounding="exact", rake_distribution="proportional_all_pots",
                minimum_chip="1", verification_status="simulation", source="manual")
    spec.update(changes)
    return AARuleProfileV2.from_dict(spec)


def seat(index, committed, *, stack="0", status=PlayerStatus.ALL_IN, street=None):
    return DecisionSeat(index, f"p{index}", Position.BTN, ChipAmount(stack),
                        ChipAmount(str(committed if street is None else street)),
                        ChipAmount(str(committed)), status, is_hero=index == 0)


def ranges(holdings):
    return tuple(RangeDistribution(i, {combo: Decimal(1)}, "manual_terminal_assumption",
                                   f"manual:seat{i}") for i, combo in holdings.items())


def scenario():
    return TerminalScenario(
        seats=(seat(0, 40, stack="60", status=PlayerStatus.ACTIVE),
               seat(1, 100), seat(2, 60), seat(3, 20),
               seat(4, 30, stack="100", status=PlayerStatus.FOLDED),
               seat(5, 10, stack="100", status=PlayerStatus.FOLDED)),
        hero_seat=0, actor_seat=0, hero_cards=(card("As"), card("Ad")),
        board_cards=tuple(map(card, ("2c", "4d", "7h", "9s", "Jc"))),
        current_bet=ChipAmount("100"), pot_before=ChipAmount("260"),
        ranges=ranges({1: "KhKd", 2: "QhQd", 3: "ThTd"}), rules=rules(),
    )


def test_three_opponents_dead_money_and_unequal_sidepots_analytical():
    result = analyze_terminal_multiway(scenario())
    assert result.status == "COMPLETE_CONDITIONAL"
    assert [p.amount for p in result.call_pots] == [110, 130, 80]
    assert [p.eligible_seats for p in result.call_pots] == [
        (0, 1, 2, 3), (0, 1, 2), (0, 1)]
    assert result.call_gross_ev == result.call_net_ev == 260
    assert result.call_cost == 60
    assert result.fold_ev == 0
    assert result.recommendation == "CALL"
    assert not result.strategy_eligible and not result.advice_emitted


def test_short_call_returns_uncalled_opponent_money():
    s = scenario()
    s = replace(s, seats=(replace(s.seats[0], stack=ChipAmount("20")), *s.seats[1:]))
    result = analyze_terminal_multiway(s)
    assert result.call_cost == 20
    assert result.call_refunds == ((1, Decimal(40)),)
    assert [p.amount for p in result.call_pots] == [110, 130]
    assert result.call_net_ev == 220


def test_hero_ineligible_outer_sidepot_cannot_increase_return():
    s = scenario()
    ss = list(s.seats)
    ss[0] = replace(ss[0], stack=ChipAmount("30"))
    ss[2] = seat(2, 100)
    s = replace(s, seats=tuple(ss), pot_before=ChipAmount("300"))
    result = analyze_terminal_multiway(s)
    assert [p.amount for p in result.call_pots] == [110, 160, 60]
    assert result.call_pots[-1].eligible_seats == (1, 2)
    assert result.call_pots[-1].hero_share == 0
    assert result.call_net_ev == 240


@pytest.mark.parametrize("n", [6, 7, 8])
def test_full_dealt_table_all_other_players_all_in(n):
    combos = ("2c3c", "4c5c", "6c7c", "8c9c", "TcJc", "QcKc", "2d3d")
    s = TerminalScenario(
        seats=(seat(0, 0, stack="100", status=PlayerStatus.ACTIVE),
               *(seat(i, 100) for i in range(1, n))),
        hero_seat=0, actor_seat=0, hero_cards=(card("4d"), card("5d")),
        board_cards=tuple(map(card, ("As", "Ks", "Qs", "Js", "Ts"))),
        current_bet=ChipAmount("100"), pot_before=ChipAmount(str(100 * (n - 1))),
        ranges=ranges(dict(enumerate(combos[:n - 1], 1))), rules=rules(n),
    )
    result = analyze_terminal_multiway(s)
    assert result.status == "COMPLETE_CONDITIONAL"
    assert result.call_pots[0].hero_share == Fraction(1, n)
    assert result.call_net_ev == 0  # No Decimal thirds/sevenths indifference error.
    assert result.recommendation == "INDIFFERENT"


def test_zero_hero_equity_is_fold():
    s = replace(scenario(), hero_cards=(card("8d"), card("6d")))
    result = analyze_terminal_multiway(s)
    assert result.call_net_ev == -60
    assert result.recommendation == "FOLD"


def test_rake_cap_and_distribution_are_explicit():
    s = replace(scenario(), rules=rules(rake_percent="0.03", rake_cap_bb="2"))
    result = analyze_terminal_multiway(s)
    assert sum(p.configured_rake for p in result.call_pots) == 4
    assert result.call_gross_ev == 260 and result.call_net_ev == 256
    assert result.call_pots[0].configured_rake == Fraction(11, 8)


def test_unsettled_prior_hero_refund_is_explicitly_blocked_not_fold_ev_zero():
    ss = (seat(0, 100, stack="20", status=PlayerStatus.ACTIVE, street=10),
          seat(1, 80, street=20), seat(2, 60, street=20), seat(3, 40, street=20),
          seat(4, 30, status=PlayerStatus.FOLDED, street=0),
          seat(5, 10, status=PlayerStatus.FOLDED, street=0))
    s = replace(scenario(), seats=ss, current_bet=ChipAmount("20"),
                pot_before=ChipAmount("320"))
    result = analyze_terminal_multiway(s)
    assert result.status == "BLOCKED"
    assert result.reasons == ("unsettled_prior_street_refund_unsupported",)
    assert result.fold_ev is None


def test_highest_river_bet_cannot_come_from_a_folded_player():
    s = scenario()
    ss = list(s.seats)
    ss[4] = seat(4, 200, status=PlayerStatus.FOLDED)
    result = analyze_terminal_multiway(replace(
        s, seats=tuple(ss), current_bet=ChipAmount("200"),
        pot_before=ChipAmount("430")))
    assert result.status == "BLOCKED"


def test_original_weighted_ranges_have_exact_thirds_and_blockers():
    s = scenario()
    first = replace(s.ranges[0], combo_weights={
        "KhKd": Decimal(1), "JhJd": Decimal(2)})
    result = analyze_terminal_multiway(replace(s, ranges=(first, *s.ranges[1:])))
    assert result.call_net_ev == Fraction(140, 3)
    first = replace(first, combo_weights={"KhKd": Decimal(1), "AsKd": Decimal(99)})
    result = analyze_terminal_multiway(replace(s, ranges=(first, *s.ranges[1:])))
    assert result.call_net_ev == 260


@pytest.mark.parametrize("mutation", [
    "wrong_actor", "other_active", "allin_with_stack", "hero_zero", "turn",
    "cards_collision", "range_collision", "missing_range", "pot_mismatch",
    "bet_mismatch", "unknown_fee", "other_fee", "rake_unknown", "live_rules",
    "gto_source", "asset_source", "confidence_claim", "duplicate_combo", "budget",
])
def test_incomplete_or_out_of_scope_is_blocked(mutation):
    s = scenario()
    limit = 4096
    if mutation == "wrong_actor":
        s = replace(s, actor_seat=1)
    elif mutation in ("other_active", "allin_with_stack", "hero_zero"):
        ss = list(s.seats)
        if mutation == "hero_zero":
            ss[0] = replace(ss[0], stack=ChipAmount("0"))
        else:
            ss[1] = replace(ss[1], stack=ChipAmount("100"), status=(
                PlayerStatus.ACTIVE if mutation == "other_active"
                else PlayerStatus.ALL_IN))
        s = replace(s, seats=tuple(ss))
    elif mutation == "turn":
        s = replace(s, board_cards=s.board_cards[:4])
    elif mutation == "cards_collision":
        s = replace(s, hero_cards=(s.board_cards[0], card("Ad")))
    elif mutation == "range_collision":
        s = replace(s, ranges=ranges({1: "AsKd", 2: "QhQd", 3: "ThTd"}))
    elif mutation == "missing_range":
        s = replace(s, ranges=s.ranges[:2])
    elif mutation == "pot_mismatch":
        s = replace(s, pot_before=ChipAmount("261"))
    elif mutation == "bet_mismatch":
        s = replace(s, current_bet=ChipAmount("101"))
    elif mutation in ("unknown_fee", "other_fee"):
        s = replace(s, other_fees=None if mutation == "unknown_fee" else Decimal(1))
    elif mutation == "rake_unknown":
        s = replace(s, rules=rules(rake_distribution="unverified"))
    elif mutation == "live_rules":
        s = replace(s, rules=rules(verification_status="live_verified"))
    else:
        r = s.ranges[0]
        if mutation == "gto_source":
            r = replace(r, source_version="manual:GTO")
        elif mutation == "asset_source":
            r = replace(r, source_version="aa-ranges-v2:simulation_only")
        elif mutation == "confidence_claim":
            r = replace(r, confidence=1)
        elif mutation == "duplicate_combo":
            r = replace(r, combo_weights={"KhKd": Decimal(1), "KdKh": Decimal(1)})
        else:
            r = replace(r, combo_weights={"KhKd": Decimal(1), "KhKc": Decimal(1)})
            limit = 1
        s = replace(s, ranges=(r, *s.ranges[1:]))
    result = analyze_terminal_multiway(s, max_joint_assignments=limit)
    assert result.status == "BLOCKED"
    assert result.recommendation is None and result.call_net_ev is None


def test_result_cannot_be_promoted_to_live_advice():
    with pytest.raises(ValueError):
        TerminalAnalysis("COMPLETE_CONDITIONAL", strategy_eligible=True)


def test_allin_zero_contribution_cannot_be_a_ghost_contender():
    s = scenario()
    ss = list(s.seats)
    ss[3] = seat(3, 0)
    result = analyze_terminal_multiway(replace(
        s, seats=tuple(ss), pot_before=ChipAmount("240")))
    assert result.status == "BLOCKED"


def test_three_way_fractional_split_and_fold_refunds_are_audited():
    s = scenario()
    ss = list(s.seats)
    ss[3] = replace(ss[3], status=PlayerStatus.FOLDED)
    s = replace(s, seats=tuple(ss), ranges=s.ranges[:2],
                hero_cards=(card("3d"), card("5d")),
                board_cards=tuple(map(card, ("As", "Ks", "Qs", "Js", "Ts"))))
    result = analyze_terminal_multiway(s)
    assert result.status == "COMPLETE_CONDITIONAL"
    assert result.call_pots[0].hero_share == Fraction(1, 3)
    assert result.call_net_ev == 60
    assert result.fold_refunds == ((1, Decimal(40)),)
    assert (sum(p[1] for p in result.fold_pots)
            + sum(v for _, v in result.fold_refunds)) == 260
