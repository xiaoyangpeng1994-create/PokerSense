"""Multiway production equity must preserve counts, splits and uncertainty."""

from dataclasses import replace

import pytest

from poker_engine.core.enums import PlayerStatus, Position, Rank, Street, Suit
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.realtime.equity import (
    ExactRandomRangeEquity, MonteCarloRandomRangeEquity,
)


def card(code):
    return Card(Rank(code[0]), Suit(code[1]))


def state(count=8, hero=("As", "Ah"), board=()):
    return PokerState(
        state_version=1, hand_id="multiway-equity",
        street={0: Street.PREFLOP, 3: Street.FLOP,
                4: Street.TURN, 5: Street.RIVER}[len(board)],
        hero_cards=tuple(map(card, hero)), board_cards=tuple(map(card, board)),
        players=tuple(PlayerState(
            player_id=f"p{seat}", seat=seat, position=Position.UNKNOWN,
            stack=ChipAmount("200"), committed_this_street=ChipAmount("0"),
            committed_this_hand=ChipAmount("0"), status=PlayerStatus.ACTIVE,
            has_cards=True, is_hero=seat == 0, is_dealer=seat == 1,
        ) for seat in range(count)),
        pot=ChipAmount("3"), current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"), actor=0,
    )


def test_same_aces_distinguish_two_six_and_eight_active_players():
    engine = MonteCarloRandomRangeEquity(trials=4000, seed=17)
    results = [engine.compute(state(count)) for count in (2, 6, 8)]
    assert [r.opponent_count for r in results] == [1, 5, 7]
    assert results[0].win_rate > results[1].win_rate > results[2].win_rate
    assert 0.80 < results[0].expected_share < 0.90
    assert 0.30 < results[2].expected_share < 0.50
    assert all(r.samples == 4000 and r.standard_error > 0 for r in results)


@pytest.mark.parametrize("count", (2, 6, 7, 8))
def test_shared_royal_board_splits_among_every_active_player(count):
    result = MonteCarloRandomRangeEquity(trials=40).compute(
        state(count, hero=("2c", "3d"), board=("As", "Ks", "Qs", "Js", "Ts"))
    )
    assert result.win_rate == 0
    assert result.tie_rate == 1
    assert result.expected_share == pytest.approx(1 / count)
    assert result.standard_error == pytest.approx(0, abs=1e-8)


def test_folded_and_sitting_out_are_excluded_all_in_is_included():
    original = state()
    players = list(original.players)
    players[1] = replace(players[1], status=PlayerStatus.ALL_IN)
    for seat in range(2, 8):
        players[seat] = replace(players[seat], has_cards=False,
                                status=PlayerStatus.FOLDED if seat % 2
                                else PlayerStatus.SITTING_OUT)
    result = MonteCarloRandomRangeEquity(trials=50).compute(
        replace(original, players=tuple(players)))
    assert result.opponent_count == 1


def test_unknown_status_withholds_equity_instead_of_undercounting():
    original = state()
    players = (original.players[0], replace(original.players[1],
               status=PlayerStatus.UNKNOWN)) + original.players[2:]
    result = MonteCarloRandomRangeEquity().compute(replace(original, players=players))
    assert result.unavailable_reason == "player_status_unavailable"
    assert result.samples == 0


def test_identical_queries_are_repeatable_and_cache_ignores_pot_for_showdown():
    engine = MonteCarloRandomRangeEquity(trials=100)
    original = state()
    result = engine.compute(original)
    assert engine.compute(replace(original, pot=ChipAmount("99"))) is result
    assert MonteCarloRandomRangeEquity(trials=100).compute(original) == result
    assert engine.compute(state(6)).opponent_count == 5


def test_heads_up_monte_carlo_matches_independent_exact_enumeration():
    spot = state(2, hero=("As", "Kd"), board=("Qh", "Jd", "Tc", "2s", "7h"))
    mc = MonteCarloRandomRangeEquity(trials=12000, seed=3).compute(spot)
    exact = ExactRandomRangeEquity().compute(spot)
    assert mc.win_rate == pytest.approx(exact.win_rate, abs=0.012)
    assert mc.expected_share == pytest.approx(exact.expected_share, abs=0.012)


def test_exact_adapter_does_not_silently_reduce_multiway_to_heads_up():
    result = ExactRandomRangeEquity().compute(state(8))
    assert result.unavailable_reason == "exact_random_equity_requires_heads_up"
