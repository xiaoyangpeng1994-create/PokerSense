"""Reviewed completion is causal, conserved, and distinct from cash/fee inference."""

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import pytest

pytest.importorskip("pokerkit")

from poker_engine.core.enums import Rank, Suit  # noqa: E402
from poker_engine.core.value_objects import Card, ChipAmount  # noqa: E402
from poker_engine.state_engine.reviewed_replay import replay_reviewed_hand  # noqa: E402
from poker_engine.state_engine.reviewed_completion import (  # noqa: E402
    complete_reviewed_hand, independent_awards,
)
from .test_reviewed_replay import synthetic_side_pot_hand  # noqa: E402


@pytest.fixture
def case():
    root = Path(__file__).resolve().parents[1] / "fixtures/wpk_reference_hands"
    return tuple(json.loads((root / name).read_text(encoding="utf-8")) for name in (
        "aq_allin_v1.json", "aq_completion_v1.json"))


def test_completes_real_reference_gross_then_observed_cash(case):
    trace, review = case
    original = deepcopy(case)
    result = complete_reviewed_hand(trace, review)
    assert case == original
    assert result.betting_steps == replay_reviewed_hand(trace)
    assert result.pokerkit_hand_ended and result.all_cash_balances_observed
    assert dict(result.gross_awards)[0] == 346
    assert dict(result.gross_awards)[6] == 0  # refund is not a pot award
    assert result.unallocated_outflows == ((0, Decimal(22)),)
    award = next(s for s in result.completion_steps if s.kind == "gross_award")
    assert award.known_by_frame == 8760
    assert award.cash_observed_seats == ()
    assert next(p.stack.value for p in award.state.players if p.is_hero) == 346
    last = result.completion_steps[-1]
    assert next(p.stack.value for p in last.state.players if p.is_hero) == 324
    assert last.state.pot.value == 0 and last.state.actor is None
    assert not result.fee_components_verified and not result.capture_to_state_verified
    all_steps = result.betting_steps + result.completion_steps
    assert [s.state.state_version for s in all_steps] == list(range(len(all_steps)))
    for step in result.completion_steps:
        accounted = (
            sum(p.stack.value for p in step.state.players)
            + step.state.pot.value
            + sum(value for _, value in step.unallocated_outflows)
        )
        assert accounted == 2644


def test_cash_observation_does_not_double_count_and_notice_does_not_debit(case):
    trace, review = case
    review["insurance_notices"][0]["amount"] = "999"
    review["cash_observations"].append(deepcopy(review["cash_observations"][-1]))
    result = complete_reviewed_hand(trace, review)
    assert result.unallocated_outflows == ((0, Decimal(22)),)
    hero_stack = next(p.stack.value
                      for p in result.completion_steps[-1].state.players if p.is_hero)
    assert hero_stack == 324


def test_incomplete_cash_coverage_stays_explicit(case):
    trace, review = case
    for observation in review["cash_observations"]:
        observation["balances"].pop("6", None)
    result = complete_reviewed_hand(trace, review)
    assert not result.all_cash_balances_observed
    assert 6 not in result.completion_steps[-1].cash_observed_seats


@pytest.mark.parametrize("mutation", [
    "early_holes", "next_ante", "external_income", "duplicate_cards",
    "multiple_boards", "missing_card",
])
def test_unsafe_completion_is_rejected(case, mutation):
    trace, review = case
    if mutation == "early_holes":
        review["evidence"]["show"]["source_frame"] = 8680
    elif mutation == "next_ante":
        review["cash_observations"][-1]["evidence"] = "next_post"
    elif mutation == "external_income":
        review["cash_observations"][0]["balances"]["0"] = "350"
    elif mutation == "duplicate_cards":
        trace["cards"][1]["cards"] = ["Ac", "Ah"]
    elif mutation == "multiple_boards":
        review["runout_count"] = 2
    else:
        trace["cards"] = trace["cards"][:1]
    with pytest.raises(ValueError):
        complete_reviewed_hand(trace, review)


def synthetic_completion(split=False):
    trace = synthetic_side_pot_hand(6)
    cards = {0: ["Kc", "Kd"], 1: ["Qc", "Qd"], 2: ["Ac", "Ad"]}
    board = ["2c", "3d", "7h", "8s", "9c"]
    balances = {"0": "120", "1": "0", "2": "60", "3": "400", "4": "400", "5": "400"}
    if split:
        cards = {0: ["2c", "2d"], 1: ["3c", "3d"], 2: ["4c", "4d"]}
        board = ["As", "Ks", "Qs", "Js", "Ts"]
        balances.update({"0": "100", "1": "60", "2": "20"})
    trace["cards"] = [{"seat": seat, "cards": value,
                       "first_verified_visible_frame": 0 if seat == 0 else 40}
                      for seat, value in cards.items()]
    trace["board"] = [{"street": street, "cards": board[:count],
                       "first_verified_visible_frame": frame}
                      for street, count, frame in (
                          ("flop", 3, 50), ("turn", 4, 60), ("river", 5, 70))]
    review = {"hand_id": trace["hand_id"], "runout_count": 1,
              "single_board_reviewed": True, "chip_unit": "1",
              "reveal_evidence": {str(seat): "show" for seat in cards},
              "board_evidence": {street: street
                                 for street in ("flop", "turn", "river")},
              "cash_observations": [{"evidence": "cash", "balances": balances}],
              "next_hand_first_post_evidence": "next", "evidence": {
                  name: {"source_frame": frame} for name, frame in (
                      ("show", 40), ("flop", 50), ("turn", 60), ("river", 70),
                      ("cash", 80), ("next", 100))}}
    return trace, review


def test_main_and_side_pots_can_have_different_winners():
    result = complete_reviewed_hand(*synthetic_completion())
    assert {seat: amount for seat, amount in result.gross_awards if amount} == {
        0: Decimal(80), 2: Decimal(60)}
    assert result.unallocated_outflows == ()
    assert result.all_cash_balances_observed


def test_exact_split_pots_are_independently_checked():
    result = complete_reviewed_hand(*synthetic_completion(split=True))
    assert dict(result.gross_awards) == {0: 60, 1: 60, 2: 20, 3: 0, 4: 0, 5: 0}


def test_odd_chip_allocation_is_not_invented():
    trace, _ = synthetic_completion(split=True)
    base = replay_reviewed_hand(trace)[-1].state
    folded = base.players[3]
    modified = replace(folded, stack=ChipAmount(folded.stack.value - 1),
                       committed_this_hand=ChipAmount(1))
    base = replace(base, pot=ChipAmount(141), players=tuple(
        modified if p.seat == 3 else p for p in base.players))
    holes = {row["seat"]: tuple(Card(Rank(c[0]), Suit(c[1])) for c in row["cards"])
             for row in trace["cards"]}
    board = tuple(Card(Rank(c[0]), Suit(c[1])) for c in trace["board"][-1]["cards"])
    with pytest.raises(ValueError, match="odd-chip"):
        independent_awards(base, holes, board, Decimal(1))
