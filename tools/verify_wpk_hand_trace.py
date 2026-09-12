"""Audit a redacted real-video hand with PokerKit and the project math core.

Card visibility is source-frame bound. Villain cards revealed after an all-in
are valid for retrospective equity checks, never for preceding decisions.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from poker_engine.core.enums import PlayerStatus, Position, Rank, Suit
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity.enumeration import EnumerationEquity
from poker_engine.strategy.contracts import DecisionSeat
from poker_engine.strategy.state import calculate_side_pots
from tools.capture_card_calibration.hashing import sha256_file
from tools.wpk_video_dataset import pixels_digest, read_image


DEFAULT_TRACE = (Path(__file__).resolve().parents[1]
                 / "tests/fixtures/wpk_reference_hands/aq_allin_v1.json")


def visible_hole_cards(trace: dict, source_frame: int) -> dict[int, tuple[str, ...]]:
    return {row["seat"]: tuple(row["cards"]) for row in trace["cards"]
            if row["first_verified_visible_frame"] <= source_frame}


def visible_board(trace: dict, source_frame: int) -> tuple[str, ...]:
    boards = [row for row in trace["board"]
              if row["first_verified_visible_frame"] <= source_frame]
    return tuple(boards[-1]["cards"]) if boards else ()


def verify_source_binding(trace: dict, window: Path) -> int:
    summary = json.loads((window / "summary.json").read_text(encoding="utf-8"))
    if summary["source_sha256"] != trace["source_video_sha256"]:
        raise ValueError("trace source does not match the extracted window")
    for row in trace["evidence"].values():
        path = window / "frames" / f"{row['source_frame']:06d}.png"
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("evidence PNG hash mismatch")
        if pixels_digest(read_image(path)) != row["pixel_sha256"]:
            raise ValueError("evidence pixel identity mismatch")
    return len(trace["evidence"])


def verify_betting(trace: dict) -> dict:
    from pokerkit import Automation, Mode, NoLimitTexasHoldem
    rules = trace["rules"]
    order = trace["pokerkit_index_to_source_seat"]
    seats = {row["seat"]: row for row in trace["source_seats"]}
    state = NoLimitTexasHoldem.create_state(
        (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
         Automation.BLIND_OR_STRADDLE_POSTING),
        True, rules["ante_each"], (rules["small_blind"], rules["big_blind"]),
        rules["big_blind"], tuple(seats[seat]["starting_stack"] for seat in order),
        rules["player_count"], mode=Mode.CASH_GAME,
    )
    for _ in order:
        state.deal_hole("????")  # no future villain cards in betting state
    assert state.total_pot_amount == 22
    street = "preflop"
    board_count = 0
    rows = []
    for action in trace["actions"]:
        if action["street"] != street:
            cards = next(row["cards"] for row in trace["board"]
                         if row["street"] == action["street"])
            # The video never identifies burn cards. Random automatic burns
            # can remove a later observed board card and make this oracle
            # warn nondeterministically. Preserve the burn as unknown.
            state.burn_card("??")
            state.deal_board("".join(cards[board_count:]))
            board_count = len(cards)
            street = action["street"]
        actor = state.actor_index
        assert order[actor] == action["source_seat"]
        before = state.stacks[actor]
        if action["kind"] == "call":
            assert state.checking_or_calling_amount == action["amount"]
            state.check_or_call()
        elif action["kind"] == "fold":
            state.fold()
        elif action["kind"] in ("raise_to", "bet_to"):
            state.complete_bet_or_raise_to(action["amount"])
        else:
            raise ValueError("unsupported trace action")
        rows.append({"order": action["order"], "source_seat": order[actor],
                     "kind": action["kind"], "paid": before - state.stacks[actor],
                     "total_pot": state.total_pot_amount})
    assert rows[8]["total_pot"] == 94
    assert rows[10]["total_pot"] == 218
    assert rows[11]["total_pot"] == 520
    assert rows[12]["total_pot"] == 346
    pots = list(state.pots)
    assert len(pots) == 1 and pots[0].unraked_amount == 346
    assert tuple(order[i] for i in pots[0].player_indices) == (6, 0)
    assert state.stacks[order.index(6)] == 238
    return {"passed": True, "actions": rows, "contestable_pot": 346,
            "uncalled_return": 238, "actor_order_basis": "rules_constrained"}


def verify_project_math(trace: dict) -> dict:
    seats = tuple(DecisionSeat(
        seat_id=row["seat"], player_id=None,
        position=getattr(Position, row["position"]),
        stack=ChipAmount(0), street_committed=ChipAmount(
            64 if row["seat"] == 0 else 302 if row["seat"] == 6 else 0),
        hand_committed=ChipAmount(row["hand_contributed"]),
        status=PlayerStatus.ALL_IN if row["seat"] in (0, 6) else PlayerStatus.FOLDED,
        is_hero=row["seat"] == 0,
    ) for row in trace["source_seats"])
    result = calculate_side_pots(seats)
    assert len(result.pots) == 1
    assert result.pots[0].amount.value == 346
    assert result.pots[0].eligible_seats == (0, 6)
    assert result.uncalled_returns[6].value == 238

    def cards(values):
        return tuple(Card(Rank(code[0]), Suit(code[1])) for code in values)

    revealed = visible_hole_cards(trace, 8700)
    assert set(visible_hole_cards(trace, 8660)) == {0}
    turn = EnumerationEquity().estimate(
        cards(revealed[0]), (cards(revealed[6]),), cards(visible_board(trace, 8700)))
    assert turn.samples == 44 and turn.tie == 0
    assert round(turn.win * turn.samples) == 37
    river = EnumerationEquity().estimate(
        cards(revealed[0]), (cards(revealed[6]),), cards(visible_board(trace, 8760)))
    assert river.win == 1
    return {"passed": True, "contestable_pot": 346, "uncalled_return": 238,
            "post_reveal_turn_equity": "37/44",
            "post_reveal_turn_equity_decimal": turn.win,
            "river_hero_wins": True,
            "call_threshold_before_fees": str(Fraction(64, 346)),
            "wrong_threshold_if_raw_total_used": str(Fraction(64, 584)),
            "prior_decision_equity_not_established": True}


def audit_trace(trace: dict, window: Path | None = None) -> dict:
    bound = verify_source_binding(trace, window) if window is not None else None
    betting = verify_betting(trace)
    math = verify_project_math(trace)
    settlement = trace["settlement"]
    assert settlement["gross_contributions"] == 346 + 238
    assert settlement["contestable_pot"] - settlement["hero_received"] == 22
    assert (
        settlement["hero_received"] - trace["source_seats"][0]["starting_stack"] == 162
    )
    return {"hand_id": trace["hand_id"], "source_evidence_frames_verified": bound,
            "betting": betting, "project_math": math,
            "settlement_status": "PARTIAL", "observed_deduction": 22,
            "displayed_insurance": 12, "remaining_if_insurance_debited": 10,
            "fee_model_verified": False, "capture_to_state_verified": False,
            "strategy_golden_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--window", type=Path)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    print(json.dumps(audit_trace(trace, args.window), indent=2))
