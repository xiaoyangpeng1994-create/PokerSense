"""Executable NLHE rule examples, independently evaluated by PokerKit.

These verify the rulebook's stated scenarios, not production WPK parity.
Cash-game house rules, straddle position and rake rounding remain configurable.
"""

from __future__ import annotations

import importlib.metadata
import json

from pokerkit import Automation, Mode, NoLimitTexasHoldem, rake


def new_hand(count=6, *, ante=0, straddle=False, stacks=None):
    state = NoLimitTexasHoldem.create_state(
        (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
         Automation.BLIND_OR_STRADDLE_POSTING),
        True, ante, (2, 4, 8) if straddle else (2, 4),
        8 if straddle else 4, stacks or (400,) * count, count,
        mode=Mode.CASH_GAME,
    )
    for _ in range(count):
        state.deal_hole("????")
    return state


def verify_rulebook() -> dict:
    cases = []
    for count in (6, 7, 8):
        for straddle in (False, True):
            state = new_hand(count, ante=2, straddle=straddle)
            expected_pot = count * 2 + (14 if straddle else 6)
            expected_call = 8 if straddle else 4
            assert state.total_pot_amount == expected_pot
            assert state.actor_index == (3 if straddle else 2)
            assert state.checking_or_calling_amount == expected_call
            assert (
                state.min_completion_betting_or_raising_to_amount == expected_call * 2
            )
            cases.append({"id": f"POST-{count}-{'STRADDLE' if straddle else 'NORMAL'}",
                          "pot": state.total_pot_amount, "actor": state.actor_index,
                          "call": expected_call, "minimum_raise_to": expected_call * 2})
    for all_in, reopens in ((15, False), (20, True)):
        state = new_hand(stacks=(400, 400, 400, all_in, 400, 400))
        state.complete_bet_or_raise_to(12)  # full increment 12 - 4 = 8
        state.complete_bet_or_raise_to(all_in)
        state.check_or_call()  # keep another non-all-in player in the pot
        for _ in range(3):
            state.fold()
        assert state.actor_index == 2
        assert state.checking_or_calling_amount == all_in - 12
        assert state.can_complete_bet_or_raise_to() is reopens
        assert state.min_completion_betting_or_raising_to_amount == (
            28 if reopens else None
        )
        cases.append({"id": f"REOPEN-{all_in}", "original_raise_to": 12,
                      "all_in_to": all_in, "call_increment": all_in - 12,
                      "raising_reopened": reopens})
    state = new_hand(stacks=(100, 60, 20, 400, 400, 400))
    state.complete_bet_or_raise_to(20)
    for _ in range(3):
        state.fold()
    state.complete_bet_or_raise_to(100)
    state.check_or_call()
    pots = list(state.pots)
    assert [pot.unraked_amount for pot in pots] == [60, 80]
    assert [pot.player_indices for pot in pots] == [(0, 1, 2), (0, 1)]
    assert state.stacks[0] == 40  # unmatched 40 returned
    assert state.total_pot_amount + state.stacks[0] == 180
    cases.append({"id": "SIDE-POTS-100-60-20", "pots": [60, 80],
                  "eligible_seats": [[0, 1, 2], [0, 1]], "uncalled_return": 40})
    for amount, expected in ((200, (6, 194)), (500, (8, 492))):
        assert rake(amount, percentage=0.03, cap=8) == expected
        cases.append({"id": f"RAKE-{amount}", "raked": expected[0], "net": expected[1]})
    state = new_hand()
    assert rake(200, state, percentage=0.03, cap=8, no_flop_no_drop=True) == (0, 200)
    cases.append({"id": "NO-FLOP-NO-DROP", "raked": 0,
                  "note": "Only when explicitly selected as the room rule"})
    return {"pokerkit_version": importlib.metadata.version("pokerkit"),
            "passed": True, "cases": cases,
            "production_wpk_parity": "NOT_YET_VERIFIED"}


if __name__ == "__main__":
    print(json.dumps(verify_rulebook(), indent=2))
