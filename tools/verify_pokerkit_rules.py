"""Independent open-source oracle for the simulated WPK ante/straddle rules.

This proves standard UTG-straddle semantics in PokerKit, not the actual WPK
platform's special rules or production StateEngine integration.
"""

import importlib.metadata
import json

from pokerkit import Automation, Mode, NoLimitTexasHoldem


def main():
    rows = []
    for count in (6, 7, 8):
        state = NoLimitTexasHoldem.create_state(
            (Automation.ANTE_POSTING, Automation.BET_COLLECTION,
             Automation.BLIND_OR_STRADDLE_POSTING),
            # This scenario explicitly assumes a live straddle establishes
            # an 8-chip minimum opening raise increment. PokerKit separately
            # parameterizes min_bet: using 4 here would permit raising to 12.
            True, 2, (2, 4, 8), 8, (400,) * count, count,
            mode=Mode.CASH_GAME,
        )
        while state.can_deal_hole():
            state.deal_hole("????")
        row = {"players": count, "ante": 2, "blinds_and_straddle": [2, 4, 8],
               "assumed_minimum_raise_increment": 8,
               "initial_pot": state.total_pot_amount,
               "actor_index": state.actor_index,
               "to_call": state.checking_or_calling_amount,
               "minimum_raise_to": state.min_completion_betting_or_raising_to_amount}
        assert state.total_pot_amount == count * 2 + 2 + 4 + 8
        assert state.actor_index == 3
        assert state.checking_or_calling_amount == 8
        assert state.min_completion_betting_or_raising_to_amount == 16
        rows.append(row)
    print(json.dumps({"pokerkit": importlib.metadata.version("pokerkit"),
                      "scope": __doc__, "standard_straddle_cases": rows}, indent=2))


if __name__ == "__main__":
    main()
