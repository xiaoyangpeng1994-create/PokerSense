"""Local WPK UI simulation. Uses no capture device and produces no actions.

Run with PYTHONPATH=src python -m tools.wpk_demo --port 8876.
The same settings API/UI is used as production; set POKERSENSE_SETTINGS_PATH
and POKERSENSE_TABLE_RULES_PATH to temporary files when testing preferences.
"""

from __future__ import annotations

import argparse
import asyncio

from poker_engine.core.enums import PlayerStatus, Position, Rank, Street, Suit
from poker_engine.core.opponents import PlayerState
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.desktop.serialize import DesktopFrame
from poker_engine.desktop.server import create_app
from poker_engine.desktop.live import _resource_root
from poker_engine.desktop.table_rules import load_user_table_rules, table_rules_revision
from poker_engine.realtime.analysis import (
    ConfidenceSnapshot, RealtimeAnalysis, StateSnapshot,
)
from poker_engine.realtime.equity import MonteCarloRandomRangeEquity


def example_state(count: int = 8, street: Street = Street.FLOP) -> PokerState:
    board = ("Qh", "Jd", "Tc", "2s", "7h")
    board = board[:{Street.PREFLOP: 0, Street.FLOP: 3,
                    Street.TURN: 4, Street.RIVER: 5}[street]]

    def card(code):
        return Card(Rank(code[0]), Suit(code[1]))

    return PokerState(
        state_version=1, hand_id="SIMULATION-wpk", street=street,
        hero_cards=(card("As"), card("Kd")), board_cards=tuple(map(card, board)),
        players=tuple(PlayerState(
            player_id=f"simulation-{seat}", seat=seat, position=Position.UNKNOWN,
            stack=ChipAmount("400"), committed_this_hand=ChipAmount("0"),
            committed_this_street=ChipAmount("0"), status=PlayerStatus.ACTIVE,
            has_cards=True, is_hero=seat == 0, is_dealer=seat == 1,
        ) for seat in range(count)),
        pot=ChipAmount("60"), current_bet=ChipAmount("0"),
        to_call=ChipAmount("0"), actor=0,
    )


async def demo_stream():
    engine = MonteCarloRandomRangeEquity()
    seq = 0
    while True:
        rules = load_user_table_rules(
            _resource_root() / "configs/game/wpk-capture-card.json")
        state = example_state(rules.get("table_size", 8))
        equity = engine.compute(state)
        analysis = RealtimeAnalysis(
            seq, StateSnapshot.from_state(state), equity,
            ConfidenceSnapshot(1.0, tuple((field, "valid") for field in (
                "hero_cards", "board_cards", "street", "pot", "stacks", "action"))),
        )
        yield DesktopFrame(
            analysis, advice_unavailable_reason="simulation_preview",
            table_rules_revision=table_rules_revision(rules),
        )
        seq += 1
        await asyncio.sleep(1)


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    uvicorn.run(create_app(demo_stream), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
