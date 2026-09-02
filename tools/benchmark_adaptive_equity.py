"""Repeatable target-hardware benchmark for adaptive multiway equity budgets."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from decimal import Decimal
from itertools import combinations

from poker_engine.core.enums import Rank, Suit
from poker_engine.core.value_objects import Card, ChipAmount
from poker_engine.equity._deck import remaining_deck
from poker_engine.strategy.contracts import PotState
from poker_engine.strategy.multiway_equity import (
    exact_multiway_pot_share,
    monte_carlo_multiway_pot_share,
)
from poker_engine.strategy.range_tracker import JointRangeAssignment


def _card(value: str) -> Card:
    return Card(Rank(value[0]), Suit(value[1]))


HERO = (_card("As"), _card("Kd"))
FLOP = (_card("2c"), _card("7d"), _card("Jh"))
POT = (PotState("main", ChipAmount("12"), (0, 1, 2)),)


def _assignments(player_count: int, count: int) -> tuple[JointRangeAssignment, ...]:
    cards_needed = 2 * (player_count - 1)
    deck = remaining_deck(HERO + FLOP)
    values = []
    for cards in combinations(deck, cards_needed):
        holdings = {
            seat: (cards[offset], cards[offset + 1])
            for seat, offset in zip(range(player_count - 1), range(0, cards_needed, 2))
        }
        values.append(JointRangeAssignment(holdings, Decimal("1")))
        if len(values) == count:
            break
    return tuple(values)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))
    return ordered[index]


def _measure(name: str, operation, units: int, repeats: int) -> dict[str, object]:
    operation()
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        elapsed.append((time.perf_counter() - started) * 1000)
    return {
        "name": name,
        "units": units,
        "repeats": repeats,
        "latency_ms": {
            "median": round(statistics.median(elapsed), 3),
            "p95": round(_percentile(elapsed, 0.95), 3),
            "maximum": round(max(elapsed), 3),
        },
        "units_per_ms_at_p95": round(units / _percentile(elapsed, 0.95), 3),
    }


def run_benchmark(repeats: int, hardware_label: str) -> dict[str, object]:
    hu_eight = _assignments(2, 8)
    three_way_eight = _assignments(3, 8)
    exact_units = len(hu_eight) * 990
    mc_trials = 10_000
    cases = [
        _measure(
            "exact_hu_flop_8_assignments",
            lambda: exact_multiway_pot_share(
                1, HERO, hu_eight, FLOP,
                (PotState("main", ChipAmount("12"), (0, 1)),),
                max_outcomes=exact_units,
            ),
            exact_units,
            repeats,
        ),
        _measure(
            "mc_hu_flop_10000",
            lambda: monte_carlo_multiway_pot_share(
                1, HERO, hu_eight, FLOP,
                (PotState("main", ChipAmount("12"), (0, 1)),),
                trials=mc_trials,
                seed=42,
            ),
            mc_trials,
            repeats,
        ),
        _measure(
            "mc_3way_flop_10000",
            lambda: monte_carlo_multiway_pot_share(
                2, HERO, three_way_eight, FLOP, POT,
                trials=mc_trials,
                seed=42,
            ),
            mc_trials,
            repeats,
        ),
    ]
    exact_rate = cases[0]["units_per_ms_at_p95"]
    mc_rate = min(item["units_per_ms_at_p95"] for item in cases[1:])
    return {
        "schema_version": 1,
        "benchmark": "adaptive-equity-target-calibration",
        "hardware_label": hardware_label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cases": cases,
        "recommended_conservative_policy": {
            "exact_outcomes_per_ms": max(1, int(exact_rate * 0.5)),
            "mc_trials_per_ms": max(1, int(mc_rate * 0.5)),
            "safety_factor": 0.5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hardware-label", required=True)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be > 0")
    print(json.dumps(
        run_benchmark(args.repeats, args.hardware_label),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
