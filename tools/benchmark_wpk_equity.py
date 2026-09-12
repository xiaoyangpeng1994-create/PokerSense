"""Cold/cached 6–8 player equity benchmark; excludes capture, strategy and UI."""

import argparse
from dataclasses import asdict
import json
import statistics
import time

from poker_engine.core.enums import Street
from poker_engine.realtime.equity import MonteCarloRandomRangeEquity
from tools.wpk_demo import example_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--evaluator", choices=("auto", "python", "phevaluator"),
                        default="auto")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    rows = []
    for count in (6, 7, 8):
        for street in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER):
            state = example_state(count, street)
            timings = []
            for _ in range(args.repeats):
                engine = MonteCarloRandomRangeEquity(
                    args.trials, seed=0, evaluator=args.evaluator)
                started = time.perf_counter()
                result = engine.compute(state)
                timings.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            assert engine.compute(state) is result
            cached_ms = (time.perf_counter() - started) * 1000
            rows.append({"players": count, "street": street.value,
                         "evaluator": engine.evaluator_backend,
                         "cold_median_ms": round(statistics.median(timings), 3),
                         "cold_max_ms": round(max(timings), 3),
                         "cached_ms": round(cached_ms, 3), "result": asdict(result)})
    print(json.dumps({"note": __doc__, "repeats": args.repeats,
                      "scenarios": rows}, indent=2))


if __name__ == "__main__":
    main()
