"""Realtime end-to-end latency benchmark (Task 8).

Measures the realtime pipeline's per-stage latency:
    Capture -> Vision -> ChangeDetect+State -> Equity -> Total end-to-end

Two paths:
  - Monte Carlo (realtime default)
  - Exact enumeration (offline verification)

Reports p50 / p95 / mean per stage (milliseconds). The realtime success
criterion is p50 < 500ms AND p95 < 500ms on the Monte Carlo path.

Does NOT modify Frozen Core. Does NOT enter strategy recommendation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, "tools")
sys.path.insert(0, "tests")
sys.path.insert(0, "tests/realtime")

import run_benchmark  # noqa: E402
from gen_wepoker_dataset import render_table  # noqa: E402

from poker_engine.memory.hand_memory import InMemoryHandMemory  # noqa: E402
from poker_engine.orchestrator import ApplicationOrchestrator  # noqa: E402
from poker_engine.realtime import (  # noqa: E402
    ExactRandomRangeEquity,
    MonteCarloRandomRangeEquity,
)
from poker_engine.realtime.change_detector import detect_change  # noqa: E402
from poker_engine.state_engine import StateEngine  # noqa: E402
from profiles import relaxed_confidence_gate  # noqa: E402
from orchestrator.fixtures import initial_state  # noqa: E402


def _frame(bc, hc, pot, bet, stacks, actions, seq):
    img = render_table(tuple(bc), tuple(hc), pot, bet, stacks, actions)
    return run_benchmark._frame(img, seq)


# A full hand progression (preflop -> river). Each stage changes the state, so
# every frame triggers a real StateEngine + equity pass (worst case, not
# de-duplicated).
_SCENARIOS = (
    ([], ["AS", "KD"], "10", "5", ("100", "200", "300"), ("CHECK", "CALL")),
    (["QH", "JD", "TC"], ["AS", "KD"], "25", "10", ("100", "200", "300"),
     ("CHECK", "BET")),
    (["QH", "JD", "TC", "2S"], ["AS", "KD"], "50", "20",
     ("150", "250", "350"), ("BET", "CALL")),
    (["QH", "JD", "TC", "2S", "7H"], ["AS", "7H"], "100", "40",
     ("200", "300", "400"), ("BET", "FOLD")),
)


def _build_pipeline(equity_strategy, scenarios):
    orch = ApplicationOrchestrator(
        StateEngine(), InMemoryHandMemory(),
        confidence_gate=relaxed_confidence_gate(),
    )
    orch.start_hand(initial_state(hand_id="h1"))
    frames = tuple(
        _frame(bc, hc, pot, bet, stacks, actions, i)
        for i, (bc, hc, pot, bet, stacks, actions) in enumerate(scenarios)
    )
    vision = run_benchmark.build_engine()
    table_map = run_benchmark.table_map()
    return orch, frames, vision, table_map, equity_strategy


# Exact enumeration is only tractable on the RIVER (board full -> no board
# completions). Monte Carlo runs the full progression.
_RIVER_ONLY = (_SCENARIOS[-1],)


def _run_path(equity_strategy, scenarios, repeats: int) -> dict[str, list[float]]:
    """Run the pipeline `repeats` times, recording per-stage latency (ms)."""
    capture_ms, vision_ms, state_ms, equity_ms, total_ms = [], [], [], [], []

    for _ in range(repeats):
        orch, frames, vision, table_map, eq = _build_pipeline(
            equity_strategy, scenarios
        )
        prev_obs = None
        for frame in frames:
            t_total = time.perf_counter()

            t = time.perf_counter()
            # capture is synthetic (no-op), but timed uniformly
            _ = frame
            capture_ms.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            obs = vision.process(frame, table_map)
            vision_ms.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            if prev_obs is not None and detect_change(prev_obs, obs).changed:
                orch.process_observation(obs)
            elif prev_obs is None:
                orch.process_observation(obs)
            state_ms.append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            active = orch._hand_memory.active_hand_id
            state = orch._hand_memory.latest_state(active)
            eq.compute(state)
            equity_ms.append((time.perf_counter() - t) * 1000)

            total_ms.append((time.perf_counter() - t_total) * 1000)
            prev_obs = obs

    return {
        "capture": capture_ms,
        "vision": vision_ms,
        "state": state_ms,
        "equity": equity_ms,
        "total": total_ms,
    }


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _summarize(values: list[float]) -> dict:
    s = sorted(values)
    return {
        "p50_ms": round(_pct(s, 50), 3),
        "p95_ms": round(_pct(s, 95), 3),
        "mean_ms": round(statistics.mean(s), 3),
        "n": len(s),
    }


def run_latency_benchmark(repeats: int) -> dict:
    mc_equity = MonteCarloRandomRangeEquity(trials=2000, seed=0)
    exact_equity = ExactRandomRangeEquity()

    mc = _run_path(mc_equity, _SCENARIOS, repeats)
    exact = _run_path(exact_equity, _RIVER_ONLY, repeats)

    report = {
        "note": (
            "Realtime pipeline latency. Realtime criterion: Monte Carlo path "
            "p50 < 500ms AND p95 < 500ms."
        ),
        "repeats": repeats,
        "monte_carlo": {k: _summarize(v) for k, v in mc.items()},
        "exact_enumeration": {k: _summarize(v) for k, v in exact.items()},
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = run_latency_benchmark(args.repeats)

    out = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)

    # verdict
    mc_total = report["monte_carlo"]["total"]
    ok = mc_total["p50_ms"] < 500 and mc_total["p95_ms"] < 500
    print(f"\nRealtime (MC) p50={mc_total['p50_ms']}ms "
          f"p95={mc_total['p95_ms']}ms -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
