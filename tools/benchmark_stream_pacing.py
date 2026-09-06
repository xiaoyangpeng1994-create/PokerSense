"""End-to-end throughput of the live capture loop (realtime migration S1).

Complements ``benchmark_realtime_latency.py``, which times a single pipeline
step in isolation. This one drives the whole ``live_analysis_stream`` loop —
the thing a UI actually consumes — and reports the frame rate it achieves.

It also projects what the *previous* fixed-sleep loop would have achieved, so
the gain from event-driven pacing is visible rather than asserted.

    python tools/benchmark_stream_pacing.py --frames 20 --capture-ms 100

Does NOT modify Frozen Core. Does NOT enter strategy recommendation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

sys.path.insert(0, "tools")
sys.path.insert(0, "tests")
sys.path.insert(0, "tests/realtime")

import run_benchmark  # noqa: E402
from gen_wepoker_dataset import render_table  # noqa: E402

from poker_engine.desktop import live  # noqa: E402
from poker_engine.memory.hand_memory import InMemoryHandMemory  # noqa: E402
from poker_engine.orchestrator import ApplicationOrchestrator  # noqa: E402
from poker_engine.realtime import RealtimePipeline  # noqa: E402
from poker_engine.state_engine import StateEngine  # noqa: E402

from profiles import relaxed_confidence_gate  # noqa: E402
from orchestrator.fixtures import initial_state  # noqa: E402

# The fixed sleep the loop used to perform after every single step,
# regardless of how long the step actually took.
LEGACY_FIXED_SLEEP_S = 1.0

_SCENARIOS = (
    ([], ["AS", "KD"], "10", "5", ("100", "200", "300"), ("CHECK", "CALL")),
    (["QH", "JD", "TC"], ["AS", "KD"], "25", "10", ("100", "200", "300"),
     ("CHECK", "BET")),
    (["QH", "JD", "TC", "2S"], ["AS", "KD"], "50", "20",
     ("150", "250", "350"), ("BET", "CALL")),
    (["QH", "JD", "TC", "2S", "7H"], ["AS", "7H"], "100", "40",
     ("200", "300", "400"), ("BET", "FOLD")),
)


class _TimedSource:
    """Replay the scenario frames, paying ``capture_ms`` per pull.

    A real device pull is not free (ADB screencap ~100ms, UVC capture card
    ~100ms at its true ~10fps), so the loop must be measured with that cost
    included or the numbers flatter the design.
    """

    def __init__(self, frames, capture_ms: float):
        self._frames = list(frames)
        self._capture_ms = capture_ms
        self._index = 0

    def next_frame(self):
        if self._capture_ms:
            time.sleep(self._capture_ms / 1000.0)
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame


def _frames(count: int):
    out = []
    for seq in range(count):
        bc, hc, pot, bet, stacks, actions = _SCENARIOS[seq % len(_SCENARIOS)]
        img = render_table(tuple(bc), tuple(hc), pot, bet, stacks, actions)
        out.append(run_benchmark._frame(img, seq))
    return tuple(out)


def _new_hand_state_factory():
    """Fresh state per hand — the scenario cycle crosses a hand boundary."""
    counter = {"n": 0}

    def factory():
        counter["n"] += 1
        return initial_state(hand_id=f"bench-{counter['n']}")

    return factory


def _pipeline(source, started_at) -> RealtimePipeline:
    orch = ApplicationOrchestrator(
        StateEngine(),
        InMemoryHandMemory(),
        confidence_gate=relaxed_confidence_gate(),
    )
    # Synthetic frames carry their own (fixed) timestamps; the hand must open
    # at or before them or the lifecycle check rejects the next-hand swap.
    orch.start_hand(initial_state(hand_id="bench"), started_at=started_at)
    return RealtimePipeline(
        frame_source=source,
        vision=run_benchmark.build_engine(),
        table_map=run_benchmark.table_map(),
        orchestrator=orch,
        equity_trials=2000,
        equity_seed=7,
        new_hand_state_factory=_new_hand_state_factory(),
    )


async def _measure_stream(pipeline, frame_count: int, interval: float) -> dict:
    live.build_pipeline = lambda *a, **k: pipeline
    gaps = []
    last = None
    started = time.perf_counter()
    seen = 0
    async for _frame in live.live_analysis_stream(interval_seconds=interval):
        now = time.perf_counter()
        if last is not None:
            gaps.append((now - last) * 1000.0)
        last = now
        seen += 1
        if seen >= frame_count:
            break
    elapsed = time.perf_counter() - started
    return {
        "frames": seen,
        "elapsed_ms": round(elapsed * 1000.0, 3),
        "achieved_fps": round(seen / elapsed, 3) if elapsed > 0 else 0.0,
        "inter_frame_ms": _summarize(gaps),
    }


def _measure_step_cost(pipeline) -> dict:
    """Cost of one step, measured directly (no loop pacing involved)."""
    costs = []
    for _ in range(5):
        started = time.perf_counter()
        pipeline.step()
        costs.append((time.perf_counter() - started) * 1000.0)
    return _summarize(costs)


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "n": 0}
    ordered = sorted(values)

    def pct(p):
        idx = int(round((p / 100.0) * (len(ordered) - 1)))
        return round(ordered[idx], 3)

    return {
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "mean_ms": round(statistics.mean(ordered), 3),
        "n": len(ordered),
    }


def run_pacing_benchmark(
    frame_count: int, capture_ms: float, interval: float
) -> dict:
    frames = _frames(frame_count)
    # Step cost is measured on its own pipeline so the stream measurement is
    # not polluted by that pipeline's own frame budget.
    step = _measure_step_cost(
        _pipeline(_TimedSource(frames, 0.0), frames[0].timestamp)
    )
    stream = asyncio.run(
        _measure_stream(
            _pipeline(
                _TimedSource(_frames(frame_count), capture_ms),
                frames[0].timestamp,
            ),
            frame_count,
            interval,
        )
    )
    legacy_period_ms = step["mean_ms"] + LEGACY_FIXED_SLEEP_S * 1000.0
    return {
        "note": (
            "Live loop throughput. 'legacy_fps' is what a fixed 1.0s sleep "
            "after every step would have achieved at the same step cost."
        ),
        "settings": {
            "frame_count": frame_count,
            "simulated_capture_ms": capture_ms,
            "interval_seconds": interval,
        },
        "step_cost": step,
        "stream": stream,
        "legacy_projection": {
            "period_ms": round(legacy_period_ms, 3),
            "fps": round(1000.0 / legacy_period_ms, 3),
        },
        "speedup_vs_legacy": (
            round(stream["achieved_fps"] / (1000.0 / legacy_period_ms), 2)
            if legacy_period_ms > 0
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--capture-ms", type=float, default=100.0)
    ap.add_argument("--interval", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = run_pacing_benchmark(args.frames, args.capture_ms, args.interval)
    out = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)

    stream = report["stream"]
    legacy = report["legacy_projection"]
    print(
        f"\nlive loop: {stream['achieved_fps']} fps "
        f"(p95 inter-frame {stream['inter_frame_ms']['p95_ms']}ms) vs "
        f"legacy fixed-sleep {legacy['fps']} fps -> "
        f"{report['speedup_vs_legacy']}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
