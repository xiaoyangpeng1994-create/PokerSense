"""Realtime pipeline: the event-driven loop that ties the layers together.

    FrameSource -> VisionEngine(process) -> RawObservation
        -> ChangeDetector -> (material change?) -> ApplicationOrchestrator
        -> StateEngine -> state snapshot + equity + confidence

This layer OWNS only the frame lifecycle and the change gating. It does not:
  - auto-operate the table / place bets (out of scope),
  - produce strategy recommendations (later tasks),
  - construct Vision/State components (they are injected).

Every step is deterministic given the injected components and frame source.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from poker_engine.core.events import StateEvent
from poker_engine.core.observation import RawObservation
from poker_engine.core.state import PokerState
from poker_engine.orchestrator import ApplicationOrchestrator

if TYPE_CHECKING:
    from poker_engine.perceptual.vision.engine import VisionEngine
    from poker_engine.perceptual.vision.table_map import TableMap
    from .frame_source import FrameSource

from .analysis import (
    ConfidenceSnapshot,
    EquitySnapshot,
    RealtimeAnalysis,
    StateSnapshot,
)
from .change_detector import ChangeReport, detect_change
from .equity import EquityStrategy, MonteCarloRandomRangeEquity
from .hand_boundary import (
    HandBoundaryDetection,
    HandBoundaryPolicy,
    HandBoundaryStatus,
    detect_hand_boundary,
)
from .temporal_consensus import TemporalConsensus


# Per-stage wall-clock cost keys recorded on :class:`PipelineStep.timings`.
# Kept as plain strings (not an Enum) so they can be serialized straight into
# a report without a lookup table.
STAGE_CAPTURE = "capture"
STAGE_VISION = "vision"
STAGE_CONSENSUS = "consensus"
STAGE_BOUNDARY = "boundary"
STAGE_CHANGE = "change"
STAGE_ADVANCE = "advance"
STAGE_EQUITY = "equity"
STAGE_TOTAL = "total"

ALL_STAGE_NAMES = (
    STAGE_CAPTURE,
    STAGE_VISION,
    STAGE_CONSENSUS,
    STAGE_BOUNDARY,
    STAGE_CHANGE,
    STAGE_ADVANCE,
    STAGE_EQUITY,
    STAGE_TOTAL,
)


@dataclass(frozen=True)
class PipelineStep:
    """Result of processing one frame.

    ``analysis_changed`` is True only when this step produced a *new* canonical
    state + equity snapshot. Every step still carries its own recognition
    confidence so a client never renders stale table data after an abstention.

    ``timings`` records per-stage wall-clock cost in milliseconds. It is
    observability only — never used to make a decision — so a caller may safely
    ignore it. Stages absent from a given step (e.g. ``equity`` on a frame that
    did not change state) are simply omitted rather than reported as zero.
    """

    frame_seq: int
    analysis: RealtimeAnalysis
    change: ChangeReport
    analysis_changed: bool
    hand_boundary: HandBoundaryDetection
    timings: tuple[tuple[str, float], ...] = ()

    def timing(self, stage: str) -> float | None:
        """Return one stage's cost in ms, or None when not recorded."""
        for name, value in self.timings:
            if name == stage:
                return value
        return None


class RealtimePipeline:
    """Event-driven realtime analysis driver."""

    def __init__(
        self,
        frame_source: FrameSource,
        vision: VisionEngine,
        table_map: TableMap,
        orchestrator: ApplicationOrchestrator,
        equity_strategy: EquityStrategy | None = None,
        equity_trials: int = 2000,
        equity_seed: int = 0,
        hero_confirmation_frames: int = 1,
        confirmation_frames: dict[str, int] | None = None,
        hand_boundary_policy: HandBoundaryPolicy | None = None,
        new_hand_state_factory: Callable[[], PokerState] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(getattr(frame_source, "next_frame", None)):
            raise TypeError("frame_source must provide next_frame()")
        if not callable(getattr(vision, "process", None)):
            raise TypeError("vision must provide process()")
        if table_map is None:
            raise TypeError("table_map must not be None")
        if not isinstance(orchestrator, ApplicationOrchestrator):
            raise TypeError("orchestrator must be an ApplicationOrchestrator")
        self._frame_source = frame_source
        self._vision = vision
        self._table_map = table_map
        self._orchestrator = orchestrator
        if equity_strategy is None:
            equity_strategy = MonteCarloRandomRangeEquity(
                trials=equity_trials, seed=equity_seed
            )
        self._equity_strategy = equity_strategy
        if isinstance(hero_confirmation_frames, bool) or not isinstance(
            hero_confirmation_frames, int
        ):
            raise TypeError("hero_confirmation_frames must be an int")
        if hero_confirmation_frames < 1:
            raise ValueError("hero_confirmation_frames must be >= 1")
        thresholds = dict(confirmation_frames or {})
        thresholds.setdefault("hero_cards", hero_confirmation_frames)
        self._temporal_consensus = TemporalConsensus(thresholds)
        if hand_boundary_policy is None:
            hand_boundary_policy = HandBoundaryPolicy()
        elif not isinstance(hand_boundary_policy, HandBoundaryPolicy):
            raise TypeError(
                "hand_boundary_policy must be a HandBoundaryPolicy or None"
            )
        self._hand_boundary_policy = hand_boundary_policy
        if new_hand_state_factory is not None and not callable(new_hand_state_factory):
            raise TypeError("new_hand_state_factory must be callable or None")
        self._new_hand_state_factory = new_hand_state_factory
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._monotonic_clock = monotonic_clock
        self._previous_obs: RawObservation | None = None
        self._current_analysis: RealtimeAnalysis | None = None

    def step(self) -> PipelineStep | None:
        """Advance one frame. Returns None when the frame source is exhausted."""
        clock = self._monotonic_clock
        started = clock()

        mark = clock()
        frame = self._frame_source.next_frame()
        capture_ms = (clock() - mark) * 1000.0
        if frame is None:
            return None

        mark = clock()
        raw_obs = self._vision.process(frame, self._table_map)
        vision_ms = (clock() - mark) * 1000.0

        mark = clock()
        obs = self._temporal_consensus.apply(raw_obs).observation
        consensus_ms = (clock() - mark) * 1000.0

        mark = clock()
        boundary = detect_hand_boundary(
            self._latest_state(), obs, self._hand_boundary_policy
        )
        boundary_ms = (clock() - mark) * 1000.0

        if boundary.status is HandBoundaryStatus.CONFIRMED:
            self._start_next_hand(obs.timestamp)
            advance_ms, equity_ms = self._advance(obs, frame)
            change = ChangeReport(
                changed=True, changed_fields=("hand_boundary",)
            )
            self._previous_obs = obs
            assert self._current_analysis is not None
            return self._step_result(
                frame, self._current_analysis, change, True, boundary,
                capture_ms, vision_ms, consensus_ms, boundary_ms,
                0.0, advance_ms, equity_ms, started,
            )

        if self._previous_obs is None:
            # First frame: no previous to diff against; still record the
            # opening state through the orchestrator.
            advance_ms, equity_ms = self._advance(obs, frame)
            change = ChangeReport(changed=True, changed_fields=())
            self._previous_obs = obs
            assert self._current_analysis is not None
            return self._step_result(
                frame, self._current_analysis, change, True, boundary,
                capture_ms, vision_ms, consensus_ms, boundary_ms,
                0.0, advance_ms, equity_ms, started,
            )

        mark = clock()
        change = detect_change(self._previous_obs, obs)
        change_ms = (clock() - mark) * 1000.0
        if change.changed:
            advance_ms, equity_ms = self._advance(obs, frame)
        else:
            advance_ms = 0.0
            equity_ms = None

        self._previous_obs = obs
        assert self._current_analysis is not None
        # The state snapshot is intentionally retained between material
        # transitions, but presentation confidence is per-frame.  Returning
        # the old confidence here made the UI show stale cards indefinitely
        # after the live recognizer had already abstained.
        fresh_analysis = replace(
            self._current_analysis,
            frame_seq=frame.frame_seq,
            confidence=ConfidenceSnapshot.from_observation(obs),
        )
        return self._step_result(
            frame, fresh_analysis, change, change.changed, boundary,
            capture_ms, vision_ms, consensus_ms, boundary_ms,
            change_ms, advance_ms, equity_ms, started,
        )

    def _step_result(
        self,
        frame,
        analysis,
        change,
        analysis_changed,
        boundary,
        capture_ms,
        vision_ms,
        consensus_ms,
        boundary_ms,
        change_ms,
        advance_ms,
        equity_ms,
        started,
    ) -> PipelineStep:
        """Assemble one step, folding the collected stage costs into it."""
        total_ms = (self._monotonic_clock() - started) * 1000.0
        stages = (
            (STAGE_CAPTURE, capture_ms),
            (STAGE_VISION, vision_ms),
            (STAGE_CONSENSUS, consensus_ms),
            (STAGE_BOUNDARY, boundary_ms),
            (STAGE_CHANGE, change_ms),
            (STAGE_ADVANCE, advance_ms),
            (STAGE_TOTAL, total_ms),
        )
        if equity_ms is not None:
            stages += ((STAGE_EQUITY, equity_ms),)
        return PipelineStep(
            frame_seq=frame.frame_seq,
            analysis=analysis,
            change=change,
            analysis_changed=analysis_changed,
            hand_boundary=boundary,
            timings=stages,
        )

    def _advance(self, obs: RawObservation, frame: Any) -> tuple[float, float]:
        """Feed one observation through; return ``(advance_ms, equity_ms)``.

        Equity is timed separately because it dominates the step budget (see
        ``tools/benchmark_realtime_latency.py``) and is the stage most likely
        to be truncated by a deadline.
        """
        clock = self._monotonic_clock
        started = clock()
        self._orchestrator.process_observation(obs)
        state = self._latest_state()
        equity_started = clock()
        equity = self._compute_equity(state)
        equity_ms = (clock() - equity_started) * 1000.0
        snapshot = RealtimeAnalysis(
            frame_seq=frame.frame_seq,
            state=StateSnapshot.from_state(state),
            equity=equity,
            confidence=ConfidenceSnapshot.from_observation(obs),
        )
        self._current_analysis = snapshot
        return (clock() - started) * 1000.0, equity_ms

    def _start_next_hand(self, boundary_time) -> None:
        if self._new_hand_state_factory is None:
            raise RuntimeError(
                "confirmed new hand requires new_hand_state_factory"
            )
        initial_state = self._new_hand_state_factory()
        self._orchestrator.start_next_hand(
            initial_state,
            ended_at=boundary_time,
            started_at=boundary_time,
        )

    def _latest_state(self) -> PokerState:
        active = self._orchestrator._hand_memory.active_hand_id
        if active is None:
            raise RuntimeError("no active hand; call start_hand before stepping")
        state = self._orchestrator._hand_memory.latest_state(active)
        if state is None:
            raise RuntimeError("active hand has no state")
        return state

    def _compute_equity(self, state: PokerState) -> EquitySnapshot:
        """Delegate equity to the injected strategy."""
        return self._equity_strategy.compute(state)

    def latest_analysis(self) -> RealtimeAnalysis | None:
        return self._current_analysis

    def current_state(self) -> PokerState:
        """Return the canonical state backing the latest live analysis."""
        return self._latest_state()

    def action_history(self) -> tuple[StateEvent, ...]:
        """Return the active hand's canonical event stream."""
        active = self._orchestrator._hand_memory.active_hand_id
        if active is None:
            return ()
        return self._orchestrator._hand_memory.events(active)


__all__ = ["RealtimePipeline", "PipelineStep"]
