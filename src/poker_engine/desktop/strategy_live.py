"""Version-safe strategy Advice for the desktop live stream.

This module deliberately has no capture or OpenCV dependency.  It binds a
canonical ``PokerState`` to a short-lived strategy request, reuses Advice only
for the exact same state version, and lets the strategy layer refuse incomplete
live inputs instead of inventing actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from poker_engine.core._freeze import _require_aware_dt, utc_now
from poker_engine.core.enums import PlayerStatus
from poker_engine.core.events import StateEvent
from poker_engine.core.state import PokerState
from poker_engine.realtime.analysis import RealtimeAnalysis
from poker_engine.strategy.contracts import ContextQuality, DecisionContext, GameConfig
from poker_engine.strategy.context_factory import RequestContextFactory
from poker_engine.strategy.orchestration import (
    RefinementState,
    SlowHandle,
    StrategyOrchestrator,
)
from poker_engine.strategy.state import build_decision_context

from .serialize import DesktopFrame


ActionLineResolver = Callable[[PokerState, tuple[StateEvent, ...]], str | None]


class LiveStrategySession:
    """Create atomic analysis + Advice frames for one live table session."""

    _REQUIRED_LIVE_FIELDS = ("hero_cards", "street", "pot", "stacks", "action")

    def __init__(
        self,
        strategy: StrategyOrchestrator,
        game_config: GameConfig,
        *,
        deadline_ms: int = 1500,
        clock: Callable[[], datetime] = utc_now,
        request_factory: RequestContextFactory | None = None,
        action_line_resolver: ActionLineResolver | None = None,
    ) -> None:
        if not isinstance(strategy, StrategyOrchestrator):
            raise TypeError("strategy must be a StrategyOrchestrator")
        if not isinstance(game_config, GameConfig):
            raise TypeError("game_config must be a GameConfig")
        if not isinstance(deadline_ms, int) or isinstance(deadline_ms, bool):
            raise TypeError("deadline_ms must be an int")
        if deadline_ms <= 0:
            raise ValueError("deadline_ms must be > 0")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if request_factory is not None and not isinstance(
            request_factory, RequestContextFactory
        ):
            raise TypeError("request_factory must be a RequestContextFactory")
        if action_line_resolver is not None and not callable(action_line_resolver):
            raise TypeError("action_line_resolver must be callable or None")
        self._strategy = strategy
        self._game_config = game_config
        self._deadline_ms = deadline_ms
        self._clock = clock
        self._request_factory = request_factory or RequestContextFactory(clock=clock)
        self._action_line_resolver = action_line_resolver
        self._context: DecisionContext | None = None
        self._advice = None
        self._slow_handle: SlowHandle | None = None
        self._quality_key: tuple | None = None
        self._action_history: tuple[StateEvent, ...] = ()
        self._math_report = {}

    def frame(
        self,
        analysis: RealtimeAnalysis,
        state: PokerState,
        *,
        action_history: tuple[StateEvent, ...] = (),
    ) -> DesktopFrame:
        """Return Advice bound to ``analysis.state`` or fail closed."""
        if not isinstance(analysis, RealtimeAnalysis):
            raise TypeError("analysis must be a RealtimeAnalysis")
        if not isinstance(state, PokerState):
            raise TypeError("state must be a PokerState")
        history = tuple(action_history)
        if not all(isinstance(event, StateEvent) for event in history):
            raise TypeError("action_history must contain StateEvent values")
        identity = (state.hand_id, state.state_version)
        if identity != (analysis.state.hand_id, analysis.state.state_version):
            raise ValueError("analysis and canonical state identity must match")
        now = self._clock()
        if not isinstance(now, datetime):
            raise TypeError("clock must return a datetime")
        _require_aware_dt(now)
        quality_key = (
            analysis.confidence.overall_confidence,
            analysis.confidence.field_status,
        )

        if (
            self._context is not None
            and (self._context.hand_id, self._context.state_version) == identity
            and self._quality_key == quality_key
            and self._action_history == history
            and not self._context.request.is_expired(now)
        ):
            self._collect_pending(now)
            return DesktopFrame(analysis, self._advice)

        context = self._new_context(analysis, state, history)
        if self._slow_handle is not None and self._context is not None:
            # The collect contract cancels a resolver Future when identity is
            # stale.  Its result must never cross a hand/state boundary.
            self._strategy.collect(self._slow_handle, context, now=now)
        math_report = {
            "win_rate": analysis.equity.win_rate,
            "tie_rate": analysis.equity.tie_rate,
        }
        cycle = self._strategy.request(
            context,
            math_report=math_report,
            now=now,
        )
        self._context = context
        self._advice = cycle.fast_advice
        self._slow_handle = cycle.slow_handle
        self._quality_key = quality_key
        self._action_history = history
        self._math_report = math_report
        return DesktopFrame(analysis, self._advice)

    def _new_context(
        self,
        analysis: RealtimeAnalysis,
        state: PokerState,
        history: tuple[StateEvent, ...],
    ) -> DecisionContext:
        request = self._request_factory.create(
            hand_id=state.hand_id,
            state_version=state.state_version,
            deadline_ms=self._deadline_ms,
        )
        statuses = dict(analysis.confidence.field_status)
        required = list(self._REQUIRED_LIVE_FIELDS)
        if state.street.value != "preflop":
            required.append("board_cards")
        failures = tuple(
            f"live_input_not_valid:{field}"
            for field in required
            if statuses.get(field) != "valid"
        )
        confidence = analysis.confidence.overall_confidence
        quality = ContextQuality(
            confidence,
            {field: confidence for field in required},
            failures,
        )
        action_line = None
        if self._action_line_resolver is not None:
            action_line = self._action_line_resolver(state, history)
        dealt_count = sum(
            player.status is not PlayerStatus.SITTING_OUT
            for player in state.players
        )
        game_config = self._game_config
        if 2 <= dealt_count <= game_config.max_seats:
            game_config = replace(
                game_config, dealt_player_count=dealt_count
            )
        return build_decision_context(
            state,
            request,
            game_config,
            action_history=history,
            input_quality=quality,
            action_line=action_line,
        )

    def _collect_pending(self, now: datetime) -> None:
        if self._slow_handle is None or self._context is None:
            return
        refinement = self._strategy.collect(
            self._slow_handle,
            self._context,
            math_report=self._math_report,
            now=now,
        )
        if refinement.state is RefinementState.APPLIED:
            self._advice = refinement.advice
            self._slow_handle = None
        elif refinement.state is not RefinementState.PENDING:
            self._slow_handle = None

    @property
    def current_context(self) -> DecisionContext | None:
        return self._context


__all__ = ["ActionLineResolver", "LiveStrategySession"]
