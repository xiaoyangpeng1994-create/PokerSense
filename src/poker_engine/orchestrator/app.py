"""Application Orchestrator — central scheduler for the Phase 0 pipeline.

Coordinates ``RawObservation -> StateEngine -> HandMemory``. It decides *when*
to call whom, never *how* to compute (no poker rules, no identity mapping, no
bet semantics here).

Deterministic: no datetime.now / random / uuid; all timestamps come from
explicit inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_engine.confidence.gate import ConfidenceGate, ConfidenceGateResult
from poker_engine.core._freeze import utc_now
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandHistory, HandSummary
from poker_engine.core.observation import RawObservation
from poker_engine.core.state import PokerState, StateContext
from poker_engine.memory.hand_memory import HandMemory
from poker_engine.state_engine.engine import StateEngine, StateTransitionResult

from .errors import OrchestratorError


@dataclass(frozen=True)
class OrchestrationResult:
    """Outcome of processing one observation through the orchestrator.

    ``transition`` is the raw StateEngine result (never duplicated here).
    ``persisted`` records whether new state/events were actually written to
    HandMemory (True only for a successful material change; False for no-op,
    invalid, or any write that did not happen).
    ``confidence_gate`` carries which fields were gated to UNKNOWN (Task 5).
    """

    transition: StateTransitionResult
    persisted: bool
    confidence_gate: ConfidenceGateResult


class ApplicationOrchestrator:
    """Central scheduler. Holds module references only — NO hand state."""

    def __init__(
        self,
        state_engine: StateEngine,
        hand_memory: HandMemory,
        confidence_gate: ConfidenceGate | None = None,
    ) -> None:
        if not isinstance(state_engine, StateEngine):
            raise TypeError("state_engine must be a StateEngine")
        # HandMemory is a typing.Protocol (structural type), not a runtime-checkable
        # class. Do a duck-type check on the methods the orchestrator relies on.
        for attr in ("start_hand", "record_state", "record_event",
                     "record_transition", "replace_active_hand",
                     "complete_hand", "latest_state", "active_hand_id"):
            if not hasattr(hand_memory, attr):
                raise TypeError(
                    f"hand_memory must provide {attr} (see HandMemory protocol)"
                )
        self._state_engine = state_engine
        self._hand_memory = hand_memory
        # Strict None semantics (avoid `or` falsy-object trap).
        if confidence_gate is None:
            self._confidence_gate = ConfidenceGate()
        elif isinstance(confidence_gate, ConfidenceGate):
            self._confidence_gate = confidence_gate
        else:
            raise TypeError("confidence_gate must be a ConfidenceGate or None")

    # ------------------------------------------------------------------ lifecycle

    def start_hand(
        self,
        initial_state: PokerState,
        started_at=None,
    ) -> None:
        """Delegate hand start to HandMemory. Does NOT invent PlayerState."""
        if not isinstance(initial_state, PokerState):
            raise TypeError("initial_state must be a PokerState")
        self._hand_memory.start_hand(
            hand_id=initial_state.hand_id,
            initial_state=initial_state,
            started_at=started_at,
        )

    def complete_hand(
        self,
        hand_id: str,
        summary: HandSummary,
        ended_at=None,
    ) -> HandHistory:
        """Delegate hand completion to HandMemory."""
        return self._hand_memory.complete_hand(
            hand_id=hand_id, summary=summary, ended_at=ended_at
        )

    def start_next_hand(
        self,
        initial_state: PokerState,
        ended_at=None,
        started_at=None,
    ) -> HandHistory:
        """Close the active capture hand and start its confirmed successor.

        Live capture cannot always observe showdown or a final pot.  A newly
        confirmed, different pair of hero cards is nevertheless an unambiguous
        hand boundary.  Keep the old record as an explicitly *unsettled*
        history (no winners or payouts are invented), then create the new
        active hand from a clean seed state.
        """
        if not isinstance(initial_state, PokerState):
            raise TypeError("initial_state must be a PokerState")
        active_hand_id = self._hand_memory.active_hand_id
        if active_hand_id is None:
            raise OrchestratorError("no active hand to replace")
        previous_state = self._hand_memory.latest_state(active_hand_id)
        if previous_state is None:
            raise OrchestratorError(
                f"active hand {active_hand_id!r} has no state"
            )
        if initial_state.hand_id == active_hand_id:
            raise OrchestratorError("successor hand_id must differ from active hand")

        boundary_time = ended_at or started_at or utc_now()
        boundary_event = StateEvent(
            event_type=EventType.HAND_END,
            hand_id=active_hand_id,
            state_version=previous_state.state_version,
            payload={
                "reason": "confirmed_hand_boundary",
                "successor_hand_id": initial_state.hand_id,
                "settled": False,
            },
            timestamp=boundary_time,
            source="application_orchestrator",
        )
        return self._hand_memory.replace_active_hand(
            initial_state,
            HandSummary(final_pot=previous_state.pot, winners=()),
            boundary_event,
            ended_at=boundary_time,
            started_at=boundary_time,
        )

    # ------------------------------------------------------------------ pipeline

    def process_observation(
        self,
        observation: RawObservation,
    ) -> OrchestrationResult:
        """Process one observation through ConfidenceGate -> StateEngine -> Memory.

        previous_state is resolved from the active hand in HandMemory (single
        source of truth). StateContext is built internally, carrying the actual
        ConfidenceGate thresholds.
        """
        if not isinstance(observation, RawObservation):
            raise TypeError("observation must be a RawObservation")

        active_hand_id = self._hand_memory.active_hand_id
        if active_hand_id is None:
            raise OrchestratorError(
                "no active hand: call start_hand() before process_observation()"
            )

        previous_state = self._hand_memory.latest_state(active_hand_id)
        if previous_state is None:
            # Should not happen given an active hand, but fail fast anyway.
            raise OrchestratorError(
                f"active hand {active_hand_id!r} has no state"
            )

        # 1. Confidence Gate: sanitize low-confidence fields to UNKNOWN.
        gate_result = self._confidence_gate.apply(observation)
        gated_observation = gate_result.observation

        # 2. Build StateContext with the actual gating thresholds.
        context = StateContext(
            previous_state=previous_state,
            confidence_thresholds=dict(self._confidence_gate.thresholds),
        )

        # 3. State Engine transition on the sanitized observation.
        result = self._state_engine.transition(
            previous_state=previous_state,
            observation=gated_observation,
            context=context,
        )

        # 4. Persist only a successful material change.
        persisted = False
        if result.changed and result.validation.is_valid:
            self._hand_memory.record_transition(result.state, result.events)
            persisted = True

        return OrchestrationResult(
            transition=result,
            persisted=persisted,
            confidence_gate=gate_result,
        )


__all__ = ["ApplicationOrchestrator", "OrchestrationResult"]
