"""Hand Memory — in-memory append-only store for poker hands.

Hand Memory is a service-layer capability. Its job is to RELIABLY remember
what happened in a hand from start to finish. It does NOT reason about what
happened (that is the State Engine's job).

Design rules (see docs/hand-memory.md):
- Append-only: history is never mutated in place.
- Single source of truth: one (hand_id, state_version) -> one PokerState.
- State versions are strictly monotonic (new > latest; gaps allowed).
- Hands are isolated; data never leaks across hand_ids.
- No database in this first version (in-memory only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from poker_engine.core._freeze import _require_aware_dt, utc_now
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.hand import HandHistory, HandSummary
from poker_engine.core.state import PokerState

from .errors import (
    HandConflictError,
    HandLifecycleError,
    HandNotFoundError,
)


# ---------------------------------------------------------------------------
# Internal record (mutable, never exposed)
# ---------------------------------------------------------------------------

@dataclass
class _HandRecord:
    hand_id: str
    started_at: datetime
    states: list[PokerState] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)
    completed: HandHistory | None = None

    @property
    def latest_state(self) -> PokerState | None:
        return self.states[-1] if self.states else None

    def state_by_version(self, version: int) -> PokerState | None:
        for s in self.states:
            if s.state_version == version:
                return s
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class HandMemory(Protocol):
    """Append-only, single-active-hand memory."""

    def start_hand(
        self,
        hand_id: str,
        initial_state: PokerState,
        started_at: datetime | None = None,
    ) -> None: ...

    def record_state(self, state: PokerState) -> None: ...

    def record_event(self, event: StateEvent) -> None: ...

    def record_transition(
        self,
        state: PokerState,
        events: tuple[StateEvent, ...],
    ) -> None: ...

    def complete_hand(
        self,
        hand_id: str,
        summary: HandSummary,
        ended_at: datetime | None = None,
    ) -> HandHistory: ...

    def replace_active_hand(
        self,
        initial_state: PokerState,
        summary: HandSummary,
        boundary_event: StateEvent,
        *,
        ended_at: datetime,
        started_at: datetime,
    ) -> HandHistory: ...

    def hand_exists(self, hand_id: str) -> bool: ...

    def is_active(self, hand_id: str) -> bool: ...

    def latest_state(self, hand_id: str) -> PokerState | None: ...

    def get_state(self, hand_id: str, state_version: int) -> PokerState | None: ...

    def states(self, hand_id: str) -> tuple[PokerState, ...]: ...

    def events(self, hand_id: str) -> tuple[StateEvent, ...]: ...

    def get_hand_history(self, hand_id: str) -> HandHistory | None: ...

    def completed_hands(self) -> tuple[HandHistory, ...]: ...

    @property
    def active_hand_id(self) -> str | None: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

class InMemoryHandMemory:
    """Single-threaded in-memory Hand Memory (MVP).

    Tracks one active hand at a time; completed hands accumulate in a dict so
    the design is not hard-wired against future multi-hand support.
    """

    def __init__(self) -> None:
        self._hands: dict[str, _HandRecord] = {}
        self._active_hand_id: str | None = None

    # ------------------------------------------------------------------ write

    def start_hand(
        self,
        hand_id: str,
        initial_state: PokerState,
        started_at: datetime | None = None,
    ) -> None:
        if not isinstance(hand_id, str) or not hand_id:
            raise ValueError("hand_id must be a non-empty str")
        if not isinstance(initial_state, PokerState):
            raise TypeError("initial_state must be a PokerState")
        if initial_state.hand_id != hand_id:
            raise HandConflictError(
                "initial_state.hand_id does not match hand_id"
            )
        resolved_start = utc_now() if started_at is None else started_at
        if not isinstance(resolved_start, datetime):
            raise TypeError("started_at must be a datetime or None")
        _require_aware_dt(resolved_start)

        existing = self._hands.get(hand_id)
        if existing is not None:
            if existing.completed is not None:
                raise HandLifecycleError(
                    f"hand {hand_id!r} is already completed"
                )
            # Idempotency: hand_id + initial_state + resolved started_at.
            if (
                existing.latest_state == initial_state
                and existing.started_at == resolved_start
            ):
                return
            raise HandConflictError(
                f"conflicting start_hand for {hand_id!r}"
            )

        # New hand — single active hand MVP constraint.
        if self._active_hand_id is not None:
            raise HandLifecycleError(
                f"a hand is already active ({self._active_hand_id!r}); "
                "single-table MVP allows only one active hand"
            )

        record = _HandRecord(hand_id=hand_id, started_at=resolved_start)
        record.states.append(initial_state)
        self._hands[hand_id] = record
        self._active_hand_id = hand_id

    def record_state(self, state: PokerState) -> None:
        if not isinstance(state, PokerState):
            raise TypeError("state must be a PokerState")
        record = self._require_hand(state.hand_id)
        self._require_active(record)
        if state.hand_id != record.hand_id:
            raise HandConflictError("state.hand_id does not match hand")

        latest = record.latest_state
        if latest is not None and state.state_version <= latest.state_version:
            # Same version -> idempotent if identical, else conflict.
            existing = record.state_by_version(state.state_version)
            if existing is not None and existing == state:
                return
            raise HandConflictError(
                f"state_version must be strictly increasing "
                f"(got {state.state_version}, latest {latest.state_version})"
            )
        record.states.append(state)

    def record_event(self, event: StateEvent) -> None:
        if not isinstance(event, StateEvent):
            raise TypeError("event must be a StateEvent")
        record = self._require_hand(event.hand_id)
        self._require_active(record)
        if event.hand_id != record.hand_id:
            raise HandConflictError("event.hand_id does not match hand")
        # The referenced state version must already exist (no event-first).
        if record.state_by_version(event.state_version) is None:
            raise HandConflictError(
                f"event references unknown state_version "
                f"{event.state_version}"
            )
        if event in record.events:
            return
        record.events.append(event)

    def record_transition(
        self,
        state: PokerState,
        events: tuple[StateEvent, ...],
    ) -> None:
        """Atomically append one canonical state and all of its events.

        Every validation happens before either list is changed.  This prevents
        a rejected event from leaving the state stream one version ahead of
        the event stream.
        """
        if not isinstance(state, PokerState):
            raise TypeError("state must be a PokerState")
        values = tuple(events)
        if not all(isinstance(event, StateEvent) for event in values):
            raise TypeError("events must contain StateEvent values")
        record = self._require_hand(state.hand_id)
        self._require_active(record)
        latest = record.latest_state
        existing = record.state_by_version(state.state_version)
        append_state = True
        if latest is not None and state.state_version <= latest.state_version:
            if existing != state:
                raise HandConflictError(
                    f"state_version must be strictly increasing "
                    f"(got {state.state_version}, latest {latest.state_version})"
                )
            append_state = False
        for event in values:
            if event.hand_id != state.hand_id:
                raise HandConflictError("event.hand_id does not match state.hand_id")
            if event.state_version != state.state_version:
                raise HandConflictError(
                    "transition event must reference transition state_version"
                )
        # Mutation starts only after every state/event invariant has passed.
        if append_state:
            record.states.append(state)
        for event in values:
            if event not in record.events:
                record.events.append(event)

    def complete_hand(
        self,
        hand_id: str,
        summary: HandSummary,
        ended_at: datetime | None = None,
    ) -> HandHistory:
        if not isinstance(summary, HandSummary):
            raise TypeError("summary must be a HandSummary")
        record = self._require_hand(hand_id)
        if record.completed is not None:
            raise HandLifecycleError(f"hand {hand_id!r} is already completed")
        if not record.states:
            raise HandLifecycleError("cannot complete a hand with no state")
        resolved_end = utc_now() if ended_at is None else ended_at
        if not isinstance(resolved_end, datetime):
            raise TypeError("ended_at must be a datetime or None")
        _require_aware_dt(resolved_end)
        if resolved_end < record.started_at:
            raise HandLifecycleError("ended_at must be >= started_at")

        history = HandHistory(
            hand_id=record.hand_id,
            players=record.latest_state.players,
            events=tuple(record.events),
            summary=summary,
            start_time=record.started_at,
            end_time=resolved_end,
        )
        record.completed = history
        if self._active_hand_id == hand_id:
            self._active_hand_id = None
        return history

    def replace_active_hand(
        self,
        initial_state: PokerState,
        summary: HandSummary,
        boundary_event: StateEvent,
        *,
        ended_at: datetime,
        started_at: datetime,
    ) -> HandHistory:
        """Atomically complete the active hand and create its successor."""
        if not isinstance(initial_state, PokerState):
            raise TypeError("initial_state must be a PokerState")
        if not isinstance(summary, HandSummary):
            raise TypeError("summary must be a HandSummary")
        if not isinstance(boundary_event, StateEvent):
            raise TypeError("boundary_event must be a StateEvent")
        for name, value in (("ended_at", ended_at), ("started_at", started_at)):
            if not isinstance(value, datetime):
                raise TypeError(f"{name} must be a datetime")
            _require_aware_dt(value)
        active_id = self._active_hand_id
        if active_id is None:
            raise HandLifecycleError("no active hand to replace")
        record = self._require_hand(active_id)
        self._require_active(record)
        latest = record.latest_state
        if latest is None:
            raise HandLifecycleError("cannot replace a hand with no state")
        if initial_state.hand_id == active_id:
            raise HandConflictError("successor hand_id must differ from active hand")
        if initial_state.hand_id in self._hands:
            raise HandConflictError(
                f"successor hand {initial_state.hand_id!r} already exists"
            )
        if boundary_event.hand_id != active_id:
            raise HandConflictError("boundary_event must reference active hand")
        if boundary_event.event_type is not EventType.HAND_END:
            raise HandConflictError("boundary_event must be a HAND_END event")
        if boundary_event.state_version != latest.state_version:
            raise HandConflictError("boundary_event must reference latest state")
        if ended_at < record.started_at:
            raise HandLifecycleError("ended_at must be >= active hand started_at")
        if started_at < ended_at:
            raise HandLifecycleError("successor started_at must be >= ended_at")
        completed_events = tuple(record.events) + (
            () if boundary_event in record.events else (boundary_event,)
        )
        history = HandHistory(
            hand_id=record.hand_id,
            players=latest.players,
            events=completed_events,
            summary=summary,
            start_time=record.started_at,
            end_time=ended_at,
        )
        successor = _HandRecord(
            hand_id=initial_state.hand_id,
            started_at=started_at,
            states=[initial_state],
        )
        # Commit the fully validated replacement without calling fallible
        # public mutators between lifecycle states.
        if boundary_event not in record.events:
            record.events.append(boundary_event)
        record.completed = history
        self._hands[initial_state.hand_id] = successor
        self._active_hand_id = initial_state.hand_id
        return history

    # ------------------------------------------------------------------- read

    def hand_exists(self, hand_id: str) -> bool:
        return hand_id in self._hands

    def is_active(self, hand_id: str) -> bool:
        record = self._hands.get(hand_id)
        return record is not None and record.completed is None

    def latest_state(self, hand_id: str) -> PokerState | None:
        record = self._hands.get(hand_id)
        return record.latest_state if record is not None else None

    def get_state(self, hand_id: str, state_version: int) -> PokerState | None:
        record = self._hands.get(hand_id)
        if record is None:
            return None
        return record.state_by_version(state_version)

    def states(self, hand_id: str) -> tuple[PokerState, ...]:
        record = self._hands.get(hand_id)
        if record is None:
            return ()
        return tuple(record.states)

    def events(self, hand_id: str) -> tuple[StateEvent, ...]:
        record = self._hands.get(hand_id)
        if record is None:
            return ()
        return tuple(record.events)

    def get_hand_history(self, hand_id: str) -> HandHistory | None:
        record = self._hands.get(hand_id)
        if record is None:
            return None
        return record.completed

    def completed_hands(self) -> tuple[HandHistory, ...]:
        return tuple(
            r.completed for r in self._hands.values() if r.completed is not None
        )

    @property
    def active_hand_id(self) -> str | None:
        return self._active_hand_id

    # --------------------------------------------------------------- helpers

    def _require_hand(self, hand_id: str) -> _HandRecord:
        record = self._hands.get(hand_id)
        if record is None:
            raise HandNotFoundError(f"hand {hand_id!r} does not exist")
        return record

    def _require_active(self, record: _HandRecord) -> None:
        if record.completed is not None:
            raise HandLifecycleError(
                f"hand {record.hand_id!r} is already completed"
            )


__all__ = ["HandMemory", "InMemoryHandMemory"]
