"""Event contracts: EventType, StateEvent.

``StateEvent`` records the change from one PokerState version to the next.
This is the Event Sourcing unit — append-only, immutable, deep-frozen.

``EventType`` is the event-stream vocabulary. It is DISTINCT from
``ActionType`` (in enums.py):

- ``ActionType`` = player action vocabulary (fold/check/call/bet/raise/
  all_in/post_*). Used to describe WHAT a player did.
- ``EventType`` = event-stream vocabulary. It includes player-action events
  AND non-action lifecycle events (HAND_START / DEAL / STREET_CHANGE /
  HAND_END). Used to tag StateEvent entries in the replayable stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from ._freeze import _require_aware_dt, freeze_mapping, utc_now


class EventType(str, Enum):
    """Event-stream event types (player action + lifecycle)."""

    # --- lifecycle ---
    HAND_START = "hand_start"
    DEAL = "deal"
    STREET_CHANGE = "street_change"
    HAND_END = "hand_end"

    # --- player actions ---
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class StateEvent:
    """A single immutable event in the hand history stream."""

    event_type: EventType
    hand_id: str
    state_version: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    source: str = "state_engine"

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise TypeError("event_type must be an EventType enum")
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")
        if not isinstance(self.source, str):
            raise TypeError("source must be a str")
        if not self.source:
            raise ValueError("source must be a non-empty str")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        _require_aware_dt(self.timestamp)
        # Deep-freeze payload (nested dict/list/set become immutable).
        object.__setattr__(self, "payload", freeze_mapping(self.payload))


__all__ = ["EventType", "StateEvent"]
