"""Hand contracts: HandSummary, HandHistory.

``HandHistory`` is the complete, append-only record of a single hand. It is
the Event Sourcing aggregate: the event stream is authoritative, and the
``summary`` is a derived settlement view.

Task 1D scope only. Serialization (to_dict/from_dict) is deferred to Task 1E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from ._freeze import _require_aware_dt, freeze_mapping
from .events import StateEvent
from .opponents import PlayerState
from .value_objects import ChipAmount, ChipDelta


@dataclass(frozen=True)
class HandSummary:
    """Settlement summary of a completed (or in-progress) hand.

    - ``final_pot``: total pot, ChipAmount (non-negative).
    - ``winners``: ordered list of winning player_ids.
    - ``winnings``: player_id -> amount won, ChipAmount (non-negative).
    - ``net_result``: player_id -> net profit/loss, ChipDelta (signed).
    """

    final_pot: ChipAmount
    winners: tuple[str, ...]
    winnings: Mapping[str, ChipAmount] = field(default_factory=dict)
    net_result: Mapping[str, ChipDelta] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.final_pot, ChipAmount):
            raise TypeError("final_pot must be a ChipAmount")

        winners = tuple(self.winners)
        if not all(isinstance(w, str) and w for w in winners):
            raise TypeError("winners must be non-empty str player_ids")
        if len(set(winners)) != len(winners):
            raise ValueError("winners must not contain duplicates")
        object.__setattr__(self, "winners", winners)

        winnings = freeze_mapping(self.winnings)
        for k, v in winnings.items():
            if not isinstance(k, str) or not k:
                raise TypeError("winnings keys must be non-empty str player_ids")
            if not isinstance(v, ChipAmount):
                raise TypeError("winnings values must be ChipAmount")
        object.__setattr__(self, "winnings", winnings)

        net = freeze_mapping(self.net_result)
        for k, v in net.items():
            if not isinstance(k, str) or not k:
                raise TypeError("net_result keys must be non-empty str player_ids")
            if not isinstance(v, ChipDelta):
                raise TypeError("net_result values must be ChipDelta")
        object.__setattr__(self, "net_result", net)


@dataclass(frozen=True)
class HandHistory:
    """Complete append-only record of one hand."""

    hand_id: str
    players: tuple[PlayerState, ...]
    events: tuple[StateEvent, ...]
    summary: HandSummary
    start_time: datetime
    end_time: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")

        players = tuple(self.players)
        if not all(isinstance(p, PlayerState) for p in players):
            raise TypeError("players must be PlayerState instances")
        player_ids = [p.player_id for p in players]
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("players must not have duplicate player_id")
        seats = [p.seat for p in players]
        if len(set(seats)) != len(seats):
            raise ValueError("players must not have duplicate seat")
        object.__setattr__(self, "players", players)

        events = tuple(self.events)
        if not all(isinstance(e, StateEvent) for e in events):
            raise TypeError("events must be StateEvent instances")
        for e in events:
            if e.hand_id != self.hand_id:
                raise ValueError(
                    "every StateEvent.hand_id must equal HandHistory.hand_id"
                )
        object.__setattr__(self, "events", events)

        if not isinstance(self.summary, HandSummary):
            raise TypeError("summary must be a HandSummary")

        if not isinstance(self.start_time, datetime):
            raise TypeError("start_time must be a datetime")
        _require_aware_dt(self.start_time)

        if self.end_time is not None:
            if not isinstance(self.end_time, datetime):
                raise TypeError("end_time must be a datetime or None")
            _require_aware_dt(self.end_time)
            if self.end_time < self.start_time:
                raise ValueError("end_time must not be before start_time")


__all__ = ["HandSummary", "HandHistory"]
