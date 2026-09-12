"""Realtime analysis output contract.

The realtime layer's observable product is a :class:`RealtimeAnalysis` snapshot:
current state + equity (win_rate / tie_rate) + confidence. This is a pure data
value — deliberately NOT strategy advice (no fold/call/raise, no bet sizing,
no outs). Those belong to later tasks built on top of a stable state + reliable
equity + opponent model.
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_engine.core.enums import Street
from poker_engine.core.observation import RawObservation
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable, serialization-safe view of the current canonical state."""

    hand_id: str
    state_version: int
    street: Street
    hero_cards: tuple[Card, ...]
    board_cards: tuple[Card, ...]
    pot: ChipAmount

    def __post_init__(self) -> None:
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")

    @classmethod
    def from_state(cls, state: PokerState) -> "StateSnapshot":
        if not isinstance(state, PokerState):
            raise TypeError("state must be a PokerState")
        return cls(
            hand_id=state.hand_id,
            state_version=state.state_version,
            street=state.street,
            hero_cards=tuple(state.hero_cards),
            board_cards=tuple(state.board_cards),
            pot=state.pot,
        )


@dataclass(frozen=True)
class EquitySnapshot:
    """Hero equity snapshot (win_rate + tie_rate only, per Task 8 boundary)."""

    win_rate: float
    tie_rate: float
    opponent_count: int | None = None
    samples: int = 0
    expected_share: float | None = None
    standard_error: float | None = None
    basis: str = "unspecified"
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("win_rate", "tie_rate"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{name} must be a float")
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {v}")
        if self.win_rate + self.tie_rate > 1.0 + 1e-12:
            raise ValueError("win_rate + tie_rate must not exceed one")
        if self.opponent_count is not None and (
            isinstance(self.opponent_count, bool)
            or not isinstance(self.opponent_count, int)
            or not 0 <= self.opponent_count <= 8
        ):
            raise ValueError("opponent_count must be in [0, 8] or None")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("samples must be an int")
        if self.samples < 0:
            raise ValueError("samples must not be negative")
        for name in ("expected_share", "standard_error"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0,1] or None")
        if not isinstance(self.basis, str) or not self.basis:
            raise ValueError("basis must be a non-empty string")
        if self.unavailable_reason is not None and (
            not isinstance(self.unavailable_reason, str)
            or not self.unavailable_reason
        ):
            raise ValueError("unavailable_reason must be a non-empty string or None")


@dataclass(frozen=True)
class ConfidenceSnapshot:
    """AI's trust in the current recognition: overall + per-field status."""

    overall_confidence: float
    # field name -> validation_status value (str)
    field_status: tuple[tuple[str, str], ...]

    @classmethod
    def from_observation(cls, obs: RawObservation) -> "ConfidenceSnapshot":
        if not isinstance(obs, RawObservation):
            raise TypeError("obs must be a RawObservation")
        statuses = tuple(
            (name, getattr(obs, name).validation_status.value)
            for name in (
                "hero_cards", "board_cards", "street", "pot",
                "stacks", "bet_size", "action",
            )
        )
        # Vision observations historically leave ``overall_confidence`` at its
        # default zero.  The desktop must therefore derive its displayed
        # confidence from the fields that are *currently valid*, rather than
        # reporting 0% beside a valid, measured hero-card reading.
        valid_confidences = [
            getattr(obs, name).confidence
            for name, status in statuses
            if status == "valid"
        ]
        overall = max(valid_confidences, default=0.0)
        return cls(overall_confidence=overall, field_status=statuses)


@dataclass(frozen=True)
class RealtimeAnalysis:
    """One frame's analysis output: state + equity + confidence."""

    frame_seq: int
    state: StateSnapshot
    equity: EquitySnapshot
    confidence: ConfidenceSnapshot


__all__ = [
    "StateSnapshot",
    "EquitySnapshot",
    "ConfidenceSnapshot",
    "RealtimeAnalysis",
]
