"""Fail-closed action-line resolver for the desktop live pipeline.

A :class:`ProviderCapability` advertises a *closed vocabulary* of action lines
(``unopened``, ``limp``, ``raise``, ``three_bet``, ...).  The live pipeline owns
the raw event stream, so something has to translate observed events into one of
those labels before a capability can match.  That translator lives here.

The derivation intentionally mirrors the one ``GTOpenPreflopProvider``
validates against internally (``gtopen_provider._validate_action_line``), so
every Provider observes the same label for the same history and the two cannot
drift apart silently.

Failure policy
--------------
Anything missing, contradictory, or outside the vocabulary resolves to
``None``.  A ``None`` action line fails every capability match with
``missing_action_line``, which yields an auditable ABSTAIN.  The resolver never
raises into the live loop and never guesses a line to satisfy a Provider.
"""

from __future__ import annotations

from typing import Sequence

from poker_engine.core.enums import Street
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.state import PokerState

__all__ = [
    "ACTION_LINES",
    "build_action_line_resolver",
    "resolve_action_line",
]


#: The closed vocabulary every production Provider draws from.
ACTION_LINES = frozenset({
    "unopened",
    "limp",
    "multi_limp",
    "raise",
    "three_bet",
    "four_bet",
    "squeeze",
    "iso_raise",
    "all_in",
})

_PLAYER_EVENTS = frozenset({
    EventType.FOLD,
    EventType.CHECK,
    EventType.CALL,
    EventType.BET,
    EventType.RAISE,
    EventType.ALL_IN,
})

_KIND_BY_EVENT = {
    EventType.FOLD: "fold",
    EventType.CHECK: "check",
    EventType.CALL: "call",
    EventType.RAISE: "raise",
    EventType.ALL_IN: "jam",
}

_AGGRESSIVE_KINDS = frozenset({"raise", "jam"})


def resolve_action_line(
    state: PokerState,
    history: Sequence[StateEvent],
) -> str | None:
    """Map the current street's action history onto one action-line label.

    Returns ``None`` whenever the evidence is insufficient or contradictory.
    """
    if not isinstance(state, PokerState):
        return None
    # The vocabulary above is defined for the preflop betting tree only.
    # Postflop has no released Provider, so there is nothing to label for.
    if state.street is not Street.PREFLOP:
        return None
    actions = _street_actions(state, history)
    if actions is None:
        return None
    return _classify(actions)


def build_action_line_resolver():
    """Return the production resolver bound for ``LiveStrategySession``.

    Kept as a factory so a future configuration flag can select an alternative
    resolver without touching the call site in ``live.py``.
    """
    return resolve_action_line


def _street_actions(
    state: PokerState,
    history: Sequence[StateEvent],
) -> tuple[tuple[int, str], ...] | None:
    """Collect ``(seat_id, kind)`` for the current street, or ``None``.

    ``None`` means the stream cannot support a label: wrong hand, a street
    change while we are still preflop, a bet event (impossible preflop), or an
    event we cannot attribute to a known seat.
    """
    if history is None:
        return None
    known_seats = {player.seat for player in state.players}
    collected: list[tuple[int, str]] = []
    for event in history:
        if not isinstance(event, StateEvent):
            return None
        if event.hand_id != state.hand_id:
            # An earlier hand still sitting in the buffer: not this street.
            continue
        if event.event_type is EventType.STREET_CHANGE:
            # The preflop segment ended; state still says preflop.
            return None
        if event.event_type not in _PLAYER_EVENTS:
            continue
        if event.event_type is EventType.BET:
            # No bet events exist preflop -- the first aggression is a raise.
            return None
        seat_id = event.payload.get("seat_id")
        if not isinstance(seat_id, int) or isinstance(seat_id, bool):
            return None
        if seat_id not in known_seats:
            return None
        collected.append((seat_id, _KIND_BY_EVENT[event.event_type]))
    return tuple(collected)


def _classify(actions: tuple[tuple[int, str], ...]) -> str:
    """Classify an ordered action sequence into the action-line vocabulary."""
    aggressive = [
        index for index, (_, kind) in enumerate(actions)
        if kind in _AGGRESSIVE_KINDS
    ]
    if any(kind == "jam" for _, kind in actions):
        return "all_in"
    if not aggressive:
        limps = sum(1 for _, kind in actions if kind == "call")
        if limps == 0:
            return "unopened"
        return "limp" if limps == 1 else "multi_limp"
    if len(aggressive) == 1:
        first = aggressive[0]
        if any(kind == "call" for _, kind in actions[:first]):
            return "iso_raise"
        return "raise"
    if len(aggressive) == 2:
        between = actions[aggressive[0] + 1:aggressive[1]]
        if any(kind == "call" for _, kind in between):
            return "squeeze"
        return "three_bet"
    return "four_bet"
