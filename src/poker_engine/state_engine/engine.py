"""State Engine — pure, deterministic state reconciliation.

Converts ``Previous PokerState + RawObservation + StateContext`` into a
``StateTransitionResult`` (canonical state + events + validation).

This is the ``Observation → Canonical State`` transformation layer. It MUST
remain pure: no datetime.now(), no random, no I/O, no global mutable state.

Task 3 v1 scope (per frozen contract + review decisions):
- Canonical updates allowed for: hero_cards, board_cards, street, pot.
- NOT updated (semantics undefined in frozen contract): actor, dealer_pos,
  stacks, bet_size, players, current_bet, to_call.
- Events: STREET_CHANGE and DEAL only (deterministic order).
- No bootstrap (previous_state must be provided).
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_engine.core.enums import Street
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.observation import RawObservation, ValidationStatus
from poker_engine.core.state import PokerState, StateContext, ValidationResult
from poker_engine.core.value_objects import Card

from .errors import StateEngineError

_STREET_ORDER: dict[Street, int] = {
    Street.PREFLOP: 0,
    Street.FLOP: 1,
    Street.TURN: 2,
    Street.RIVER: 3,
    Street.SHOWDOWN: 4,
}

# Core invariant exceptions raised while constructing a PokerState. These are
# domain conflicts, not programmer errors, so they become an invalid result.
_CORE_VALIDATION_ERRORS = (InvalidStateError, ValueError, TypeError)


@dataclass(frozen=True)
class StateTransitionResult:
    """Outcome of one transition.

    ``state`` is always a PokerState (never None): material change -> new
    state; no-op or invalid -> previous_state.
    """

    state: PokerState
    events: tuple[StateEvent, ...]
    validation: ValidationResult
    changed: bool


def _card_tuple(value: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(value)


def _observe_status(
    status: ValidationStatus, name: str, warnings: list[str]
) -> bool:
    """Return True only when an observation may be merged as a candidate.

    VALID          -> candidate (caller still checks value is not None).
    UNKNOWN        -> retain previous, no warning.
    LOW_CONFIDENCE  -> retain previous + deterministic warning.
    CONFLICT        -> retain previous + deterministic warning.

    No confidence numeric threshold is read here.
    """
    if status is ValidationStatus.VALID:
        return True
    if status is ValidationStatus.UNKNOWN:
        return False
    # LOW_CONFIDENCE / CONFLICT
    warnings.append(f"{name} ignored (status={status.value})")
    return False


def _is_board_prefix(previous: tuple[Card, ...], new: tuple[Card, ...]) -> bool:
    """True if ``previous`` is a strict prefix of ``new`` (board growth)."""
    return len(new) > len(previous) and new[: len(previous)] == previous


class StateEngine:
    """Stateless, pure state reconciliation engine."""

    def transition(
        self,
        previous_state: PokerState,
        observation: RawObservation,
        context: StateContext,
    ) -> StateTransitionResult:
        # --- programmer / contract errors (raise) ---
        if not isinstance(previous_state, PokerState):
            raise TypeError("previous_state must be a PokerState")
        if not isinstance(observation, RawObservation):
            raise TypeError("observation must be a RawObservation")
        if not isinstance(context, StateContext):
            raise TypeError("context must be a StateContext")
        if context.previous_state is not None and (
            context.previous_state != previous_state
        ):
            raise StateEngineError(
                "context.previous_state does not match previous_state argument"
            )

        warnings: list[str] = []

        # --- merge candidates (start from previous canonical values) ---
        hero = previous_state.hero_cards
        board = previous_state.board_cards
        street = previous_state.street
        pot = previous_state.pot

        hero_changed = False
        board_changed = False
        street_changed = False
        pot_changed = False

        # --- hero_cards ---
        hf = observation.hero_cards
        if _observe_status(hf.validation_status, "hero_cards", warnings) and (
            hf.value is not None
        ):
            new_hero = _card_tuple(hf.value)
            prev_hero = previous_state.hero_cards
            if len(new_hero) < len(prev_hero):
                # 2 -> 0, or any count decrease: invalid (no hand boundary).
                return self._invalid(
                    previous_state, "hero_cards regression"
                )
            if len(new_hero) == len(prev_hero) and new_hero != prev_hero:
                # existing 2 -> different 2: card identity changed -> invalid.
                return self._invalid(
                    previous_state, "hero_cards identity changed"
                )
            if new_hero != prev_hero:
                hero = new_hero
                hero_changed = True

        # --- board_cards ---
        bf = observation.board_cards
        if _observe_status(bf.validation_status, "board_cards", warnings) and (
            bf.value is not None
        ):
            new_board = _card_tuple(bf.value)
            prev_board = previous_state.board_cards
            if len(new_board) < len(prev_board):
                # 4 -> 3 / 5 -> 4 / etc: invalid.
                return self._invalid(
                    previous_state, "board_cards regression"
                )
            if len(new_board) == len(prev_board) and new_board != prev_board:
                # same size but a confirmed card was replaced: invalid.
                return self._invalid(
                    previous_state, "board_cards identity changed"
                )
            if len(new_board) > len(prev_board):
                # growth: previous board must be a strict prefix.
                if not _is_board_prefix(prev_board, new_board):
                    return self._invalid(
                        previous_state, "board_cards prefix violated"
                    )
                board = new_board
                board_changed = True

        # --- street ---
        sf = observation.street
        if _observe_status(sf.validation_status, "street", warnings) and (
            sf.value is not None
        ):
            new_street = sf.value
            prev_order = _STREET_ORDER[previous_state.street]
            new_order = _STREET_ORDER[new_street]
            if new_order < prev_order:
                return self._invalid(previous_state, "street regression")
            if new_order > prev_order:
                street = new_street
                street_changed = True

        # --- pot ---
        pf = observation.pot
        if _observe_status(pf.validation_status, "pot", warnings) and (
            pf.value is not None
        ):
            new_pot = pf.value
            if new_pot < previous_state.pot:
                # pot regression: keep canonical, warn (NOT invalid whole frame)
                warnings.append("pot regression ignored (value decrease)")
            elif new_pot != previous_state.pot:
                pot = new_pot
                pot_changed = True

        # --- un-resolved semantics: emit warnings, never mutate state ---
        for name, field in (
            ("stacks", observation.stacks),
            ("bet_size", observation.bet_size),
            ("action", observation.action),
            ("dealer_pos", observation.dealer_pos),
            ("actor", observation.actor),
        ):
            if (
                field.validation_status is not ValidationStatus.VALID
                and field.validation_status is not ValidationStatus.UNKNOWN
                or field.value is not None
            ):
                # Only warn when the field actually carried some signal we are
                # choosing to ignore (LOW_CONFIDENCE/CONFLICT or a present value
                # whose semantics we cannot resolve).
                if field.validation_status in (
                    ValidationStatus.LOW_CONFIDENCE,
                    ValidationStatus.CONFLICT,
                ):
                    warnings.append(
                        f"{name} ignored (semantics unresolved in v1)"
                    )

        # --- no-op? ---
        material_change = (
            hero_changed or board_changed or street_changed or pot_changed
        )
        if not material_change:
            return StateTransitionResult(
                state=previous_state,
                events=(),
                validation=ValidationResult(
                    is_valid=True, errors=(), warnings=tuple(warnings)
                ),
                changed=False,
            )

        # --- material change: construct a new canonical state ---
        try:
            new_state = PokerState(
                state_version=previous_state.state_version + 1,
                hand_id=previous_state.hand_id,
                street=street,
                hero_cards=hero,
                board_cards=board,
                players=previous_state.players,
                pot=pot,
                current_bet=previous_state.current_bet,
                to_call=previous_state.to_call,
                actor=previous_state.actor,
            )
        except _CORE_VALIDATION_ERRORS as exc:
            return self._invalid(previous_state, str(exc))

        # --- events (deterministic order: STREET_CHANGE then DEAL) ---
        events: list[StateEvent] = []
        if street_changed:
            events.append(
                StateEvent(
                    event_type=EventType.STREET_CHANGE,
                    hand_id=previous_state.hand_id,
                    state_version=new_state.state_version,
                    payload={
                        "from": previous_state.street.value,
                        "to": street.value,
                    },
                    timestamp=observation.timestamp,
                    source="state_engine",
                )
            )
        if board_changed and len(board) > len(previous_state.board_cards):
            events.append(
                StateEvent(
                    event_type=EventType.DEAL,
                    hand_id=previous_state.hand_id,
                    state_version=new_state.state_version,
                    payload={"board_count": len(board)},
                    timestamp=observation.timestamp,
                    source="state_engine",
                )
            )

        return StateTransitionResult(
            state=new_state,
            events=tuple(events),
            validation=ValidationResult(
                is_valid=True, errors=(), warnings=tuple(warnings)
            ),
            changed=True,
        )

    def _invalid(
        self, previous_state: PokerState, message: str
    ) -> StateTransitionResult:
        return StateTransitionResult(
            state=previous_state,
            events=(),
            validation=ValidationResult(
                is_valid=False, errors=(message,), warnings=()
            ),
            changed=False,
        )


__all__ = ["StateEngine", "StateTransitionResult"]
