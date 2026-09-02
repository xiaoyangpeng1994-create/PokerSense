"""Rules-derived legal actions, side pots, and DecisionContext building."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from poker_engine.core._freeze import freeze_mapping
from poker_engine.core.enums import ActionType, PlayerStatus, Position
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.events import StateEvent
from poker_engine.core.opponents import PlayerState
from poker_engine.core.request_context import RequestContext
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import ChipAmount

from .contracts import (
    ActionAmountSemantics,
    ContextQuality,
    DecisionContext,
    DecisionSeat,
    EffectiveStack,
    GameConfig,
    InputProvenance,
    LegalAction,
    PotState,
    RangeDistribution,
)
from .context_factory import ContextQualityPolicy, aggregate_context_quality
from .input_provenance import ProvenanceCollection


@dataclass(frozen=True)
class PotCalculation:
    pots: tuple[PotState, ...]
    uncalled_returns: Mapping[int, ChipAmount]

    def __post_init__(self) -> None:
        pots = tuple(self.pots)
        if not all(isinstance(pot, PotState) for pot in pots):
            raise TypeError("pots must contain PotState values")
        object.__setattr__(self, "pots", pots)
        returns = dict(self.uncalled_returns)
        if not all(
            isinstance(seat, int) and not isinstance(seat, bool) and seat >= 0
            for seat in returns
        ):
            raise TypeError("uncalled return keys must be non-negative seats")
        if not all(isinstance(amount, ChipAmount) for amount in returns.values()):
            raise TypeError("uncalled return values must be ChipAmount")
        object.__setattr__(self, "uncalled_returns", freeze_mapping(returns))


def calculate_side_pots(
    seats: tuple[DecisionSeat, ...],
    *,
    settle_uncalled: bool = True,
) -> PotCalculation:
    """Split commitments into pots and, when settled, unmatched returns.

    During an open betting round, active players who have not matched the
    current level can still contest it. In that mode the unmatched tranche
    remains a provisional pot. Once betting is closed, a one-contributor top
    tranche is returned as uncalled chips.
    """
    seats = tuple(seats)
    if not seats:
        raise ValueError("seats cannot be empty")
    if not all(isinstance(seat, DecisionSeat) for seat in seats):
        raise TypeError("seats must contain DecisionSeat values")
    if not isinstance(settle_uncalled, bool):
        raise TypeError("settle_uncalled must be a bool")
    commitments = {
        seat.seat_id: seat.hand_committed.value
        for seat in seats if seat.occupied and seat.hand_committed.value > 0
    }
    if not commitments:
        return PotCalculation((), {})
    by_id = {seat.seat_id: seat for seat in seats}
    levels = sorted(set(commitments.values()))
    previous = Decimal("0")
    tranches: list[tuple[Decimal, tuple[int, ...]]] = []
    returns: dict[int, Decimal] = {}
    for level in levels:
        contributors = tuple(sorted(
            seat_id for seat_id, amount in commitments.items()
            if amount >= level
        ))
        amount = (level - previous) * len(contributors)
        previous = level
        if amount <= 0:
            continue
        if len(contributors) == 1 and settle_uncalled:
            seat_id = contributors[0]
            returns[seat_id] = returns.get(seat_id, Decimal("0")) + amount
            continue
        if settle_uncalled:
            eligible = tuple(
                seat_id for seat_id in contributors
                if by_id[seat_id].status in (
                    PlayerStatus.ACTIVE, PlayerStatus.ALL_IN,
                )
            )
        else:
            eligible = tuple(sorted({
                seat.seat_id
                for seat in seats
                if seat.occupied
                and seat.status is PlayerStatus.ACTIVE
                and seat.stack.value > 0
            } | {
                seat_id
                for seat_id in contributors
                if by_id[seat_id].status is PlayerStatus.ALL_IN
            }))
        if not eligible:
            raise InvalidStateError("pot tranche has no eligible player")
        if tranches and tranches[-1][1] == eligible:
            prior_amount, _ = tranches[-1]
            tranches[-1] = (prior_amount + amount, eligible)
        else:
            tranches.append((amount, eligible))
    pots = tuple(
        PotState(
            pot_id="main" if index == 0 else f"side-{index}",
            amount=ChipAmount(amount),
            eligible_seats=eligible,
        )
        for index, (amount, eligible) in enumerate(tranches)
    )
    return PotCalculation(
        pots,
        {seat_id: ChipAmount(amount) for seat_id, amount in returns.items()},
    )


def calculate_legal_actions(
    state: PokerState,
    game_config: GameConfig,
    *,
    minimum_raise_increment: ChipAmount | None = None,
) -> tuple[LegalAction, ...]:
    """Return legal Hero/actor actions using explicit amount semantics."""
    if not isinstance(state, PokerState):
        raise TypeError("state must be a PokerState")
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    if minimum_raise_increment is not None and not isinstance(
        minimum_raise_increment, ChipAmount
    ):
        raise TypeError("minimum_raise_increment must be ChipAmount or None")
    if state.actor is None:
        return ()
    actor = next((item for item in state.players if item.seat == state.actor), None)
    if actor is None or actor.status is not PlayerStatus.ACTIVE:
        return ()
    if actor.stack.value <= 0:
        return ()
    actions: list[LegalAction] = []
    to_call = state.to_call.value
    stack = actor.stack.value
    committed = actor.committed_this_street.value
    if to_call > 0:
        actions.append(_zero_action(ActionType.FOLD))
        if stack <= to_call:
            actions.append(LegalAction(
                ActionType.ALL_IN,
                ChipAmount(stack),
                ChipAmount(stack),
                ActionAmountSemantics.ADDITIONAL,
            ))
            return tuple(actions)
        actions.append(LegalAction(
            ActionType.CALL,
            ChipAmount(to_call),
            ChipAmount(to_call),
            ActionAmountSemantics.ADDITIONAL,
        ))
    else:
        actions.append(_zero_action(ActionType.CHECK))

    max_total = committed + stack
    if state.current_bet.value == 0:
        minimum = max(
            game_config.big_blind.value,
            game_config.minimum_chip.value,
        )
        if max_total >= minimum:
            actions.append(LegalAction(
                ActionType.BET,
                ChipAmount(minimum),
                ChipAmount(max_total),
                ActionAmountSemantics.TOTAL_STREET,
            ))
        else:
            actions.append(LegalAction(
                ActionType.ALL_IN,
                ChipAmount(stack),
                ChipAmount(stack),
                ActionAmountSemantics.ADDITIONAL,
            ))
        return tuple(actions)

    increment = (
        minimum_raise_increment.value
        if minimum_raise_increment is not None
        else game_config.big_blind.value
    )
    if increment <= 0:
        raise ValueError("minimum raise increment must be > 0")
    minimum_raise_to = state.current_bet.value + increment
    if max_total >= minimum_raise_to:
        actions.append(LegalAction(
            ActionType.RAISE,
            ChipAmount(minimum_raise_to),
            ChipAmount(max_total),
            ActionAmountSemantics.TOTAL_STREET,
        ))
    elif stack > to_call:
        actions.append(LegalAction(
            ActionType.ALL_IN,
            ChipAmount(stack),
            ChipAmount(stack),
            ActionAmountSemantics.ADDITIONAL,
        ))
    return tuple(actions)


def build_decision_context(
    state: PokerState,
    request: RequestContext,
    game_config: GameConfig,
    *,
    action_history: tuple[StateEvent, ...] = (),
    input_quality: ContextQuality | None = None,
    input_provenance: tuple[InputProvenance, ...] = (),
    collected_inputs: ProvenanceCollection | None = None,
    quality_policy: ContextQualityPolicy | None = None,
    hero_range: RangeDistribution | None = None,
    villain_ranges: tuple[RangeDistribution, ...] = (),
    action_line: str | None = None,
    assumptions: tuple[str, ...] = (),
    minimum_raise_increment: ChipAmount | None = None,
) -> DecisionContext:
    """Build a strategy context from the current immutable PokerState."""
    if not isinstance(state, PokerState):
        raise TypeError("state must be a PokerState")
    if not isinstance(request, RequestContext):
        raise TypeError("request must be a RequestContext")
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    if request.hand_id != state.hand_id or request.state_version != state.state_version:
        raise InvalidStateError("request must reference the supplied PokerState")
    seats = tuple(_decision_seat(player) for player in state.players)
    hero = next((seat for seat in seats if seat.is_hero), None)
    if hero is None:
        raise InvalidStateError("PokerState must identify one Hero seat")
    players_by_seat = {player.seat: player for player in state.players}
    active = tuple(
        seat.seat_id for seat in seats
        if seat.status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN)
        and players_by_seat[seat.seat_id].has_cards
    )
    if len(active) < 2:
        raise InvalidStateError("strategy context requires two active players")
    if input_quality is not None and quality_policy is not None:
        raise ValueError("provide input_quality or quality_policy, not both")
    if collected_inputs is not None:
        if not isinstance(collected_inputs, ProvenanceCollection):
            raise TypeError("collected_inputs must be a ProvenanceCollection")
        if input_provenance:
            raise ValueError(
                "provide input_provenance or collected_inputs, not both"
            )
        input_provenance = collected_inputs.provenance
    if quality_policy is not None:
        quality = aggregate_context_quality(
            tuple(input_provenance), quality_policy
        )
    else:
        quality = input_quality or ContextQuality(1.0)
    missing = []
    hard_failures = list(quality.hard_failures)
    if len(state.hero_cards) != 2:
        missing.append("hero_cards")
    if hero.position is Position.UNKNOWN:
        missing.append("hero_position")
    if state.actor is None:
        missing.append("actor")
    if action_line is None:
        missing.append("action_line")
    legal_actions = calculate_legal_actions(
        state,
        game_config,
        minimum_raise_increment=minimum_raise_increment,
    )
    if state.actor == hero.seat_id and not legal_actions:
        missing.append("legal_actions")
    betting_open = state.actor is not None and any(
        seat.status is PlayerStatus.ACTIVE and seat.stack.value > 0
        for seat in seats
    )
    pot_calculation = calculate_side_pots(
        seats,
        settle_uncalled=not betting_open,
    )
    calculated_total = sum(
        (pot.amount.value for pot in pot_calculation.pots), Decimal("0")
    )
    if pot_calculation.uncalled_returns:
        hard_failures.append("uncalled_return_pending")
    if calculated_total != state.pot.value:
        hard_failures.append("commitment_breakdown_mismatch")
    pots = pot_calculation.pots
    if not pots and state.pot.value > 0:
        pots = (PotState("main", state.pot, active),)
    quality = ContextQuality(
        quality.overall_confidence,
        quality.field_confidences,
        tuple(dict.fromkeys(hard_failures)),
    )
    effective = tuple(
        EffectiveStack(
            seat.seat_id,
            ChipAmount(min(hero.stack.value, seat.stack.value)),
        )
        for seat in seats if seat.seat_id in active and seat.seat_id != hero.seat_id
    )
    effective_stack_bb = min(
        (item.amount.value for item in effective),
        default=Decimal("0"),
    ) / game_config.big_blind.value
    return DecisionContext(
        request=request,
        game_config=game_config,
        seats=seats,
        hero_seat=hero.seat_id,
        actor_seat=state.actor,
        active_seats=active,
        hero_cards=state.hero_cards,
        board_cards=state.board_cards,
        street=state.street,
        pots=pots,
        legal_actions=legal_actions,
        action_history=tuple(action_history),
        effective_stacks=effective,
        hero_range=hero_range,
        villain_ranges=tuple(villain_ranges),
        input_quality=quality,
        input_provenance=tuple(input_provenance),
        missing_fields=tuple(dict.fromkeys(missing)),
        assumptions=tuple(assumptions),
        action_line=action_line,
        effective_stack_bb=effective_stack_bb,
    )


def _decision_seat(player: PlayerState) -> DecisionSeat:
    return DecisionSeat(
        seat_id=player.seat,
        player_id=player.player_id,
        position=player.position,
        stack=player.stack,
        street_committed=player.committed_this_street,
        hand_committed=player.committed_this_hand,
        status=player.status,
        occupied=True,
        is_hero=player.is_hero,
        is_dealer=player.is_dealer,
    )


def _zero_action(action: ActionType) -> LegalAction:
    return LegalAction(
        action,
        ChipAmount.zero(),
        ChipAmount.zero(),
        ActionAmountSemantics.NONE,
    )


__all__ = [
    "PotCalculation",
    "build_decision_context",
    "calculate_legal_actions",
    "calculate_side_pots",
]
