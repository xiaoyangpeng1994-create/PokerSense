"""Fail-closed AA visual-state to strategy-context bridge.

The bridge consumes only explicitly authoritative base-game evidence.  It does
not promote perception candidates, fill missing commitments, or reinterpret a
special-mode pause as an ordinary poker state.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping

from poker_engine.core.enums import PlayerStatus, Position, Rank, Street, Suit
from poker_engine.core.errors import InvalidStateError
from poker_engine.core.opponents import PlayerState
from poker_engine.core.request_context import RequestContext
from poker_engine.core.state import PokerState
from poker_engine.core.value_objects import Card, ChipAmount

from .contracts import (
    ContextQuality,
    DecisionContext,
    GameConfig,
    InputProvenance,
    InputSource,
    QualityStatus,
)
from .state import build_decision_context


AUTHORITY_FIELDS = (
    "hand_boundary",
    "cards",
    "pot",
    "stacks",
    "street_wagers",
    "hand_commitments",
    "actions",
    "participation",
    "dealer",
    "action_line",
)
_POSITION_ORDER = {
    6: (Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ,
        Position.CO),
    7: (Position.BTN, Position.SB, Position.BB, Position.UTG, Position.LJ,
        Position.HJ, Position.CO),
    8: (Position.BTN, Position.SB, Position.BB, Position.UTG, Position.UTG1,
        Position.LJ, Position.HJ, Position.CO),
}
_STATUS = {
    "active": PlayerStatus.ACTIVE,
    "folded": PlayerStatus.FOLDED,
    "all_in": PlayerStatus.ALL_IN,
    "waiting": PlayerStatus.SITTING_OUT,
    "empty": PlayerStatus.SITTING_OUT,
}


class AA8ShadowStatus(str, Enum):
    READY = "READY"
    WAITING = "WAITING"
    ABSTAIN = "ABSTAIN"
    DEFERRED_SPECIAL_MODE = "DEFERRED_SPECIAL_MODE"


@dataclass(frozen=True)
class AA8ShadowResult:
    status: AA8ShadowStatus
    reasons: tuple[str, ...]
    state: PokerState | None = None
    context: DecisionContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AA8ShadowStatus):
            raise TypeError("status must be AA8ShadowStatus")
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise TypeError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)
        if self.status is AA8ShadowStatus.READY and (
            self.state is None or self.context is None
        ):
            raise ValueError("READY requires state and context")
        if self.context is not None and self.state is None:
            raise ValueError("context cannot exist without state")


def assess_aa8_shadow_input(
    row: Mapping[str, Any], game_config: GameConfig,
) -> tuple[str, ...]:
    """Return deterministic blockers without constructing a canonical state."""
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    reasons = []
    if game_config.max_seats != 8 or game_config.dealt_player_count not in (6, 7, 8):
        reasons.append("aa_base_strategy_requires_6_to_8_players_on_8_seat_layout")
    scope = row.get("visual_scope") or {}
    deferred = tuple(scope.get("observed_deferred_modes") or ())
    if deferred:
        reasons.extend(f"deferred_special_mode:{mode}" for mode in deferred)
    elif (scope.get("scope_id") != "aa8_base_visual_v1"
          or scope.get("status") != "BASE_VISUAL_PASS"
          or scope.get("normal_mode_verified") is not True):
        reasons.append("base_visual_scope_not_verified")
    if row.get("scene_supported") is not True:
        reasons.append("scene_not_supported")
    authority = row.get("strategy_input_authority") or {}
    if authority.get("schema_version") != 1:
        reasons.append("strategy_authority_schema_missing")
    flags = authority.get("fields") or {}
    reasons.extend(
        f"canonical_field_missing:{name}"
        for name in AUTHORITY_FIELDS if flags.get(name) is not True
    )
    if authority.get("source_frame") != row.get("frame"):
        reasons.append("strategy_authority_frame_mismatch")
    if authority.get("source_sha256") != row.get("source_sha256"):
        reasons.append("strategy_authority_source_mismatch")
    state = row.get("observed_state_v2") or {}
    if state.get("authoritative_hand_boundary") is not True:
        reasons.append("hand_boundary_not_authoritative")
    if state.get("complete_legal_state") is not True:
        reasons.append("legal_state_incomplete")
    wagers = row.get("causal_street_wagers_v2") or {}
    if wagers.get("canonical_verified") is not True:
        reasons.append("street_wagers_not_canonical")
    if row.get("actions_complete_and_canonical_verified") is not True:
        reasons.append("actions_not_complete_or_canonical")
    if row.get("dealer_seat_canonical_verified") is not True:
        reasons.append("dealer_not_canonical")
    if row.get("hand_commitments_canonical_verified") is not True:
        reasons.append("hand_commitments_not_canonical")
    return tuple(dict.fromkeys(reasons))


def build_aa8_shadow_context(
    row: Mapping[str, Any],
    request: RequestContext,
    game_config: GameConfig,
) -> AA8ShadowResult:
    """Build a reusable DecisionContext or fail closed with exact blockers."""
    if not isinstance(request, RequestContext):
        raise TypeError("request must be RequestContext")
    blockers = assess_aa8_shadow_input(row, game_config)
    special = tuple(
        reason for reason in blockers if reason.startswith("deferred_special_mode:")
    )
    if special:
        return AA8ShadowResult(AA8ShadowStatus.DEFERRED_SPECIAL_MODE, blockers)
    if blockers:
        return AA8ShadowResult(AA8ShadowStatus.ABSTAIN, blockers)
    try:
        state = _build_state(row, request, game_config)
        authority = row["strategy_input_authority"]
        confidence = authority.get("confidence")
        if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not 0 <= confidence <= 1):
            raise ValueError("invalid_authority_confidence")
        evidence = authority.get("evidence_ref")
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("strategy_authority_evidence_missing")
        provenance = tuple(
            InputProvenance(
                name, InputSource.VISION, QualityStatus.VALID, float(confidence),
                f"{evidence}#{name}",
            )
            for name in AUTHORITY_FIELDS
        )
        action_line = row.get("action_line")
        if not isinstance(action_line, str) or not action_line:
            raise ValueError("action_line_missing")
        context = build_decision_context(
            state,
            request,
            game_config,
            input_quality=ContextQuality(
                float(confidence),
                {name: float(confidence) for name in AUTHORITY_FIELDS},
            ),
            input_provenance=provenance,
            action_line=action_line,
            assumptions=(
                "aa8_base_visual_scope_only",
                "insurance_mushroom_bomb_semantics_deferred",
            ),
        )
    except (
        KeyError, TypeError, ValueError, InvalidOperation, InvalidStateError,
    ) as exc:
        return AA8ShadowResult(
            AA8ShadowStatus.ABSTAIN,
            (f"invalid_canonical_strategy_input:{exc}",),
        )
    if context.actor_seat != context.hero_seat:
        return AA8ShadowResult(
            AA8ShadowStatus.WAITING, ("hero_not_actor",), state, context,
        )
    if not context.is_decision_ready:
        return AA8ShadowResult(
            AA8ShadowStatus.ABSTAIN,
            tuple(context.missing_fields) + context.input_quality.hard_failures,
            state,
            context,
        )
    return AA8ShadowResult(AA8ShadowStatus.READY, (), state, context)


def _card(value: str) -> Card:
    if not isinstance(value, str) or len(value) != 2:
        raise ValueError("invalid_card")
    return Card(Rank(value[0]), Suit(value[1]))


def _money(value: Any, name: str) -> ChipAmount:
    if not isinstance(value, str):
        raise ValueError(f"{name}_missing")
    return ChipAmount(value)


def _build_state(
    row: Mapping[str, Any], request: RequestContext, game_config: GameConfig,
) -> PokerState:
    observed = row["observed_state_v2"]
    hand_id = observed.get("observed_epoch")
    if hand_id != request.hand_id:
        raise ValueError("request_hand_mismatch")
    version = row.get("canonical_state_version")
    if version != request.state_version:
        raise ValueError("request_state_version_mismatch")
    street = Street(observed.get("street_candidate"))
    hero_cards = tuple(_card(value) for value in row["cards"]["hero"])
    board = observed.get("board_candidate")
    board_cards = tuple(_card(value) for value in (board or ()))
    participants = observed.get("participants") or {}
    wagers = row["causal_street_wagers_v2"]["wagers"]
    commitments = row.get("hand_commitments") or {}
    stacks = row.get("stacks") or {}
    if any(set(values) != set(map(str, range(8))) for values in (
        participants, wagers, commitments, stacks,
    )):
        raise ValueError("eight_physical_seat_vectors_required")
    dealer = row.get("dealer_seat")
    if type(dealer) is not int or not 0 <= dealer < 8:
        raise ValueError("invalid_dealer_seat")
    occupied = sorted(
        int(seat) for seat, item in participants.items()
        if item.get("state") not in ("waiting", "empty")
    )
    if len(occupied) != game_config.dealt_player_count or dealer not in occupied:
        raise ValueError("dealt_count_or_dealer_mismatch")
    start = occupied.index(dealer)
    clockwise = occupied[start:] + occupied[:start]
    positions = dict(zip(clockwise, _POSITION_ORDER[len(occupied)]))
    players = []
    for seat in range(8):
        key = str(seat)
        status_name = participants[key].get("state")
        if status_name not in _STATUS:
            raise ValueError(f"participant_unknown:{key}")
        status = _STATUS[status_name]
        street_value = wagers[key]
        if isinstance(street_value, Mapping):
            if street_value.get("status") != "NOT_APPLICABLE":
                raise ValueError(f"street_wager_unknown:{key}")
            street_value = "0"
        stack_value = stacks[key].get("value")
        if stack_value is None and status is PlayerStatus.SITTING_OUT:
            stack_value = "0"
        player = PlayerState(
            player_id="hero" if seat == 4 else f"aa8-seat-{seat}",
            seat=seat,
            position=positions.get(seat, Position.UNKNOWN),
            stack=_money(stack_value, f"stack_{key}"),
            committed_this_street=_money(street_value, f"street_wager_{key}"),
            committed_this_hand=_money(commitments[key], f"hand_commitment_{key}"),
            status=status,
            has_cards=status in (PlayerStatus.ACTIVE, PlayerStatus.ALL_IN),
            is_hero=seat == 4,
            is_dealer=seat == dealer,
        )
        players.append(player)
    actor = row.get("current_actor")
    if type(actor) is not int or actor not in occupied:
        raise ValueError("invalid_actor")
    by_seat = {player.seat: player for player in players}
    if by_seat[actor].status is not PlayerStatus.ACTIVE:
        raise ValueError("actor_not_active")
    current_bet = max(player.committed_this_street.value for player in players)
    to_call = current_bet - by_seat[actor].committed_this_street.value
    pot = _money(row["pot"].get("value"), "pot")
    total_committed = sum(
        (player.committed_this_hand.value for player in players), Decimal(0)
    )
    if total_committed != pot.value:
        raise ValueError("hand_commitments_do_not_equal_pot")
    return PokerState(
        state_version=version,
        hand_id=hand_id,
        street=street,
        hero_cards=hero_cards,
        board_cards=board_cards,
        players=tuple(players),
        pot=pot,
        current_bet=ChipAmount(current_bet),
        to_call=ChipAmount(to_call),
        actor=actor,
    )


__all__ = [
    "AA8ShadowResult",
    "AA8ShadowStatus",
    "AUTHORITY_FIELDS",
    "assess_aa8_shadow_input",
    "build_aa8_shadow_context",
]
