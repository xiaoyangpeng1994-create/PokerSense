"""Explicit JSON-safe serialization for strategy context and Advice v1."""

from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from poker_engine.core.enums import (
    ActionType,
    PlayerStatus,
    Position,
    Rank,
    Street,
    Suit,
)
from poker_engine.core.errors import SerializationError
from poker_engine.core.events import StateEvent
from poker_engine.core.request_context import RequestContext
from poker_engine.core.serialization import deserialize, serialize
from poker_engine.core.value_objects import Card, ChipAmount, ChipDelta

from .advice import Advice, AdviceStatus
from .contracts import (
    ActionAmountSemantics,
    ContextQuality,
    DecisionContext,
    DecisionSeat,
    EffectiveStack,
    GameConfig,
    GameType,
    InputProvenance,
    InputSource,
    LegalAction,
    PotState,
    QualityStatus,
    RangeDistribution,
)
from .provider import ActionOption, MatchDimension, MatchKind
from .safety import GateResult, GateStatus


STRATEGY_SCHEMA_VERSION = 1


def strategy_serialize(value: Advice | DecisionContext) -> dict[str, Any]:
    if isinstance(value, Advice):
        return _advice_to_dict(value)
    if isinstance(value, DecisionContext):
        return _context_to_dict(value)
    raise SerializationError(
        f"unsupported strategy type {type(value).__name__}"
    )


def strategy_deserialize(
    type_: type[Advice] | type[DecisionContext],
    data: Any,
) -> Advice | DecisionContext:
    _require_envelope(data)
    if type_ is Advice and data["type"] == "Advice":
        return _advice_from_dict(data)
    if type_ is DecisionContext and data["type"] == "DecisionContext":
        return _context_from_dict(data)
    raise SerializationError(
        f"strategy type mismatch: requested {type_.__name__}, "
        f"payload has {data.get('type')!r}"
    )


def _require_envelope(data: Any) -> None:
    if not isinstance(data, dict):
        raise SerializationError("strategy payload must be a dict")
    if data.get("schema_version") != STRATEGY_SCHEMA_VERSION:
        raise SerializationError("unsupported strategy schema_version")
    if data.get("type") not in ("Advice", "DecisionContext"):
        raise SerializationError("unknown strategy payload type")


def _advice_to_dict(value: Advice) -> dict[str, Any]:
    return {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "type": "Advice",
        "hand_id": value.hand_id,
        "state_version": value.state_version,
        "request_id": value.request_id,
        "player_count": value.player_count,
        "active_player_count": value.active_player_count,
        "status": value.status.value,
        "action_probabilities": _action_decimal_map(value.action_probabilities),
        "recommended_sizes": {
            action.value: [_money(item) for item in sizes]
            for action, sizes in sorted(
                value.recommended_sizes.items(), key=lambda item: item[0].value
            )
        },
        "action_options": [
            {
                "action": option.action.value,
                "probability": str(option.probability),
                "amount": (
                    _money(option.amount) if option.amount is not None else None
                ),
                "source_label": option.source_label,
            }
            for option in value.action_options
        ],
        "action_ev": {
            action.value: _money(amount)
            for action, amount in sorted(
                value.action_ev.items(), key=lambda item: item[0].value
            )
        },
        "ev_gap": _money(value.ev_gap) if value.ev_gap is not None else None,
        "preferred_action": (
            value.preferred_action.value
            if value.preferred_action is not None else None
        ),
        "math_report": _generic(value.math_report),
        "strategy_source": value.strategy_source,
        "strategy_version": value.strategy_version,
        "match_kind": value.match_kind.value if value.match_kind else None,
        "state_match_score": value.state_match_score,
        "match_dimensions": [
            {
                "name": item.name,
                "requested": item.requested,
                "matched": item.matched,
                "distance": str(item.distance),
                "maximum_distance": str(item.maximum_distance),
            }
            for item in value.match_dimensions
        ],
        "confidence": value.confidence,
        "confidence_factors": dict(value.confidence_factors),
        "missing_confidence_factors": list(value.missing_confidence_factors),
        "evidence": list(value.evidence),
        "input_provenance": [
            {
                "field_name": item.field_name,
                "source": item.source.value,
                "status": item.status.value,
                "confidence": item.confidence,
                "evidence_ref": item.evidence_ref,
                "observed_at": (
                    item.observed_at.isoformat() if item.observed_at else None
                ),
            }
            for item in value.input_provenance
        ],
        "evidence_chain_id": value.evidence_chain_id,
        "evidence_complete": value.evidence_complete,
        "missing_evidence": list(value.missing_evidence),
        "assumptions": list(value.assumptions),
        "missing_inputs": list(value.missing_inputs),
        "rejection_reasons": list(value.rejection_reasons),
        "gate_results": [
            {
                "name": item.name,
                "status": item.status.value,
                "reasons": list(item.reasons),
            }
            for item in value.gate_results
        ],
        "expires_at": value.expires_at.isoformat(),
    }


def _advice_from_dict(data: Mapping[str, Any]) -> Advice:
    return Advice(
        hand_id=data["hand_id"],
        state_version=data["state_version"],
        request_id=data["request_id"],
        player_count=data["player_count"],
        active_player_count=data["active_player_count"],
        status=AdviceStatus(data["status"]),
        action_probabilities={
            ActionType(action): Decimal(probability)
            for action, probability in data["action_probabilities"].items()
        },
        recommended_sizes={
            ActionType(action): tuple(ChipAmount(item) for item in sizes)
            for action, sizes in data["recommended_sizes"].items()
        },
        action_options=tuple(
            ActionOption(
                action=ActionType(item["action"]),
                probability=Decimal(item["probability"]),
                amount=(
                    ChipAmount(item["amount"])
                    if item["amount"] is not None else None
                ),
                source_label=item.get("source_label"),
            )
            for item in data.get("action_options", [])
        ),
        action_ev={
            ActionType(action): ChipDelta(value)
            for action, value in data["action_ev"].items()
        },
        ev_gap=ChipDelta(data["ev_gap"]) if data["ev_gap"] is not None else None,
        preferred_action=(
            ActionType(data["preferred_action"])
            if data["preferred_action"] is not None else None
        ),
        math_report=_generic_back(data["math_report"]),
        strategy_source=data["strategy_source"],
        strategy_version=data["strategy_version"],
        match_kind=(
            MatchKind(data["match_kind"])
            if data["match_kind"] is not None else None
        ),
        state_match_score=data["state_match_score"],
        match_dimensions=tuple(
            MatchDimension(
                item["name"],
                item["requested"],
                item["matched"],
                Decimal(item["distance"]),
                Decimal(item["maximum_distance"]),
            )
            for item in data.get("match_dimensions", [])
        ),
        confidence=data["confidence"],
        confidence_factors=data.get("confidence_factors", {}),
        missing_confidence_factors=tuple(
            data.get("missing_confidence_factors", [])
        ),
        evidence=tuple(data["evidence"]),
        input_provenance=tuple(
            InputProvenance(
                field_name=item["field_name"],
                source=InputSource(item["source"]),
                status=QualityStatus(item["status"]),
                confidence=item["confidence"],
                evidence_ref=item["evidence_ref"],
                observed_at=(
                    _datetime(item["observed_at"])
                    if item["observed_at"] is not None else None
                ),
            )
            for item in data.get("input_provenance", [])
        ),
        evidence_chain_id=data.get("evidence_chain_id"),
        evidence_complete=data.get("evidence_complete", False),
        missing_evidence=tuple(data.get("missing_evidence", [])),
        assumptions=tuple(data["assumptions"]),
        missing_inputs=tuple(data["missing_inputs"]),
        rejection_reasons=tuple(data["rejection_reasons"]),
        gate_results=tuple(
            GateResult(
                item["name"],
                GateStatus(item["status"]),
                tuple(item.get("reasons", [])),
            )
            for item in data.get("gate_results", [])
        ),
        expires_at=_datetime(data["expires_at"]),
    )


def _context_to_dict(value: DecisionContext) -> dict[str, Any]:
    return {
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "type": "DecisionContext",
        "request": {
            "hand_id": value.request.hand_id,
            "state_version": value.request.state_version,
            "request_id": value.request.request_id,
            "requested_at": value.request.requested_at.isoformat(),
            "expires_at": (
                value.request.expires_at.isoformat()
                if value.request.expires_at else None
            ),
            "deadline_ms": value.request.deadline_ms,
        },
        "game_config": {
            "variant": value.game_config.variant,
            "game_type": value.game_config.game_type.value,
            "max_seats": value.game_config.max_seats,
            "dealt_player_count": value.game_config.dealt_player_count,
            "small_blind": _money(value.game_config.small_blind),
            "big_blind": _money(value.game_config.big_blind),
            "ante": _money(value.game_config.ante),
            "rake_percent": str(value.game_config.rake_percent),
            "rake_cap": _money(value.game_config.rake_cap),
            "minimum_chip": _money(value.game_config.minimum_chip),
        },
        "seats": [_seat_to_dict(item) for item in value.seats],
        "hero_seat": value.hero_seat,
        "actor_seat": value.actor_seat,
        "active_seats": list(value.active_seats),
        "hero_cards": [str(item) for item in value.hero_cards],
        "board_cards": [str(item) for item in value.board_cards],
        "street": value.street.value,
        "pots": [
            {
                "pot_id": item.pot_id,
                "amount": _money(item.amount),
                "eligible_seats": list(item.eligible_seats),
            }
            for item in value.pots
        ],
        "legal_actions": [
            {
                "action": item.action.value,
                "min_amount": _money(item.min_amount),
                "max_amount": _money(item.max_amount),
                "amount_semantics": item.amount_semantics.value,
            }
            for item in value.legal_actions
        ],
        "action_history": [serialize(item) for item in value.action_history],
        "effective_stacks": [
            {"opponent_seat": item.opponent_seat,
             "amount": _money(item.amount)}
            for item in value.effective_stacks
        ],
        "hero_range": (
            _range_to_dict(value.hero_range) if value.hero_range else None
        ),
        "villain_ranges": [
            _range_to_dict(item) for item in value.villain_ranges
        ],
        "input_quality": {
            "overall_confidence": value.input_quality.overall_confidence,
            "field_confidences": dict(value.input_quality.field_confidences),
            "hard_failures": list(value.input_quality.hard_failures),
        },
        "input_provenance": [
            {
                "field_name": item.field_name,
                "source": item.source.value,
                "status": item.status.value,
                "confidence": item.confidence,
                "evidence_ref": item.evidence_ref,
                "observed_at": (
                    item.observed_at.isoformat() if item.observed_at else None
                ),
            }
            for item in value.input_provenance
        ],
        "missing_fields": list(value.missing_fields),
        "assumptions": list(value.assumptions),
        "action_line": value.action_line,
        "effective_stack_bb": (
            str(value.effective_stack_bb)
            if value.effective_stack_bb is not None else None
        ),
    }


def _context_from_dict(data: Mapping[str, Any]) -> DecisionContext:
    request = data["request"]
    config = data["game_config"]
    quality = data["input_quality"]
    return DecisionContext(
        request=RequestContext(
            hand_id=request["hand_id"],
            state_version=request["state_version"],
            request_id=request["request_id"],
            requested_at=_datetime(request["requested_at"]),
            expires_at=(
                _datetime(request["expires_at"])
                if request["expires_at"] is not None else None
            ),
            deadline_ms=request["deadline_ms"],
        ),
        game_config=GameConfig(
            variant=config["variant"],
            game_type=GameType(config["game_type"]),
            max_seats=config["max_seats"],
            dealt_player_count=config["dealt_player_count"],
            small_blind=ChipAmount(config["small_blind"]),
            big_blind=ChipAmount(config["big_blind"]),
            ante=ChipAmount(config["ante"]),
            rake_percent=Decimal(config["rake_percent"]),
            rake_cap=ChipAmount(config["rake_cap"]),
            minimum_chip=ChipAmount(config["minimum_chip"]),
        ),
        seats=tuple(_seat_from_dict(item) for item in data["seats"]),
        hero_seat=data["hero_seat"],
        actor_seat=data["actor_seat"],
        active_seats=tuple(data["active_seats"]),
        hero_cards=tuple(_card(item) for item in data["hero_cards"]),
        board_cards=tuple(_card(item) for item in data["board_cards"]),
        street=Street(data["street"]),
        pots=tuple(
            PotState(
                pot_id=item["pot_id"],
                amount=ChipAmount(item["amount"]),
                eligible_seats=tuple(item["eligible_seats"]),
            )
            for item in data["pots"]
        ),
        legal_actions=tuple(
            LegalAction(
                action=ActionType(item["action"]),
                min_amount=ChipAmount(item["min_amount"]),
                max_amount=ChipAmount(item["max_amount"]),
                amount_semantics=ActionAmountSemantics(
                    item.get("amount_semantics", "none" if item["action"] in (
                        "fold", "check"
                    ) else "additional" if item["action"] in (
                        "call", "all_in", "post_sb", "post_bb", "post_ante"
                    ) else "total_street")
                ),
            )
            for item in data["legal_actions"]
        ),
        action_history=tuple(
            deserialize(StateEvent, item) for item in data["action_history"]
        ),
        effective_stacks=tuple(
            EffectiveStack(
                opponent_seat=item["opponent_seat"],
                amount=ChipAmount(item["amount"]),
            )
            for item in data["effective_stacks"]
        ),
        hero_range=(
            _range_from_dict(data["hero_range"])
            if data["hero_range"] is not None else None
        ),
        villain_ranges=tuple(
            _range_from_dict(item) for item in data["villain_ranges"]
        ),
        input_quality=ContextQuality(
            overall_confidence=quality["overall_confidence"],
            field_confidences=quality["field_confidences"],
            hard_failures=tuple(quality["hard_failures"]),
        ),
        input_provenance=tuple(
            InputProvenance(
                field_name=item["field_name"],
                source=InputSource(item["source"]),
                status=QualityStatus(item["status"]),
                confidence=item["confidence"],
                evidence_ref=item["evidence_ref"],
                observed_at=(
                    _datetime(item["observed_at"])
                    if item["observed_at"] is not None else None
                ),
            )
            for item in data["input_provenance"]
        ),
        missing_fields=tuple(data["missing_fields"]),
        assumptions=tuple(data["assumptions"]),
        action_line=data["action_line"],
        effective_stack_bb=(
            Decimal(data["effective_stack_bb"])
            if data["effective_stack_bb"] is not None else None
        ),
    )


def _seat_to_dict(value: DecisionSeat) -> dict[str, Any]:
    return {
        "seat_id": value.seat_id,
        "player_id": value.player_id,
        "position": value.position.value,
        "stack": _money(value.stack),
        "street_committed": _money(value.street_committed),
        "hand_committed": _money(value.hand_committed),
        "status": value.status.value,
        "occupied": value.occupied,
        "is_hero": value.is_hero,
        "is_dealer": value.is_dealer,
    }


def _seat_from_dict(data: Mapping[str, Any]) -> DecisionSeat:
    return DecisionSeat(
        seat_id=data["seat_id"],
        player_id=data["player_id"],
        position=Position(data["position"]),
        stack=ChipAmount(data["stack"]),
        street_committed=ChipAmount(data["street_committed"]),
        hand_committed=ChipAmount(data["hand_committed"]),
        status=PlayerStatus(data["status"]),
        occupied=data["occupied"],
        is_hero=data["is_hero"],
        is_dealer=data["is_dealer"],
    )


def _range_to_dict(value: RangeDistribution) -> dict[str, Any]:
    return {
        "seat_id": value.seat_id,
        "combo_weights": {
            combo: str(weight)
            for combo, weight in sorted(value.combo_weights.items())
        },
        "source": value.source,
        "source_version": value.source_version,
        "entropy": str(value.entropy) if value.entropy is not None else None,
        "effective_sample_size": value.effective_sample_size,
        "confidence": value.confidence,
    }


def _range_from_dict(data: Mapping[str, Any]) -> RangeDistribution:
    return RangeDistribution(
        seat_id=data["seat_id"],
        combo_weights={
            combo: Decimal(weight)
            for combo, weight in data["combo_weights"].items()
        },
        source=data["source"],
        source_version=data["source_version"],
        entropy=Decimal(data["entropy"]) if data["entropy"] is not None else None,
        effective_sample_size=data["effective_sample_size"],
        confidence=data["confidence"],
    )


def _action_decimal_map(value: Mapping[ActionType, Decimal]) -> dict[str, str]:
    return {
        action.value: str(probability)
        for action, probability in sorted(
            value.items(), key=lambda item: item[0].value
        )
    }


def _money(value: ChipAmount | ChipDelta) -> str:
    return str(value.value)


def _datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SerializationError("datetime must be an ISO-8601 string")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.tzinfo.utcoffset(result) is None:
        raise SerializationError("datetime must be timezone-aware")
    return result


def _card(value: str) -> Card:
    if not isinstance(value, str) or len(value) != 2:
        raise SerializationError("card must be a two-character string")
    try:
        return Card(Rank(value[0]), Suit(value[1]))
    except ValueError as exc:
        raise SerializationError(f"invalid card {value!r}") from exc


def _generic(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("cannot serialize non-finite float")
        return value
    if isinstance(value, Decimal):
        return {"__strategy_type__": "Decimal", "value": str(value)}
    if isinstance(value, ChipAmount):
        return {"__strategy_type__": "ChipAmount", "value": _money(value)}
    if isinstance(value, ChipDelta):
        return {"__strategy_type__": "ChipDelta", "value": _money(value)}
    if isinstance(value, datetime):
        return {"__strategy_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, Enum):
        return {"__strategy_type__": "enum", "value": value.value}
    if isinstance(value, (tuple, list)):
        return [_generic(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _generic(item) for key, item in value.items()}
    raise SerializationError(f"unsupported generic type {type(value).__name__}")


def _generic_back(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return tuple(_generic_back(item) for item in value)
    if isinstance(value, dict):
        tag = value.get("__strategy_type__")
        if tag is None:
            return {key: _generic_back(item) for key, item in value.items()}
        if tag == "Decimal":
            return Decimal(value["value"])
        if tag == "ChipAmount":
            return ChipAmount(value["value"])
        if tag == "ChipDelta":
            return ChipDelta(value["value"])
        if tag == "datetime":
            return _datetime(value["value"])
        if tag == "enum":
            return value["value"]
        raise SerializationError(f"unknown generic strategy tag {tag!r}")
    raise SerializationError("unsupported generic serialized value")


__all__ = [
    "STRATEGY_SCHEMA_VERSION",
    "strategy_deserialize",
    "strategy_serialize",
]
