"""Serialization for Poker Intelligence Engine core contracts.

Centralized, explicit, reversible serializer for Phase 0 core objects.

Public API:
    serialize(obj) -> JSON-safe primitive (top-level carries schema_version)
    deserialize(type_, data) -> domain object

Constraints (see docs/serialization.md):
- Money stringified exactly (no float/round/quantize).
- datetime ISO-8601, timezone-aware preserved.
- Enum uses ``.value``; unknown value fails fast.
- No pickle / reflection / class-path magic.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .enums import ActionType, PlayerStatus, Position, Rank, Street, Suit
from .errors import SerializationError
from .events import EventType, StateEvent
from .hand import HandHistory, HandSummary
from .observation import (
    ObservationField,
    RawObservation,
    SlotObservation,
    ValidationStatus,
)
from .opponents import OpponentProfile, PlayerState
from .reports import (
    Decision,
    DecisionPath,
    EquityMethod,
    EquityReport,
    ReasoningReport,
    StrategyReport,
    StrategySource,
)
from .request_context import RequestContext
from .state import PokerState, StateContext, ValidationResult
from .value_objects import Card, ChipAmount, ChipDelta

SCHEMA_VERSION = 1
_TYPE_KEY = "__type__"
_SCHEMA_KEY = "schema_version"
_RESERVED_KEYS = frozenset({_TYPE_KEY, _SCHEMA_KEY})

# Enum registry: stable enum name -> Enum class, for recovering Enum values in
# generic positions (e.g. ObservationField.value, free mapping payloads).
_ENUM_REGISTRY: dict[str, type[Enum]] = {
    "ActionType": ActionType,
    "PlayerStatus": PlayerStatus,
    "Position": Position,
    "Rank": Rank,
    "Street": Street,
    "Suit": Suit,
    "EventType": EventType,
    "ValidationStatus": ValidationStatus,
    "EquityMethod": EquityMethod,
    "StrategySource": StrategySource,
    "DecisionPath": DecisionPath,
}
_ENUM_NAME_REGISTRY: dict[type[Enum], str] = {
    cls: name for name, cls in _ENUM_REGISTRY.items()
}


# --------------------------------------------------------------------------
# Serialization primitives
# --------------------------------------------------------------------------

def _enum(value: Enum) -> str:
    return value.value


def _dt(value: datetime) -> str:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise SerializationError("cannot serialize a naive datetime")
    return value.isoformat()


def _dt_back(data: Any) -> datetime:
    if not isinstance(data, str):
        raise SerializationError("datetime must be an ISO-8601 string")
    dt = datetime.fromisoformat(data)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise SerializationError("datetime must be timezone-aware")
    return dt


def _money_str(o: ChipAmount | ChipDelta) -> str:
    return str(o._value)


def _mapping(value: Mapping[Any, Any]) -> dict:
    out: dict[Any, Any] = {}
    for k, v in value.items():
        if k in _RESERVED_KEYS:
            raise SerializationError(
                f"free mapping key {k!r} collides with reserved key"
            )
        if isinstance(k, Enum):
            out[k.value] = _any(v)
        elif isinstance(k, str):
            out[k] = _any(v)
        else:
            raise SerializationError(
                f"unsupported mapping key type {type(k).__name__}"
            )
    return out


def _any(value: Any) -> Any:
    # NOTE: Enum must be checked BEFORE str, because ActionType etc. subclass
    # str. Otherwise an enum value would be mistaken for a plain string.
    if isinstance(value, Enum):
        name = _ENUM_NAME_REGISTRY.get(type(value))
        if name is None:
            raise SerializationError(
                f"unregistered enum type {type(value).__name__}"
            )
        return {_TYPE_KEY: "Enum", "enum": name, "value": value.value}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # reject non-finite floats (NaN / Infinity) in generic positions
        import math

        if not math.isfinite(value):
            raise SerializationError("cannot serialize a non-finite float")
        return value
    if isinstance(value, datetime):
        return _dt(value)
    if isinstance(value, Card):
        return _card(value)
    if isinstance(value, ChipAmount):
        return {_TYPE_KEY: "ChipAmount", "value": _money_str(value)}
    if isinstance(value, ChipDelta):
        return {_TYPE_KEY: "ChipDelta", "value": _money_str(value)}
    if isinstance(value, Decimal):
        return {_TYPE_KEY: "Decimal", "value": str(value)}
    if isinstance(value, tuple):
        return [_any(x) for x in value]
    if isinstance(value, list):
        return [_any(x) for x in value]
    if isinstance(value, (set, frozenset)):
        # tagged so it round-trips back to frozenset (deterministic order)
        try:
            items = sorted(value, key=lambda x: str(x))
        except Exception:
            items = list(value)
        return {_TYPE_KEY: "FrozenSet", "items": [_any(x) for x in items]}
    if isinstance(value, (dict, MappingProxyType, Mapping)):
        return _mapping(value)
    # Registered domain object (RequestContext / PlayerState / PokerState / ...)
    handler = _HANDLERS.get(type(value))
    if handler is not None:
        # handler output already carries __type__ and NOT schema_version
        return handler["serialize"](value)
    raise SerializationError(
        f"cannot serialize nested value of type {type(value).__name__}"
    )


def _card(o: Card) -> dict:
    return {_TYPE_KEY: "Card", "rank": o.rank.value, "suit": o.suit.value}


def _card_back(data: Any) -> Card:
    if not isinstance(data, dict) or data.get(_TYPE_KEY) != "Card":
        raise SerializationError("expected Card type tag")
    try:
        return Card(rank=Rank(data["rank"]), suit=Suit(data["suit"]))
    except (KeyError, ValueError) as e:
        raise SerializationError(f"invalid Card data: {e}") from e


def _any_back(data: Any) -> Any:
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, list):
        return [_any_back(x) for x in data]
    if isinstance(data, dict):
        tag = data.get(_TYPE_KEY)
        if tag is None:
            # plain mapping: recurse values (but reject reserved keys below)
            return _plain_mapping_back(data)
        # has a __type__ tag -> must be recognized
        if tag == "Card":
            return _card_back(data)
        if tag == "ChipAmount":
            return ChipAmount(_money_back(data, "ChipAmount"))
        if tag == "ChipDelta":
            return ChipDelta(_money_back(data, "ChipDelta"))
        if tag == "Decimal":
            return _decimal_back(data)
        if tag == "FrozenSet":
            return frozenset(_any_back(x) for x in data["items"])
        if tag == "Enum":
            return _enum_back(data)
        # Registered domain type tag (RequestContext / PlayerState / ...)
        handler = _HANDLERS_BY_NAME.get(tag)
        if handler is not None:
            return handler["deserialize"](data)
        raise SerializationError(f"unknown type tag {tag!r}")
    raise SerializationError(
        f"cannot deserialize value of type {type(data).__name__}"
    )


def _plain_mapping_back(data: dict) -> dict:
    """Deserialize a plain (untagged) mapping's keys and values."""
    out = {}
    for k, v in data.items():
        if k in _RESERVED_KEYS:
            raise SerializationError(
                f"free mapping key {k!r} collides with reserved key"
            )
        out[k] = _any_back(v)
    return out


def _decimal_back(data: dict) -> Decimal:
    value = data.get("value")
    if not isinstance(value, str):
        raise SerializationError("Decimal value must be a string")
    try:
        return Decimal(value)
    except Exception as e:
        raise SerializationError(f"invalid Decimal {value!r}") from e


def _enum_back(data: dict) -> Enum:
    name = data.get("enum")
    if not isinstance(name, str) or name not in _ENUM_REGISTRY:
        raise SerializationError(f"unknown enum type {name!r}")
    cls = _ENUM_REGISTRY[name]
    value = data.get("value")
    try:
        return cls(value)
    except (ValueError, TypeError) as e:
        raise SerializationError(f"invalid enum value {value!r} for {name}") from e


def _money_back(data: Any, tag: str) -> str:
    if not isinstance(data, dict) or data.get(_TYPE_KEY) != tag:
        raise SerializationError(f"expected {tag} type tag")
    value = data.get("value")
    if not isinstance(value, str):
        raise SerializationError(f"{tag} value must be a string")
    return value


def _expect_tag(data: Any, name: str) -> None:
    """Require a serialized nested domain object to carry the expected tag."""
    if not isinstance(data, dict) or data.get(_TYPE_KEY) != name:
        raise SerializationError(f"expected {name} type tag")


def _mapping_back(data: Any) -> dict:
    """Recursively deserialize a free mapping's values (reject reserved keys)."""
    if not isinstance(data, dict):
        raise SerializationError("expected a dict for mapping")
    return _plain_mapping_back(data)


# --------------------------------------------------------------------------
# Handler registry
# --------------------------------------------------------------------------

_HANDLERS: dict[type, dict[str, Any]] = {}
_HANDLERS_BY_NAME: dict[str, dict[str, Any]] = {}


def _register(t, name, ser, deser):
    handler = {"name": name, "serialize": ser, "deserialize": deser}
    _HANDLERS[t] = handler
    _HANDLERS_BY_NAME[name] = handler


def _obj(handlers, o, fields):
    d = {name: getattr(o, name) for name in fields}
    return d


# --- Card / ChipAmount / ChipDelta ---
_register(Card, "Card", _card, _card_back)
_register(ChipAmount, "ChipAmount",
          lambda o: {_TYPE_KEY: "ChipAmount", "value": _money_str(o)},
          lambda d: ChipAmount(_money_back(d, "ChipAmount")))
_register(ChipDelta, "ChipDelta",
          lambda o: {_TYPE_KEY: "ChipDelta", "value": _money_str(o)},
          lambda d: ChipDelta(_money_back(d, "ChipDelta")))


# --- PlayerState ---
def _ser_player(o: PlayerState) -> dict:
    return {
        _TYPE_KEY: "PlayerState",
        "player_id": o.player_id,
        "seat": o.seat,
        "position": _enum(o.position),
        "stack": _any(o.stack),
        "committed_this_street": _any(o.committed_this_street),
        "committed_this_hand": _any(o.committed_this_hand),
        "status": _enum(o.status),
        "has_cards": o.has_cards,
        "is_hero": o.is_hero,
        "is_dealer": o.is_dealer,
    }


def _deser_player(d: dict) -> PlayerState:
    _expect_tag(d, "PlayerState")
    return PlayerState(
        player_id=d["player_id"],
        seat=d["seat"],
        position=Position(d["position"]),
        stack=_any_back(d["stack"]),
        committed_this_street=_any_back(d["committed_this_street"]),
        committed_this_hand=_any_back(d["committed_this_hand"]),
        status=PlayerStatus(d["status"]),
        has_cards=d["has_cards"],
        is_hero=d["is_hero"],
        is_dealer=d["is_dealer"],
    )


_register(PlayerState, "PlayerState", _ser_player, _deser_player)


# --- PokerState ---
def _ser_poker_state(o: PokerState) -> dict:
    return {
        _TYPE_KEY: "PokerState",
        "state_version": o.state_version,
        "hand_id": o.hand_id,
        "street": _enum(o.street),
        "hero_cards": [_card(c) for c in o.hero_cards],
        "board_cards": [_card(c) for c in o.board_cards],
        "players": [_ser_player(p) for p in o.players],
        "pot": _any(o.pot),
        "current_bet": _any(o.current_bet),
        "to_call": _any(o.to_call),
        "actor": o.actor,
    }


def _deser_poker_state(d: dict) -> PokerState:
    _expect_tag(d, "PokerState")
    return PokerState(
        state_version=d["state_version"],
        hand_id=d["hand_id"],
        street=Street(d["street"]),
        hero_cards=tuple(_card_back(c) for c in d["hero_cards"]),
        board_cards=tuple(_card_back(c) for c in d["board_cards"]),
        players=tuple(_deser_player(p) for p in d["players"]),
        pot=_any_back(d["pot"]),
        current_bet=_any_back(d["current_bet"]),
        to_call=_any_back(d["to_call"]),
        actor=d["actor"],
    )


_register(PokerState, "PokerState", _ser_poker_state, _deser_poker_state)


# --- StateContext ---
def _ser_state_context(o: StateContext) -> dict:
    return {
        _TYPE_KEY: "StateContext",
        "previous_state":
            _ser_poker_state(o.previous_state) if o.previous_state is not None
            else None,
        "platform_rules": _mapping(o.platform_rules),
        "confidence_thresholds": _mapping(o.confidence_thresholds),
        "recent_events": [_ser_event(e) for e in o.recent_events],
    }


def _deser_state_context(d: dict) -> StateContext:
    _expect_tag(d, "StateContext")
    return StateContext(
        previous_state=(
            _deser_poker_state(d["previous_state"])
            if d["previous_state"] is not None else None
        ),
        platform_rules=_mapping_back(d["platform_rules"]),
        confidence_thresholds=_mapping_back(d["confidence_thresholds"]),
        recent_events=tuple(_deser_event(e) for e in d["recent_events"]),
    )


_register(StateContext, "StateContext",
          _ser_state_context, _deser_state_context)


# --- ValidationResult ---
def _ser_validation(o: ValidationResult) -> dict:
    return {
        _TYPE_KEY: "ValidationResult",
        "is_valid": o.is_valid,
        "errors": list(o.errors),
        "warnings": list(o.warnings),
    }


def _deser_validation(d: dict) -> ValidationResult:
    _expect_tag(d, "ValidationResult")
    return ValidationResult(
        is_valid=d["is_valid"],
        errors=tuple(d["errors"]),
        warnings=tuple(d["warnings"]),
    )


_register(ValidationResult, "ValidationResult",
          _ser_validation, _deser_validation)


# --- StateEvent ---
def _ser_event(o: StateEvent) -> dict:
    return {
        _TYPE_KEY: "StateEvent",
        "event_type": _enum(o.event_type),
        "hand_id": o.hand_id,
        "state_version": o.state_version,
        "payload": _mapping(o.payload),
        "timestamp": _dt(o.timestamp),
        "source": o.source,
    }


def _deser_event(d: dict) -> StateEvent:
    _expect_tag(d, "StateEvent")
    return StateEvent(
        event_type=EventType(d["event_type"]),
        hand_id=d["hand_id"],
        state_version=d["state_version"],
        payload=_mapping_back(d["payload"]),
        timestamp=_dt_back(d["timestamp"]),
        source=d["source"],
    )


_register(StateEvent, "StateEvent", _ser_event, _deser_event)


# --- RequestContext ---
def _ser_request(o: RequestContext) -> dict:
    return {
        _TYPE_KEY: "RequestContext",
        "hand_id": o.hand_id,
        "state_version": o.state_version,
        "request_id": o.request_id,
        "requested_at": _dt(o.requested_at),
        "expires_at": _dt(o.expires_at) if o.expires_at is not None else None,
        "deadline_ms": o.deadline_ms,
    }


def _deser_request(d: dict) -> RequestContext:
    _expect_tag(d, "RequestContext")
    return RequestContext(
        hand_id=d["hand_id"],
        state_version=d["state_version"],
        request_id=d["request_id"],
        requested_at=_dt_back(d["requested_at"]),
        expires_at=(
            _dt_back(d["expires_at"])
            if d.get("expires_at") is not None else None
        ),
        deadline_ms=d.get("deadline_ms"),
    )


_register(RequestContext, "RequestContext", _ser_request, _deser_request)


# --- HandSummary ---
def _ser_summary(o: HandSummary) -> dict:
    return {
        _TYPE_KEY: "HandSummary",
        "final_pot": _any(o.final_pot),
        "winners": list(o.winners),
        "winnings": _mapping(o.winnings),
        "net_result": _mapping(o.net_result),
    }


def _deser_summary(d: dict) -> HandSummary:
    _expect_tag(d, "HandSummary")
    return HandSummary(
        final_pot=_any_back(d["final_pot"]),
        winners=tuple(d["winners"]),
        winnings=_mapping_back(d["winnings"]),
        net_result=_mapping_back(d["net_result"]),
    )


_register(HandSummary, "HandSummary", _ser_summary, _deser_summary)


# --- HandHistory ---
def _ser_hand_history(o: HandHistory) -> dict:
    return {
        _TYPE_KEY: "HandHistory",
        "hand_id": o.hand_id,
        "players": [_ser_player(p) for p in o.players],
        "events": [_ser_event(e) for e in o.events],
        "summary": _ser_summary(o.summary),
        "start_time": _dt(o.start_time),
        "end_time": _dt(o.end_time) if o.end_time is not None else None,
    }


def _deser_hand_history(d: dict) -> HandHistory:
    _expect_tag(d, "HandHistory")
    return HandHistory(
        hand_id=d["hand_id"],
        players=tuple(_deser_player(p) for p in d["players"]),
        events=tuple(_deser_event(e) for e in d["events"]),
        summary=_deser_summary(d["summary"]),
        start_time=_dt_back(d["start_time"]),
        end_time=_dt_back(d["end_time"]) if d["end_time"] is not None else None,
    )


_register(HandHistory, "HandHistory", _ser_hand_history, _deser_hand_history)


# --- ObservationField ---
def _ser_observation_field(o: ObservationField) -> dict:
    return {
        _TYPE_KEY: "ObservationField",
        "value": _any(o.value),
        "confidence": o.confidence,
        "source": o.source,
        "evidence": _mapping(o.evidence),
        "timestamp": _dt(o.timestamp),
        "validation_status": _enum(o.validation_status),
    }


def _deser_observation_field(d: dict) -> ObservationField:
    _expect_tag(d, "ObservationField")
    return ObservationField(
        value=_any_back(d["value"]),
        confidence=d["confidence"],
        source=d["source"],
        evidence=_mapping_back(d["evidence"]),
        timestamp=_dt_back(d["timestamp"]),
        validation_status=ValidationStatus(d["validation_status"]),
    )


_register(ObservationField, "ObservationField",
          _ser_observation_field, _deser_observation_field)


# --- SlotObservation ---
def _ser_slot_observation(o: SlotObservation) -> dict:
    return {
        _TYPE_KEY: "SlotObservation",
        "slot_id": o.slot_id,
        "field": _ser_observation_field(o.field),
    }


def _deser_slot_observation(d: dict) -> SlotObservation:
    _expect_tag(d, "SlotObservation")
    return SlotObservation(
        slot_id=d["slot_id"],
        field=_deser_observation_field(d["field"]),
    )


_register(SlotObservation, "SlotObservation",
          _ser_slot_observation, _deser_slot_observation)


# --- RawObservation ---
def _ser_raw(o: RawObservation) -> dict:
    return {
        _TYPE_KEY: "RawObservation",
        "frame_seq": o.frame_seq,
        "timestamp": _dt(o.timestamp),
        "hero_cards": _ser_observation_field(o.hero_cards),
        "board_cards": _ser_observation_field(o.board_cards),
        "pot": _ser_observation_field(o.pot),
        "stacks": _ser_observation_field(o.stacks),
        "bet_size": _ser_observation_field(o.bet_size),
        "action": _ser_observation_field(o.action),
        "street": _ser_observation_field(o.street),
        "dealer_pos": _ser_observation_field(o.dealer_pos),
        "actor": _ser_observation_field(o.actor),
        "slot_stacks": [_ser_slot_observation(s) for s in o.slot_stacks],
        "slot_actions": [_ser_slot_observation(s) for s in o.slot_actions],
        "slot_occupancies": [
            _ser_slot_observation(s) for s in o.slot_occupancies
        ],
        "overall_confidence": o.overall_confidence,
    }


def _deser_raw(d: dict) -> RawObservation:
    _expect_tag(d, "RawObservation")
    # Historical v1 payloads omit slot_*; default to empty tuples.
    slot_stacks = tuple(
        _deser_slot_observation(s) for s in d.get("slot_stacks", [])
    )
    slot_actions = tuple(
        _deser_slot_observation(s) for s in d.get("slot_actions", [])
    )
    slot_occupancies = tuple(
        _deser_slot_observation(s) for s in d.get("slot_occupancies", [])
    )
    return RawObservation(
        frame_seq=d["frame_seq"],
        timestamp=_dt_back(d["timestamp"]),
        hero_cards=_deser_observation_field(d["hero_cards"]),
        board_cards=_deser_observation_field(d["board_cards"]),
        pot=_deser_observation_field(d["pot"]),
        stacks=_deser_observation_field(d["stacks"]),
        bet_size=_deser_observation_field(d["bet_size"]),
        action=_deser_observation_field(d["action"]),
        street=_deser_observation_field(d["street"]),
        dealer_pos=_deser_observation_field(d["dealer_pos"]),
        actor=_deser_observation_field(d["actor"]),
        slot_stacks=slot_stacks,
        slot_actions=slot_actions,
        slot_occupancies=slot_occupancies,
        overall_confidence=d["overall_confidence"],
    )


_register(RawObservation, "RawObservation", _ser_raw, _deser_raw)


# --- EquityReport ---
def _ser_equity(o: EquityReport) -> dict:
    return {
        _TYPE_KEY: "EquityReport",
        "win_rate": o.win_rate,
        "tie_rate": o.tie_rate,
        "pot_odds": o.pot_odds,
        "implied_odds": o.implied_odds,
        "estimated_ev": _any(o.estimated_ev),
        "method": _enum(o.method),
        "timestamp": _dt(o.timestamp),
    }


def _deser_equity(d: dict) -> EquityReport:
    _expect_tag(d, "EquityReport")
    return EquityReport(
        win_rate=d["win_rate"],
        tie_rate=d["tie_rate"],
        pot_odds=d["pot_odds"],
        implied_odds=d["implied_odds"],
        estimated_ev=_any_back(d["estimated_ev"]),
        method=EquityMethod(d["method"]),
        timestamp=_dt_back(d["timestamp"]),
    )


_register(EquityReport, "EquityReport", _ser_equity, _deser_equity)


# --- StrategyReport ---
def _ser_strategy(o: StrategyReport) -> dict:
    # action_frequencies: deterministic array sorted by ActionType.value
    af = sorted(
        o.action_frequencies.items(), key=lambda kv: kv[0].value
    )
    return {
        _TYPE_KEY: "StrategyReport",
        "action_frequencies": [
            {"action": _enum(k), "frequency": v} for k, v in af
        ],
        "bet_sizes": [_any(b) for b in o.bet_sizes],
        "ev": _any(o.ev),
        "strategy_source": _enum(o.strategy_source),
        "confidence": o.confidence,
        "cache_hit": o.cache_hit,
        "solver_metadata": _mapping(o.solver_metadata),
    }


def _deser_strategy(d: dict) -> StrategyReport:
    _expect_tag(d, "StrategyReport")
    af = {
        ActionType(item["action"]): item["frequency"]
        for item in d["action_frequencies"]
    }
    return StrategyReport(
        action_frequencies=af,
        bet_sizes=tuple(_any_back(b) for b in d["bet_sizes"]),
        ev=_any_back(d["ev"]),
        strategy_source=StrategySource(d["strategy_source"]),
        confidence=d["confidence"],
        cache_hit=d["cache_hit"],
        solver_metadata=_mapping_back(d["solver_metadata"]),
    )


_register(StrategyReport, "StrategyReport", _ser_strategy, _deser_strategy)


# --- ReasoningReport ---
def _ser_reasoning(o: ReasoningReport) -> dict:
    return {
        _TYPE_KEY: "ReasoningReport",
        "analysis_summary": o.analysis_summary,
        "key_factors": list(o.key_factors),
        "suggested_action": _enum(o.suggested_action),
        "suggested_size": _any(o.suggested_size),
        "confidence": o.confidence,
        "source": o.source,
        "hand_id": o.hand_id,
        "request_id": o.request_id,
        "model_metadata": _mapping(o.model_metadata),
        "state_version": o.state_version,
        "timestamp": _dt(o.timestamp),
    }


def _deser_reasoning(d: dict) -> ReasoningReport:
    _expect_tag(d, "ReasoningReport")
    return ReasoningReport(
        analysis_summary=d["analysis_summary"],
        key_factors=tuple(d["key_factors"]),
        suggested_action=ActionType(d["suggested_action"]),
        suggested_size=_any_back(d["suggested_size"]),
        confidence=d["confidence"],
        source=d["source"],
        hand_id=d["hand_id"],
        request_id=d["request_id"],
        model_metadata=_mapping_back(d["model_metadata"]),
        state_version=d["state_version"],
        timestamp=_dt_back(d["timestamp"]),
    )


_register(ReasoningReport, "ReasoningReport", _ser_reasoning, _deser_reasoning)


# --- Decision ---
def _ser_decision(o: Decision) -> dict:
    return {
        _TYPE_KEY: "Decision",
        "action": _enum(o.action),
        "confidence": o.confidence,
        "evidence_chain": list(o.evidence_chain),
        "raise_size": _any(o.raise_size) if o.raise_size is not None else None,
        "fast_or_slow": _enum(o.fast_or_slow),
        "timestamp": _dt(o.timestamp),
        "state_version": o.state_version,
    }


def _deser_decision(d: dict) -> Decision:
    _expect_tag(d, "Decision")
    return Decision(
        action=ActionType(d["action"]),
        confidence=d["confidence"],
        evidence_chain=tuple(d["evidence_chain"]),
        raise_size=(
            _any_back(d["raise_size"]) if d["raise_size"] is not None else None
        ),
        fast_or_slow=DecisionPath(d["fast_or_slow"]),
        timestamp=_dt_back(d["timestamp"]),
        state_version=d["state_version"],
    )


_register(Decision, "Decision", _ser_decision, _deser_decision)


# --- OpponentProfile ---
def _ser_opponent_profile(o: OpponentProfile) -> dict:
    return {
        _TYPE_KEY: "OpponentProfile",
        "player_id": o.player_id,
        "vpip": o.vpip,
        "pfr": o.pfr,
        "af": o.af,
        "cbet_freq": o.cbet_freq,
        "threebet_freq": o.threebet_freq,
        "bluff_freq": o.bluff_freq,
        "sample_size": o.sample_size,
        "last_updated": _dt(o.last_updated),
    }


def _deser_opponent_profile(d: dict) -> OpponentProfile:
    _expect_tag(d, "OpponentProfile")
    return OpponentProfile(
        player_id=d["player_id"],
        vpip=d["vpip"],
        pfr=d["pfr"],
        af=d["af"],
        cbet_freq=d["cbet_freq"],
        threebet_freq=d["threebet_freq"],
        bluff_freq=d["bluff_freq"],
        sample_size=d["sample_size"],
        last_updated=_dt_back(d["last_updated"]),
    )


_register(OpponentProfile, "OpponentProfile",
          _ser_opponent_profile, _deser_opponent_profile)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def serialize(obj: Any) -> dict:
    """Serialize a top-level domain object to a JSON-safe dict.

    The handler output already carries ``__type__``; we add ``schema_version``
    at the top level only. Raises SerializationError if unsupported.
    """
    handler = _HANDLERS.get(type(obj))
    if handler is None:
        raise SerializationError(
            f"no serializer registered for type {type(obj).__name__}"
        )
    fields = dict(handler["serialize"](obj))
    fields[_SCHEMA_KEY] = SCHEMA_VERSION
    return fields


def deserialize(type_: type, data: Any) -> Any:
    """Deserialize a top-level domain object from a JSON-safe dict.

    Checks the explicit ``type_`` against ``data["__type__"]`` and validates
    ``schema_version``; mismatches raise SerializationError.
    """
    handler = _HANDLERS.get(type_)
    if handler is None:
        raise SerializationError(
            f"no deserializer registered for type {type_.__name__}"
        )
    if not isinstance(data, dict):
        raise SerializationError("serialized data must be a dict")
    tag = data.get(_TYPE_KEY)
    if tag != handler["name"]:
        raise SerializationError(
            f"type mismatch: expected {handler['name']!r}, got {tag!r}"
        )
    version = data.get(_SCHEMA_KEY)
    if not isinstance(version, int) or isinstance(version, bool):
        raise SerializationError("schema_version must be an int")
    if version != SCHEMA_VERSION:
        raise SerializationError(
            f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})"
        )
    return handler["deserialize"](data)


__all__ = [
    "SCHEMA_VERSION",
    "serialize",
    "deserialize",
]
