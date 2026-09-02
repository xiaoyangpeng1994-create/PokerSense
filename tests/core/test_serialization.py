"""Task 1E serialization round-trip + JSON + precision tests."""

import json

import pytest

from poker_engine.core.errors import SerializationError
from poker_engine.core.serialization import deserialize, serialize
from poker_engine.core.value_objects import Card, ChipAmount, ChipDelta
from poker_engine.core.enums import Rank, Suit

from . import helpers as H


def roundtrip(obj):
    """Full JSON chain: obj -> serialize -> json.dumps -> json.loads -> deserialize."""
    return deserialize(type(obj), json.loads(json.dumps(serialize(obj))))


# --- Round-trip (19 domain types) ---

def test_roundtrip_card():
    obj = Card(Rank.QUEEN, Suit.HEARTS)
    assert roundtrip(obj) == obj


def test_roundtrip_chipamount():
    assert roundtrip(ChipAmount("2.345")) == ChipAmount("2.345")


def test_roundtrip_chipdelta():
    assert roundtrip(ChipDelta("-123.456789")) == ChipDelta("-123.456789")


def test_roundtrip_observation_field_card_tuple():
    obj = H.observation_field((H.Ac, H.Kh))
    rt = roundtrip(obj)
    assert rt.value == obj.value
    assert rt.confidence == obj.confidence
    assert rt.source == obj.source
    assert rt.validation_status == obj.validation_status


def test_roundtrip_observation_field_none():
    obj = H.observation_field(None, H.ValidationStatus.UNKNOWN)
    rt = roundtrip(obj)
    assert rt.value is None
    assert rt.validation_status is H.ValidationStatus.UNKNOWN


def test_roundtrip_raw_observation():
    obj = H.raw_observation()
    rt = roundtrip(obj)
    assert rt.frame_seq == obj.frame_seq
    assert rt.hero_cards.value == obj.hero_cards.value
    assert rt.pot.value == obj.pot.value
    assert rt.street.value == obj.street.value
    assert rt.overall_confidence == obj.overall_confidence


def test_roundtrip_player_state():
    obj = H.player(0, hero=True)
    rt = roundtrip(obj)
    assert rt == obj  # PlayerState dataclass equality
    assert isinstance(rt.stack, ChipAmount)
    assert rt.position is H.Position.BTN


def test_roundtrip_poker_state():
    obj = H.poker_state()
    rt = roundtrip(obj)
    assert rt == obj
    assert rt.hero_cards == (H.Ac, H.Kh)
    assert rt.board_cards == (H.Kc, H.Qh, H.Jh)
    assert rt.pot == ChipAmount("30.5")
    assert isinstance(rt.players[0], H.PlayerState)
    # actor preserved
    assert rt.actor == 1


def test_roundtrip_state_event():
    obj = H.state_event()
    rt = roundtrip(obj)
    assert rt == obj
    assert rt.payload["amount"] == ChipAmount("12.5")
    assert rt.payload["cards"] == (H.Ac, H.Kh)


def test_roundtrip_state_context():
    obj = H.state_context()
    rt = roundtrip(obj)
    assert rt.previous_state == obj.previous_state
    assert rt.platform_rules["blind"]["sb"] == "1"
    assert rt.recent_events[0] == obj.recent_events[0]


def test_roundtrip_request_context():
    obj = H.request_context()
    rt = roundtrip(obj)
    assert rt == obj
    assert rt.request_id == "req-abc"


def test_request_context_v1_legacy_payload_defaults_new_deadline_fields():
    payload = serialize(H.request_context())
    payload.pop("expires_at")
    payload.pop("deadline_ms")
    rt = deserialize(H.RequestContext, payload)
    assert rt.expires_at is None
    assert rt.deadline_ms is None


def test_roundtrip_validation_result():
    obj = H.validation_result()
    rt = roundtrip(obj)
    assert rt.is_valid is False
    assert rt.errors == ("e1", "e2")
    assert rt.warnings == ("w1",)


def test_roundtrip_hand_summary():
    obj = H.hand_summary()
    rt = roundtrip(obj)
    assert rt.final_pot == obj.final_pot
    assert rt.winners == ("p0",)
    assert rt.winnings["p0"] == ChipAmount("100")
    assert rt.net_result["p1"] == ChipDelta("-50")


def test_roundtrip_hand_history():
    obj = H.hand_history()
    rt = roundtrip(obj)
    assert rt.hand_id == obj.hand_id
    assert rt.end_time is None
    assert rt.summary.final_pot == obj.summary.final_pot
    assert len(rt.events) == 1
    assert rt.events[0].event_type is H.EventType.RAISE


def test_roundtrip_equity_report():
    obj = H.equity_report()
    rt = roundtrip(obj)
    assert rt == obj
    assert rt.method is H.EquityMethod.MONTECARLO
    assert rt.estimated_ev == ChipAmount("10")


def test_roundtrip_strategy_report():
    obj = H.strategy_report()
    rt = roundtrip(obj)
    assert rt.action_frequencies == obj.action_frequencies
    assert rt.bet_sizes == (ChipAmount("10"), ChipAmount("20"))
    assert rt.strategy_source is H.StrategySource.CACHE
    assert rt.cache_hit is True


def test_roundtrip_reasoning_report():
    obj = H.reasoning_report()
    rt = roundtrip(obj)
    assert rt.hand_id == "h1"
    assert rt.request_id == "r1"
    assert rt.suggested_action is H.ActionType.RAISE
    assert rt.key_factors == obj.key_factors


def test_roundtrip_decision():
    obj = H.decision()
    rt = roundtrip(obj)
    assert rt.action is H.ActionType.RAISE
    assert rt.raise_size == ChipAmount("15")
    assert rt.fast_or_slow is H.DecisionPath.FAST


def test_roundtrip_opponent_profile():
    obj = H.opponent_profile()
    rt = roundtrip(obj)
    assert rt == obj
    assert rt.player_id == "p1"
    assert rt.af == 2.0


# --- JSON-safety ---

def test_serialize_output_is_json_dumpable():
    for obj in [
        H.poker_state(), H.raw_observation(), H.hand_history(),
        H.strategy_report(), H.state_event(), H.opponent_profile(),
    ]:
        serialized = serialize(obj)
        json.dumps(serialized)  # must not raise


# --- Precision ---

def test_money_precision_exact():
    values = ["2.345", "0.00000001", "123.456789", "0.999999999999"]
    for v in values:
        obj = ChipAmount(v)
        rt = roundtrip(obj)
        assert rt == obj
        assert rt.value == obj.value


def test_delta_precision_exact():
    values = ["-123.456789", "-0.00000001", "0.1", "-999999.999999"]
    for v in values:
        obj = ChipDelta(v)
        rt = roundtrip(obj)
        assert rt.value == obj.value


def test_money_no_float_in_json():
    s = serialize(ChipAmount("0.1"))
    assert s["value"] == "0.1"
    assert isinstance(s["value"], str)
    assert "0.1" in json.dumps(s)


# --- schema_version / type tag ---

def test_top_level_has_schema_version():
    s = serialize(H.poker_state())
    assert s["schema_version"] == 1
    assert s["__type__"] == "PokerState"


def test_nested_no_schema_version():
    # nested domain object (e.g. players' stack) must not carry schema_version
    s = serialize(H.poker_state())
    player0 = s["players"][0]
    assert "schema_version" not in player0
    stack = player0["stack"]
    assert stack["__type__"] == "ChipAmount"
    assert "schema_version" not in stack


def test_type_mismatch_raises():
    s = serialize(H.poker_state())
    with pytest.raises(SerializationError):
        deserialize(H.PlayerState, s)  # wrong type


def test_unknown_type_tag_raises():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    s["__type__"] = "Nope"
    with pytest.raises(SerializationError):
        deserialize(H.PokerState, s)


def test_unsupported_schema_version_raises():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    s["schema_version"] = 99
    with pytest.raises(SerializationError):
        deserialize(H.PokerState, s)


def test_missing_schema_version_raises():
    s = serialize(H.poker_state())
    s = json.loads(json.dumps(s))
    del s["schema_version"]
    with pytest.raises(SerializationError):
        deserialize(H.PokerState, s)


def test_unsupported_type_raises():
    with pytest.raises(SerializationError):
        serialize(object())


# --- action_frequencies determinism ---

def test_action_frequencies_deterministic_order():
    obj = H.strategy_report()
    s1 = serialize(obj)
    s2 = serialize(obj)
    af1 = [item["action"] for item in s1["action_frequencies"]]
    af2 = [item["action"] for item in s2["action_frequencies"]]
    assert af1 == af2
    # sorted by ActionType.value
    assert af1 == sorted(af1)


# --- v2 返修：nested domain object __type__ ---

def test_nested_player_state_has_type_tag():
    s = serialize(H.poker_state())
    assert s["players"][0]["__type__"] == "PlayerState"
    assert "schema_version" not in s["players"][0]


def test_nested_poker_state_has_type_tag():
    s = serialize(H.state_context())
    assert s["previous_state"]["__type__"] == "PokerState"
    assert "schema_version" not in s["previous_state"]


def test_nested_state_event_has_type_tag():
    s = serialize(H.state_context())
    assert s["recent_events"][0]["__type__"] == "StateEvent"
    assert "schema_version" not in s["recent_events"][0]


def test_nested_hand_summary_has_type_tag():
    s = serialize(H.hand_history())
    assert s["summary"]["__type__"] == "HandSummary"
    assert "schema_version" not in s["summary"]


def test_nested_observation_field_has_type_tag():
    s = serialize(H.raw_observation())
    assert s["hero_cards"]["__type__"] == "ObservationField"
    assert "schema_version" not in s["hero_cards"]


def test_nested_hand_history_players_type_tag():
    s = serialize(H.hand_history())
    assert s["players"][0]["__type__"] == "PlayerState"
    assert s["events"][0]["__type__"] == "StateEvent"


# --- v2 返修：frozenset round-trip ---

def test_frozenset_roundtrip_in_payload():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    obj = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={"tags": frozenset({"a", "b", "c"})},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s = serialize(obj)
    # tagged representation
    assert s["payload"]["tags"]["__type__"] == "FrozenSet"
    rt = H.roundtrip(obj)
    assert isinstance(rt.payload["tags"], frozenset)
    assert rt.payload["tags"] == frozenset({"a", "b", "c"})


def test_frozenset_deterministic_order():
    from poker_engine.core.events import EventType, StateEvent
    from datetime import datetime, timezone
    obj = StateEvent(
        event_type=EventType.FOLD, hand_id="h", state_version=0,
        payload={"tags": frozenset({"c", "a", "b"})},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s1 = serialize(obj)
    s2 = serialize(obj)
    assert s1["payload"]["tags"]["items"] == s2["payload"]["tags"]["items"]


# --- v2 返修：generic Decimal round-trip ---

def test_decimal_roundtrip_exact():
    from decimal import Decimal
    for v in ["2.345", "0.00000001", "-123.456789"]:
        json.dumps(serialize(H.observation_field(Decimal(v))))
        rt = H.roundtrip(H.observation_field(Decimal(v)))
        assert rt.value == Decimal(v)


def test_decimal_no_float_trap():
    from decimal import Decimal
    json.dumps(serialize(H.observation_field(Decimal("0.1"))))
    rt = H.roundtrip(H.observation_field(Decimal("0.1")))
    assert rt.value == Decimal("0.1")


# --- v3 返修：generic/free Mapping 嵌套 registered Domain Object ---

def test_payload_nested_request_context_roundtrip():
    from poker_engine.core.events import EventType, StateEvent
    from poker_engine.core.request_context import RequestContext
    from datetime import datetime, timezone

    req = RequestContext(
        hand_id="h1", state_version=3, request_id="r1",
        requested_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    ev = StateEvent(
        event_type=EventType.FOLD, hand_id="h1", state_version=0,
        payload={"request": req},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s = serialize(ev)
    # nested __type__ present, schema_version absent
    assert s["payload"]["request"]["__type__"] == "RequestContext"
    assert "schema_version" not in s["payload"]["request"]
    # full JSON round-trip restores correct type
    rt = deserialize(StateEvent, json.loads(json.dumps(s)))
    assert isinstance(rt.payload["request"], RequestContext)
    assert rt.payload["request"].request_id == "r1"


def test_payload_nested_player_state_roundtrip():
    from poker_engine.core.events import EventType, StateEvent
    from poker_engine.core.opponents import PlayerState
    from datetime import datetime, timezone

    ev = StateEvent(
        event_type=EventType.FOLD, hand_id="h1", state_version=0,
        payload={"player": H.player(0, hero=True)},
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    s = serialize(ev)
    assert s["payload"]["player"]["__type__"] == "PlayerState"
    assert "schema_version" not in s["payload"]["player"]
    rt = deserialize(StateEvent, json.loads(json.dumps(s)))
    assert isinstance(rt.payload["player"], PlayerState)
    assert rt.payload["player"].is_hero is True
    # outer StateEvent deep immutability still holds
    with pytest.raises(TypeError):
        rt.payload["player"] = H.player(1)
