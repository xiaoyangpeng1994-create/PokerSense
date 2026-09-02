"""Tests for observation contracts: ObservationField, RawObservation."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from poker_engine.core.enums import ActionType, Rank, Street, Suit
from poker_engine.core.observation import (
    ObservationField,
    RawObservation,
    ValidationStatus,
)
from poker_engine.core.value_objects import Card, ChipAmount

UTC = timezone.utc


def _aware() -> datetime:
    return datetime(2026, 8, 18, 14, 0, 0, tzinfo=UTC)


def _make_field(value=None, status=ValidationStatus.VALID):
    return ObservationField(
        value=value,
        confidence=1.0,
        source="test",
        evidence={},
        timestamp=_aware(),
        validation_status=status,
    )


def _make_raw() -> RawObservation:
    return RawObservation(
        frame_seq=1,
        timestamp=_aware(),
        hero_cards=_make_field(()),
        board_cards=_make_field(()),
        pot=_make_field(ChipAmount("10")),
        stacks=_make_field(()),
        bet_size=_make_field(ChipAmount("0")),
        action=_make_field(None, ValidationStatus.UNKNOWN),
        street=_make_field(Street.PREFLOP),
        dealer_pos=_make_field(0),
        actor=_make_field(0),
    )


# ---------- ObservationField ----------

def test_confidence_low_rejected():
    with pytest.raises(ValueError):
        ObservationField(
            value=None, confidence=-0.1, source="x",
            evidence={}, timestamp=_aware(),
        )


def test_confidence_high_rejected():
    with pytest.raises(ValueError):
        ObservationField(
            value=None, confidence=1.5, source="x",
            evidence={}, timestamp=_aware(),
        )


def test_confidence_bounds_ok():
    ObservationField(value=None, confidence=0.0, source="x",
                     timestamp=_aware())
    ObservationField(value=None, confidence=1.0, source="x",
                     timestamp=_aware())


def test_unknown_allows_none_value():
    f = ObservationField(
        value=None, confidence=0.0, source="x",
        validation_status=ValidationStatus.UNKNOWN, timestamp=_aware(),
    )
    assert f.value is None
    assert f.validation_status is ValidationStatus.UNKNOWN


def test_evidence_external_mutation_no_effect():
    evidence = {"box": [1, 2, 3]}
    f = ObservationField(
        value="x", confidence=0.9, source="x",
        evidence=evidence, timestamp=_aware(),
    )
    evidence["box"].append(999)
    evidence["new_key"] = "injected"
    assert "new_key" not in f.evidence
    assert list(f.evidence["box"]) == [1, 2, 3]


def test_nested_evidence_immutable():
    inner = {"cards": ["Ah", "Kd"]}
    f = ObservationField(
        value="x", confidence=0.9, source="x",
        evidence={"inner": inner}, timestamp=_aware(),
    )
    inner["cards"].append("Qs")
    assert list(f.evidence["inner"]["cards"]) == ["Ah", "Kd"]
    # nested list was deep-frozen into a tuple, so .append no longer exists
    with pytest.raises(AttributeError):
        f.evidence["inner"]["cards"].append("Qs")  # type: ignore[index]


def test_observation_field_frozen():
    f = _make_field("x")
    with pytest.raises(FrozenInstanceError):
        f.value = "y"  # type: ignore[misc]


def test_naive_timestamp_rejected():
    with pytest.raises(TypeError):
        ObservationField(
            value="x", confidence=0.9, source="x",
            timestamp=datetime(2026, 8, 18),  # naive
        )


# ---------- RawObservation ----------

def test_raw_observation_valid():
    r = _make_raw()
    assert r.frame_seq == 1
    assert r.street.value is Street.PREFLOP


def test_raw_observation_frozen():
    r = _make_raw()
    with pytest.raises(FrozenInstanceError):
        r.frame_seq = 2  # type: ignore[misc]


def test_raw_observation_timestamp_aware():
    r = _make_raw()
    assert r.timestamp.tzinfo is UTC


def test_raw_observation_naive_rejected():
    with pytest.raises(TypeError):
        RawObservation(
            frame_seq=1,
            timestamp=datetime(2026, 8, 18),  # naive
            hero_cards=_make_field(()),
            board_cards=_make_field(()),
            pot=_make_field(ChipAmount("10")),
            stacks=_make_field(()),
            bet_size=_make_field(ChipAmount("0")),
            action=_make_field(None, ValidationStatus.UNKNOWN),
            street=_make_field(Street.PREFLOP),
            dealer_pos=_make_field(0),
            actor=_make_field(0),
        )


# ---------- 返修 v2 新增测试 ----------

def test_default_timestamp_is_aware():
    # 默认构造时 timestamp 必须是 timezone-aware（不再 naive）
    f = ObservationField(value="x", confidence=0.9, source="x")
    assert f.timestamp.tzinfo is not None
    assert f.timestamp.utcoffset() is not None


def test_value_deep_freeze_external_mutation_no_effect():
    # value 传入外部 list，随后修改原 list 不得影响 field.value
    original = ["Ah", "Kd"]
    f = ObservationField(
        value=original, confidence=0.9, source="x",
        timestamp=_aware(),
    )
    original.append("Qs")
    assert list(f.value) == ["Ah", "Kd"]


def test_value_deep_freeze_nested_dict_no_effect():
    inner = {"cards": ["Ah", "Kd"]}
    f = ObservationField(
        value=inner, confidence=0.9, source="x",
        timestamp=_aware(),
    )
    inner["cards"].append("Qs")
    assert list(f.value["cards"]) == ["Ah", "Kd"]  # type: ignore[index]


def test_value_itself_not_mutatable_in_place():
    f = ObservationField(
        value=["Ah", "Kd"], confidence=0.9, source="x",
        timestamp=_aware(),
    )
    # value 被转成 tuple，不能原地 append
    with pytest.raises(AttributeError):
        f.value.append("Qs")  # type: ignore[union-attr]


def test_source_empty_rejected():
    with pytest.raises(ValueError):
        ObservationField(value="x", confidence=0.9, source="", timestamp=_aware())


def test_frame_seq_negative_rejected():
    with pytest.raises(ValueError):
        RawObservation(
            frame_seq=-1, timestamp=_aware(),
            hero_cards=_make_field(()), board_cards=_make_field(()),
            pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
            bet_size=_make_field(ChipAmount("0")),
            action=_make_field(None, ValidationStatus.UNKNOWN),
            street=_make_field(Street.PREFLOP),
            dealer_pos=_make_field(0), actor=_make_field(0),
        )


def test_overall_confidence_bool_rejected():
    with pytest.raises(TypeError):
        RawObservation(
            frame_seq=1, timestamp=_aware(),
            hero_cards=_make_field(()), board_cards=_make_field(()),
            pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
            bet_size=_make_field(ChipAmount("0")),
            action=_make_field(None, ValidationStatus.UNKNOWN),
            street=_make_field(Street.PREFLOP),
            dealer_pos=_make_field(0), actor=_make_field(0),
            overall_confidence=True,  # bool 拒绝
        )


def test_raw_action_must_be_actiontype():
    # action 现在用 ActionType，不再是任意 str
    with pytest.raises(TypeError):
        RawObservation(
            frame_seq=1, timestamp=_aware(),
            hero_cards=_make_field(()), board_cards=_make_field(()),
            pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
            bet_size=_make_field(ChipAmount("0")),
            action=_make_field("fold"),  # str, 非 ActionType
            street=_make_field(Street.PREFLOP),
            dealer_pos=_make_field(0), actor=_make_field(0),
        )


def test_raw_action_actiontype_ok():
    r = RawObservation(
        frame_seq=1, timestamp=_aware(),
        hero_cards=_make_field(()), board_cards=_make_field(()),
        pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
        bet_size=_make_field(ChipAmount("0")),
        action=_make_field(ActionType.FOLD),
        street=_make_field(Street.PREFLOP),
        dealer_pos=_make_field(0), actor=_make_field(0),
    )
    assert r.action.value is ActionType.FOLD


def test_raw_hero_cards_elements_must_be_card():
    with pytest.raises(TypeError):
        RawObservation(
            frame_seq=1, timestamp=_aware(),
            hero_cards=_make_field(("As", "Kh")),  # 字符串，非 Card
            board_cards=_make_field(()),
            pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
            bet_size=_make_field(ChipAmount("0")),
            action=_make_field(None, ValidationStatus.UNKNOWN),
            street=_make_field(Street.PREFLOP),
            dealer_pos=_make_field(0), actor=_make_field(0),
        )


def test_raw_hero_cards_cards_ok():
    ac = Card(Rank.ACE, Suit.CLUBS)
    kh = Card(Rank.KING, Suit.HEARTS)
    r = RawObservation(
        frame_seq=1, timestamp=_aware(),
        hero_cards=_make_field((ac, kh)),
        board_cards=_make_field(()),
        pot=_make_field(ChipAmount("10")), stacks=_make_field(()),
        bet_size=_make_field(ChipAmount("0")),
        action=_make_field(None, ValidationStatus.UNKNOWN),
        street=_make_field(Street.PREFLOP),
        dealer_pos=_make_field(0), actor=_make_field(0),
    )
    assert len(r.hero_cards.value) == 2
