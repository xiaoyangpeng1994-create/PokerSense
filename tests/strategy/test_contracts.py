"""DecisionContext and supporting strategy input contract tests."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from decimal import Decimal

import pytest

from poker_engine.core.enums import Street
from poker_engine.strategy.contracts import GameConfig
from poker_engine.strategy.serialization import (
    strategy_deserialize,
    strategy_serialize,
)

from .helpers import context


def test_preflop_strategy_count_uses_dealt_count():
    ctx = context(9)
    assert ctx.strategy_player_count == 9


def test_postflop_strategy_count_uses_active_count():
    ctx = context(9, street=Street.FLOP, active_count=2)
    assert ctx.strategy_player_count == 2
    assert ctx.game_config.dealt_player_count == 9


def test_context_reports_readiness_and_missing_fields():
    ctx = context(missing_fields=("position",))
    assert ctx.is_decision_ready is False
    assert ctx.missing_fields == ("position",)


def test_context_requires_hero_to_be_actor_for_live_advice():
    assert context(actor_is_hero=False).is_decision_ready is False


def test_context_nested_mappings_are_immutable():
    ctx = context()
    with pytest.raises(TypeError):
        ctx.input_quality.field_confidences["hero_cards"] = 0.1
    with pytest.raises(TypeError):
        ctx.hero_range.combo_weights["AA"] = Decimal("1")


def test_context_is_frozen():
    ctx = context()
    with pytest.raises(FrozenInstanceError):
        ctx.hero_seat = 0


def test_context_rejects_duplicate_known_cards():
    ctx = context()
    args = dict(ctx.__dict__)
    args["board_cards"] = ()
    args["hero_cards"] = (ctx.hero_cards[0], ctx.hero_cards[0])
    with pytest.raises(ValueError, match="distinct"):
        type(ctx)(**args)


def test_context_round_trip_is_json_safe_and_equal():
    ctx = context(6)
    payload = strategy_serialize(ctx)
    restored = strategy_deserialize(type(ctx), payload)
    assert restored == ctx
    assert payload["schema_version"] == 1
    assert payload["type"] == "DecisionContext"
    assert payload["legal_actions"][0]["amount_semantics"] == "none"
    assert payload["legal_actions"][1]["amount_semantics"] == "total_street"


def test_context_deserializes_legacy_legal_actions_without_semantics():
    ctx = context(6)
    payload = strategy_serialize(ctx)
    for action in payload["legal_actions"]:
        del action["amount_semantics"]

    restored = strategy_deserialize(type(ctx), payload)

    assert restored == ctx


@pytest.mark.parametrize("count", (1, 10))
def test_game_config_rejects_player_count_outside_2_to_9(count):
    original = context().game_config
    args = dict(original.__dict__)
    args["dealt_player_count"] = count
    args["max_seats"] = max(2, count)
    with pytest.raises(ValueError):
        GameConfig(**args)


def test_request_context_rejects_naive_expiry():
    ctx = context()
    request = ctx.request
    args = dict(request.__dict__)
    args["expires_at"] = datetime(2026, 8, 22)
    with pytest.raises(TypeError):
        type(request)(**args)
