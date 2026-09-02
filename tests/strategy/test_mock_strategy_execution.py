"""Execute the first strategy slice against generated multiplayer fixtures."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from poker_engine.core.enums import ActionType, Street
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.advice import AdviceStatus, build_advice
from poker_engine.strategy.contracts import LegalAction
from poker_engine.strategy.provider import FakeProvider
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, context, hit_result


DATA = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "strategy" / "v1" / "fixtures.jsonl"
)


def _fixtures_with(tag):
    return [
        item for item in (
            json.loads(line) for line in DATA.read_text().splitlines()
        )
        if tag in item["tags"]
    ]


PREFLOP_POSITIVE = [
    item for item in _fixtures_with("positive")
    if item["input"]["state"]["street"] == "preflop"
]
POSTFLOP_POSITIVE = [
    item for item in _fixtures_with("postflop")
    if "positive" in item["tags"]
]


def _with_legal_actions(ctx, fixture):
    args = dict(ctx.__dict__)
    args["legal_actions"] = tuple(
        LegalAction(
            ActionType(item["action"]),
            ChipAmount(item["min"]),
            ChipAmount(item["max"]),
        )
        for item in fixture["input"]["state"]["legal_actions"]
    )
    return type(ctx)(**args)


@pytest.mark.parametrize(
    "fixture", PREFLOP_POSITIVE,
    ids=[item["fixture_id"] for item in PREFLOP_POSITIVE],
)
def test_all_preflop_player_counts_and_action_lines_reach_ready(fixture):
    state = fixture["input"]["state"]
    count = state["dealt_player_count"]
    stack = Decimal(state["seats"][state["hero_seat"]]["starting_stack"])
    ctx = context(
        count,
        action_line=state["action_line"],
        effective_stack_bb=stack,
    )
    ctx = _with_legal_actions(ctx, fixture)
    metadata = fixture["input"]["providers"][0]
    probabilities = {
        ActionType(action): Decimal(value)
        for action, value in metadata["mock_result"][
            "action_probabilities"
        ].items()
    }
    value = candidate(
        ctx,
        metadata["provider_id"],
        metadata["source_version"],
        probabilities=probabilities,
    )
    provider = FakeProvider(
        metadata["provider_id"],
        metadata["source_version"],
        capability(
            (count,),
            streets=(Street.PREFLOP,),
        ),
        hit_result(value),
    )
    route = StrategyRouter((provider,)).route(ctx, now=NOW)
    advice = build_advice(ctx, route, now=NOW)
    assert advice.status is AdviceStatus.READY
    assert advice.strategy_source == metadata["provider_id"]
    assert advice.action_probabilities == probabilities


@pytest.mark.parametrize(
    "fixture", POSTFLOP_POSITIVE,
    ids=[item["fixture_id"] for item in POSTFLOP_POSITIVE],
)
def test_all_postflop_streets_and_active_counts_route_matching_provider(fixture):
    state = fixture["input"]["state"]
    count = state["active_player_count"]
    street = Street(state["street"])
    ctx = context(count, street=street, active_count=count)
    metadata = fixture["input"]["providers"][0]
    value = candidate(
        ctx,
        metadata["provider_id"],
        metadata["source_version"],
        probabilities={
            ActionType.CHECK: Decimal("0.4"),
            ActionType.BET: Decimal("0.6"),
        },
    )
    provider = FakeProvider(
        metadata["provider_id"], metadata["source_version"],
        capability((count,), streets=(street,)), hit_result(value),
    )
    route = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert route.selected.provider_id == metadata["provider_id"]


@pytest.mark.parametrize("dealt_count", (6, 9))
def test_multiplayer_preflop_history_can_route_hu_postflop(dealt_count):
    ctx = context(dealt_count, street=Street.FLOP, active_count=2)
    value = candidate(ctx, "hu-postflop", "v1")
    provider = FakeProvider(
        "hu-postflop", "v1",
        capability((2,), streets=(Street.FLOP,)), hit_result(value),
    )
    route = StrategyRouter((provider,)).route(ctx, now=NOW)
    assert ctx.game_config.dealt_player_count == dealt_count
    assert ctx.strategy_player_count == 2
    assert route.selected.provider_id == "hu-postflop"
