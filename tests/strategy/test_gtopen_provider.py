"""Bounded local GTOpen API adapter and Slow Path integration tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.events import EventType, StateEvent
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.contracts import EffectiveStack, LegalAction
from poker_engine.strategy.gtopen_provider import (
    GTOpenConfig,
    GTOpenError,
    GTOpenPreflopProvider,
    gtopen_hand_class_index,
)
from poker_engine.strategy.orchestration import (
    RefinementState,
    StrategyOrchestrator,
    ThreadedSlowResolver,
)
from poker_engine.strategy.provider import LookupState, MatchKind
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, card, context


REVISION = "4aee435bdeb155b25f0c8140e707a8342ce4356f"


def _strategy(actions, values):
    result = [0.0] * (len(actions) * 169)
    hand_index = gtopen_hand_class_index((card("As"), card("Kd")))
    for action_index, value in enumerate(values):
        result[action_index * 169 + hand_index] = value
    return result


def _node(actor, actor_pos, actions, values):
    return {
        "kind": "action",
        "actor": actor,
        "actor_pos": actor_pos,
        "positions": ["BTN", "SB", "BB"],
        "pot": 1.5,
        "actions": actions,
        "strategy": _strategy(actions, values),
        "reach": [1.0] * 169,
        "history": [],
        "reaches_all": [[1.0] * 169 for _ in range(3)],
        "exportable": False,
    }


ROOT_ACTIONS = [
    {"label": "Fold", "kind": "fold", "to": 0.0, "freq": 0.6},
    {"label": "Raise 2", "kind": "raise", "to": 2.0, "freq": 0.4},
]


class FakeTransport:
    def __init__(self, nodes=None, statuses=None):
        self.nodes = nodes or {(): _node(0, "BTN", ROOT_ACTIONS, (0.2, 0.8))}
        self.statuses = list(statuses or [{
            "state": "done",
            "iteration": 25,
            "gap_total": 0.0004,
            "gaps": [0.0001, 0.0001, 0.0002],
            "evs": [0.1, -0.05, -0.05],
        }])
        self.calls = []

    def get(self, path, timeout_seconds):
        self.calls.append(("GET", path, None, timeout_seconds))
        if path != "/api/preflop/status":
            raise AssertionError(path)
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def post(self, path, payload, timeout_seconds):
        self.calls.append(("POST", path, payload, timeout_seconds))
        if path == "/api/preflop/spot":
            return {"nodes": 13, "action_nodes": 6, "arena_mb": 0.016224}
        if path == "/api/preflop/solve":
            return {"ok": True}
        if path == "/api/preflop/stop":
            return {"ok": True}
        if path == "/api/preflop/node":
            return self.nodes[tuple(payload["path"])]
        raise AssertionError(path)


class BrokenTransport(FakeTransport):
    def post(self, path, payload, timeout_seconds):
        raise GTOpenError("gtopen_transport_error:ConnectionRefusedError")


def _context(
    *,
    hero_seat=0,
    stacks=("20", "20", "20"),
    history=(),
    legal_actions=None,
    action_line="unopened",
):
    value = context(
        3,
        action_line=action_line,
        effective_stack_bb=Decimal("20"),
        expires_at=NOW.replace(day=23),
    )
    seats = tuple(
        replace(
            seat,
            stack=ChipAmount(stacks[seat.seat_id]),
            is_hero=seat.seat_id == hero_seat,
        )
        for seat in value.seats
    )
    legal_actions = legal_actions or (
        LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
        LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("20")),
    )
    return replace(
        value,
        seats=seats,
        hero_seat=hero_seat,
        actor_seat=hero_seat,
        legal_actions=tuple(legal_actions),
        action_history=tuple(history),
        effective_stacks=tuple(
            EffectiveStack(seat_id, ChipAmount("20"))
            for seat_id in range(3)
            if seat_id != hero_seat
        ),
    )


def _provider(transport, **kwargs):
    values = {
        "source_revision": REVISION,
        "realization": "raw",
        "iterations": 100,
        "check_every": 25,
        "target_gap_bb": Decimal("0.01"),
        "timeout_ms": 1_000,
    }
    values.update(kwargs)
    return GTOpenPreflopProvider(
        GTOpenConfig(**values),
        transport,
        clock=lambda: NOW,
    )


def _player_count_context(player_count):
    value = context(
        player_count,
        action_line="unopened",
        effective_stack_bb=Decimal("20"),
        expires_at=NOW.replace(day=23),
    )
    seats = tuple(
        replace(
            seat,
            stack=ChipAmount("20"),
            is_hero=seat.seat_id == 0,
        )
        for seat in value.seats
    )
    return replace(
        value,
        seats=seats,
        hero_seat=0,
        actor_seat=0,
        legal_actions=(
            LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
            LegalAction(ActionType.RAISE, ChipAmount("2"), ChipAmount("20")),
        ),
        effective_stacks=tuple(
            EffectiveStack(seat_id, ChipAmount("20"))
            for seat_id in range(player_count)
            if seat_id != 0
        ),
    )


def test_root_multiway_result_becomes_audited_heuristic_candidate():
    transport = FakeTransport()
    result = _provider(transport).query(_context())

    assert result.state is LookupState.HIT_APPROXIMATE
    candidate = result.candidate
    assert candidate.match_kind is MatchKind.HEURISTIC
    assert candidate.provider_version.endswith(REVISION)
    assert candidate.action_probabilities == {
        ActionType.FOLD: Decimal("0.2"),
        ActionType.RAISE: Decimal("0.8"),
    }
    assert candidate.recommended_sizes == {
        ActionType.RAISE: (ChipAmount("2"),),
    }
    assert candidate.confidence == 0.6
    assert "gtopen_multiway_product_equity_approximation" in candidate.assumptions
    assert (
        "gtopen_server_revision_configured_not_remotely_attested"
        in candidate.assumptions
    )
    assert "gtopen_model_gap_bb:0.0004" in candidate.evidence
    spot = next(
        call[2] for call in transport.calls
        if call[:2] == ("POST", "/api/preflop/spot")
    )
    assert spot["positions"] == ["BTN", "SB", "BB"]
    assert spot["posts"] == [0.0, 0.5, 1.0]
    assert spot["stack"] == 20.0
    assert spot["realization"] == "raw"


@pytest.mark.parametrize("player_count", range(2, 10))
def test_every_declared_player_count_builds_the_exact_position_and_blind_map(
    player_count,
):
    ctx = _player_count_context(player_count)
    actor_position = ctx.seats[0].position.value
    transport = FakeTransport({
        (): _node(0, actor_position, ROOT_ACTIONS, (0.2, 0.8)),
    })

    result = _provider(transport).query(ctx)

    assert result.state is LookupState.HIT_APPROXIMATE
    spot = next(call[2] for call in transport.calls if call[1].endswith("/spot"))
    assert spot["positions"] == [seat.position.value for seat in ctx.seats]
    assert spot["posts"][-1] == 1.0
    assert spot["posts"][-2] == 0.5
    assert all(value == 0.0 for value in spot["posts"][:-2])


def test_history_is_walked_by_exact_actor_kind_and_raise_to_amount():
    first_raise = StateEvent(
        EventType.RAISE,
        "h-3-preflop",
        1,
        payload={"seat_id": 0, "amount_total_street": "2.2"},
        timestamp=NOW,
    )
    current_actions = [
        {"label": "Fold", "kind": "fold", "to": 0.0, "freq": 0.4},
        {"label": "Call 2.2", "kind": "call", "to": 2.2, "freq": 0.4},
        {"label": "3-bet 6.6", "kind": "raise", "to": 6.6, "freq": 0.2},
    ]
    nodes = {
        (): _node(0, "BTN", [
            ROOT_ACTIONS[0],
            {"label": "Raise 2.2", "kind": "raise", "to": 2.2, "freq": 0.5},
        ], (0.5, 0.5)),
        (1,): _node(1, "SB", current_actions, (0.1, 0.6, 0.3)),
    }
    legal = (
        LegalAction(ActionType.FOLD, ChipAmount("0"), ChipAmount("0")),
        LegalAction(ActionType.CALL, ChipAmount("1.7"), ChipAmount("1.7")),
        LegalAction(ActionType.RAISE, ChipAmount("3.4"), ChipAmount("20")),
    )
    transport = FakeTransport(nodes)

    result = _provider(transport).query(_context(
        hero_seat=1,
        history=(first_raise,),
        legal_actions=legal,
        action_line="raise",
    ))

    assert result.state is LookupState.HIT_APPROXIMATE
    assert result.candidate.action_probabilities == {
        ActionType.FOLD: Decimal("0.1"),
        ActionType.CALL: Decimal("0.6"),
        ActionType.RAISE: Decimal("0.3"),
    }
    spot = next(call[2] for call in transport.calls if call[1].endswith("/spot"))
    assert 2.2 in spot["open_raises"]
    node_paths = [
        tuple(call[2]["path"])
        for call in transport.calls
        if call[:2] == ("POST", "/api/preflop/node")
    ]
    assert node_paths == [(), (1,)]


def test_unequal_stacks_fail_closed_before_building_a_tree():
    transport = FakeTransport()
    result = _provider(transport).query(_context(stacks=("20", "19", "20")))

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_requires_equal_starting_stacks",)
    assert not transport.calls


def test_non_exact_history_size_is_rejected_instead_of_nearest_matching():
    event = StateEvent(
        EventType.RAISE,
        "h-3-preflop",
        1,
        payload={"seat_id": 0, "amount_total_street": "2.2"},
        timestamp=NOW,
    )
    transport = FakeTransport()

    result = _provider(transport).query(_context(
        hero_seat=1,
        history=(event,),
        action_line="raise",
    ))

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_history_action_not_exact",)


def test_declared_action_line_must_match_authoritative_events():
    event = StateEvent(
        EventType.RAISE,
        "h-3-preflop",
        1,
        payload={"seat_id": 0, "amount_total_street": "2"},
        timestamp=NOW,
    )
    transport = FakeTransport()

    result = _provider(transport).query(_context(
        hero_seat=1,
        history=(event,),
        action_line="unopened",
    ))

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_action_line_mismatch",)
    assert not transport.calls


def test_convergence_gap_over_threshold_is_rejected():
    transport = FakeTransport(statuses=[{
        "state": "done", "iteration": 100, "gap_total": 0.02,
    }])
    result = _provider(transport).query(_context())

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_convergence_threshold_not_met",)


def test_transport_failure_is_contained():
    result = _provider(BrokenTransport()).query(_context())

    assert result.state is LookupState.REJECTED
    assert result.reasons == (
        "gtopen_transport_error:ConnectionRefusedError",
    )


def test_unknown_or_context_illegal_action_is_rejected():
    actions = [
        {"label": "Check", "kind": "check", "to": 1.0, "freq": 1.0},
    ]
    transport = FakeTransport({(): _node(0, "BTN", actions, (1.0,))})

    result = _provider(transport).query(_context())

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_action_not_legal",)


def test_multiple_raise_sizes_are_preserved_and_aggregate_exactly():
    actions = [
        {"label": "Fold", "kind": "fold", "to": 0.0, "freq": 0.1},
        {"label": "Raise 2", "kind": "raise", "to": 2.0, "freq": 0.2},
        {"label": "Raise 3", "kind": "raise", "to": 3.0, "freq": 0.7},
    ]
    result = _provider(FakeTransport({
        (): _node(0, "BTN", actions, (0.1, 0.2, 0.7)),
    })).query(_context())

    assert result.state is LookupState.HIT_APPROXIMATE
    assert result.candidate.action_probabilities == {
        ActionType.FOLD: Decimal("0.1"),
        ActionType.RAISE: Decimal("0.9"),
    }
    assert result.candidate.recommended_sizes == {
        ActionType.RAISE: (ChipAmount("2"), ChipAmount("3")),
    }
    assert len(result.candidate.action_options) == 3


def test_timeout_stops_the_upstream_solve_best_effort():
    class FakeTime:
        value = 0.0

        def monotonic(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds

    fake_time = FakeTime()
    transport = FakeTransport(statuses=[{"state": "running"}])
    provider = GTOpenPreflopProvider(
        GTOpenConfig(
            source_revision=REVISION,
            realization="raw",
            timeout_ms=50,
            poll_interval_ms=25,
        ),
        transport,
        clock=lambda: NOW,
        monotonic=fake_time.monotonic,
        sleep=fake_time.sleep,
    )

    result = provider.query(_context())

    assert result.state is LookupState.REJECTED
    assert result.reasons == ("gtopen_timeout",)
    assert any(call[1] == "/api/preflop/stop" for call in transport.calls)


def test_provider_runs_as_the_existing_asynchronous_slow_refinement():
    ctx = _context()
    provider = _provider(FakeTransport())
    with ThreadPoolExecutor(max_workers=1) as executor:
        slow = ThreadedSlowResolver(provider, executor)
        orchestrator = StrategyOrchestrator(StrategyRouter(), slow)
        cycle = orchestrator.request(ctx, now=NOW)
        assert cycle.fast_advice.status is AdviceStatus.ABSTAIN
        cycle.slow_handle.future.result(timeout=1)
        refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)

    assert refinement.state is RefinementState.APPLIED
    assert refinement.advice.status is AdviceStatus.READY
    assert refinement.advice.strategy_source == provider.provider_id


def test_hand_class_index_matches_upstream_layout():
    assert gtopen_hand_class_index((card("As"), card("Ah"))) == 168
    assert gtopen_hand_class_index((card("As"), card("Ks"))) == 167
    assert gtopen_hand_class_index((card("As"), card("Kd"))) == 155


def test_config_rejects_remote_service_and_unpinned_source():
    for values in (
        {"source_revision": "main"},
        {"source_revision": REVISION, "base_url": "https://example.com:3737"},
        {"source_revision": REVISION, "base_url": "http://127.0.0.1"},
    ):
        try:
            GTOpenConfig(**values)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe GTOpen configuration was accepted")
