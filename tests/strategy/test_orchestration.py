from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta

from poker_engine.core.request_context import RequestContext
from poker_engine.strategy.advice import AdviceStatus
from poker_engine.strategy.orchestration import (
    RefinementState,
    StrategyOrchestrator,
    ThreadedSlowResolver,
)
from poker_engine.strategy.provider import (
    FakeProvider,
    LookupState,
    MatchKind,
    ProviderResult,
)
from poker_engine.strategy.router import StrategyRouter

from .helpers import NOW, candidate, capability, context, hit_result


class ManualResolver:
    def __init__(self, *, provider_id="slow", version="v2", cap=None):
        self.provider_id = provider_id
        self.source_version = version
        self.capability = cap or capability((2,))
        self.future = Future()
        self.submit_count = 0

    def submit(self, value):
        self.submit_count += 1
        self.submitted = value
        return self.future


def slow_hit(ctx, resolver, *, kind=MatchKind.EXACT, expires_at=None):
    value = candidate(
        ctx,
        resolver.provider_id,
        resolver.source_version,
        match_kind=kind,
        expires_at=expires_at,
    )
    return hit_result(value)


def test_fast_advice_returns_before_unresolved_slow_future():
    ctx = context()
    resolver = ManualResolver()
    cycle = StrategyOrchestrator(
        StrategyRouter(), resolver
    ).request(ctx, math_report={"equity": "0.55"}, now=NOW)

    assert cycle.fast_advice.status is AdviceStatus.PARTIAL
    assert cycle.slow_handle is not None
    assert not resolver.future.done()
    assert resolver.submit_count == 1


def test_same_context_better_slow_result_is_applied():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    resolver.future.set_result(slow_hit(ctx, resolver))

    refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)
    assert refinement.state is RefinementState.APPLIED
    assert refinement.advice.status is AdviceStatus.READY
    assert refinement.advice.strategy_source == resolver.provider_id


def test_pending_slow_result_does_not_replace_fast_advice():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)

    refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)
    assert refinement.state is RefinementState.PENDING
    assert refinement.advice is None


def test_new_state_discards_and_cancels_old_slow_result():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    updated_request = replace(
        ctx.request,
        state_version=ctx.state_version + 1,
        request_id="new-request",
    )
    updated = replace(ctx, request=updated_request)

    refinement = orchestrator.collect(cycle.slow_handle, updated, now=NOW)
    assert refinement.state is RefinementState.DISCARDED
    assert refinement.reasons == ("stale_context",)
    assert resolver.future.cancelled()


def test_expired_request_discards_slow_result():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)

    refinement = orchestrator.collect(
        cycle.slow_handle, ctx, now=ctx.request.expires_at
    )
    assert refinement.state is RefinementState.DISCARDED
    assert refinement.reasons == ("expired_request",)


def test_expired_candidate_is_discarded():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    resolver.future.set_result(slow_hit(
        ctx, resolver, expires_at=NOW + timedelta(milliseconds=10)
    ))

    refinement = orchestrator.collect(
        cycle.slow_handle, ctx, now=NOW + timedelta(milliseconds=20)
    )
    assert refinement.state is RefinementState.DISCARDED
    assert refinement.reasons == ("expired_candidate",)


def test_slow_exception_is_contained():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    resolver.future.set_exception(RuntimeError("solver crashed"))

    refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)
    assert refinement.state is RefinementState.FAILED
    assert refinement.reasons == ("slow_resolver_error:RuntimeError",)


def test_exact_fast_result_does_not_schedule_slow():
    ctx = context()
    exact = candidate(ctx, "fast", "v1")
    fast = FakeProvider("fast", "v1", capability((2,)), hit_result(exact))
    resolver = ManualResolver()

    cycle = StrategyOrchestrator(
        StrategyRouter((fast,)), resolver
    ).request(ctx, now=NOW)
    assert cycle.fast_advice.status is AdviceStatus.READY
    assert cycle.slow_handle is None
    assert resolver.submit_count == 0


def test_same_or_lower_quality_slow_result_is_not_applied():
    ctx = context()
    fast_value = candidate(
        ctx, "fast", "v1", match_kind=MatchKind.INTERPOLATED
    )
    fast = FakeProvider(
        "fast",
        "v1",
        capability((2,), match_kind=MatchKind.INTERPOLATED),
        hit_result(fast_value),
    )
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter((fast,)), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    resolver.future.set_result(slow_hit(
        ctx, resolver, kind=MatchKind.INTERPOLATED
    ))

    refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)
    assert refinement.state is RefinementState.NO_UPDATE
    assert refinement.reasons == ("slow_result_not_better",)


def test_non_applicable_slow_capability_is_not_scheduled():
    ctx = context(3)
    resolver = ManualResolver(cap=capability((2,)))
    cycle = StrategyOrchestrator(
        StrategyRouter(), resolver
    ).request(ctx, now=NOW)
    assert cycle.slow_handle is None
    assert resolver.submit_count == 0


def test_slow_provider_identity_mismatch_fails_closed():
    ctx = context()
    resolver = ManualResolver()
    orchestrator = StrategyOrchestrator(StrategyRouter(), resolver)
    cycle = orchestrator.request(ctx, now=NOW)
    resolver.future.set_result(ProviderResult(LookupState.NOT_FOUND, "other"))

    refinement = orchestrator.collect(cycle.slow_handle, ctx, now=NOW)
    assert refinement.state is RefinementState.FAILED
    assert refinement.reasons == ("provider_id_mismatch",)


def test_deadline_derived_request_expiry_prevents_scheduling():
    ctx = context()
    request = RequestContext(
        hand_id=ctx.hand_id,
        state_version=ctx.state_version,
        request_id=ctx.request_id,
        requested_at=NOW,
        deadline_ms=10,
    )
    ctx = replace(ctx, request=request)
    resolver = ManualResolver()

    cycle = StrategyOrchestrator(
        StrategyRouter(), resolver
    ).request(ctx, now=NOW + timedelta(milliseconds=10))
    assert cycle.fast_advice.status is AdviceStatus.STALE
    assert cycle.slow_handle is None


def test_threaded_resolver_adapts_sync_provider_without_owning_executor():
    ctx = context()
    value = candidate(ctx, "threaded", "v1")
    provider = FakeProvider(
        "threaded", "v1", capability((2,)), hit_result(value)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        resolver = ThreadedSlowResolver(provider, executor)
        result = resolver.submit(ctx).result(timeout=1)
    assert result.state is LookupState.HIT_EXACT
