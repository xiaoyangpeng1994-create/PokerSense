from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from poker_engine.strategy.context_factory import (
    ContextQualityPolicy,
    RequestContextFactory,
    aggregate_context_quality,
)
from poker_engine.strategy.contracts import (
    InputProvenance,
    InputSource,
    QualityStatus,
)


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def evidence(
    field,
    *,
    confidence=0.95,
    status=QualityStatus.VALID,
    source=InputSource.VISION,
):
    return InputProvenance(
        field, source, status, confidence, f"test://{field}", NOW
    )


def test_quality_uses_required_field_minimum_not_average():
    result = aggregate_context_quality(
        (evidence("cards", confidence=0.99),
         evidence("pot", confidence=0.81),
         evidence("optional", confidence=0.10)),
        ContextQualityPolicy(("cards", "pot"), 0.8),
    )
    assert result.overall_confidence == 0.81
    assert result.is_decision_ready
    assert result.field_confidences["optional"] == 0.10


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ((evidence("cards"),), "missing_provenance:pot"),
        ((evidence("cards"), evidence("pot", status=QualityStatus.UNKNOWN)),
         "unknown:pot"),
        ((evidence("cards"), evidence("pot", status=QualityStatus.CONFLICT)),
         "conflict:pot"),
        ((evidence("cards"), evidence(
            "pot", confidence=0.7, status=QualityStatus.LOW_CONFIDENCE
        )), "low_confidence:pot"),
        ((evidence("cards"), evidence("pot", confidence=0.79)),
         "below_threshold:pot"),
    ],
)
def test_required_quality_failure_is_machine_readable_and_caps_zero(items, reason):
    result = aggregate_context_quality(
        items, ContextQualityPolicy(("cards", "pot"), 0.8)
    )
    assert result.overall_confidence == 0.0
    assert not result.is_decision_ready
    assert reason in result.hard_failures


def test_consistency_failures_are_deduplicated_and_fail_closed():
    result = aggregate_context_quality(
        (evidence("cards"), evidence("pot")),
        ContextQualityPolicy(("cards", "pot")),
        consistency_failures=("chip_conservation", "chip_conservation"),
    )
    assert result.hard_failures == ("chip_conservation",)
    assert result.overall_confidence == 0.0


def test_duplicate_field_provenance_is_rejected():
    with pytest.raises(ValueError, match="duplicate provenance"):
        aggregate_context_quality(
            (evidence("pot"), evidence("pot", source=InputSource.MANUAL)),
            ContextQualityPolicy(("pot",)),
        )


def test_request_factory_creates_aware_expiring_unique_contexts():
    ids = iter(("r1", "r2"))
    factory = RequestContextFactory(
        clock=lambda: NOW, request_id_factory=lambda: next(ids)
    )
    first = factory.create(hand_id="h1", state_version=1, deadline_ms=300)
    second = factory.create(hand_id="h1", state_version=1, deadline_ms=300)
    assert first.request_id == "r1"
    assert second.request_id == "r2"
    assert first.expires_at.isoformat() == "2026-08-22T00:00:00.300000+00:00"
    assert factory.issued_count == 2


def test_request_factory_retries_duplicate_then_uses_new_id():
    ids = iter(("same", "same", "new"))
    factory = RequestContextFactory(
        clock=lambda: NOW,
        request_id_factory=lambda: next(ids),
        maximum_id_attempts=2,
    )
    assert factory.create(
        hand_id="h1", state_version=1, deadline_ms=10
    ).request_id == "same"
    assert factory.create(
        hand_id="h1", state_version=1, deadline_ms=10
    ).request_id == "new"


def test_request_factory_repeated_ids_fail_without_reusing_identity():
    factory = RequestContextFactory(
        clock=lambda: NOW,
        request_id_factory=lambda: "same",
        maximum_id_attempts=2,
    )
    factory.create(hand_id="h1", state_version=1, deadline_ms=10)
    with pytest.raises(RuntimeError, match="repeated IDs"):
        factory.create(hand_id="h1", state_version=1, deadline_ms=10)
    assert factory.issued_count == 1


def test_request_factory_is_thread_safe():
    counter = iter(f"r{i}" for i in range(20))
    factory = RequestContextFactory(
        clock=lambda: NOW, request_id_factory=lambda: next(counter)
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = tuple(executor.map(
            lambda version: factory.create(
                hand_id="h1", state_version=version, deadline_ms=100
            ),
            range(20),
        ))
    assert len({request.request_id for request in requests}) == 20
    assert factory.issued_count == 20


def test_naive_clock_result_is_rejected_and_id_reservation_rolled_back():
    factory = RequestContextFactory(
        clock=lambda: datetime(2026, 8, 22),
        request_id_factory=lambda: "r1",
    )
    with pytest.raises(TypeError, match="timezone-aware"):
        factory.create(hand_id="h1", state_version=1, deadline_ms=100)
    assert factory.issued_count == 0
