"""Context provenance/quality aggregation and unique request creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Callable
from uuid import uuid4

from poker_engine.core.request_context import RequestContext

from .contracts import ContextQuality, InputProvenance, QualityStatus


@dataclass(frozen=True)
class ContextQualityPolicy:
    required_fields: tuple[str, ...]
    minimum_confidence: float = 0.8

    def __post_init__(self) -> None:
        fields = tuple(self.required_fields)
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("required_fields must be non-empty and unique")
        if not all(isinstance(field, str) and field for field in fields):
            raise TypeError("required_fields must contain non-empty strings")
        object.__setattr__(self, "required_fields", fields)
        value = self.minimum_confidence
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("minimum_confidence must be a float")
        if not 0 <= value <= 1:
            raise ValueError("minimum_confidence must be in [0, 1]")


def aggregate_context_quality(
    provenance: tuple[InputProvenance, ...],
    policy: ContextQualityPolicy,
    *,
    consistency_failures: tuple[str, ...] = (),
) -> ContextQuality:
    """Aggregate required input quality conservatively using the minimum.

    Duplicate field provenance is rejected instead of selecting a source
    silently. Missing, UNKNOWN, CONFLICT, LOW_CONFIDENCE, and below-threshold
    required fields become machine-readable hard failures.
    """
    if not isinstance(policy, ContextQualityPolicy):
        raise TypeError("policy must be a ContextQualityPolicy")
    items = tuple(provenance)
    if not all(isinstance(item, InputProvenance) for item in items):
        raise TypeError("provenance must contain InputProvenance values")
    by_field = {}
    for item in items:
        if item.field_name in by_field:
            raise ValueError(f"duplicate provenance field {item.field_name!r}")
        by_field[item.field_name] = item
    failures = list(consistency_failures)
    if not all(isinstance(value, str) and value for value in failures):
        raise TypeError("consistency_failures must contain non-empty strings")
    confidences = {item.field_name: item.confidence for item in items}
    required_confidences = []
    for field in policy.required_fields:
        item = by_field.get(field)
        if item is None:
            failures.append(f"missing_provenance:{field}")
            continue
        required_confidences.append(item.confidence)
        if item.status is QualityStatus.UNKNOWN:
            failures.append(f"unknown:{field}")
        elif item.status is QualityStatus.CONFLICT:
            failures.append(f"conflict:{field}")
        elif item.status is QualityStatus.LOW_CONFIDENCE:
            failures.append(f"low_confidence:{field}")
        elif item.confidence < policy.minimum_confidence:
            failures.append(f"below_threshold:{field}")
    failures = list(dict.fromkeys(failures))
    overall = min(required_confidences, default=0.0)
    if failures:
        overall = 0.0
    return ContextQuality(overall, confidences, tuple(failures))


class RequestContextFactory:
    """Thread-safe request factory with process-local duplicate protection."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        request_id_factory: Callable[[], str] | None = None,
        maximum_id_attempts: int = 3,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        if request_id_factory is not None and not callable(request_id_factory):
            raise TypeError("request_id_factory must be callable")
        if not isinstance(maximum_id_attempts, int) or isinstance(
            maximum_id_attempts, bool
        ) or maximum_id_attempts <= 0:
            raise ValueError("maximum_id_attempts must be a positive int")
        self._clock = clock
        self._request_id_factory = request_id_factory or (
            lambda: str(uuid4())
        )
        self._maximum_id_attempts = maximum_id_attempts
        self._issued: set[str] = set()
        self._lock = Lock()

    def create(
        self,
        *,
        hand_id: str,
        state_version: int,
        deadline_ms: int,
    ) -> RequestContext:
        requested_at = self._clock()
        if not isinstance(requested_at, datetime):
            raise TypeError("clock must return a datetime")
        with self._lock:
            request_id = None
            for _ in range(self._maximum_id_attempts):
                candidate = self._request_id_factory()
                if candidate not in self._issued:
                    request_id = candidate
                    self._issued.add(candidate)
                    break
            if request_id is None:
                raise RuntimeError("request ID factory produced repeated IDs")
        try:
            return RequestContext(
                hand_id=hand_id,
                state_version=state_version,
                request_id=request_id,
                requested_at=requested_at,
                expires_at=requested_at + timedelta(milliseconds=deadline_ms),
                deadline_ms=deadline_ms,
            )
        except Exception:
            with self._lock:
                self._issued.discard(request_id)
            raise

    @property
    def issued_count(self) -> int:
        with self._lock:
            return len(self._issued)


__all__ = [
    "ContextQualityPolicy",
    "RequestContextFactory",
    "aggregate_context_quality",
]
