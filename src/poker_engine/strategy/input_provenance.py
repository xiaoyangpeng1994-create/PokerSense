"""Deterministic collection of strategy-input provenance.

The collector is the boundary between heterogeneous input channels and the
canonical ``DecisionContext``.  It emits exactly one provenance record per
field and never resolves disagreeing values silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping
from urllib.parse import quote

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.observation import ObservationField, ValidationStatus

from .contracts import InputProvenance, InputSource, QualityStatus


_SOURCE_PRIORITY = {
    InputSource.MANUAL: 0,
    InputSource.VISION: 1,
    InputSource.CONFIG: 2,
    InputSource.DERIVED: 3,
    InputSource.INFERRED: 4,
}

_VALIDATION_STATUS = {
    ValidationStatus.VALID: QualityStatus.VALID,
    ValidationStatus.LOW_CONFIDENCE: QualityStatus.LOW_CONFIDENCE,
    ValidationStatus.UNKNOWN: QualityStatus.UNKNOWN,
    ValidationStatus.CONFLICT: QualityStatus.CONFLICT,
}


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("provenance values cannot contain non-finite floats")
        return value
    if isinstance(value, Decimal):
        return {"$decimal": format(value, "f")}
    if isinstance(value, datetime):
        _require_aware_dt(value)
        return {"$datetime": value.isoformat()}
    if isinstance(value, Enum):
        return {
            "$enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_value(value.value),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("provenance mapping keys must be strings")
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(
        f"unsupported provenance value type: {type(value).__name__}"
    )


def canonical_value_digest(value: Any) -> str:
    """Return a stable digest used only to compare source values."""
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SuppliedInput:
    """A manually/configured/derived/inferred value with source metadata."""

    value: Any
    confidence: float = 1.0
    status: QualityStatus = QualityStatus.VALID
    evidence_ref: str | None = None
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("confidence must be a float")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.status, QualityStatus):
            raise TypeError("status must be a QualityStatus")
        if self.evidence_ref is not None and (
            not isinstance(self.evidence_ref, str) or not self.evidence_ref
        ):
            raise ValueError("evidence_ref must be a non-empty str or None")
        if self.observed_at is not None:
            if not isinstance(self.observed_at, datetime):
                raise TypeError("observed_at must be a datetime or None")
            _require_aware_dt(self.observed_at)


@dataclass(frozen=True)
class ProvenanceCandidate:
    field_name: str
    value_digest: str
    value_present: bool
    source: InputSource
    status: QualityStatus
    confidence: float
    evidence_ref: str
    observed_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name:
            raise ValueError("field_name must be a non-empty str")
        if not isinstance(self.value_digest, str) or len(self.value_digest) != 64:
            raise ValueError("value_digest must be a SHA-256 hex digest")
        try:
            int(self.value_digest, 16)
        except ValueError as exc:
            raise ValueError("value_digest must be a SHA-256 hex digest") from exc
        if not isinstance(self.value_present, bool):
            raise TypeError("value_present must be a bool")
        InputProvenance(
            self.field_name,
            self.source,
            self.status,
            self.confidence,
            self.evidence_ref,
            self.observed_at,
        )


@dataclass(frozen=True)
class ProvenanceCollection:
    """Unique canonical provenance plus auditable source candidates."""

    provenance: tuple[InputProvenance, ...]
    candidates: tuple[ProvenanceCandidate, ...]

    def __post_init__(self) -> None:
        provenance = tuple(self.provenance)
        candidates = tuple(self.candidates)
        names = [item.field_name for item in provenance]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("provenance fields must be unique and sorted")
        if not all(isinstance(item, InputProvenance) for item in provenance):
            raise TypeError("provenance must contain InputProvenance values")
        if not all(isinstance(item, ProvenanceCandidate) for item in candidates):
            raise TypeError("candidates must contain ProvenanceCandidate values")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "candidates", candidates)


def _candidate(
    field_name: str,
    supplied: SuppliedInput | Any,
    source: InputSource,
) -> ProvenanceCandidate:
    item = supplied if isinstance(supplied, SuppliedInput) else SuppliedInput(supplied)
    status = item.status
    if item.value is None and status is not QualityStatus.CONFLICT:
        status = QualityStatus.UNKNOWN
    digest = canonical_value_digest(item.value)
    evidence_ref = item.evidence_ref or (
        f"{source.value}://{quote(field_name, safe='')}/{digest}"
    )
    return ProvenanceCandidate(
        field_name,
        digest,
        item.value is not None,
        source,
        status,
        item.confidence,
        evidence_ref,
        item.observed_at,
    )


def candidate_from_observation(
    field_name: str,
    observation: ObservationField[Any],
) -> ProvenanceCandidate:
    """Convert raw vision evidence without trusting its free-form source tag."""
    if not isinstance(observation, ObservationField):
        raise TypeError("observation must be an ObservationField")
    digest = canonical_value_digest(observation.value)
    evidence_ref = observation.evidence.get("evidence_ref")
    if not isinstance(evidence_ref, str) or not evidence_ref:
        evidence_ref = (
            f"vision://{quote(observation.source, safe='')}/"
            f"{quote(field_name, safe='')}/{digest}"
        )
    status = _VALIDATION_STATUS[observation.validation_status]
    if observation.value is None and status is not QualityStatus.CONFLICT:
        status = QualityStatus.UNKNOWN
    return ProvenanceCandidate(
        field_name,
        digest,
        observation.value is not None,
        InputSource.VISION,
        status,
        observation.confidence,
        evidence_ref,
        observation.timestamp,
    )


def _winner(candidates: tuple[ProvenanceCandidate, ...]) -> ProvenanceCandidate:
    return min(
        candidates,
        key=lambda item: (
            _SOURCE_PRIORITY[item.source],
            -item.confidence,
            item.evidence_ref,
        ),
    )


def _resolve_field(
    field_name: str,
    candidates: tuple[ProvenanceCandidate, ...],
) -> InputProvenance:
    usable = tuple(
        item for item in candidates
        if item.value_present and item.status is not QualityStatus.UNKNOWN
    )
    distinct_values = {item.value_digest for item in usable}
    explicit_conflict = any(
        item.status is QualityStatus.CONFLICT for item in candidates
    )
    if explicit_conflict or len(distinct_values) > 1:
        winner = _winner(usable or candidates)
        conflict_digest = hashlib.sha256(
            "|".join(sorted(
                f"{item.source.value}:{item.value_digest}:{item.evidence_ref}"
                for item in candidates
            )).encode("utf-8")
        ).hexdigest()
        return InputProvenance(
            field_name,
            winner.source,
            QualityStatus.CONFLICT,
            min(item.confidence for item in candidates),
            f"conflict://{quote(field_name, safe='')}/{conflict_digest}",
            max(
                (item.observed_at for item in candidates if item.observed_at),
                default=None,
            ),
        )
    valid = tuple(item for item in usable if item.status is QualityStatus.VALID)
    low = tuple(
        item for item in usable
        if item.status is QualityStatus.LOW_CONFIDENCE
    )
    pool = valid or low or candidates
    winner = _winner(pool)
    status = (
        QualityStatus.VALID if valid
        else QualityStatus.LOW_CONFIDENCE if low
        else QualityStatus.UNKNOWN
    )
    return InputProvenance(
        field_name,
        winner.source,
        status,
        winner.confidence,
        winner.evidence_ref,
        winner.observed_at,
    )


def collect_input_provenance(
    *,
    observations: Mapping[str, ObservationField[Any]] | None = None,
    manual_inputs: Mapping[str, SuppliedInput | Any] | None = None,
    config_inputs: Mapping[str, SuppliedInput | Any] | None = None,
    derived_inputs: Mapping[str, SuppliedInput | Any] | None = None,
    inferred_inputs: Mapping[str, SuppliedInput | Any] | None = None,
) -> ProvenanceCollection:
    """Collect all supported channels into one deterministic record per field."""
    candidates: list[ProvenanceCandidate] = []
    for field_name, observation in (observations or {}).items():
        candidates.append(candidate_from_observation(field_name, observation))
    for source, values in (
        (InputSource.MANUAL, manual_inputs or {}),
        (InputSource.CONFIG, config_inputs or {}),
        (InputSource.DERIVED, derived_inputs or {}),
        (InputSource.INFERRED, inferred_inputs or {}),
    ):
        for field_name, supplied in values.items():
            candidates.append(_candidate(field_name, supplied, source))
    ordered_candidates = tuple(sorted(
        candidates,
        key=lambda item: (
            item.field_name,
            _SOURCE_PRIORITY[item.source],
            item.evidence_ref,
        ),
    ))
    by_field: dict[str, list[ProvenanceCandidate]] = {}
    for candidate in ordered_candidates:
        by_field.setdefault(candidate.field_name, []).append(candidate)
    provenance = tuple(
        _resolve_field(field_name, tuple(by_field[field_name]))
        for field_name in sorted(by_field)
    )
    return ProvenanceCollection(provenance, ordered_candidates)


__all__ = [
    "ProvenanceCandidate",
    "ProvenanceCollection",
    "SuppliedInput",
    "candidate_from_observation",
    "canonical_value_digest",
    "collect_input_provenance",
]
