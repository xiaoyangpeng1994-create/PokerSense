"""Structured evidence-chain construction and completeness auditing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from .contracts import DecisionContext, QualityStatus, RangeDistribution
from .provider import StrategyCandidate


class EvidenceStage(str, Enum):
    INPUT = "input"
    STATE = "state"
    RANGE = "range"
    PROVIDER = "provider"


@dataclass(frozen=True)
class EvidenceReference:
    stage: EvidenceStage
    key: str
    uri: str
    version: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, EvidenceStage):
            raise TypeError("stage must be an EvidenceStage")
        for name in ("key", "uri"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if self.version is not None and (
            not isinstance(self.version, str) or not self.version
        ):
            raise ValueError("version must be a non-empty str or None")
        if self.confidence is not None and (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be in [0, 1] or None")


@dataclass(frozen=True)
class EvidenceAuditPolicy:
    required_input_fields: tuple[str, ...] | None = None
    require_hero_range: bool = True
    require_villain_ranges: bool = True
    require_provider_evidence: bool = True
    incomplete_confidence_cap: float = 0.49

    def __post_init__(self) -> None:
        fields = self.required_input_fields
        if fields is not None:
            fields = tuple(fields)
            if len(fields) != len(set(fields)) or not all(
                isinstance(field, str) and field for field in fields
            ):
                raise ValueError(
                    "required_input_fields must be unique non-empty strings"
                )
            object.__setattr__(self, "required_input_fields", fields)
        for name in (
            "require_hero_range",
            "require_villain_ranges",
            "require_provider_evidence",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        cap = self.incomplete_confidence_cap
        if (
            not isinstance(cap, (int, float))
            or isinstance(cap, bool)
            or not 0 <= cap <= 0.49
        ):
            raise ValueError("incomplete_confidence_cap must be in [0, 0.49]")


@dataclass(frozen=True)
class EvidenceAudit:
    chain_id: str
    complete: bool
    references: tuple[EvidenceReference, ...]
    missing: tuple[str, ...]
    satisfied_requirements: int
    total_requirements: int
    confidence_limit: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chain_id, str)
            or len(self.chain_id) != 64
            or any(char not in "0123456789abcdef"
                   for char in self.chain_id.lower())
        ):
            raise ValueError("chain_id must be a SHA-256 hex digest")
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be a bool")
        references = tuple(self.references)
        if not all(isinstance(item, EvidenceReference) for item in references):
            raise TypeError("references must contain EvidenceReference values")
        object.__setattr__(self, "references", references)
        missing = tuple(self.missing)
        if not all(isinstance(item, str) and item for item in missing):
            raise TypeError("missing must contain non-empty strings")
        object.__setattr__(self, "missing", missing)
        if self.complete != (not missing):
            raise ValueError("complete and missing must agree")
        if not 0 <= self.satisfied_requirements <= self.total_requirements:
            raise ValueError("invalid evidence requirement counts")
        if self.complete and self.satisfied_requirements != self.total_requirements:
            raise ValueError("complete audit must satisfy every requirement")
        if not 0 <= self.confidence_limit <= 1:
            raise ValueError("confidence_limit must be in [0, 1]")

    @property
    def evidence_uris(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.uri for item in self.references))


def audit_evidence_chain(
    context: DecisionContext,
    candidate: StrategyCandidate | None = None,
    *,
    policy: EvidenceAuditPolicy | None = None,
) -> EvidenceAudit:
    if not isinstance(context, DecisionContext):
        raise TypeError("context must be DecisionContext")
    if candidate is not None and not isinstance(candidate, StrategyCandidate):
        raise TypeError("candidate must be StrategyCandidate or None")
    if candidate is not None and (
        candidate.hand_id != context.hand_id
        or candidate.state_version != context.state_version
        or candidate.request_id != context.request_id
    ):
        raise ValueError("candidate_context_mismatch")
    policy = policy or EvidenceAuditPolicy()
    if not isinstance(policy, EvidenceAuditPolicy):
        raise TypeError("policy must be EvidenceAuditPolicy")

    references: list[EvidenceReference] = []
    missing: list[str] = []
    required = 0
    satisfied = 0
    provenance = {item.field_name: item for item in context.input_provenance}
    required_fields = policy.required_input_fields
    if required_fields is None:
        required_fields = tuple(sorted(context.input_quality.field_confidences))
    for field in required_fields:
        required += 1
        item = provenance.get(field)
        if item is None:
            missing.append(f"input:{field}")
            continue
        references.append(EvidenceReference(
            EvidenceStage.INPUT,
            field,
            item.evidence_ref,
            confidence=item.confidence,
        ))
        if item.status is QualityStatus.VALID:
            satisfied += 1
        else:
            missing.append(f"input_not_valid:{field}:{item.status.value}")

    required += 1
    satisfied += 1
    references.append(EvidenceReference(
        EvidenceStage.STATE,
        "canonical_state",
        f"state://{context.hand_id}/{context.state_version}",
        version=str(context.state_version),
        confidence=context.input_quality.overall_confidence,
    ))

    if policy.require_hero_range:
        required += 1
        if _append_range_reference(references, context.hero_range):
            satisfied += 1
        else:
            missing.append(f"range:seat:{context.hero_seat}")
    if policy.require_villain_ranges:
        by_seat = {item.seat_id: item for item in context.villain_ranges}
        for seat_id in context.active_seats:
            if seat_id == context.hero_seat:
                continue
            required += 1
            if _append_range_reference(references, by_seat.get(seat_id)):
                satisfied += 1
            else:
                missing.append(f"range:seat:{seat_id}")

    if candidate is not None and policy.require_provider_evidence:
        required += 1
        if candidate.evidence:
            satisfied += 1
            references.extend(
                EvidenceReference(EvidenceStage.PROVIDER, str(index), uri)
                for index, uri in enumerate(candidate.evidence)
            )
        else:
            missing.append(f"provider:{candidate.provider_id}")

    references = sorted(
        references,
        key=lambda item: (item.stage.value, item.key, item.uri),
    )
    digest_payload = [
        {
            "stage": item.stage.value,
            "key": item.key,
            "uri": item.uri,
            "version": item.version,
            "confidence": item.confidence,
        }
        for item in references
    ]
    chain_id = hashlib.sha256(json.dumps(
        digest_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    missing_values = tuple(dict.fromkeys(missing))
    complete = not missing_values
    return EvidenceAudit(
        chain_id,
        complete,
        tuple(references),
        missing_values,
        satisfied,
        required,
        1.0 if complete else policy.incomplete_confidence_cap,
    )


def _append_range_reference(
    references: list[EvidenceReference],
    value: RangeDistribution | None,
) -> bool:
    if value is None or not value.combo_weights:
        return False
    references.append(EvidenceReference(
        EvidenceStage.RANGE,
        f"seat:{value.seat_id}",
        f"range://seat/{value.seat_id}/{value.source}/{value.source_version}",
        version=value.source_version,
        confidence=value.confidence,
    ))
    return True


__all__ = [
    "EvidenceAudit",
    "EvidenceAuditPolicy",
    "EvidenceReference",
    "EvidenceStage",
    "audit_evidence_chain",
]
