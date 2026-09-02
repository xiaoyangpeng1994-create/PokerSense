"""Structured evidence-chain audits and Advice confidence gating."""

from __future__ import annotations

from dataclasses import replace

import pytest

from poker_engine.strategy.advice import Advice, AdviceStatus, build_advice
from poker_engine.strategy.contracts import ContextQuality, QualityStatus
from poker_engine.strategy.evidence import (
    EvidenceAuditPolicy,
    EvidenceStage,
    audit_evidence_chain,
)
from poker_engine.strategy.provider import LookupState
from poker_engine.strategy.router import RouteResult
from poker_engine.strategy.serialization import (
    strategy_deserialize,
    strategy_serialize,
)

from .helpers import NOW, candidate, context


def _route(value):
    return RouteResult(LookupState.HIT_EXACT, value, ())


def test_complete_chain_contains_input_state_range_and_provider_stages():
    ctx = context(3)
    audit = audit_evidence_chain(ctx, candidate(ctx))

    assert audit.complete
    assert audit.missing == ()
    assert audit.satisfied_requirements == audit.total_requirements
    assert audit.confidence_limit == 1.0
    assert {item.stage for item in audit.references} == set(EvidenceStage)
    assert len(audit.chain_id) == 64


def test_chain_digest_is_stable_across_provenance_order():
    ctx = context()
    baseline = audit_evidence_chain(ctx, candidate(ctx))
    reordered = replace(ctx, input_provenance=tuple(reversed(ctx.input_provenance)))

    assert audit_evidence_chain(reordered, candidate(reordered)).chain_id == (
        baseline.chain_id
    )


def test_state_version_or_provider_evidence_changes_chain_digest():
    ctx = context()
    value = candidate(ctx)
    baseline = audit_evidence_chain(ctx, value)
    changed_provider = replace(value, evidence=("mock://different",))

    assert audit_evidence_chain(ctx, changed_provider).chain_id != baseline.chain_id
    changed_context = replace(ctx, request=replace(ctx.request, state_version=2))
    changed_candidate = replace(value, state_version=2)
    assert audit_evidence_chain(changed_context, changed_candidate).chain_id != (
        baseline.chain_id
    )


def test_missing_required_input_provenance_is_named_and_caps_confidence():
    ctx = context()
    ctx = replace(ctx, input_quality=ContextQuality(
        0.9,
        {"hero_cards": 0.99, "stacks": 0.9, "pot": 0.9},
    ))
    audit = audit_evidence_chain(ctx, candidate(ctx))

    assert not audit.complete
    assert audit.missing == ("input:pot",)
    assert audit.confidence_limit == 0.49


def test_non_valid_input_provenance_does_not_satisfy_evidence_requirement():
    ctx = context()
    provenance = list(ctx.input_provenance)
    provenance[0] = replace(provenance[0], status=QualityStatus.UNKNOWN)
    ctx = replace(ctx, input_provenance=tuple(provenance))
    audit = audit_evidence_chain(ctx, candidate(ctx))

    assert "input_not_valid:hero_cards:UNKNOWN" in audit.missing


@pytest.mark.parametrize(
    ("mutation", "missing"),
    (
        (lambda value: replace(value, hero_range=None), "range:seat:1"),
        (lambda value: replace(value, villain_ranges=()), "range:seat:0"),
    ),
)
def test_missing_required_range_link_is_named(mutation, missing):
    ctx = mutation(context())
    audit = audit_evidence_chain(ctx, candidate(ctx))

    assert missing in audit.missing


def test_context_only_audit_does_not_require_provider_evidence():
    audit = audit_evidence_chain(context())

    assert audit.complete
    assert all(item.stage is not EvidenceStage.PROVIDER
               for item in audit.references)


def test_candidate_without_provider_evidence_breaks_only_provider_link():
    ctx = context()
    value = replace(candidate(ctx), evidence=())
    audit = audit_evidence_chain(ctx, value)

    assert not audit.complete
    assert audit.missing == ("provider:mock-2p",)


def test_complete_advice_exposes_chain_id_and_round_trips():
    ctx = context()
    advice = build_advice(ctx, _route(candidate(ctx)), now=NOW)
    payload = strategy_serialize(advice)
    decoded = strategy_deserialize(Advice, payload)
    assert advice.status is AdviceStatus.READY
    assert advice.evidence_complete
    assert advice.missing_evidence == ()
    assert advice.confidence_factors["evidence_chain"] == 1.0
    assert decoded == advice


def test_older_schema_v1_advice_without_evidence_audit_fields_still_loads():
    ctx = context()
    advice = build_advice(ctx, _route(candidate(ctx)), now=NOW)
    payload = strategy_serialize(advice)
    for field in (
        "evidence_chain_id",
        "evidence_complete",
        "missing_evidence",
        "input_provenance",
    ):
        payload.pop(field)

    decoded = strategy_deserialize(Advice, payload)
    assert decoded.evidence_chain_id is None
    assert not decoded.evidence_complete
    assert decoded.missing_evidence == ()
    assert decoded.input_provenance == ()


def test_incomplete_provider_chain_remains_visible_but_never_high_confidence():
    ctx = context()
    value = replace(candidate(ctx), evidence=())
    advice = build_advice(ctx, _route(value), now=NOW)

    assert advice.status is AdviceStatus.READY
    assert not advice.evidence_complete
    assert advice.missing_evidence == ("provider:mock-2p",)
    assert advice.confidence == 0.49
    assert advice.confidence_factors["evidence_chain"] == 0.49
    assert "evidence_chain_incomplete" in advice.assumptions


def test_custom_policy_can_explicitly_disable_range_requirements():
    ctx = replace(context(), hero_range=None, villain_ranges=())
    audit = audit_evidence_chain(
        ctx,
        candidate(ctx),
        policy=EvidenceAuditPolicy(
            require_hero_range=False,
            require_villain_ranges=False,
        ),
    )

    assert audit.complete


def test_candidate_from_another_context_is_rejected():
    ctx = context()
    with pytest.raises(ValueError, match="candidate_context_mismatch"):
        audit_evidence_chain(ctx, replace(candidate(ctx), request_id="other"))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"required_input_fields": ("pot", "pot")},
        {"required_input_fields": ("",)},
        {"incomplete_confidence_cap": 0.5},
        {"incomplete_confidence_cap": -0.1},
    ),
)
def test_invalid_audit_policy_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        EvidenceAuditPolicy(**kwargs)


def test_audit_references_are_immutable_tuples():
    audit = audit_evidence_chain(context())
    assert isinstance(audit.references, tuple)
    with pytest.raises(AttributeError):
        audit.references.append("bad")
