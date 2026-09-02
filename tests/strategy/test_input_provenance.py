from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from poker_engine.core.observation import ObservationField, ValidationStatus
from poker_engine.core.value_objects import ChipAmount
from poker_engine.strategy.context_factory import (
    ContextQualityPolicy,
    aggregate_context_quality,
)
from poker_engine.strategy.contracts import InputSource, QualityStatus
from poker_engine.strategy.input_provenance import (
    SuppliedInput,
    candidate_from_observation,
    canonical_value_digest,
    collect_input_provenance,
)


NOW = datetime(2026, 8, 22, tzinfo=timezone.utc)


def observation(
    value,
    *,
    confidence=0.95,
    status=ValidationStatus.VALID,
    evidence=None,
):
    return ObservationField(
        value,
        confidence,
        "wepoker.hero-card-detector-v2",
        evidence or {},
        NOW,
        status,
    )


@pytest.mark.parametrize(
    ("channel", "expected_source"),
    [
        ("manual_inputs", InputSource.MANUAL),
        ("config_inputs", InputSource.CONFIG),
        ("derived_inputs", InputSource.DERIVED),
        ("inferred_inputs", InputSource.INFERRED),
    ],
)
def test_each_non_vision_channel_is_labeled_without_source_spoofing(
    channel, expected_source
):
    result = collect_input_provenance(**{channel: {"pot": ChipAmount("12")}})
    assert len(result.provenance) == 1
    assert result.provenance[0].source is expected_source
    assert result.provenance[0].status is QualityStatus.VALID
    assert result.provenance[0].evidence_ref.startswith(
        f"{expected_source.value}://"
    )


def test_observation_source_is_always_vision_and_preserves_adapter_in_ref():
    candidate = candidate_from_observation("pot", observation(ChipAmount("12")))
    assert candidate.source is InputSource.VISION
    assert "wepoker.hero-card-detector-v2" in candidate.evidence_ref


def test_observation_uses_explicit_evidence_reference_when_available():
    candidate = candidate_from_observation(
        "pot", observation(ChipAmount("12"), evidence={
            "evidence_ref": "frame://table-1/123/pot"
        })
    )
    assert candidate.evidence_ref == "frame://table-1/123/pot"


@pytest.mark.parametrize(
    ("validation", "quality"),
    [
        (ValidationStatus.VALID, QualityStatus.VALID),
        (ValidationStatus.LOW_CONFIDENCE, QualityStatus.LOW_CONFIDENCE),
        (ValidationStatus.UNKNOWN, QualityStatus.UNKNOWN),
        (ValidationStatus.CONFLICT, QualityStatus.CONFLICT),
    ],
)
def test_observation_validation_status_is_mapped_exactly(validation, quality):
    candidate = candidate_from_observation(
        "pot", observation(ChipAmount("12"), status=validation)
    )
    assert candidate.status is quality


def test_none_value_is_unknown_even_when_caller_marks_it_valid():
    result = collect_input_provenance(manual_inputs={"actor": None})
    assert result.provenance[0].status is QualityStatus.UNKNOWN


def test_same_value_from_multiple_sources_forms_valid_consensus():
    result = collect_input_provenance(
        observations={"pot": observation(ChipAmount("12"), confidence=0.93)},
        manual_inputs={"pot": SuppliedInput(
            ChipAmount("12"), evidence_ref="manual://session/pot", observed_at=NOW
        )},
        config_inputs={"pot": ChipAmount("12")},
    )
    assert len(result.provenance) == 1
    assert len(result.candidates) == 3
    resolved = result.provenance[0]
    assert resolved.source is InputSource.MANUAL
    assert resolved.status is QualityStatus.VALID
    assert resolved.confidence == 1.0
    assert resolved.evidence_ref == "manual://session/pot"


def test_disagreeing_values_are_conflict_not_silent_manual_override():
    result = collect_input_provenance(
        observations={"pot": observation(ChipAmount("12"), confidence=0.94)},
        manual_inputs={"pot": SuppliedInput(ChipAmount("13"), confidence=1.0)},
    )
    resolved = result.provenance[0]
    assert resolved.source is InputSource.MANUAL
    assert resolved.status is QualityStatus.CONFLICT
    assert resolved.confidence == 0.94
    assert resolved.evidence_ref.startswith("conflict://pot/")


def test_low_confidence_disagreement_is_still_recorded_as_conflict():
    result = collect_input_provenance(
        observations={"actor": observation(
            1, confidence=0.4, status=ValidationStatus.LOW_CONFIDENCE
        )},
        inferred_inputs={"actor": SuppliedInput(
            2, confidence=0.5, status=QualityStatus.LOW_CONFIDENCE
        )},
    )
    assert result.provenance[0].status is QualityStatus.CONFLICT


def test_one_low_confidence_value_stays_low_confidence():
    result = collect_input_provenance(inferred_inputs={
        "actor": SuppliedInput(
            2, confidence=0.5, status=QualityStatus.LOW_CONFIDENCE
        )
    })
    assert result.provenance[0].status is QualityStatus.LOW_CONFIDENCE
    assert result.provenance[0].confidence == 0.5


def test_explicit_source_conflict_cannot_be_overridden_by_valid_source():
    result = collect_input_provenance(
        observations={"actor": observation(
            1, confidence=0.8, status=ValidationStatus.CONFLICT
        )},
        manual_inputs={"actor": 1},
    )
    assert result.provenance[0].status is QualityStatus.CONFLICT


def test_output_is_unique_sorted_and_independent_of_mapping_order():
    first = collect_input_provenance(
        manual_inputs={"street": "flop", "actor": 1},
        config_inputs={"blinds": (1, 2)},
    )
    second = collect_input_provenance(
        config_inputs={"blinds": (1, 2)},
        manual_inputs={"actor": 1, "street": "flop"},
    )
    assert first == second
    assert [item.field_name for item in first.provenance] == [
        "actor", "blinds", "street"
    ]


def test_canonical_digest_handles_mapping_order_decimal_and_chip_amount():
    left = {
        "chips": ChipAmount("12.50"),
        "weights": {"AA": Decimal("0.5"), "KK": Decimal("0.5")},
    }
    right = {
        "weights": {"KK": Decimal("0.5"), "AA": Decimal("0.5")},
        "chips": ChipAmount("12.50"),
    }
    assert canonical_value_digest(left) == canonical_value_digest(right)


def test_canonical_digest_rejects_unsupported_and_non_finite_values():
    with pytest.raises(TypeError, match="unsupported provenance"):
        canonical_value_digest(object())
    with pytest.raises(ValueError, match="non-finite"):
        canonical_value_digest(float("nan"))


def test_conflict_is_consumed_by_existing_quality_gate():
    result = collect_input_provenance(
        observations={"pot": observation(ChipAmount("12"))},
        manual_inputs={"pot": ChipAmount("13")},
    )
    quality = aggregate_context_quality(
        result.provenance, ContextQualityPolicy(("pot",), 0.8)
    )
    assert quality.overall_confidence == 0.0
    assert quality.hard_failures == ("conflict:pot",)


def test_unknown_field_is_consumed_by_existing_quality_gate():
    result = collect_input_provenance(inferred_inputs={"actor": None})
    quality = aggregate_context_quality(
        result.provenance, ContextQualityPolicy(("actor",), 0.8)
    )
    assert quality.hard_failures == ("unknown:actor",)


def test_invalid_input_metadata_is_rejected_early():
    with pytest.raises(ValueError, match="confidence"):
        SuppliedInput("x", confidence=1.1)
    with pytest.raises(TypeError, match="QualityStatus"):
        SuppliedInput("x", status="VALID")
    with pytest.raises(TypeError, match="timezone-aware"):
        SuppliedInput("x", observed_at=datetime(2026, 8, 22))


def test_empty_collection_is_well_defined():
    result = collect_input_provenance()
    assert result.provenance == ()
    assert result.candidates == ()
