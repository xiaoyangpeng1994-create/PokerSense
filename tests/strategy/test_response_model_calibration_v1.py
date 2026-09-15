from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import math

import pytest

from poker_engine.strategy.response_model_calibration_v1 import (
    CalibrationReport, DecisionObservation, ModelCandidate,
    calibrate_response_models, freeze_candidates, observation_probabilities,
    select_candidate, validate_selection,
)
from poker_engine.strategy.threeway_river_v1 import ResponseModel, RiverAction


F = Fraction


def candidates():
    return tuple(ModelCandidate(name, (ResponseModel(1, (
        ("fold", F(fold)), ("call", F(call)))),))
        for name, fold, call in (("a", 9, 1), ("b", 1, 9)))


def observation(index, *, session="train", hand=None, action="fold", category=None):
    identity = f"{session}-{index}"
    return DecisionObservation(
        "synthetic", session, hand or identity, identity,
        hashlib.sha256(identity.encode()).hexdigest(), 1, category,
        (RiverAction(1, "fold"), RiverAction(1, "call")), F(1, 4),
        RiverAction(1, action),
    )


def test_validation_cannot_select_its_preferred_candidate():
    frozen = freeze_candidates(candidates())
    selected = select_candidate(frozen, (observation(0), observation(1)))
    result = validate_selection(frozen, selected, (
        observation(2, session="validation", action="call"),))
    assert selected.selected_id == result.selected_id == "a"
    assert result.validation_score.mean_log_loss == pytest.approx(-math.log(0.1))
    assert result.validation_score.mean_brier == F(162, 100)
    assert result.status == "NOT_REAL_CALIBRATION"
    assert result.provenance_unverified and not result.empirical_approval
    assert not result.strategy_eligible and not result.advice_emitted


def test_deterministic_90_10_sample_selects_known_finite_candidate():
    train = tuple(observation(i, action="call" if i == 9 else "fold")
                  for i in range(10))
    val = tuple(observation(i, session="validation",
                            action="call" if i == 9 else "fold")
                for i in range(10))
    report = calibrate_response_models(candidates(), train, val)
    assert report.selected_id == "a"
    assert report.validation_score.samples == 10
    assert report.validation_score.mean_brier == F(18, 100)
    assert report.validation_score.mean_log_loss == pytest.approx(
        -0.9 * math.log(0.9) - 0.1 * math.log(0.1))


def test_tie_break_is_candidate_id_not_input_order():
    row = observation(0)
    twins = (ModelCandidate("z", candidates()[0].models),
             ModelCandidate("a", candidates()[0].models))
    assert select_candidate(freeze_candidates(twins), (row,)).selected_id == "a"


@pytest.mark.parametrize("field,value", [
    ("session_id", "train"), ("hand_id", "train-0"),
    ("row_id", "train-0"),
    ("source_hash", hashlib.sha256(b"train-0").hexdigest()),
])
def test_group_or_evidence_leakage_is_rejected(field, value):
    with pytest.raises(ValueError, match="overlap|duplicate"):
        calibrate_response_models(candidates(), (observation(0),), (
            replace(observation(1, session="validation"), **{field: value}),))


def test_showdown_selected_data_never_becomes_all_decisions():
    with pytest.raises(ValueError, match="all_decisions"):
        select_candidate(freeze_candidates(candidates()), (
            replace(observation(0), sampling_frame="showdown_only"),))


def test_mixed_real_and_synthetic_rejected_even_across_splits():
    with pytest.raises(ValueError, match="mixed"):
        calibrate_response_models(candidates(), (observation(0),), (
            replace(observation(1, session="validation"),
                    source_kind="reviewed_all_decisions"),))


def test_reviewed_claim_is_never_empirical_approval():
    report = calibrate_response_models(candidates(), (
        replace(observation(0), source_kind="reviewed_all_decisions"),), (
        replace(observation(1, session="validation"),
                source_kind="reviewed_all_decisions"),))
    assert report.status == "REVIEW_REQUIRED_NOT_EMPIRICALLY_APPROVED"
    assert report.provenance_unverified and not report.empirical_approval


def test_none_category_only_accepted_for_public_models():
    freeze = freeze_candidates(candidates())
    assert select_candidate(freeze, (observation(0),)).selected_id == "a"
    cs = list(candidates())
    cs[0] = replace(cs[0], models=(replace(cs[0].models[0], category_weights=(
        (1, (("fold", F(1)), ("call", F(1)))),)),))
    with pytest.raises(ValueError, match="own_category_missing"):
        select_candidate(freeze_candidates(tuple(cs)), (observation(0),))


@pytest.mark.parametrize("change", [
    {"legal_actions": (RiverAction(1, "call"),)},
    {"legal_actions": (RiverAction(1, "check"), RiverAction(1, "fold"))},
    {"legal_actions": (RiverAction(2, "fold"), RiverAction(2, "call"))},
    {"observed_action": RiverAction(1, "raise", Decimal(10))},
    {"price_ratio": 0.25}, {"own_category": True},
    {"source_hash": "not-a-hash"}, {"actor_seat": True},
])
def test_bad_public_or_provenance_contract_rejected(change):
    with pytest.raises(ValueError):
        select_candidate(freeze_candidates(candidates()), (
            replace(observation(0), **change),))


def test_zero_likelihood_is_explicit_not_epsilon_clipped():
    cs = (ModelCandidate("a", (ResponseModel(1, (("fold", F(1)),)),)),
          ModelCandidate("b", (ResponseModel(1, (("call", F(1)),)),)))
    report = calibrate_response_models(cs, (observation(0),), (
        observation(0, session="validation", action="call"),))
    assert report.selected_id == "a"
    assert report.validation_score.mean_log_loss is None
    assert report.validation_score.zero_likelihood_rows == ("validation-0",)
    assert report.validation_score.mean_brier == 2
    assert report.training_scores[1].zero_likelihood_rows == ("train-0",)


def test_zero_legal_mass_rejected_not_uniform_policy():
    cs = list(candidates())
    cs[0] = replace(cs[0], models=(ResponseModel(1, (("check", F(1)),)),))
    with pytest.raises(ValueError, match="zero_legal_response_mass"):
        select_candidate(freeze_candidates(tuple(cs)), (observation(0),))


def test_parameter_hash_and_selection_receipt_bind_full_content():
    frozen = freeze_candidates(candidates())
    selected = select_candidate(frozen, (observation(0),))
    changed = replace(frozen, candidates=(
        replace(candidates()[0], models=candidates()[1].models), candidates()[1]))
    with pytest.raises(ValueError, match="changed_after_freeze"):
        select_candidate(changed, (observation(0),))
    for forged in (replace(selected, selected_id="b"), replace(
            selected, training=(replace(observation(0), price_ratio=F(1, 3)),))):
        with pytest.raises(ValueError, match="receipt"):
            validate_selection(frozen, forged, (observation(1, session="validation"),))


def test_duplicate_candidate_and_model_seat_mappings_rejected():
    with pytest.raises(ValueError, match="candidate_ids"):
        freeze_candidates((candidates()[0], candidates()[0]))
    bad = replace(candidates()[0], models=candidates()[0].models * 2)
    with pytest.raises(ValueError, match="model_seats"):
        freeze_candidates((bad, candidates()[1]))
    changed_model = replace(candidates()[0].models[0], seat_id=2)
    bad = replace(candidates()[0], models=(changed_model,))
    with pytest.raises(ValueError, match="seats_must_match"):
        freeze_candidates((bad, candidates()[1]))


def test_exact_existing_price_formula_and_per_size_normalization():
    model = ResponseModel(1, (("fold", F(1)), ("call", F(2)), ("raise", F(1))),
                          price_multipliers=((F(1), (("call", F(3)),)),))
    row = replace(observation(0), legal_actions=(
        RiverAction(1, "fold"), RiverAction(1, "call"),
        RiverAction(1, "raise", Decimal(20)), RiverAction(1, "raise", Decimal(40))))
    assert observation_probabilities(model, row) == (F(1, 9), F(6, 9), F(1, 9), F(1, 9))


def test_report_cannot_be_promoted():
    report = calibrate_response_models(candidates(), (observation(0),), (
        observation(1, session="validation"),))
    assert isinstance(report, CalibrationReport)
    for field in ("empirical_approval", "strategy_eligible", "advice_emitted"):
        with pytest.raises(ValueError):
            replace(report, **{field: True})
