"""Finite preregistered response-model selection, never empirical approval.

Source hashes identify individual decision evidence, not an entire shared video.
Coverage, own-category truth and legal-menu completeness remain declarations.
"""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import re
from types import SimpleNamespace

from .threeway_river_v1 import ResponseModel, RiverAction, _Tree, _weight_table


@dataclass(frozen=True)
class DecisionObservation:
    source_kind: str
    session_id: str
    hand_id: str
    row_id: str
    source_hash: str
    actor_seat: int
    own_category: int | None
    legal_actions: tuple[RiverAction, ...]
    price_ratio: Fraction
    observed_action: RiverAction
    sampling_frame: str = "all_decisions"


@dataclass(frozen=True)
class ModelCandidate:
    candidate_id: str
    models: tuple[ResponseModel, ...]


@dataclass(frozen=True)
class FrozenCandidates:
    candidates: tuple[ModelCandidate, ...]
    sha256: str


@dataclass(frozen=True)
class ModelScore:
    candidate_id: str
    samples: int
    mean_log_loss: float | None
    mean_brier: Fraction
    zero_likelihood_rows: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    selected_id: str
    candidates_sha256: str
    training_sha256: str
    training: tuple[DecisionObservation, ...]
    training_scores: tuple[ModelScore, ...]
    selection_rule: str = "minimum_training_log_loss_then_lexical_candidate_id"


@dataclass(frozen=True)
class CalibrationReport:
    status: str
    selected_id: str
    candidates_sha256: str
    training_sha256: str
    validation_sha256: str
    training_scores: tuple[ModelScore, ...]
    validation_score: ModelScore
    training_sessions: int
    validation_sessions: int
    training_hands: int
    validation_hands: int
    provenance_unverified: bool = True
    empirical_approval: bool = False
    strategy_eligible: bool = False
    advice_emitted: bool = False
    limitations: tuple[str, ...] = (
        "finite_candidate_selection_not_parameter_learning_or_range_learning",
        "candidate_freeze_chronology_and_all_decisions_coverage_not_authenticated",
        "legal_menu_completeness_and_own_category_truth_not_authenticated",
        "strict_session_and_hand_disjoint_validation_scored_only_after_selection",
        "validation_metrics_not_profitability_or_live_strategy_evidence",
        "Brier_is_sum_over_declared_legal_actions_not_mean_over_actions",
    )

    def __post_init__(self):
        if (self.provenance_unverified is not True
                or self.empirical_approval is not False
                or self.strategy_eligible is not False
                or self.advice_emitted is not False):
            raise ValueError("calibration_cannot_authorize_real_or_live_strategy")


def _encoded(value):
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: _encoded(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_encoded(v) for v in value]
    return value


def _digest(value):
    payload = json.dumps(_encoded(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_model(model):
    if not isinstance(model, ResponseModel) or type(model.seat_id) is not int:
        raise ValueError("explicit_response_model_and_integer_seat_required")
    _weight_table(model.weights)
    if not isinstance(model.category_weights, tuple):
        raise ValueError("immutable_category_weights_required")
    seen = set()
    for category, weights in model.category_weights:
        if type(category) is not int or not 0 <= category <= 8 or category in seen:
            raise ValueError("invalid_or_duplicate_category")
        seen.add(category)
        _weight_table(weights)
    if not isinstance(model.price_multipliers, tuple):
        raise ValueError("immutable_price_bands_required")
    previous = Fraction(0)
    for bound, weights in model.price_multipliers:
        if not isinstance(bound, Fraction) or not previous < bound <= 1:
            raise ValueError("invalid_price_bounds")
        previous = bound
        _weight_table(weights, allow_empty=True)
    if model.price_multipliers and previous != 1:
        raise ValueError("price_bands_must_end_at_one")


def freeze_candidates(candidates):
    """Freeze parameters before selection; this does not certify chronology."""
    if not isinstance(candidates, tuple) or not 2 <= len(candidates) <= 64:
        raise ValueError("two_to_64_predeclared_candidates_required")
    ids, expected_seats = set(), None
    for candidate in candidates:
        if (not isinstance(candidate, ModelCandidate)
                or not isinstance(candidate.candidate_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", candidate.candidate_id) is None
                or candidate.candidate_id in ids):
            raise ValueError("unique_explicit_candidate_ids_required")
        ids.add(candidate.candidate_id)
        if (not isinstance(candidate.models, tuple)
                or not 1 <= len(candidate.models) <= 7):
            raise ValueError("immutable_candidate_models_required")
        for model in candidate.models:
            _validate_model(model)
        seats = {m.seat_id for m in candidate.models}
        if len(seats) != len(candidate.models) or any(s < 0 for s in seats):
            raise ValueError("unique_nonnegative_model_seats_required")
        if expected_seats is not None and seats != expected_seats:
            raise ValueError("candidate_model_seats_must_match")
        expected_seats = seats
    ordered = tuple(sorted(candidates, key=lambda c: c.candidate_id))
    return FrozenCandidates(ordered, _digest(ordered))


def _verify_frozen(frozen):
    if not isinstance(frozen, FrozenCandidates):
        raise ValueError("frozen_candidates_required")
    if freeze_candidates(frozen.candidates) != frozen:
        raise ValueError("candidate_parameters_or_ids_changed_after_freeze")


def _observations(rows, frozen):
    if not isinstance(rows, tuple) or not 1 <= len(rows) <= 10000:
        raise ValueError("bounded_nonempty_immutable_observations_required")
    identities, hashes, kinds = set(), set(), set()
    actors = {m.seat_id for m in frozen.candidates[0].models}
    category_needed = {m.seat_id for c in frozen.candidates for m in c.models
                       if m.category_weights}
    for row in rows:
        if not isinstance(row, DecisionObservation):
            raise ValueError("decision_observation_required")
        if (row.source_kind not in ("synthetic", "reviewed_all_decisions")
                or row.sampling_frame != "all_decisions"):
            raise ValueError("all_decisions_required_not_showdown_subset")
        kinds.add(row.source_kind)
        if any(not isinstance(v, str) or not v.strip()
               for v in (row.session_id, row.hand_id, row.row_id)):
            raise ValueError("explicit_session_hand_row_ids_required")
        if (not isinstance(row.source_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", row.source_hash) is None):
            raise ValueError("per_decision_source_hash_required")
        if row.row_id in identities or row.source_hash in hashes:
            raise ValueError("duplicate_row_or_decision_evidence")
        identities.add(row.row_id)
        hashes.add(row.source_hash)
        if type(row.actor_seat) is not int or row.actor_seat not in actors:
            raise ValueError("observation_actor_not_in_model_map")
        if row.own_category is None:
            if row.actor_seat in category_needed:
                raise ValueError("own_category_missing_for_category_model")
        elif type(row.own_category) is not int or not 0 <= row.own_category <= 8:
            raise ValueError("invalid_own_category")
        if (not isinstance(row.price_ratio, Fraction) or not 0 <= row.price_ratio <= 1
                or not isinstance(row.legal_actions, tuple) or not row.legal_actions):
            raise ValueError("explicit_public_price_and_legal_menu_required")
        for action in (*row.legal_actions, row.observed_action):
            if (not isinstance(action, RiverAction) or type(action.actor) is not int
                    or action.actor != row.actor_seat
                    or action.kind not in ("check", "fold", "call", "bet", "raise")
                    or not isinstance(action.target, Decimal)
                    or not action.target.is_finite()
                    or (action.target <= 0 if action.kind in ("bet", "raise")
                        else action.target != 0)):
                raise ValueError("invalid_declared_legal_action")
        if len(set(row.legal_actions)) != len(row.legal_actions):
            raise ValueError("duplicate_legal_actions")
        kinds_at_node = {a.kind for a in row.legal_actions}
        passive = kinds_at_node & {"check", "call", "fold"}
        if (passive != ({"check"} if row.price_ratio == 0 else {"fold", "call"})
                or ("raise" in kinds_at_node if row.price_ratio == 0
                    else "bet" in kinds_at_node)):
            raise ValueError("incomplete_or_inconsistent_declared_legal_menu")
        if row.observed_action not in row.legal_actions:
            raise ValueError("observed_action_not_legal")
    if len(kinds) != 1:
        raise ValueError("mixed_synthetic_and_reviewed_sources_forbidden")


def observation_probabilities(model, row):
    """Invoke the existing river formula using only own/public information."""
    return _Tree.probabilities(SimpleNamespace(models={model.seat_id: model}),
                               row.actor_seat, row.own_category, row.legal_actions,
                               row.price_ratio)


def _score(candidate, rows):
    models = {m.seat_id: m for m in candidate.models}
    log_losses, briers, zeros = [], [], []
    for row in rows:
        probabilities = observation_probabilities(models[row.actor_seat], row)
        index = row.legal_actions.index(row.observed_action)
        p = probabilities[index]
        if not p:
            zeros.append(row.row_id)
        else:
            log_losses.append(math.log(p.denominator) - math.log(p.numerator))
        briers.append(sum(((q - int(i == index)) ** 2
                           for i, q in enumerate(probabilities)), Fraction(0)))
    return ModelScore(candidate.candidate_id, len(rows),
                      None if zeros else math.fsum(log_losses) / len(rows),
                      sum(briers, Fraction(0)) / len(rows), tuple(zeros))


def select_candidate(frozen, training):
    """No validation rows or labels are accepted by this selection function."""
    _verify_frozen(frozen)
    _observations(training, frozen)
    scores = tuple(_score(c, training) for c in frozen.candidates)
    finite = [s for s in scores if s.mean_log_loss is not None]
    if not finite:
        raise ValueError("all_candidates_have_zero_training_likelihood")
    selected = min(finite, key=lambda s: (s.mean_log_loss, s.candidate_id))
    return Selection(selected.candidate_id, frozen.sha256, _digest(training), training,
                     scores)


def validate_selection(frozen, selection, validation):
    """Score only the training-selected model on strictly separate groups."""
    _verify_frozen(frozen)
    if (not isinstance(selection, Selection)
            or select_candidate(frozen, selection.training) != selection):
        raise ValueError("selection_receipt_does_not_match_frozen_training")
    _observations(validation, frozen)
    training = selection.training
    _observations(training + validation, frozen)
    for field in ("session_id", "hand_id"):
        if ({getattr(r, field) for r in training}
                & {getattr(r, field) for r in validation}):
            raise ValueError("training_validation_session_or_hand_overlap")
    chosen = next(c for c in frozen.candidates
                  if c.candidate_id == selection.selected_id)
    return CalibrationReport(
        "NOT_REAL_CALIBRATION" if training[0].source_kind == "synthetic"
        else "REVIEW_REQUIRED_NOT_EMPIRICALLY_APPROVED",
        selection.selected_id, frozen.sha256, selection.training_sha256,
        _digest(validation), selection.training_scores, _score(chosen, validation),
        len({r.session_id for r in training}), len({r.session_id for r in validation}),
        len({r.hand_id for r in training}), len({r.hand_id for r in validation}),
    )


def calibrate_response_models(candidates, training, validation):
    frozen = freeze_candidates(candidates)
    return validate_selection(frozen, select_candidate(frozen, training), validation)
