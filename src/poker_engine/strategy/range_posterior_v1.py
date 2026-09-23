"""Offline, non-authoritative turn-action filter for manual river-start ranges.

The input range is explicitly a prior *before* the supplied turn action.  The
river planner subsequently conditions river history itself, exactly once.
This module does not infer an action model or accept a revealed holding.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import re

from poker_engine.equity.evaluator import evaluate
from .range_tracker import parse_concrete_combo
from .threeway_river_v1 import ThreewayRiverScenario, _validate


@dataclass(frozen=True)
class PublicTurnObservation:
    seat_id: int
    action: str  # pressure or passive, from a reviewed public action vocabulary
    board_cards: tuple
    source_kind: str
    source_sha256: str
    event_ordinal: int
    river_decision_ordinal: int


@dataclass(frozen=True)
class ActionLikelihoodProfile:
    p_pressure_strong: Fraction
    p_pressure_weak: Fraction
    source_kind: str
    source_sha256: str


@dataclass(frozen=True)
class PreActionRangeBinding:
    stage: str
    source_kind: str
    source_sha256: str


@dataclass(frozen=True)
class RangePosterior:
    ranges: tuple
    prior_sha256: str
    observations_sha256: str
    profile_sha256: str
    status: str = "SHADOW_ONLY_MANUAL_LIKELIHOOD"
    strategy_eligible: bool = False
    advice_emitted: bool = False

    def __post_init__(self):
        if (self.status != "SHADOW_ONLY_MANUAL_LIKELIHOOD"
                or self.strategy_eligible is not False
                or self.advice_emitted is not False):
            raise ValueError("range_posterior_cannot_authorize_strategy_or_advice")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def strength_buckets(ranges, turn_board):
    """Within each declared support, rank holdings using only turn-known cards.

    Ties stay in the same bucket.  This intentionally small ordinal feature is
    not a calibrated probability or an estimate of the full 1326-combo range.
    """
    result = {}
    for distribution in ranges:
        strengths = {combo: evaluate(parse_concrete_combo(combo) + turn_board)
                     for combo in distribution.combo_weights}
        if not strengths:
            raise ValueError("empty_pre_action_range")
        best = max(strengths.values())
        result[distribution.seat_id] = {
            combo: strength == best for combo, strength in strengths.items()}
    return result


def posterior_ranges(scenario: ThreewayRiverScenario,
                     observations: tuple[PublicTurnObservation, ...],
                     profile: ActionLikelihoodProfile, *,
                     prior_binding: PreActionRangeBinding) -> RangePosterior:
    """Apply declared likelihoods to public actions; fail closed on provenance.

    The caller must establish that the supplied ranges predate these turn
    actions.  Their manual source is preserved; a posterior is not poker fact.
    """
    if not isinstance(scenario, ThreewayRiverScenario):
        raise ValueError("threeway_scenario_required")
    _validate(scenario, 128, 20000)
    if (not isinstance(prior_binding, PreActionRangeBinding)
            or prior_binding.stage != "before_turn_observation"
            or prior_binding.source_kind not in (
                "manual_synthetic", "human_reviewed")
            or not _hash(prior_binding.source_sha256)):
        raise ValueError("pre_action_prior_binding_required")
    if (not isinstance(observations, tuple) or len(observations) != 2
            or not all(isinstance(o, PublicTurnObservation) for o in observations)
            or {o.seat_id for o in observations}
            != {r.seat_id for r in scenario.ranges}):
        raise ValueError("one_turn_observation_per_opponent_required")
    if (not isinstance(profile, ActionLikelihoodProfile)
            or profile.source_kind != "manual_unvalidated"
            or not _hash(profile.source_sha256)
            or not all(isinstance(p, Fraction) and 0 < p < 1 for p in (
                profile.p_pressure_strong, profile.p_pressure_weak))):
        raise ValueError("explicit_manual_nonzero_likelihood_profile_required")
    turn_board = tuple(scenario.board_cards[:4])
    for o in observations:
        if (o.action not in ("pressure", "passive")
                or o.board_cards != turn_board
                or o.source_kind not in ("manual_synthetic", "reviewed_public")
                or not _hash(o.source_sha256)
                or type(o.event_ordinal) is not int
                or type(o.river_decision_ordinal) is not int
                or not 0 <= o.event_ordinal < o.river_decision_ordinal):
            raise ValueError("future_or_unbound_public_observation")
    buckets = strength_buckets(scenario.ranges, turn_board)
    output = []
    for distribution in scenario.ranges:
        if (distribution.source not in ("manual_river_assumption",
                                        "manual_river_start_assumption")
                or not distribution.source_version.startswith("manual:")
                or distribution.effective_sample_size != 0
                or distribution.confidence != 0):
            raise ValueError("pre_action_manual_prior_required")
        observation = next(o for o in observations
                           if o.seat_id == distribution.seat_id)
        weights = {}
        for combo, weight in distribution.combo_weights.items():
            if weight <= 0:
                raise ValueError("positive_prior_combo_weights_required")
            p = (profile.p_pressure_strong if buckets[distribution.seat_id][combo]
                 else profile.p_pressure_weak)
            likelihood = p if observation.action == "pressure" else 1 - p
            weights[combo] = weight * (Decimal(likelihood.numerator)
                                       / Decimal(likelihood.denominator))
        output.append(replace(distribution, combo_weights=weights,
                              source="manual_river_start_assumption",
                              source_version="manual:range_posterior_v1",
                              effective_sample_size=0, confidence=0.0))
    prior = [(r.seat_id, r.source, r.source_version,
              sorted((c, str(w)) for c, w in r.combo_weights.items()))
             for r in scenario.ranges]
    observed = [(o.seat_id, o.action, tuple(map(str, o.board_cards)),
                 o.source_kind, o.source_sha256, o.event_ordinal,
                 o.river_decision_ordinal) for o in observations]
    prior_hash = _sha((prior, prior_binding.stage, prior_binding.source_kind,
                       prior_binding.source_sha256))
    profile_hash = _sha((
        str(profile.p_pressure_strong), str(profile.p_pressure_weak),
        profile.source_kind, profile.source_sha256))
    return RangePosterior(tuple(output), prior_hash, _sha(observed), profile_hash)
