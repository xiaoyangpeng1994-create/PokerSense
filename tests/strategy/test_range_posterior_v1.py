"""Causal and provenance checks for the offline range filter."""

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from poker_engine.strategy.range_posterior_v1 import (
    ActionLikelihoodProfile, PreActionRangeBinding, PublicTurnObservation,
    posterior_ranges,
)
from tools.strategy_evaluation_v1 import (
    DEFAULT_INPUT, DEFAULT_PROTOCOL, planning_scenarios,
)


def _inputs():
    plans, _, _, _ = planning_scenarios(DEFAULT_INPUT, DEFAULT_PROTOCOL)
    plan = plans["n6-facing_bet"]
    observations = tuple(PublicTurnObservation(
        seat, action, plan.board_cards[:4], "manual_synthetic",
        hashlib.sha256(f"test-event-{seat}".encode()).hexdigest(), 1, 2)
        for seat, action in ((1, "pressure"), (2, "passive")))
    profile = ActionLikelihoodProfile(Fraction(3, 4), Fraction(1, 4),
                                      "manual_unvalidated", "a" * 64)
    binding = PreActionRangeBinding(
        "before_turn_observation", "manual_synthetic", "b" * 64)
    return plan, observations, profile, binding


def test_turn_action_updates_only_pre_action_manual_prior():
    plan, observations, profile, binding = _inputs()
    result = posterior_ranges(plan, observations, profile, prior_binding=binding)
    by_seat = {r.seat_id: r for r in result.ranges}
    assert by_seat[1].combo_weights["JhJd"] == by_seat[1].combo_weights["TcTd"]
    assert by_seat[2].combo_weights["7c7s"] == by_seat[2].combo_weights["KhTh"] / 9
    assert all(r.confidence == 0 and r.effective_sample_size == 0
               and r.source == "manual_river_start_assumption"
               for r in result.ranges)
    assert result.strategy_eligible is False and result.advice_emitted is False
    assert plan.history  # River history stays for the frozen planner to condition.
    assert result.observations_sha256 != result.prior_sha256


@pytest.mark.parametrize("change", [
    lambda o: replace(o, action="showdown"),
    lambda o: replace(o, board_cards=o.board_cards[:3]),
    lambda o: replace(o, source_kind="revealed_holding"),
    lambda o: replace(o, source_sha256="not-a-hash"),
    lambda o: replace(o, event_ordinal=2),
    lambda o: replace(o, event_ordinal=-1),
])
def test_unavailable_or_unbound_event_rejected(change):
    plan, observations, profile, binding = _inputs()
    altered = (change(observations[0]), observations[1])
    with pytest.raises(ValueError, match="future_or_unbound"):
        posterior_ranges(plan, altered, profile, prior_binding=binding)


def test_missing_or_duplicate_opponent_event_rejected():
    plan, observations, profile, binding = _inputs()
    for invalid in ((observations[0],),
                    (observations[0], observations[0])):
        with pytest.raises(ValueError, match="one_turn_observation"):
            posterior_ranges(plan, invalid, profile, prior_binding=binding)


def test_unvalidated_likelihood_and_prior_boundaries():
    plan, observations, profile, binding = _inputs()
    for invalid in (replace(profile, p_pressure_strong=Fraction(0)),
                    replace(profile, source_kind="learned_confirmed"),
                    replace(profile, source_sha256="fake")):
        with pytest.raises(ValueError, match="explicit_manual"):
            posterior_ranges(plan, observations, invalid, prior_binding=binding)
    bad_prior = replace(plan.ranges[0], source="confirmed_fact")
    with pytest.raises(ValueError, match="manual_unvalidated_range_source"):
        posterior_ranges(replace(plan, ranges=(bad_prior, plan.ranges[1])),
                         observations, profile, prior_binding=binding)


def test_prior_must_be_bound_before_the_turn_event():
    plan, observations, profile, binding = _inputs()
    for invalid in (replace(binding, stage="river_start"),
                    replace(binding, source_kind="showdown_selected"),
                    replace(binding, source_sha256="unknown")):
        with pytest.raises(ValueError, match="pre_action_prior_binding"):
            posterior_ranges(plan, observations, profile,
                             prior_binding=invalid)


def test_public_observation_variation_changes_posterior_not_hidden_input():
    plan, observations, profile, binding = _inputs()
    pressure = posterior_ranges(plan, observations, profile,
                                prior_binding=binding)
    passive_events = (replace(observations[0], action="passive"), observations[1])
    passive = posterior_ranges(plan, passive_events, profile,
                               prior_binding=binding)
    assert pressure.ranges[0].combo_weights != passive.ranges[0].combo_weights
    assert pressure.ranges[1].combo_weights == passive.ranges[1].combo_weights
    assert pressure.prior_sha256 == passive.prior_sha256
    assert pressure.observations_sha256 != passive.observations_sha256
    with pytest.raises(TypeError):
        PublicTurnObservation(**{**observations[0].__dict__, "showdown_cards": "JhJd"})
