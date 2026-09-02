"""Whole-hand debrief aggregation, pairing, and partial-EV boundaries."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta
from poker_engine.strategy.advice import build_advice
from poker_engine.strategy.provider import LookupState
from poker_engine.strategy.router import RouteResult
from poker_engine.strategy.training import (
    ActualActionRecord,
    build_hand_review,
)

from .helpers import NOW, candidate, context


def _ready(version=1, request_id="r1", *, action_ev=True):
    ctx = context()
    value = candidate(ctx)
    advice = build_advice(
        ctx,
        RouteResult(LookupState.HIT_EXACT, value, ()),
        now=NOW,
    )
    return replace(
        advice,
        state_version=version,
        request_id=request_id,
        action_ev=advice.action_ev if action_ev else {},
        ev_gap=advice.ev_gap if action_ev else None,
    )


def _partial(version=1, request_id="r1"):
    ctx = context()
    advice = build_advice(
        ctx,
        RouteResult(LookupState.NO_STRATEGY, None, (), ("no_strategy",)),
        math_report={"equity": "0.5"},
        now=NOW,
    )
    return replace(advice, state_version=version, request_id=request_id)


def _actual(
    advice,
    action=ActionType.RAISE,
    amount=ChipAmount("2.5"),
    *,
    seconds=0,
    evidence=None,
):
    return ActualActionRecord(
        hand_id=advice.hand_id,
        state_version=advice.state_version,
        request_id=advice.request_id,
        action=action,
        amount=amount,
        observed_at=NOW + timedelta(seconds=seconds),
        evidence_ref=evidence or f"replay://actual/{advice.request_id}",
    )


def test_complete_hand_sums_ev_and_identifies_largest_loss_decision():
    first = _ready(1, "r1")
    second = _ready(2, "r2")
    third = _ready(3, "r3")
    review = build_hand_review(
        (third, first, second),
        (
            _actual(second, ActionType.CHECK, None, seconds=2),
            _actual(third, seconds=3),
            _actual(first, seconds=1),
        ),
    )

    assert review.decision_count == 3
    assert review.ev_evaluated_count == 3
    assert review.ev_unavailable_count == 0
    assert review.known_ev_loss_total == ChipDelta("1.25")
    assert review.ev_loss_complete
    assert review.max_ev_loss == ChipDelta("1.25")
    assert review.max_loss_state_version == 2
    assert review.max_loss_request_id == "r2"
    assert review.action_deviation_count == 1
    assert review.training_tags == ("hand_ev_complete", "action_deviation")


def test_decisions_are_ordered_by_observed_action_time_not_input_order():
    first = _ready(1, "r1")
    second = _ready(2, "r2")
    review = build_hand_review(
        (first, second),
        (_actual(first, seconds=5), _actual(second, seconds=1)),
    )

    assert tuple(item.request_id for item in review.decisions) == ("r2", "r1")


def test_missing_counterfactual_ev_keeps_known_total_but_marks_hand_partial():
    complete = _ready(1, "r1")
    missing = _ready(2, "r2", action_ev=False)
    review = build_hand_review(
        (complete, missing),
        (
            _actual(complete, ActionType.CHECK, None),
            _actual(missing, ActionType.CHECK, None, seconds=1),
        ),
    )

    assert review.known_ev_loss_total == ChipDelta("1.25")
    assert review.ev_evaluated_count == 1
    assert review.ev_unavailable_count == 1
    assert not review.ev_loss_complete
    assert review.training_tags[0] == "hand_ev_partial"


def test_non_ready_decision_is_recorded_without_ev_or_strategy_judgment():
    advice = _partial()
    review = build_hand_review(
        (advice,), (_actual(advice, ActionType.CHECK, None),)
    )

    assert review.ready_decision_count == 0
    assert review.strategy_unavailable_count == 1
    assert review.ev_evaluated_count == 0
    assert review.known_ev_loss_total is None
    assert not review.ev_loss_complete


def test_missing_actual_action_is_disclosed_and_never_inferred():
    first = _ready(1, "r1")
    missing = _ready(2, "r2")
    review = build_hand_review((first, missing), (_actual(first),))

    assert review.decision_count == 1
    assert review.missing_actual_request_ids == ("r2",)
    assert "missing_actual_action" in review.training_tags
    assert not review.ev_loss_complete


def test_orphan_actual_action_is_disclosed_and_never_attached_to_nearest_state():
    advice = _ready(1, "r1")
    orphan = replace(_actual(advice), state_version=2, request_id="orphan")
    review = build_hand_review((advice,), (orphan,))

    assert review.decisions == ()
    assert review.missing_actual_request_ids == ("r1",)
    assert review.orphan_actual_request_ids == ("orphan",)
    assert review.known_ev_loss_total is None
    assert "orphan_actual_action" in review.training_tags


def test_size_deviations_are_aggregated_separately():
    advice = _ready()
    review = build_hand_review(
        (advice,), (_actual(advice, amount=ChipAmount("3")),)
    )

    assert review.action_deviation_count == 0
    assert review.size_deviation_count == 1
    assert "size_deviation" in review.training_tags


def test_evidence_is_deduplicated_across_decisions():
    first = _ready(1, "r1")
    second = _ready(2, "r2")
    review = build_hand_review(
        (first, second),
        (
            _actual(first, evidence="replay://same"),
            _actual(second, evidence="replay://same", seconds=1),
        ),
    )

    assert review.evidence.count("replay://same") == 1
    assert len(review.evidence) == len(set(review.evidence))


def test_equal_max_losses_choose_later_state_deterministically():
    first = _ready(1, "r1")
    second = _ready(2, "r2")
    review = build_hand_review(
        (first, second),
        (
            _actual(first, ActionType.CHECK, None),
            _actual(second, ActionType.CHECK, None, seconds=1),
        ),
    )

    assert review.max_ev_loss == ChipDelta("1.25")
    assert review.max_loss_state_version == 2


def test_duplicate_advice_or_actual_identity_is_rejected():
    advice = _ready()
    action = _actual(advice)
    with pytest.raises(ValueError, match="duplicate Advice identity"):
        build_hand_review((advice, advice), (action,))
    with pytest.raises(ValueError, match="duplicate actual action identity"):
        build_hand_review((advice,), (action, action))


def test_multiple_actual_actions_for_same_state_are_rejected_even_with_retries():
    first = _ready(1, "r1")
    retry = _ready(1, "r2")
    with pytest.raises(ValueError, match="multiple actual actions"):
        build_hand_review(
            (first, retry),
            (_actual(first), _actual(retry, seconds=1)),
        )


def test_records_from_different_hands_are_rejected():
    advice = _ready()
    action = replace(_actual(advice), hand_id="another-hand")
    with pytest.raises(ValueError, match="one hand"):
        build_hand_review((advice,), (action,))


def test_empty_or_invalid_record_collections_are_rejected():
    with pytest.raises(ValueError, match="requires"):
        build_hand_review((), ())
    with pytest.raises(TypeError, match="Advice values"):
        build_hand_review(("bad",), ())
    with pytest.raises(TypeError, match="ActualActionRecord"):
        build_hand_review((_ready(),), ("bad",))


def test_review_collections_are_immutable():
    advice = _ready()
    review = build_hand_review((advice,), (_actual(advice),))

    assert isinstance(review.decisions, tuple)
    with pytest.raises(AttributeError):
        review.decisions.append("bad")
