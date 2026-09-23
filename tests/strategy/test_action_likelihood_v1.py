"""Synthetic opportunity, leakage, smoothing and holdout gates."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from poker_engine.strategy.action_likelihood_v1 import (
    fit_action_likelihood, predict_action, score_action_likelihood,
)
from tools.generate_action_likelihood_synthetic_v1 import OUTPUT


def _rows():
    return json.loads(OUTPUT.read_text(encoding="utf-8"))["opportunities"]


def _receipt(row):
    core = {key: row[key] for key in (
        "opportunity_id", "session_id", "hand_id", "opponent_id", "public",
        "actual_action")}
    raw = (json.dumps(core, sort_keys=True, ensure_ascii=False, indent=2)
           + "\n").encode()
    row["source_sha256"] = hashlib.sha256(raw).hexdigest()


def test_hand_count_dirichlet_has_positive_unseen_action_mass():
    rows = [deepcopy(item) for item in _rows()[:2]]
    rows[0]["actual_action"] = "check"
    rows[1]["actual_action"] = "bet:20"
    # Both first rows belong to one hand-context group but have unique hands.
    # Make their public menus identical without using any future label.
    rows[1]["public"] = deepcopy(rows[0]["public"])
    for row in rows:
        _receipt(row)
    one = fit_action_likelihood(rows[:1],
                                train_sessions=(rows[0]["session_id"],),
                                scope="pooled")
    one_probability, _ = predict_action(
        one, rows[0]["public"], opponent_id=rows[0]["opponent_id"])
    assert one_probability == {"check": Fraction(2, 3),
                               "bet:20": Fraction(1, 3)}
    model = fit_action_likelihood(rows,
                                  train_sessions=(rows[0]["session_id"],),
                                  scope="pooled")
    probabilities, fallback = predict_action(
        model, rows[0]["public"], opponent_id=rows[0]["opponent_id"])
    assert probabilities == {"check": Fraction(1, 2),
                             "bet:20": Fraction(1, 2)}
    assert fallback is False
    assert model.strategy_eligible is False and model.advice_emitted is False
    with pytest.raises(ValueError, match="exact_public_predecision"):
        predict_action(model, rows[0], opponent_id=rows[0]["opponent_id"])
    with pytest.raises(ValueError, match="authoritative_synthetic"):
        replace(model, advice_emitted=True)


def test_future_label_and_incomplete_or_tampered_row_refuse_fit():
    row = deepcopy(_rows()[0])
    for bad, reason in (
        ({**row, "future_showdown_strength": "strong"}, "no_future_labels"),
        ({**row, "unknown_reasons": ["legal_menu_unknown"]},
         "only_complete_synthetic"),
        ({**row, "source_sha256": "0" * 64}, "receipt_mismatch"),
        ({**row, "source_kind": "reviewed_real"},
         "only_complete_synthetic"),
    ):
        with pytest.raises(ValueError, match=reason):
            fit_action_likelihood([bad],
                                  train_sessions=(row["session_id"],),
                                  scope="pooled")


def test_no_heldout_session_can_enter_training_or_duplicate():
    rows = _rows()
    train = [item for item in rows if item["session_id"].startswith("syn-train")]
    sessions = tuple(sorted({item["session_id"] for item in train}))
    with pytest.raises(ValueError, match="heldout_session_entered_training"):
        fit_action_likelihood(train + [rows[-1]], train_sessions=sessions,
                              scope="pooled")
    with pytest.raises(ValueError, match="duplicate_action_opportunity"):
        fit_action_likelihood(train + [train[0]], train_sessions=sessions,
                              scope="pooled")
    duplicate_hand = deepcopy(train[1])
    duplicate_hand["hand_id"] = train[0]["hand_id"]
    _receipt(duplicate_hand)
    with pytest.raises(ValueError, match="duplicate_or_cross_session"):
        fit_action_likelihood([train[0], duplicate_hand],
                              train_sessions=(train[0]["session_id"],),
                              scope="pooled")
    model = fit_action_likelihood(train, train_sessions=sessions, scope="player")
    with pytest.raises(ValueError, match="whole_session_holdout_required"):
        score_action_likelihood(model, train,
                                heldout_sessions=(sessions[0],))


def test_both_predeclared_models_score_every_synthetic_heldout_opportunity():
    rows = _rows()
    train = [item for item in rows if item["session_id"].startswith("syn-train")]
    sessions = tuple(sorted({item["session_id"] for item in train}))
    stable = [item for item in rows if item["session_id"].startswith("syn-stable")]
    drift = [item for item in rows if item["session_id"].startswith("syn-drift")]
    stable_sessions = tuple(sorted({item["session_id"] for item in stable}))
    drift_sessions = tuple(sorted({item["session_id"] for item in drift}))
    pooled = fit_action_likelihood(train, train_sessions=sessions, scope="pooled")
    player = fit_action_likelihood(train, train_sessions=sessions, scope="player")
    p_stable = score_action_likelihood(pooled, stable,
                                       heldout_sessions=stable_sessions)
    p_drift = score_action_likelihood(pooled, drift,
                                      heldout_sessions=drift_sessions)
    a_stable = score_action_likelihood(player, stable,
                                       heldout_sessions=stable_sessions)
    a_drift = score_action_likelihood(player, drift,
                                      heldout_sessions=drift_sessions)
    assert all(score["opportunities"] == 96
               and score["zero_probability_count"] == 0
               and score["refusal_count"] == 0
               for score in (p_stable, p_drift, a_stable, a_drift))
    assert a_stable["mean_log_loss"] < p_stable["mean_log_loss"]
    assert a_drift["mean_log_loss"] > p_drift["mean_log_loss"]
    assert a_drift["top_action_ece_5bin"] > p_drift["top_action_ece_5bin"]
