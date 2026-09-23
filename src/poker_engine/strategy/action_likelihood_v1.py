"""Synthetic-only action-opportunity likelihood control, never live advice.

The historical audit has zero eligible opportunities.  This module deliberately
refuses real-source fitting; it tests a small public-menu Dirichlet estimator
on precommitted complete synthetic opportunities.  Later labels are excluded.
"""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import re


ROW_KEYS = {"opportunity_id", "session_id", "hand_id", "opponent_id",
            "public", "actual_action", "source_kind", "source_sha256",
            "unknown_reasons"}
PUBLIC_KEYS = {"platform_id", "rule_fingerprint", "table_size", "active_seats",
               "street", "board", "seat_id", "position", "action_order", "pot",
               "stacks", "street_committed", "to_call", "prior_public_actions",
               "legal_actions"}
ACTION_KINDS = {"fold", "check", "call", "bet", "raise"}


def _encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n").encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _positive_amount(value, *, allow_zero=False):
    if not isinstance(value, str):
        raise ValueError("exact_money_string_required")
    amount = Decimal(value)
    if not amount.is_finite() or amount < 0 or (not allow_zero and amount == 0):
        raise ValueError("finite_positive_money_required")
    return amount


def public_context(public):
    """Derive the context using only fields available before the action."""
    if not isinstance(public, dict) or set(public) != PUBLIC_KEYS:
        raise ValueError("exact_public_predecision_fields_required")
    if (public["platform_id"] != "AA_SYNTHETIC_CONTROL"
            or not isinstance(public["rule_fingerprint"], str)
            or re.fullmatch(r"[0-9a-f]{64}", public["rule_fingerprint"]) is None
            or public["table_size"] != 6 or public["active_seats"] != [0, 1, 2]
            or public["street"] != "river"
            or not isinstance(public["board"], list)
            or len(public["board"]) != 5
            or len(set(public["board"])) != 5):
        raise ValueError("synthetic_public_scope_mismatch")
    seat = public["seat_id"]
    if (type(seat) is not int or seat not in (1, 2)
            or public["position"] not in ("BB", "BTN")
            or not isinstance(public["action_order"], list)
            or sorted(public["action_order"]) != [0, 1, 2]
            or not isinstance(public["prior_public_actions"], list)
            or not isinstance(public["stacks"], dict)
            or set(public["stacks"]) != {"0", "1", "2"}
            or not isinstance(public["street_committed"], dict)
            or set(public["street_committed"]) != {"0", "1", "2"}):
        raise ValueError("synthetic_seat_order_or_stack_mismatch")
    for amount in (*public["stacks"].values(),
                   *public["street_committed"].values()):
        _positive_amount(amount, allow_zero=True)
    pot = _positive_amount(public["pot"])
    owed = _positive_amount(public["to_call"], allow_zero=True)
    menu = public["legal_actions"]
    if (not isinstance(menu, list) or not 2 <= len(menu) <= 3
            or len(set(menu)) != len(menu)
            or not all(isinstance(action, str) and action.partition(":")[0]
                       in ACTION_KINDS for action in menu)):
        raise ValueError("complete_legal_menu_required")
    if (owed == 0 and menu != ["check", "bet:20"]
            or owed > 0 and menu != ["fold", "call", "raise:40"]):
        raise ValueError("legal_menu_price_conflict")
    price = Fraction(owed) / Fraction(pot + owed) if owed else Fraction(0)
    context = json.dumps({"rule_fingerprint": public["rule_fingerprint"],
                          "legal_actions": menu,
                          "price": str(price)},
                         sort_keys=True, separators=(",", ":"))
    return context


def validate_opportunity(row):
    """Require a complete, provenance-bound synthetic action opportunity.

    Actual action is a label for fitting/scoring; offline revealed strength and
    future actions are absent from this exact record schema.
    """
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        raise ValueError("exact_opportunity_fields_required_no_future_labels")
    for name in ("opportunity_id", "session_id", "hand_id", "opponent_id"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError("complete_opportunity_identity_required")
    if (row["source_kind"] != "synthetic" or row["unknown_reasons"] != []
            or not isinstance(row["source_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) is None):
        raise ValueError("only_complete_synthetic_opportunities_fit")
    context = public_context(row["public"])
    if row["actual_action"] not in row["public"]["legal_actions"]:
        raise ValueError("actual_action_outside_legal_menu")
    receipt = {name: row[name] for name in (
        "opportunity_id", "session_id", "hand_id", "opponent_id", "public",
        "actual_action")}
    if row["source_sha256"] != _sha(_encoded(receipt)):
        raise ValueError("opportunity_source_receipt_mismatch")
    return context


@dataclass(frozen=True)
class ActionLikelihoodModelV1:
    model_id: str
    scope: str
    training_sessions: tuple[str, ...]
    training_sha256: str
    counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    source_kind: str = "synthetic_control_only"
    strategy_eligible: bool = False
    advice_emitted: bool = False

    def __post_init__(self):
        if (self.scope not in ("pooled", "player")
                or self.model_id != ("player_dirichlet_v1" if self.scope == "player"
                                     else "pooled_dirichlet_v1")
                or not isinstance(self.training_sessions, tuple)
                or not self.training_sessions
                or len(set(self.training_sessions)) != len(self.training_sessions)
                or re.fullmatch(r"[0-9a-f]{64}", self.training_sha256) is None
                or not isinstance(self.counts, tuple)
                or any(not isinstance(key, str) or not isinstance(values, tuple)
                       or any(not isinstance(action, str) or type(count) is not int
                              or count <= 0 for action, count in values)
                       for key, values in self.counts)
                or self.source_kind != "synthetic_control_only"
                or self.strategy_eligible is not False
                or self.advice_emitted is not False):
            raise ValueError("invalid_or_authoritative_synthetic_likelihood")


def fit_action_likelihood(rows, *, train_sessions, scope):
    """Fit fixed alpha=1 counts; real or incomplete inputs are rejected."""
    if (scope not in ("pooled", "player")
            or not isinstance(train_sessions, tuple)
            or not train_sessions or len(set(train_sessions)) != len(train_sessions)
            or not isinstance(rows, list) or not rows):
        raise ValueError("declared_synthetic_train_scope_required")
    seen, hands = set(), set()
    counts, sessions = defaultdict(lambda: defaultdict(int)), set()
    for row in rows:
        context = validate_opportunity(row)
        if row["opportunity_id"] in seen:
            raise ValueError("duplicate_action_opportunity")
        seen.add(row["opportunity_id"])
        if row["hand_id"] in hands:
            raise ValueError("duplicate_or_cross_session_synthetic_hand")
        hands.add(row["hand_id"])
        sessions.add(row["session_id"])
        if row["session_id"] not in train_sessions:
            raise ValueError("heldout_session_entered_training")
        actor = row["opponent_id"] if scope == "player" else "population"
        counts[json.dumps((actor, context))][row["actual_action"]] += 1
    if sessions != set(train_sessions):
        raise ValueError("declared_training_session_missing")
    frozen_counts = tuple((key, tuple(sorted(value.items())))
                          for key, value in sorted(counts.items()))
    return ActionLikelihoodModelV1(
        "player_dirichlet_v1" if scope == "player" else "pooled_dirichlet_v1",
        scope, train_sessions, _sha(_encoded(rows)), frozen_counts)


def predict_action(model, public, *, opponent_id):
    """Inference takes a public snapshot and known identity, never the action."""
    if not isinstance(model, ActionLikelihoodModelV1):
        raise ValueError("frozen_likelihood_model_required")
    if not isinstance(opponent_id, str) or not opponent_id:
        raise ValueError("known_synthetic_opponent_identity_required")
    context = public_context(public)
    actor = opponent_id if model.scope == "player" else "population"
    key = json.dumps((actor, context))
    learned = dict(dict(model.counts).get(key, ()))
    menu = public["legal_actions"]
    masses = {action: learned.get(action, 0) + 1 for action in menu}
    total = sum(masses.values())
    return ({action: Fraction(mass, total) for action, mass in masses.items()},
            key not in dict(model.counts))


def score_action_likelihood(model, rows, *, heldout_sessions):
    if (not isinstance(heldout_sessions, tuple) or not heldout_sessions
            or len(set(heldout_sessions)) != len(heldout_sessions)
            or set(heldout_sessions) & set(model.training_sessions)
            or not isinstance(rows, list) or not rows):
        raise ValueError("whole_session_holdout_required")
    seen, sessions = set(), set()
    logloss, brier = 0.0, Fraction(0)
    bins = [list() for _ in range(5)]
    fallback, zero = 0, 0
    for row in rows:
        validate_opportunity(row)
        probabilities, used_fallback = predict_action(
            model, row["public"], opponent_id=row["opponent_id"])
        if row["opportunity_id"] in seen:
            raise ValueError("duplicate_heldout_opportunity")
        seen.add(row["opportunity_id"])
        sessions.add(row["session_id"])
        if row["session_id"] not in heldout_sessions:
            raise ValueError("undeclared_heldout_session")
        actual = row["actual_action"]
        p = probabilities[actual]
        zero += p == 0
        fallback += used_fallback
        if p:
            logloss -= math.log(float(p))
        else:
            logloss = math.inf
        brier += sum((prediction - (1 if action == actual else 0)) ** 2
                     for action, prediction in probabilities.items())
        top = max(probabilities, key=probabilities.__getitem__)
        confidence = float(probabilities[top])
        bins[min(int(confidence * 5), 4)].append((confidence, int(top == actual)))
    if sessions != set(heldout_sessions):
        raise ValueError("declared_heldout_session_missing")
    ece = sum(len(bucket) / len(rows) * abs(
        sum(item[0] for item in bucket) / len(bucket)
        - sum(item[1] for item in bucket) / len(bucket))
        for bucket in bins if bucket)
    return {"status": "SYNTHETIC_HELDOUT_ONLY", "opportunities": len(rows),
            "sessions": list(heldout_sessions),
            "mean_log_loss": round(logloss / len(rows), 9),
            "mean_brier": str(brier / len(rows)),
            "top_action_ece_5bin": round(ece, 9),
            "zero_probability_count": zero,
            "unseen_context_fallback_count": fallback,
            "refusal_count": 0,
            "strategy_eligible": False, "advice_emitted": False}
