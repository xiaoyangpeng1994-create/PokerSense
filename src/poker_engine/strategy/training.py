"""Actual-action binding and deterministic post-hand debrief contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from poker_engine.core._freeze import _require_aware_dt
from poker_engine.core.enums import ActionType
from poker_engine.core.value_objects import ChipAmount, ChipDelta

from .advice import Advice, AdviceStatus


@dataclass(frozen=True)
class ActualActionRecord:
    hand_id: str
    state_version: int
    request_id: str
    action: ActionType
    amount: ChipAmount | None
    observed_at: datetime
    evidence_ref: str

    def __post_init__(self) -> None:
        for name in ("hand_id", "request_id", "evidence_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        if not isinstance(self.state_version, int) or isinstance(
            self.state_version, bool
        ):
            raise TypeError("state_version must be an int")
        if self.state_version < 0:
            raise ValueError("state_version must be >= 0")
        if not isinstance(self.action, ActionType):
            raise TypeError("action must be an ActionType")
        if self.amount is not None and not isinstance(self.amount, ChipAmount):
            raise TypeError("amount must be a ChipAmount or None")
        if self.amount is not None and self.action not in (
            ActionType.BET, ActionType.RAISE, ActionType.ALL_IN,
        ):
            raise ValueError("only bet/raise/all-in actions can carry amount")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        _require_aware_dt(self.observed_at)


@dataclass(frozen=True)
class HandDebrief:
    hand_id: str
    state_version: int
    request_id: str
    advice_status: AdviceStatus
    preferred_action: ActionType | None
    actual_action: ActionType
    actual_amount: ChipAmount | None
    action_deviation: bool | None
    size_deviation: bool | None
    ev_loss: ChipDelta | None
    training_tags: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class HandReview:
    hand_id: str
    decisions: tuple[HandDebrief, ...]
    decision_count: int
    ready_decision_count: int
    strategy_unavailable_count: int
    action_deviation_count: int
    size_deviation_count: int
    ev_evaluated_count: int
    ev_unavailable_count: int
    known_ev_loss_total: ChipDelta | None
    ev_loss_complete: bool
    max_ev_loss: ChipDelta | None
    max_loss_state_version: int | None
    max_loss_request_id: str | None
    missing_actual_request_ids: tuple[str, ...]
    orphan_actual_request_ids: tuple[str, ...]
    training_tags: tuple[str, ...]
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hand_id, str) or not self.hand_id:
            raise ValueError("hand_id must be a non-empty str")
        decisions = tuple(self.decisions)
        if not all(isinstance(item, HandDebrief) for item in decisions):
            raise TypeError("decisions must contain HandDebrief values")
        if any(item.hand_id != self.hand_id for item in decisions):
            raise ValueError("all decisions must belong to hand_id")
        object.__setattr__(self, "decisions", decisions)
        count_fields = (
            "decision_count", "ready_decision_count",
            "strategy_unavailable_count", "action_deviation_count",
            "size_deviation_count", "ev_evaluated_count",
            "ev_unavailable_count",
        )
        for name in count_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.decision_count != len(decisions):
            raise ValueError("decision_count must equal decisions length")
        if self.ready_decision_count + self.strategy_unavailable_count != (
            self.decision_count
        ):
            raise ValueError("ready and unavailable counts must cover decisions")
        if self.ev_evaluated_count + self.ev_unavailable_count != (
            self.decision_count
        ):
            raise ValueError("EV counts must cover decisions")
        for name in ("known_ev_loss_total", "max_ev_loss"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, ChipDelta):
                raise TypeError(f"{name} must be ChipDelta or None")
            if value is not None and value.value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.ev_loss_complete, bool):
            raise TypeError("ev_loss_complete must be a bool")
        if self.ev_loss_complete != (
            self.decision_count > 0
            and self.ev_unavailable_count == 0
            and not self.missing_actual_request_ids
            and not self.orphan_actual_request_ids
        ):
            raise ValueError("ev_loss_complete disagrees with review coverage")
        max_identity = (
            self.max_loss_state_version is not None,
            self.max_loss_request_id is not None,
            self.max_ev_loss is not None,
        )
        if len(set(max_identity)) != 1:
            raise ValueError("max-loss value and identity must appear together")
        for name in (
            "missing_actual_request_ids", "orphan_actual_request_ids",
            "training_tags", "evidence",
        ):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)


def build_hand_debrief(
    advice: Advice,
    actual: ActualActionRecord,
) -> HandDebrief:
    """Compare an observed human action with the exact Advice version."""
    if not isinstance(advice, Advice):
        raise TypeError("advice must be an Advice")
    if not isinstance(actual, ActualActionRecord):
        raise TypeError("actual must be an ActualActionRecord")
    if (
        advice.hand_id != actual.hand_id
        or advice.state_version != actual.state_version
        or advice.request_id != actual.request_id
    ):
        raise ValueError("actual action does not match Advice identity")

    preferred = advice.preferred_action
    if advice.status is not AdviceStatus.READY or preferred is None:
        return HandDebrief(
            hand_id=advice.hand_id,
            state_version=advice.state_version,
            request_id=advice.request_id,
            advice_status=advice.status,
            preferred_action=None,
            actual_action=actual.action,
            actual_amount=actual.amount,
            action_deviation=None,
            size_deviation=None,
            ev_loss=None,
            training_tags=("strategy_unavailable",),
            evidence=(actual.evidence_ref,) + advice.evidence,
        )

    action_deviation = actual.action is not preferred
    size_deviation = _size_deviation(advice, actual)
    ev_loss = _ev_loss(advice, preferred, actual.action)
    tags = []
    if action_deviation:
        tags.append("action_deviation")
    if size_deviation:
        tags.append("size_deviation")
    if ev_loss is None:
        tags.append("ev_loss_unavailable")
    elif ev_loss.value > 0:
        tags.append("positive_ev_loss")
    if not tags:
        tags.append("matched_advice")
    return HandDebrief(
        hand_id=advice.hand_id,
        state_version=advice.state_version,
        request_id=advice.request_id,
        advice_status=advice.status,
        preferred_action=preferred,
        actual_action=actual.action,
        actual_amount=actual.amount,
        action_deviation=action_deviation,
        size_deviation=size_deviation,
        ev_loss=ev_loss,
        training_tags=tuple(tags),
        evidence=(actual.evidence_ref,) + advice.evidence,
    )


def build_hand_review(
    advice_records: tuple[Advice, ...],
    actual_records: tuple[ActualActionRecord, ...],
) -> HandReview:
    """Aggregate exact matched decision points without inferring missing actions."""
    advice_records = tuple(advice_records)
    actual_records = tuple(actual_records)
    if not advice_records and not actual_records:
        raise ValueError("hand review requires Advice or actual-action records")
    if not all(isinstance(item, Advice) for item in advice_records):
        raise TypeError("advice_records must contain Advice values")
    if not all(isinstance(item, ActualActionRecord) for item in actual_records):
        raise TypeError("actual_records must contain ActualActionRecord values")
    hand_ids = {
        item.hand_id for item in advice_records
    } | {
        item.hand_id for item in actual_records
    }
    if len(hand_ids) != 1:
        raise ValueError("all review records must belong to one hand")
    hand_id = next(iter(hand_ids))
    advice_by_identity = _unique_by_identity(advice_records, "Advice")
    actual_by_identity = _unique_by_identity(actual_records, "actual action")
    actual_states = [item.state_version for item in actual_records]
    if len(actual_states) != len(set(actual_states)):
        raise ValueError("multiple actual actions for one state_version")

    matched_identities = set(advice_by_identity) & set(actual_by_identity)
    matched = sorted(
        matched_identities,
        key=lambda identity: (
            actual_by_identity[identity].observed_at,
            identity[0],
            identity[1],
        ),
    )
    decisions = tuple(
        build_hand_debrief(
            advice_by_identity[identity], actual_by_identity[identity]
        )
        for identity in matched
    )
    missing_actual = tuple(sorted(
        identity[1] for identity in set(advice_by_identity) - matched_identities
    ))
    orphan_actual = tuple(sorted(
        identity[1] for identity in set(actual_by_identity) - matched_identities
    ))
    known_losses = tuple(
        item for item in decisions if item.ev_loss is not None
    )
    total = (
        ChipDelta(sum((item.ev_loss.value for item in known_losses), Decimal("0")))
        if known_losses else None
    )
    maximum = max(
        known_losses,
        key=lambda item: (item.ev_loss.value, item.state_version, item.request_id),
        default=None,
    )
    unavailable = sum(item.ev_loss is None for item in decisions)
    complete = (
        bool(decisions)
        and unavailable == 0
        and not missing_actual
        and not orphan_actual
    )
    tags = ["hand_ev_complete" if complete else "hand_ev_partial"]
    if any(item.action_deviation for item in decisions):
        tags.append("action_deviation")
    if any(item.size_deviation for item in decisions):
        tags.append("size_deviation")
    if missing_actual:
        tags.append("missing_actual_action")
    if orphan_actual:
        tags.append("orphan_actual_action")
    evidence = tuple(dict.fromkeys(
        tuple(ref for item in decisions for ref in item.evidence)
        + tuple(
            ref
            for identity, advice in advice_by_identity.items()
            if identity not in matched_identities
            for ref in advice.evidence
        )
        + tuple(
            actual.evidence_ref
            for identity, actual in actual_by_identity.items()
            if identity not in matched_identities
        )
    ))
    return HandReview(
        hand_id=hand_id,
        decisions=decisions,
        decision_count=len(decisions),
        ready_decision_count=sum(
            item.advice_status is AdviceStatus.READY for item in decisions
        ),
        strategy_unavailable_count=sum(
            item.advice_status is not AdviceStatus.READY for item in decisions
        ),
        action_deviation_count=sum(
            item.action_deviation is True for item in decisions
        ),
        size_deviation_count=sum(
            item.size_deviation is True for item in decisions
        ),
        ev_evaluated_count=len(known_losses),
        ev_unavailable_count=unavailable,
        known_ev_loss_total=total,
        ev_loss_complete=complete,
        max_ev_loss=maximum.ev_loss if maximum is not None else None,
        max_loss_state_version=(
            maximum.state_version if maximum is not None else None
        ),
        max_loss_request_id=(maximum.request_id if maximum is not None else None),
        missing_actual_request_ids=missing_actual,
        orphan_actual_request_ids=orphan_actual,
        training_tags=tuple(tags),
        evidence=evidence,
    )


def _unique_by_identity(records, label: str):
    result = {}
    for item in records:
        identity = item.state_version, item.request_id
        if identity in result:
            raise ValueError(f"duplicate {label} identity")
        result[identity] = item
    return result


def _size_deviation(advice: Advice, actual: ActualActionRecord) -> bool | None:
    expected = advice.recommended_sizes.get(actual.action, ())
    if not expected:
        return None
    if actual.amount is None:
        return True
    return actual.amount not in expected


def _ev_loss(
    advice: Advice,
    preferred: ActionType,
    actual: ActionType,
) -> ChipDelta | None:
    if preferred not in advice.action_ev or actual not in advice.action_ev:
        return None
    loss = advice.action_ev[preferred].value - advice.action_ev[actual].value
    return ChipDelta(max(loss, 0))


__all__ = [
    "ActualActionRecord",
    "HandDebrief",
    "HandReview",
    "build_hand_debrief",
    "build_hand_review",
]
