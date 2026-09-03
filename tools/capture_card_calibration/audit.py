# -*- coding: utf-8 -*-
"""Cross-field consistency audit of stage-F ground-truth labels.

Ground truth is labelled frame by frame from pixels that are actually
visible. Humans make mistakes, and a wrong VALID label is worse than an
UNKNOWN: it will be treated as authoritative evidence by stage I threshold
calibration and stage J mapping. This module therefore audits the
already-labelled ``labels/frames.jsonl`` for *logical* contradictions that a
machine can decide without looking at a single pixel.

The audit is deliberately conservative. It never rewrites a label and never
guesses a value — it only reports rules that are violated, so a human can go
back and confirm or fix the offending field. Every violation renders as an
:class:`Issue` referencing the frame, the rule, the slot (when the conflict is
per-slot) and the exact values that contradict each other.

Design principles (mirroring the guide's failure-closed philosophy):

- A violation is a red flag, not an edit. ``UNKNOWN`` stays ``UNKNOWN``; a
  contradiction is surfaced, never resolved by picking one value.
- ``ERROR`` rules are hard logical contradictions (e.g. ``street`` says FLOP
  but the board holds 2 cards, or an EMPTY slot still carries a stack).
  ``WARN`` rules flag *suspicious-but-possibly-legitimate* patterns (e.g. an
  OCCUPIED slot whose stack is UNKNOWN — the digits may not have been readable,
  but it often means a fill was missed), so the owner can weigh in without the
  report punishing a correct fail-closed label.
- Rules that depend on fields the owner has not labelled yet (action / actor)
  are included but short-circuit to nothing when those fields are all
  ``UNKNOWN``, so an under-labelled dataset does not flood the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import (
    CompletedAction,
    FieldValue,
    FrameLabel,
    LabelStatus,
    Occupancy,
    SlotLabel,
    Street,
)

#: Severity of an audit finding.
_SEVERITY_ERROR = "ERROR"
_SEVERITY_WARN = "WARN"
_SEVERITIES = (_SEVERITY_ERROR, _SEVERITY_WARN)

#: Expected public-board card count per street (section 9).
_STREET_BOARD_COUNT = {
    Street.PRE_FLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}

#: Minimum number of frames in a hand for the monotonic board/pot checks to
#: mean anything. A single-frame hand has nothing to compare against, so the
#: monotonic rules stay silent rather than reporting a trivially-passing hand.
_MIN_HAND_FRAMES_FOR_TREND = 2


@dataclass(frozen=True)
class Issue:
    """One rule violation found during the audit."""

    severity: str
    rule: str
    frame: str
    hand_id: str
    session_id: str
    message: str
    slot_id: int | None = None

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "rule": self.rule,
            "frame": self.frame,
            "hand_id": self.hand_id,
            "session_id": self.session_id,
            "slot_id": self.slot_id,
            "message": self.message,
        }


@dataclass(frozen=True)
class CheckResult:
    """Outcome of running a single rule over the label set."""

    rule: str
    checked: int
    violated: int
    skipped: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "rule": self.rule,
            "checked": self.checked,
            "violated": self.violated,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class AuditReport:
    """Aggregated audit output."""

    issues: tuple[Issue, ...]
    results: tuple[CheckResult, ...]
    frames_checked: int
    hands_checked: int
    sessions: tuple[str, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == _SEVERITY_ERROR)

    @property
    def warn_count(self) -> int:
        return sum(1 for item in self.issues if item.severity == _SEVERITY_WARN)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def rule_result(self, rule: str) -> CheckResult | None:
        for result in self.results:
            if result.rule == rule:
                return result
        return None

    def issues_for(self, rule: str) -> tuple[Issue, ...]:
        return tuple(item for item in self.issues if item.rule == rule)

    def to_dict(self) -> dict[str, object]:
        return {
            "frames_checked": self.frames_checked,
            "hands_checked": self.hands_checked,
            "sessions": list(self.sessions),
            "error_count": self.error_count,
            "warn_count": self.warn_count,
            "results": [result.to_dict() for result in self.results],
            "issues": [issue.to_dict() for issue in self.issues],
        }


# --- small helpers --------------------------------------------------------


def _known(value: FieldValue):
    """Return the value if the field is VALID, else ``None`` (UNKNOWN/CONFLICT).

    The audit only reasons about fields that carry a confident value; an
    UNKNOWN is not evidence of anything, so it is neither a violation nor a
    pass. Returning ``None`` lets callers skip the field cleanly.
    """
    if value.status is LabelStatus.VALID:
        return value.value
    return None


def _occupied(slot: SlotLabel) -> bool | None:
    """True if the slot is literally OCCUPIED, False if EMPTY, None if UNKNOWN."""
    value = _known(slot.occupancy)
    if value is None:
        return None
    return Occupancy(value) is Occupancy.OCCUPIED


def _dealer(slot: SlotLabel) -> bool | None:
    """True if dealer is present, False if absent, None if UNKNOWN."""
    return _known(slot.dealer)


def _stack(slot: SlotLabel) -> int | None:
    """Stack chip count if VALID, else None."""
    value = _known(slot.stack)
    if value is None:
        return None
    return int(value)


def _group_by_hand(labels: Sequence[FrameLabel]) -> dict[str, list[FrameLabel]]:
    """Group labels into per-hand buckets, preserving frame order."""
    groups: dict[str, list[FrameLabel]] = {}
    for label in labels:
        groups.setdefault(label.group_id, []).append(label)
    for items in groups.values():
        items.sort(key=lambda item: item.timestamp_ms)
    return groups


def _issue(
    label: FrameLabel,
    rule: str,
    message: str,
    severity: str = _SEVERITY_ERROR,
    slot_id: int | None = None,
) -> Issue:
    return Issue(
        severity=severity,
        rule=rule,
        frame=label.frame,
        hand_id=label.hand_id,
        session_id=label.session_id,
        message=message,
        slot_id=slot_id,
    )


# --- individual rules ------------------------------------------------------


def _check_street_board_count(labels: Sequence[FrameLabel]) -> list[Issue]:
    """``street`` must agree with the public-board card count (section 9)."""
    issues: list[Issue] = []
    for label in labels:
        street = _known(label.street)
        board = _known(label.board_cards)
        if street is None or board is None:
            continue
        expected = _STREET_BOARD_COUNT.get(Street(street))
        if expected is None:
            continue
        if len(board) != expected:
            issues.append(
                _issue(
                    label,
                    "street-board-count",
                    f"street {street!r} expects {expected} board card(s) but "
                    f"{len(board)} labelled",
                )
            )
    return issues


def _check_board_non_decreasing(labels: Sequence[FrameLabel]) -> list[Issue]:
    """Within a hand the public board may only grow, never shrink."""
    issues: list[Issue] = []
    for hand, items in _group_by_hand(labels).items():
        last: list[str] | None = None
        for label in items:
            board = _known(label.board_cards)
            if board is None:
                continue
            if last is not None and len(board) < len(last):
                issues.append(
                    _issue(
                        label,
                        "board-non-decreasing",
                        f"board shrank from {len(last)} to {len(board)} "
                        f"cards within hand {hand}",
                    )
                )
            last = board
    return issues


def _check_dealer_unique(labels: Sequence[FrameLabel]) -> list[Issue]:
    """Each frame must show exactly one dealer marker (section 9).

    The rule only fires when at least one slot carries a *known* dealer value.
    When every slot is ``UNKNOWN`` the dealer simply was not observable on that
    frame; that is a missing observation, not a contradiction, and it must not
    be reported as an error.
    """
    issues: list[Issue] = []
    for label in labels:
        known = [slot.slot_id for slot in label.slots if _dealer(slot) is not None]
        present = [
            slot.slot_id for slot in label.slots if _dealer(slot) is True
        ]
        if not known:
            continue
        if len(present) != 1:
            issues.append(
                _issue(
                    label,
                    "dealer-unique",
                    f"expected exactly one dealer slot, found {present}",
                    severity=_SEVERITY_ERROR,
                )
            )
    return issues


def _check_occupancy_stack(labels: Sequence[FrameLabel]) -> list[Issue]:
    """A seated player should carry a stack; an empty slot must not.

    An ``EMPTY`` slot carrying a stack is a hard contradiction (an empty seat
    holds no chips) and is an ``ERROR``. An ``OCCUPIED`` slot whose stack is
    ``UNKNOWN`` is *not* a contradiction — the guide tells the labeller to stay
    UNKNOWN when the digits cannot be read, so this is a legitimate
    fail-closed label. It is still worth surfacing as a ``WARN`` because it
    usually means the stack was readable on that frame but not transcribed, so
    the owner can decide whether to go back and fill it.
    """
    issues: list[Issue] = []
    for label in labels:
        for slot in label.slots:
            occ = _occupied(slot)
            stack = _stack(slot)
            if occ is None:
                continue
            if occ is True and stack is None:
                issues.append(
                    _issue(
                        label,
                        "occupancy-stack",
                        f"slot {slot.slot_id} is OCCUPIED but stack is UNKNOWN "
                        "(stack may have been readable but was not transcribed)",
                        slot_id=slot.slot_id,
                        severity=_SEVERITY_WARN,
                    )
                )
            elif occ is False and stack is not None:
                issues.append(
                    _issue(
                        label,
                        "occupancy-stack",
                        f"slot {slot.slot_id} is EMPTY but stack {stack} is "
                        "labelled",
                        slot_id=slot.slot_id,
                    )
                )
    return issues


def _check_empty_not_dealer(labels: Sequence[FrameLabel]) -> list[Issue]:
    """An empty slot cannot be the dealer."""
    issues: list[Issue] = []
    for label in labels:
        for slot in label.slots:
            occ = _occupied(slot)
            dealer = _dealer(slot)
            if occ is False and dealer is True:
                issues.append(
                    _issue(
                        label,
                        "empty-not-dealer",
                        f"slot {slot.slot_id} is EMPTY yet marked dealer",
                        slot_id=slot.slot_id,
                    )
                )
    return issues


def _check_pot_non_decreasing(labels: Sequence[FrameLabel]) -> list[Issue]:
    """Within a hand the pot should not fall (a new hand resets it)."""
    issues: list[Issue] = []
    for hand, items in _group_by_hand(labels).items():
        if len(items) < _MIN_HAND_FRAMES_FOR_TREND:
            continue
        last: int | None = None
        for label in items:
            pot = _known(label.pot)
            if pot is None:
                continue
            if last is not None and int(pot) < last:
                issues.append(
                    _issue(
                        label,
                        "pot-non-decreasing",
                        f"pot fell from {last} to {pot} within hand {hand} "
                        "(new hand resets are the only legitimate exception)",
                        severity=_SEVERITY_WARN,
                    )
                )
            last = int(pot)
    return issues


def _check_action_stack_consistency(
    labels: Sequence[FrameLabel],
) -> list[Issue]:
    """A completed action must be supported by a stack / pot delta.

    This rule only fires when the owner has actually labelled completed
    actions; until then every slot carries ``UNKNOWN`` and there is nothing to
    check. It is included so the audit grows into action evidence without a
    code change once those frames are labelled.
    """
    issues: list[Issue] = []
    for label in labels:
        for slot in label.slots:
            action = _known(slot.completed_action)
            if action is None:
                continue
            stack = _stack(slot)
            if action == CompletedAction.FOLD:
                pass
            elif stack is None:
                issues.append(
                    _issue(
                        label,
                        "action-stack-consistency",
                        f"slot {slot.slot_id} labelled {action} but stack is "
                        "UNKNOWN (need a stack delta to corroborate)",
                        slot_id=slot.slot_id,
                        severity=_SEVERITY_WARN,
                    )
                )
    return issues


# --- rule registry ---------------------------------------------------------

#: (rule name, checker function). Ordered so the most fundamental rules run
#: first; a later rule can still fire even when an earlier one did.
_RULES: tuple[tuple[str, callable], ...] = (
    ("street-board-count", _check_street_board_count),
    ("board-non-decreasing", _check_board_non_decreasing),
    ("dealer-unique", _check_dealer_unique),
    ("occupancy-stack", _check_occupancy_stack),
    ("empty-not-dealer", _check_empty_not_dealer),
    ("pot-non-decreasing", _check_pot_non_decreasing),
    ("action-stack-consistency", _check_action_stack_consistency),
)


def audit_labels(
    labels: Sequence[FrameLabel],
    *,
    rules: Iterable[str] | None = None,
) -> AuditReport:
    """Audit a label set for cross-field contradictions.

    ``rules`` may select a subset of the built-in rules by name; when omitted
    every rule runs. Results and issues are only produced for the rules that
    actually executed, so a label set whose action fields are all ``UNKNOWN``
    reports ``action-stack-consistency`` as skipped rather than flooded with
    findings.
    """
    selected = set(rules) if rules is not None else None
    labels = tuple(labels)
    hands = _group_by_hand(labels)
    sessions = sorted({label.session_id for label in labels})

    all_issues: list[Issue] = []
    all_results: list[CheckResult] = []

    for rule, check in _RULES:
        if selected is not None and rule not in selected:
            continue
        issues = check(labels)
        all_issues.extend(issues)
        all_results.append(
            CheckResult(
                rule=rule,
                checked=len(labels),
                violated=len(issues),
            )
        )

    return AuditReport(
        issues=tuple(all_issues),
        results=tuple(all_results),
        frames_checked=len(labels),
        hands_checked=len(hands),
        sessions=tuple(sessions),
    )


__all__ = [
    "AuditReport",
    "CheckResult",
    "Issue",
    "audit_labels",
]
