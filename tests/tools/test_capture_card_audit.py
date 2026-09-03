# -*- coding: utf-8 -*-
"""Tests for the cross-field consistency audit (stage F).

The audit must never rewrite a label or guess a value: it only reports
contradictions a machine can decide without pixels, so a human can confirm or
fix the offending field. These tests pin the rule behaviours — including the
cases where a rule must stay *silent* (UNKNOWN fields, an all-UNKNOWN action
column, a single-frame hand) rather than report a spurious finding.
"""

from __future__ import annotations

from tools.capture_card_calibration.audit import audit_labels
from tools.capture_card_calibration.schema import (
    FieldValue,
    FrameLabel,
    Occupancy,
    Review,
    Scene,
    SlotLabel,
    Street,
)


# --- helpers ---------------------------------------------------------------


def _slot(slot_id: int, **overrides: object) -> SlotLabel:
    data: dict[str, object] = {
        "slot_id": slot_id,
        "occupancy": FieldValue.valid(Occupancy.OCCUPIED.value).to_dict(),
        "stack": FieldValue.valid(100 + slot_id).to_dict(),
        "dealer": FieldValue.valid(False).to_dict(),
        "completed_action": FieldValue.unknown().to_dict(),
        "current_actor": FieldValue.unknown().to_dict(),
    }
    data.update(overrides)
    return SlotLabel.from_dict(data)


def _label(
    frame: str = "a.png",
    *,
    hand_id: str = "h1",
    timestamp_ms: int = 0,
    board: object = FieldValue.valid(["4H", "KC", "TS"]),
    street: object = FieldValue.valid(Street.FLOP.value),
    pot: object = FieldValue.valid(120),
    **overrides: object,
) -> FrameLabel:
    kwargs: dict[str, object] = {
        "frame": frame,
        "sha256": "a" * 64,
        "session_id": "session_001",
        "hand_id": hand_id,
        "timestamp_ms": timestamp_ms,
        "stable": True,
        "scene": Scene.TABLE,
        "hero_cards": FieldValue.valid(["TS", "JD"]),
        "board_cards": board,
        "street": street,
        "pot": pot,
        "slots": tuple(
            _slot(i, dealer=FieldValue.valid(i == 3).to_dict()) for i in range(8)
        ),
        "review": Review(reviewer="tester"),
    }
    kwargs.update(overrides)
    return FrameLabel(**kwargs)


def _dealer_slots(dealer_id: int) -> tuple[SlotLabel, ...]:
    return tuple(
        _slot(i, dealer=FieldValue.valid(i == dealer_id).to_dict())
        for i in range(8)
    )


def _one_dealer() -> tuple[SlotLabel, ...]:
    return _dealer_slots(3)


# --- street / board card count ---------------------------------------------


def test_street_board_count_ok() -> None:
    labels = [_label()]
    report = audit_labels(labels)
    assert report.error_count == 0


def test_street_board_count_mismatch_is_error() -> None:
    # FLOP expects 3 cards, but the board holds 2.
    labels = [_label(board=FieldValue.valid(["4H", "KC"]))]
    report = audit_labels(labels)
    assert report.has_errors
    issue = report.issues_for("street-board-count")
    assert len(issue) == 1
    assert issue[0].severity == "ERROR"


def test_street_board_count_skips_unknown_street() -> None:
    labels = [_label(street=FieldValue.unknown())]
    report = audit_labels(labels)
    assert report.issues_for("street-board-count") == ()


def test_street_board_count_quiet_when_preflop_board_unknown() -> None:
    # PRE_FLOP has no public cards; schema forbids a VALID empty board list,
    # so the board must be UNKNOWN and the rule must stay silent.
    labels = [
        _label(
            board=FieldValue.unknown(),
            street=FieldValue.valid(Street.PRE_FLOP.value),
        )
    ]
    assert audit_labels(labels).issues_for("street-board-count") == ()


# --- board monotonicity -----------------------------------------------------


def test_board_non_decreasing_ok() -> None:
    labels = [
        _label(board=FieldValue.valid(["4H", "KC", "TS"]), timestamp_ms=0),
        _label(board=FieldValue.valid(["4H", "KC", "TS", "9D"]), timestamp_ms=10000),
    ]
    assert audit_labels(labels).issues_for("board-non-decreasing") == ()


def test_board_shrinks_is_error() -> None:
    labels = [
        _label(board=FieldValue.valid(["4H", "KC", "TS", "9D"]), timestamp_ms=0),
        _label(board=FieldValue.valid(["4H", "KC", "TS"]), timestamp_ms=10000),
    ]
    issues = audit_labels(labels).issues_for("board-non-decreasing")
    assert len(issues) == 1


# --- dealer uniqueness ------------------------------------------------------


def test_dealer_unique_ok() -> None:
    labels = [_label(slots=_one_dealer())]
    assert audit_labels(labels).issues_for("dealer-unique") == ()


def test_dealer_unique_missing_is_error() -> None:
    # All slots dealer=False -> no dealer at all.
    labels = [_label(slots=tuple(_slot(i) for i in range(8)))]
    assert audit_labels(labels).issues_for("dealer-unique")


def test_dealer_unique_multiple_is_error() -> None:
    # Two dealer slots -> contradiction.
    slots = tuple(
        _slot(i, dealer=FieldValue.valid(i in (2, 5)).to_dict()) for i in range(8)
    )
    labels = [_label(slots=slots)]
    assert audit_labels(labels).issues_for("dealer-unique")


def test_dealer_unique_ignores_unknown_dealer() -> None:
    # Dealer UNKNOWN on every slot is a missing observation, not a contradiction.
    slots = tuple(
        _slot(i, dealer=FieldValue.unknown().to_dict()) for i in range(8)
    )
    labels = [_label(slots=slots)]
    assert audit_labels(labels).issues_for("dealer-unique") == ()


# --- occupancy vs stack -----------------------------------------------------


def test_occupancy_stack_ok() -> None:
    labels = [_label()]
    assert audit_labels(labels).issues_for("occupancy-stack") == ()


def test_occupied_with_unknown_stack_is_warn() -> None:
    # OCCUPIED with UNKNOWN stack is a legitimate fail-closed label (the
    # digits may not have been readable), so it surfaces as a WARN, not an
    # ERROR — the owner may still want to fill it in if it was readable.
    slots = tuple(
        _slot(i, stack=FieldValue.unknown().to_dict()) for i in range(8)
    )
    labels = [_label(slots=slots)]
    issues = audit_labels(labels).issues_for("occupancy-stack")
    assert len(issues) == 8
    assert all(issue.severity == "WARN" for issue in issues)


def test_empty_with_stack_is_error() -> None:
    slots = tuple(
        _slot(i, occupancy=FieldValue.valid(Occupancy.EMPTY.value).to_dict())
        for i in range(8)
    )
    labels = [_label(slots=slots)]
    assert audit_labels(labels).issues_for("occupancy-stack")


def test_occupancy_stack_ignores_unknown_occupancy() -> None:
    slots = tuple(
        _slot(i, occupancy=FieldValue.unknown().to_dict()) for i in range(8)
    )
    labels = [_label(slots=slots)]
    assert audit_labels(labels).issues_for("occupancy-stack") == ()


# --- empty slot cannot be dealer --------------------------------------------


def test_empty_not_dealer_ok() -> None:
    labels = [_label(slots=_one_dealer())]
    assert audit_labels(labels).issues_for("empty-not-dealer") == ()


def test_empty_dealer_is_error() -> None:
    slots = tuple(
        _slot(i, occupancy=FieldValue.valid(Occupancy.EMPTY.value).to_dict())
        for i in range(8)
    )
    # Dealer on slot 4, which is EMPTY.
    slots = tuple(
        _slot(
            i,
            occupancy=FieldValue.valid(Occupancy.EMPTY.value).to_dict(),
            dealer=FieldValue.valid(i == 4).to_dict(),
        )
        for i in range(8)
    )
    labels = [_label(slots=slots)]
    assert audit_labels(labels).issues_for("empty-not-dealer")


# --- pot monotonicity --------------------------------------------------------


def test_pot_non_decreasing_ok() -> None:
    labels = [
        _label(pot=FieldValue.valid(120), timestamp_ms=0),
        _label(pot=FieldValue.valid(240), timestamp_ms=10000),
    ]
    assert audit_labels(labels).issues_for("pot-non-decreasing") == ()


def test_pot_falls_is_warn() -> None:
    labels = [
        _label(pot=FieldValue.valid(240), timestamp_ms=0),
        _label(pot=FieldValue.valid(120), timestamp_ms=10000),
    ]
    issues = audit_labels(labels).issues_for("pot-non-decreasing")
    assert len(issues) == 1
    assert issues[0].severity == "WARN"


def test_pot_monotonic_silent_on_single_frame_hand() -> None:
    # A one-frame hand has nothing to compare, so no spurious finding.
    labels = [_label(pot=FieldValue.valid(120))]
    assert audit_labels(labels).issues_for("pot-non-decreasing") == ()


# --- action / actor (expected to be silent while all UNKNOWN) ---------------


def test_action_consistency_silent_while_unknown() -> None:
    # No completed_action is VALID, so the rule must not fire.
    labels = [_label()]
    assert audit_labels(labels).issues_for("action-stack-consistency") == ()
    result = audit_labels(labels).rule_result("action-stack-consistency")
    assert result is not None
    assert result.violated == 0


def test_action_with_unknown_stack_is_warn() -> None:
    slots = tuple(
        _slot(
            i,
            completed_action=FieldValue.valid("CALL").to_dict(),
            stack=FieldValue.unknown().to_dict(),
        )
        for i in range(8)
    )
    labels = [_label(slots=slots)]
    issues = audit_labels(labels).issues_for("action-stack-consistency")
    assert issues
    assert all(issue.severity == "WARN" for issue in issues)


# --- report shape -----------------------------------------------------------


def test_report_summary_shape() -> None:
    report = audit_labels([_label()])
    assert report.frames_checked == 1
    assert report.hands_checked == 1
    assert report.sessions == ("session_001",)
    assert report.has_errors is False


def test_rule_subset_selection() -> None:
    report = audit_labels([_label()], rules=["dealer-unique"])
    names = {result.rule for result in report.results}
    assert names == {"dealer-unique"}


def test_known_rule_names() -> None:
    report = audit_labels([_label()])
    names = {result.rule for result in report.results}
    expected = {
        "street-board-count",
        "board-non-decreasing",
        "dealer-unique",
        "occupancy-stack",
        "empty-not-dealer",
        "pot-non-decreasing",
        "action-stack-consistency",
    }
    assert names == expected
