"""Ordinary (non all-in) river close, confirmed only by the next hand boundary.

The existing all-in terminal is untouched: ``closed`` still needs two all-in
seats, and an all-in showdown never produces an ordinary close. This covers ONE
ending only -- an ordinary hand that reaches a complete river and settles -- and
the row says so in ``ordinary_terminal_semantics``. Preflop / flop / turn
fold-out endings are deliberately not covered.
"""

from poker_engine.desktop.aa_semantics import AAObservationSemantics

DEFAULT_STATES = ((1, "active"), (4, "active"))


def river_row(frame, *, epoch="h", pot="623", credited_at=None, street="river",
              board=("5h", "6c", "6s", "Tc", "3d"), hero=("5d", "6d"),
              states=None, actor=None, pending=0):
    seats = DEFAULT_STATES if states is None else states
    participants = {str(seat): {"epoch": epoch, "state": state}
                    for seat, state in seats}
    credits = [] if credited_at is None else [{
        "epoch": epoch, "seat": 4, "amount": "516", "confirmed_frame": credited_at,
        "source": "stable_visual_balance_increase"}]
    return {
        "frame": frame, "source_id": "source-a", "source_sha256": "a" * 64,
        "source_frame": 1000 + frame, "scene_supported": True, "special_modes": {},
        "observed_state_v2": {
            "observed_epoch": epoch, "street_candidate": street,
            "participants": participants, "pending_actions": pending,
            "unallocated_positive_cash": credits},
        "hand_ledger_v2": {"epoch": epoch, "status": "HAND_COMMITMENTS_UNKNOWN",
                           "opening_evidence": None},
        "pot": {"value": pot}, "current_actor": actor,
        "cards": {"hero": list(hero) if hero else None,
                  "board_slots": list(board) if board else [None] * 5},
        "hand_transition": {"center_deal": {"visible": None}},
    }


def test_ordinary_river_close_is_confirmed_only_by_the_next_hand_boundary():
    subject = AAObservationSemantics()
    subject.observe(river_row(0))
    pending = subject.observe(river_row(1, credited_at=1))["hand_phase"]
    assert pending["ordinary_terminal"] is None
    assert pending["ordinary_river_pending"]["status"] == (
        "AWAITING_NEXT_HAND_BOUNDARY")
    assert pending["ordinary_river_pending"]["river_complete_frame"] == 0

    confirmed = subject.observe(river_row(2, epoch="next"))["hand_phase"]
    terminal = confirmed["ordinary_terminal"]
    assert terminal["kind"] == "ORDINARY_RIVER_CLOSED"
    assert terminal["epoch"] == "h"
    assert terminal["river_complete_frame"] == 0
    assert terminal["confirmed_by_epoch_frame"] == 2
    assert terminal["canonical_verified"] is False
    assert terminal["strategy_eligible"] is False
    assert terminal["card_showdown_verified"] is False
    assert terminal["rake_verified"] is False
    assert confirmed["ordinary_river_pending"] is None


def test_ordinary_terminal_says_what_it_does_not_cover():
    subject = AAObservationSemantics()
    result = subject.observe(river_row(0))["hand_phase"]
    assert result["ordinary_terminal_semantics"] == (
        "ORDINARY_RIVER_CLOSED_ONLY; preflop/flop/turn fold-out endings "
        "are NOT covered")
    assert result["ordinary_terminal"] is None


def test_zero_pot_without_a_payout_never_settles():
    """A cleared pot alone is not settlement evidence."""
    subject = AAObservationSemantics()
    subject.observe(river_row(0))
    for frame in (1, 2, 3):
        result = subject.observe(river_row(frame, pot="0"))["hand_phase"]
    assert result["ordinary_river_pending"] is None
    assert result["ordinary_terminal"] is None


def test_a_lost_epoch_after_the_river_never_confirms():
    """Losing the epoch is a suspension, never a next-hand boundary."""
    subject = AAObservationSemantics()
    subject.observe(river_row(0))
    subject.observe(river_row(1, credited_at=1))
    gone = river_row(2, epoch=None, street=None, board=None, hero=None)
    result = subject.observe(gone)["hand_phase"]
    assert result["phase"] == "WAITING_OPENING"
    assert result["ordinary_terminal"] is None
    assert result["ordinary_river_pending"] is None


def test_recording_ending_right_after_the_river_stays_pending():
    subject = AAObservationSemantics()
    subject.observe(river_row(0))
    last = subject.observe(river_row(1, credited_at=1))["hand_phase"]
    assert last["ordinary_terminal"] is None
    assert last["ordinary_river_pending"] is not None


def test_all_in_showdown_never_produces_an_ordinary_close():
    subject = AAObservationSemantics()
    ledger = {"epoch": "h", "opening_evidence": {"debits": {"1": "2", "4": "2"}}}
    first = river_row(0, states=((1, "all_in"), (4, "all_in")))
    first["hand_ledger_v2"] = ledger
    subject.observe(first)
    second = river_row(1, credited_at=1, states=((1, "all_in"), (4, "all_in")))
    second["hand_ledger_v2"] = ledger
    result = subject.observe(second)["hand_phase"]
    assert result["terminal_observation_frame"] == 0
    assert result["ordinary_terminal"] is None
    assert result["ordinary_river_pending"] is None


def test_a_blocked_frame_neither_confirms_nor_reports_a_pending_close():
    """An overlay must not be able to stand in for a next-hand boundary."""
    subject = AAObservationSemantics()
    subject.observe(river_row(0))
    subject.observe(river_row(1, credited_at=1))
    overlay = river_row(2, credited_at=1)
    overlay["special_modes"] = {"insurance": "VISIBLE"}
    blocked = subject.observe(overlay)["hand_phase"]
    assert blocked["phase"] == "SUSPENDED"
    assert blocked["ordinary_terminal"] is None
    assert blocked["ordinary_river_pending"] is None
    assert blocked["ordinary_terminal_semantics"].startswith(
        "ORDINARY_RIVER_CLOSED_ONLY")


def test_preflop_fold_out_is_not_covered_by_this_terminal():
    subject = AAObservationSemantics()
    subject.observe(river_row(0, street=None, board=None, hero=None))
    for frame in (1, 2, 3):
        result = subject.observe(river_row(
            frame, pot="0", street=None, board=None, hero=None,
            states=((1, "active"), (4, "folded"))))["hand_phase"]
    assert result["ordinary_terminal"] is None
    assert result["ordinary_river_pending"] is None
    assert result["ordinary_terminal_semantics"].startswith(
        "ORDINARY_RIVER_CLOSED_ONLY")
