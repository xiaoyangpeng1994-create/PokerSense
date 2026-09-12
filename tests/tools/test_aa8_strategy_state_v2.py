from types import SimpleNamespace

from tools.aa8_strategy_state_v2 import (
    AA8HandLedgerCandidate,
    candidate_action_line,
)


def adapter(epoch="h1", *, actions=()):
    return SimpleNamespace(
        epoch_events=[{
            "frame": 2, "first_frame": 1, "epoch": epoch,
            "status": "MULTI_POST_DEAL_CANDIDATE",
            "posting_comparison": {
                "debits": {"0": "1", "1": "2", "2": "4"},
                "excluded_na_slots": ["7"],
            },
            "authoritative_boundary": False,
        }],
        actions=list(actions),
    )


def row(frame, epoch="h1", *, pot="7", modes=None):
    return {
        "frame": frame,
        "pot": {"value": pot},
        "observed_state_v2": {
            "observed_epoch": epoch, "street_candidate": "preflop",
        },
        "special_modes": modes or {},
    }


def action(frame=3, slot=3, kind="aggressive", amount="10", epoch="h1"):
    return {"frame": frame, "slot": slot, "kind": kind, "amount": amount,
            "epoch": epoch, "street": "preflop"}


def test_posts_and_actions_accumulate_once_without_assigning_difference():
    ledger = AA8HandLedgerCandidate()
    source = adapter(actions=[action()])
    value = ledger.observe(row(3, pot="17"), source)
    assert value["hand_commitments"]["3"] == "10"
    assert value["observed_total"] == "17"
    assert value["unallocated_difference"] == "0"
    assert value["applied_action_count"] == 1
    repeated = ledger.observe(row(4, pot="16"), source)
    assert repeated["hand_commitments"]["3"] == "10"
    assert repeated["unallocated_difference"] == "1"
    assert "NOT_RAKE_OR_FEE" in repeated["difference_semantics"]
    assert not repeated["complete_and_canonical_verified"]


def test_epoch_change_resets_old_commitments_and_actions():
    ledger = AA8HandLedgerCandidate()
    ledger.observe(row(3), adapter(actions=[action()]))
    next_adapter = adapter("h2")
    value = ledger.observe(row(4, "h2"), next_adapter)
    assert value["epoch"] == "h2"
    assert value["observed_total"] == "7"
    assert value["applied_action_count"] == 0


def test_special_mode_and_gap_suspend_candidate():
    ledger = AA8HandLedgerCandidate()
    source = adapter()
    ledger.observe(row(2), source)
    special = ledger.observe(
        row(3, modes={"insurance": "VISIBLE"}), source
    )
    assert special["status"] == "HAND_COMMITMENTS_SUSPENDED"
    assert "special_mode_or_overlay_observed" in special["taint_reasons"]
    gap = ledger.observe(row(5), source)
    assert gap["status"] == "HAND_COMMITMENTS_SUSPENDED"
    assert "source_frame_gap" in gap["taint_reasons"]


def test_missing_opening_vector_stays_unknown():
    ledger = AA8HandLedgerCandidate()
    source = adapter()
    source.epoch_events[0]["posting_comparison"] = {"debits": {"0": "1"}}
    value = ledger.observe(row(2), source)
    assert value["status"] == "HAND_COMMITMENTS_UNKNOWN"
    assert value["hand_commitments"] is None


def test_candidate_action_line_uses_observed_order_but_is_not_canonical():
    actions = [
        action(3, 2, "call", "2"),
        action(4, 3, "aggressive", "10"),
        action(5, 4, "call", "10"),
        action(6, 5, "aggressive", "30"),
    ]
    value = candidate_action_line(actions, "h1", "preflop")
    assert value["value"] == "squeeze"
    assert value["source_action_frames"] == [3, 4, 5, 6]
    assert not value["complete_and_canonical_verified"]


def test_action_line_abstains_postflop_or_unknown_action():
    assert candidate_action_line([], "h1", "flop")["value"] is None
    unknown = action(kind="mystery")
    assert candidate_action_line([unknown], "h1", "preflop")["value"] is None
    bad_seat = action(slot=9)
    assert candidate_action_line([bad_seat], "h1", "preflop")["value"] is None


def test_future_action_taints_ledger_instead_of_preapplying():
    ledger = AA8HandLedgerCandidate()
    value = ledger.observe(row(2), adapter(actions=[action(frame=3)]))
    assert value["hand_commitments"]["3"] == "0"
    assert "action_amount_or_seat_unknown" in value["taint_reasons"]
