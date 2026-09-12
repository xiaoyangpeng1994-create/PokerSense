from types import SimpleNamespace

from tools.aa8_live_wagers_v2 import CausalWagersV2


def inputs(frame, anchored=True):
    states = {str(s): {"current": "EMPTY_CANDIDATE" if s == 6 else "DEALT_IN_CANDIDATE"}
              for s in range(8)}
    row = {"frame": frame, "scene_supported": True, "current_actor": 3,
           "actor_evidence": {"timer_suffix_verified": True}, "special_modes": {},
           "participation": {"slots": states}}
    wagers = {str(s): None for s in range(8)}
    wagers.update({"0": "1", "1": "2", "2": "4", "4": "2"})
    part = {"title": "23", "center": {"value": "14"}, "wagers": wagers,
            "visibility": {s: {
                "status": "VISIBLE_COIN_CANDIDATE" if v is not None
                else "VISIBLE_EMPTY_CANDIDATE"} for s, v in wagers.items()}}
    epoch = {"epoch": "e", "frame": 0, "status": "MULTI_POST_DEAL_CANDIDATE"
             if anchored else "UNANCHORED_DEAL_CONTEXT", "posting_comparison": {}}
    state = SimpleNamespace(last=frame, epoch_events=[epoch],
                            actions=[], pending=[], cash=[],
                            snapshot=lambda f: {"street_candidate": "preflop",
                                                "observed_epoch": "e"})
    return row, state, part


def test_context_automatic_and_each_debit_applied_only_once():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    value = tracker.observe(*inputs(2))
    assert value["status"] == "OBSERVED_STREET_WAGERS_CANDIDATE"
    assert value["context_automated"]
    for frame in (3, 4):
        row, state, part = inputs(frame)
        state.actions = [{"frame": 3, "slot": 3, "kind": "aggressive", "amount": "20",
                          "epoch": "e", "street": "preflop"}]
        part["title"] = "43"
        part["wagers"]["3"] = "20"
        part["visibility"]["3"]["status"] = "VISIBLE_COIN_CANDIDATE"
        value = tracker.observe(row, state, part)
        assert value["wagers"]["3"] == "20"
        assert value["observed_debits_applied"] == 1
        assert not value["canonical_verified"]


def test_initial_unanchored_preflop_never_initializes():
    tracker = CausalWagersV2()
    for frame in (1, 2):
        assert tracker.observe(*inputs(frame, False))["wagers"] is None


def test_missing_center_abstains_then_requires_current_reconciliation():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    assert tracker.observe(*inputs(2))["wagers"] is not None
    row, state, part = inputs(3)
    part["center"] = {"value": None}
    value = tracker.observe(row, state, part)
    assert value["wagers"] is None
    assert value["reason"] == "center_unreadable_current_frame"
    assert tracker.baseline is not None
    row, state, part = inputs(4)
    part["title"] = "999"
    assert tracker.observe(row, state, part)["wagers"] is None
    assert tracker.observe(*inputs(5))["wagers"] is not None


def test_center_dropout_cannot_survive_gap_or_insurance():
    for fault in ("gap", "insurance", "changed_center"):
        tracker = CausalWagersV2()
        tracker.observe(*inputs(1))
        tracker.observe(*inputs(2))
        row, state, part = inputs(3)
        part["center"] = {"value": None}
        tracker.observe(row, state, part)
        row, state, part = inputs(5 if fault == "gap" else 4)
        if fault == "insurance":
            row["special_modes"]["insurance"] = "VISIBLE"
        if fault == "changed_center":
            part["center"] = {"value": "23"}
        assert tracker.observe(row, state, part)["wagers"] is None
        assert tracker.baseline is None


def test_unknown_rectangle_does_not_become_positive_conflict_after_timeout():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    for frame in range(3, 20):
        row, state, part = inputs(frame)
        part["wagers"]["2"] = None
        part["visibility"]["2"] = {"status": "UNKNOWN"}
        assert tracker.observe(row, state, part)["wagers"] is None
    assert tracker.baseline is not None
    assert tracker.observe(*inputs(20))["wagers"]["2"] == "4"


def test_known_display_contradiction_still_invalidates_after_timeout():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    for frame in range(3, 17):
        row, state, part = inputs(frame)
        part["title"] = "999"
        assert tracker.observe(row, state, part)["wagers"] is None
    assert tracker.baseline is None


def test_confirmed_debit_during_dropout_is_applied_once_after_recovery():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    for frame in (3, 4, 5):
        row, state, part = inputs(frame)
        state.actions = [{"frame": 3, "slot": 3, "kind": "aggressive", "amount": "20",
                          "epoch": "e", "street": "preflop"}]
        part["title"] = "43"
        part["wagers"]["3"] = "20"
        part["visibility"]["3"] = {"status": "VISIBLE_COIN_CANDIDATE"}
        if frame == 3:
            part["center"] = {"value": None}
        value = tracker.observe(row, state, part)
        if frame == 3:
            assert value["wagers"] is None
        else:
            assert value["wagers"]["3"] == "20"
            assert value["observed_debits_applied"] == 1


def test_insurance_clears_ledger_without_pretending_no_fee():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    row, state, part = inputs(3)
    row["special_modes"]["insurance"] = "VISIBLE"
    assert tracker.observe(row, state, part)["status"] == "WAGERS_UNKNOWN"
    assert tracker.totals is None


def test_collection_changes_center_invalidates_old_street():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    row, state, part = inputs(3)
    part["center"]["value"] = "23"
    assert tracker.observe(row, state, part)["wagers"] is None


def test_positive_credit_retained_once_unallocated():
    tracker = CausalWagersV2()
    for f in (1, 2):
        row, state, part = inputs(f)
        state.epoch_events[0]["posting_comparison"] = {"credits": {"1": "192"}}
        value = tracker.observe(row, state, part)
        assert len(value["unallocated_credits"]) == 1
        assert value["unallocated_credits"][0]["amount"] == "192"
        assert value["unallocated_credits"][0]["semantics"] == "UNALLOCATED_NOT_PROFIT"


def test_posting_and_balance_credit_sources_do_not_double_count():
    tracker = CausalWagersV2()
    for f in (1, 2):
        row, state, part = inputs(f)
        state.epoch_events[0].update(
            first_frame=0, frame=1, posting_comparison={"credits": {"1": "192"}})
        state.credits = [{"first_frame": 0, "confirmed_frame": 1,
                          "seat": 1, "amount": "192", "source": "balance"}]
        value = tracker.observe(row, state, part)
        assert len(value["unallocated_credits"]) == 1


def test_existing_action_before_observer_start_prevents_new_zero_baseline():
    tracker = CausalWagersV2()
    for frame in (20, 21):
        row, state, part = inputs(frame)
        state.actions = [{"frame": 3, "slot": 3, "kind": "aggressive", "amount": "20",
                          "epoch": "e", "street": "preflop"}]
        assert tracker.observe(row, state, part)["wagers"] is None


def test_lagging_glyph_confirmation_never_publishes_unaccounted_totals():
    tracker = CausalWagersV2()
    tracker.observe(*inputs(1))
    tracker.observe(*inputs(2))
    for f in (3, 4):
        row, state, part = inputs(f)
        part["title"] = "43"
        part["wagers"]["3"] = "20"
        assert tracker.observe(row, state, part)["wagers"] is None
    row, state, part = inputs(5)
    part["title"] = "43"
    part["wagers"]["3"] = "20"
    state.actions = [{"frame": 5, "slot": 3, "kind": "aggressive", "amount": "20",
                      "epoch": "e", "street": "preflop"}]
    assert tracker.observe(row, state, part)["wagers"]["3"] == "20"


def test_two_geometry_frames_can_initialize_before_rank_identity_finishes():
    tracker = CausalWagersV2()
    for f in (4402, 4403):
        row, state, part = inputs(f)
        state.snapshot = lambda frame: {
            "street_candidate": "flop" if frame == 4403 else None,
            "observed_epoch": "e", "positive_board_geometry": {
                "count": 3, "first_frame": 4402,
                "last_frame": frame, "streak": frame - 4401}}
        part["title"], part["center"]["value"] = "114", "114"
        part["wagers"] = dict.fromkeys(map(str, range(8)))
        part["visibility"] = {str(s): {"status": "VISIBLE_EMPTY_CANDIDATE"}
                              for s in range(8)}
        value = tracker.observe(row, state, part)
    assert value["baseline_evidence_frames"] == [4402, 4403]
    assert value["status"] == "OBSERVED_STREET_WAGERS_CANDIDATE"


def test_partial_river_animation_cannot_restart_a_flop_baseline():
    tracker = CausalWagersV2()
    for f in (10, 11):
        row, state, part = inputs(f)
        state.street = "river"
        state.snapshot = lambda frame: {
            "street_candidate": None, "observed_epoch": "e",
            "positive_board_geometry": {"count": 3, "streak": 2}}
        part["title"] = part["center"]["value"] = "114"
        part["wagers"] = dict.fromkeys(map(str, range(8)))
        assert tracker.observe(row, state, part)["wagers"] is None
