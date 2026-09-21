"""Synthetic source/temporal safety; no visual-accuracy claim or private media."""

from copy import deepcopy

import pytest

from poker_engine.desktop.aa_critical_perception import (
    CriticalPerceptionBoundary, checked_view, target_s_candidate_screen,
)
from poker_engine.desktop.aa_live_context_v3 import LiveStateAdapterV3
from tools.aa8_critical_fields_v3 import normalize_fields


def observation(frame, stage="river", *, actor=4):
    size = 4 if stage == "turn" else 5
    board = ["2c", "3d", "4h", "5s", "7c"][:size]
    states = {str(s): "active" if s in (4, 5, 7) else "folded" for s in range(8)}
    return {
        "source_id": "synthetic-session", "source_frame": frame + 100,
        "frame": frame, "pts_seconds": frame / 10, "source_sha256": "a" * 64,
        "scene_supported": True, "special_modes": {}, "board_count": size,
        "cards": {"hero": ["Ah", "As"], "board_slots": board + [None] * (5 - size)},
        "observed_state_v2": {
            "observed_epoch": "synthetic-hand", "street_candidate": stage,
            "board_candidate": board,
            "positive_board_geometry": {"last_frame": frame, "streak": 2},
            "pending_actions": 0,
            "participants": {s: {
                "state": v, "epoch": "synthetic-hand", "evidence_frame": frame,
                "evidence": "DEALT_IN_CANDIDATE" if v == "active"
                else "FOLDED_CANDIDATE"} for s, v in states.items()}},
        "participation": {"slots": {s: {
            "current": "DEALT_IN_CANDIDATE" if v == "active" else "FOLDED_CANDIDATE",
            "conflict": False} for s, v in states.items()}},
        "current_actor": actor,
        "actor_evidence": {"reason": "hero_buttons", "hero_turn": True},
        "dealer_seat": 3,
        "dealer_observation_v2": {"dealer_seat": 3},
        "dealer_evidence_v2": {
            "dealer_seat": 3, "epoch": "synthetic-hand", "frame": frame},
        "stacks": {str(s): {"value": "100"} for s in range(8)},
        "glyphs": {}, "glyph_transitions": [], "observed_actions_v2": [],
        "action_history_candidate": [],
        "causal_street_wagers_v2": {
            "status": "OBSERVED_STREET_WAGERS_CANDIDATE",
            "title_center_ledger_reconciled": True,
            "wagers": {str(s): "0" for s in range(8)}}}


def qualify(boundary, row):
    row["critical_perception_v1"] = boundary.observe(row)
    return row["critical_perception_v1"]["fields"]


def test_contiguous_river_first_actor_candidate_is_not_truth():
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    assert qualify(boundary, observation(1))["river_first_actor"]["status"] == "UNKNOWN"
    row = observation(2)
    fields = qualify(boundary, row)
    assert all(f["status"] == "KNOWN" for f in fields.values())
    assert fields["river_first_actor"]["value"] == 4
    assert checked_view(row) is not None
    assert row["critical_perception_v1"]["candidate_only"] is True
    assert row["critical_perception_v1"]["strategy_eligible"] is False
    screen = target_s_candidate_screen(row)
    assert screen["status"] == "SCREEN_ELIGIBLE"
    assert screen["real_hand_confirmed"] is False


@pytest.mark.parametrize("change", [
    "hero_folded", "two_active", "all_in", "actor_later"])
def test_target_s_screen_rejects_ineligible_machine_shapes(change):
    boundary = CriticalPerceptionBoundary()
    for frame in range(3):
        row = observation(frame, "turn" if frame == 0 else "river")
        if change in ("hero_folded", "two_active"):
            seat = "4" if change == "hero_folded" else "5"
            row["observed_state_v2"]["participants"][seat]["state"] = "folded"
        elif change == "all_in":
            row["observed_state_v2"]["participants"]["0"]["state"] = "all_in"
            row["stacks"]["0"]["value"] = "0"
        else:
            row["current_actor"] = 5
        qualify(boundary, row)
    assert target_s_candidate_screen(row)["status"] == "SCREEN_BLOCKED"


def test_target_s_screen_rejects_unknown_and_forged_view():
    row = observation(0)
    qualify(CriticalPerceptionBoundary(), row)
    assert target_s_candidate_screen(row)["status"] == "SCREEN_BLOCKED"
    row["critical_perception_v1"]["fields"]["board"]["value"][0] = "9c"
    assert target_s_candidate_screen(row)["status"] == "SCREEN_BLOCKED"


def test_snapshot_prefill_requires_qualified_view_and_preserves_provenance():
    from poker_engine.desktop.aa_hand_input import facts_from_snapshot
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    qualify(boundary, observation(1))
    row = observation(2)
    row.update(hero_seat=4, action_order=[4, 5, 7],
               hand_ledger_v2={"status": "OBSERVED_HAND_COMMITMENTS_CANDIDATE",
                               "epoch": "synthetic-hand", "taint_reasons": [],
                               "hand_commitments": {str(s): "20" for s in range(8)}})
    qualify(boundary, row)
    facts, _ = facts_from_snapshot(row)
    assert facts["board_cards"]["provenance"] == "observed"
    assert facts["seats"]["provenance"] == "observed"
    assert facts["action_order"]["provenance"] == "observed"
    row.pop("critical_perception_v1")
    facts, _ = facts_from_snapshot(row)
    assert all(facts[key]["provenance"] == "unknown"
               for key in ("board_cards", "seats", "action_order"))


@pytest.mark.parametrize("change", [
    "prior_check", "raw_gap", "actor_not_first", "partial_board", "pending_action",
    "current_conflict", "stale_active", "all_in_unknown_stack", "old_epoch",
    "missing_source"])
def test_ambiguous_or_late_river_never_claims_first(change):
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    qualify(boundary, observation(1))
    row = observation(2)
    if change == "prior_check":
        row["action_history_candidate"] = [{
            "epoch": "synthetic-hand", "street": "river", "frame": 1,
            "kind": "check", "amount": "0"}]
    elif change == "raw_gap":
        row["source_frame"] += 1
    elif change == "actor_not_first":
        row["current_actor"] = 5
    elif change == "partial_board":
        row["cards"]["board_slots"][4] = None
    elif change == "pending_action":
        row["observed_state_v2"]["pending_actions"] = 1
    elif change == "current_conflict":
        row["participation"]["slots"]["4"]["conflict"] = True
    elif change == "stale_active":
        row["observed_state_v2"]["participants"]["4"]["evidence_frame"] = 0
        row["participation"]["slots"]["4"]["current"] = "UNKNOWN"
    elif change == "all_in_unknown_stack":
        row["observed_state_v2"]["participants"]["5"]["state"] = "all_in"
        row["stacks"]["5"]["value"] = None
    elif change == "old_epoch":
        row["observed_state_v2"]["participants"]["5"]["epoch"] = "old"
    elif change == "missing_source":
        row["source_frame"] = None
    assert qualify(boundary, row)["river_first_actor"]["status"] != "KNOWN"


def test_board_maturation_requires_uninterrupted_action_free_transition():
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    partial = observation(1)
    partial["cards"]["board_slots"][4] = None
    assert qualify(boundary, partial)["board"]["status"] == "UNKNOWN"
    row = observation(2)
    row["glyphs"] = {"0": "fold", "1": "fold"}
    row["cards"]["board_slots"][4] = None
    assert qualify(boundary, row)["river_first_actor"]["status"] == "UNKNOWN"
    row = observation(3)
    row["glyphs"] = {"0": "fold", "1": "fold"}
    fields = qualify(boundary, row)
    assert fields["board"]["status"] == "KNOWN"
    assert fields["river_first_actor"]["status"] == "KNOWN"
    assert row["critical_perception_v1"]["binding"]["frame"] == 3
    assert checked_view(row) is not None


def test_late_board_after_intervening_check_cannot_backdate_first_actor():
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    partial = observation(1)
    partial["cards"]["board_slots"][4] = None
    qualify(boundary, partial)
    partial = observation(2)
    partial["glyph_transitions"] = [{"frame": 2, "slot": 4, "glyph": "check"}]
    qualify(boundary, partial)
    assert qualify(boundary, observation(3))["river_first_actor"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("failure", [
    None, "check", "unknown_rectangle", "pot_mismatch"])
def test_actual_adapter_geometry_transition_requires_no_intervening_action(
        failure):
    from poker_engine.desktop.aa_live_context import LiveCausalWagers
    adapter, boundary = LiveStateAdapterV3(), CriticalPerceptionBoundary()
    wagers = LiveCausalWagers()
    adapter.epoch = "synthetic-hand"
    for frame, count in enumerate((4, 4, 5, 5, 5, 5)):
        row = observation(frame, "turn" if count == 4 else "river")
        # Feed raw geometry/cues through the real adapter. Its first five-card
        # row has street=None until geometry confirms on the next observation.
        row["cards"]["evidence"] = {f"board_{s}": {
            "rect": [110 + s * 56, 473, 53, 78] if s < count else None,
            "face_support": "nominal_face_boundary_supported" if s < count
            else "no_positive_face_background"} for s in range(5)}
        row["pot"] = {"value": "160"}
        row["observed_center_v2"] = {"value": "160"}
        row["street_wagers"] = {str(s): None for s in range(8)}
        row["wager_visibility_v2"] = {
            str(s): {"status": "VISIBLE_EMPTY_CANDIDATE"} for s in range(8)}
        if frame in (2, 3):
            row["cards"]["board_slots"][4] = None
        if failure == "check" and frame == 3:
            row["glyph_transitions"] = [{"frame": 3, "slot": 4, "glyph": "check"}]
        if frame == 2 and failure == "unknown_rectangle":
            row["wager_visibility_v2"]["0"]["status"] = "UNKNOWN"
        if frame == 2 and failure == "pot_mismatch":
            row["pot"]["value"] = "161"
        row["observed_state_v2"] = adapter.observe(row)
        row["causal_street_wagers_v2"] = wagers.observe(row, adapter, {
            "title": row["pot"]["value"], "center": row["observed_center_v2"],
            "wagers": row["street_wagers"], "visibility": row["wager_visibility_v2"]})
        fields = qualify(boundary, row)
        if frame == 2:
            assert row["observed_state_v2"]["street_candidate"] is None
            assert fields["river_first_actor"]["status"] == "UNKNOWN"
            assert row["causal_street_wagers_v2"]["status"] == "WAGERS_UNKNOWN"
        if frame >= 4:
            assert fields["board"]["status"] == "KNOWN"
            assert fields["river_first_actor"]["status"] == (
                "UNKNOWN" if failure else "KNOWN")
            assert checked_view(row) is not None


def test_within_hand_board_change_conflicts_until_new_source_context():
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    row = observation(1)
    row["cards"]["board_slots"][0] = "9c"
    row["observed_state_v2"]["board_candidate"][0] = "9c"
    assert qualify(boundary, row)["board"]["status"] == "CONFLICT"
    assert qualify(boundary, observation(2))["board"]["status"] == "CONFLICT"


def test_all_in_is_separate_and_normalizer_uses_qualified_view():
    row = observation(0)
    row["observed_state_v2"]["participants"]["5"]["state"] = "all_in"
    row["stacks"]["5"]["value"] = "0"
    fields = qualify(CriticalPerceptionBoundary(), row)
    assert fields["all_in_seats"]["value"] == [5]
    assert normalize_fields(row)["participation"]["value"]["5"] == "all_in"


@pytest.mark.parametrize("stack", ["0", None, "NaN", True, 100])
def test_back_cards_with_zero_or_unknown_stack_do_not_claim_active(stack):
    row = observation(0)
    row["stacks"]["5"]["value"] = stack
    fields = qualify(CriticalPerceptionBoundary(), row)
    assert fields["participation"]["status"] == "UNKNOWN"
    assert fields["all_in_seats"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("previous", [None, "active", "folded", "all_in"])
def test_current_hero_buttons_cannot_reopen_terminal_participation(previous):
    adapter = LiveStateAdapterV3()
    adapter.epoch = "synthetic-hand"
    if previous:
        adapter.participants["4"] = {
            "state": previous, "epoch": "synthetic-hand", "evidence_frame": 0}
    row = observation(1)
    row["participation"]["slots"]["4"]["current"] = "UNKNOWN"
    row["glyph_transitions"] = []
    value = adapter.observe(row)
    if previous in ("folded", "all_in"):
        assert value["participants"]["4"]["state"] == previous
        assert "4" in value["critical_status_conflicts"]
    else:
        assert value["participants"]["4"]["state"] == "active"
        assert value["participants"]["4"]["evidence_frame"] == 1


@pytest.mark.parametrize("key,value", [("source_id", "other"), ("source_frame", 99)])
def test_cached_view_does_not_survive_source_substitution(key, value):
    row = observation(0)
    qualify(CriticalPerceptionBoundary(), row)
    row[key] = value
    assert checked_view(row) is None
    assert normalize_fields(row)["board_cards"]["status"] == "UNKNOWN"


def test_raw_board_change_or_cached_value_change_rejects_projection():
    row = observation(0)
    qualify(CriticalPerceptionBoundary(), row)
    changed = deepcopy(row)
    changed["cards"]["board_slots"][0] = "9c"
    assert checked_view(changed) is None
    row["critical_perception_v1"]["fields"]["board"]["value"][0] = "9c"
    assert checked_view(row) is None


def test_board_count_change_invalidates_cached_projection_and_direct_candidate():
    boundary = CriticalPerceptionBoundary()
    qualify(boundary, observation(0, "turn"))
    qualify(boundary, observation(1))
    row = observation(2)
    qualify(boundary, row)
    assert target_s_candidate_screen(row)["status"] == "SCREEN_ELIGIBLE"
    row["board_count"] = 4
    assert checked_view(row) is None
    assert target_s_candidate_screen(row)["status"] == "SCREEN_BLOCKED"
    fields = qualify(CriticalPerceptionBoundary(), row)
    assert fields["board"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("terminal", ["folded", "all_in"])
def test_later_paired_call_cannot_resurrect_terminal_state(terminal):
    adapter = LiveStateAdapterV3()
    adapter.epoch = "same_hand"
    adapter.participants["4"] = {
        "state": terminal, "evidence_frame": 0,
        "evidence": "confirmed_action_evidence", "epoch": "same_hand"}
    for frame in range(1, 5):
        row = {"frame": frame, "scene_supported": True, "source_sha256": "a" * 64,
               "stacks": {str(s): {"value": "100"} for s in range(8)},
               "pot": {"value": "0"}, "board_count": 0,
               "cards": {"hero": None, "board_slots": [None] * 5},
               "participation": {"slots": {}}, "glyph_transitions": []}
        if frame >= 3:
            row["stacks"]["4"]["value"] = "90"
            row["pot"]["value"] = "10"
        if frame == 3:
            row["glyph_transitions"] = [{"frame": 3, "slot": 4, "glyph": "call"}]
        value = adapter.observe(row)
    assert value["participants"]["4"]["state"] == terminal
    assert value["critical_status_conflicts"] == ["4"]
