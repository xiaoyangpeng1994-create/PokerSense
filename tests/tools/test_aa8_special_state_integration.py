"""Deterministic integration contracts; no claim of independent visual accuracy."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools.aa8_candidate_v2 import BombGuard
from tools.aa8_live_wagers_v2 import CausalWagersV2
from tools.aa8_special_modes import separate_cash
from tools.aa8_state_adapter_v2 import AA8StateAdapterV2, posting_comparison


def observation(frame, value="100", modes=None, glyph=None):
    return {"frame": frame, "scene_supported": True, "source_sha256": "a" * 64,
            "stacks": {str(s): {"value": value} for s in range(8)},
            "pot": {"value": "0"}, "board_count": 0, "special_modes": modes or {},
            "cards": {"hero": None, "board_slots": [None] * 5},
            "participation": {"slots": {
                str(s): {"current": "DEALT_IN_CANDIDATE"} for s in range(8)}},
            "hand_transition": {"center_deal": {"visible": frame <= 2}},
            "glyph_transitions": [] if glyph is None else [
                {"frame": frame, "slot": 3, "glyph": glyph}]}


@pytest.mark.parametrize("modes", [
    {"insurance": "VISIBLE"},
    {"block_state_updates": True, "blocking_overlay": "BUYIN_APPLICATION"},
    {"block_state_updates": True, "blocking_overlay": "LUCKY_BOMB_TRANSITION"},
])
def test_mode_breaks_adapter_and_causal_pairing_together(modes):
    adapter, causal = AA8StateAdapterV2(), CausalWagersV2()
    for frame in (1, 2):
        adapter.observe(observation(frame))
    # A pre-modal action waiting for cash cannot consume insurance debits later.
    adapter.observe(observation(3, glyph="aggressive"))
    assert adapter.pending
    blocked = observation(4, value="94", modes=modes, glyph="check")
    state = adapter.observe(blocked)
    ledger = causal.observe(blocked, adapter, {})
    assert state["observation_blocked"]
    assert state["observed_epoch"] is None
    assert not adapter.pending and not adapter.cash
    assert not adapter.actions
    assert ledger["wagers"] is None
    assert not ledger["strategy_eligible"]
    for frame in range(5, 21):
        adapter.observe(observation(frame, value="94"))
    assert not adapter.actions, "pre-modal glyph cannot reappear after modal exit"


def test_bombguard_positive_is_consumed_by_adapter_not_betting():
    base = SimpleNamespace(recognize=lambda image: {
        "insurance": "UNKNOWN", "block_state_updates": False})
    rng = np.random.default_rng(71)
    image = rng.integers(0, 256, (1080, 498, 3), dtype=np.uint8)
    guard = BombGuard(base, image)
    # Controlled detector positive tests wiring, not synthetic title accuracy.
    guard.bomb = SimpleNamespace(recognize=lambda image: {
        "critical_hit_animation": True, "score": 1., "strategy_eligible": False})
    modes = guard.recognize(image)
    assert modes["critical_hit_title"]["critical_hit_animation"] is True
    adapter = AA8StateAdapterV2()
    blocked = observation(1, modes=modes, glyph="check")
    assert adapter.observe(blocked)["observation_blocked"]
    assert not adapter.actions


def test_static_rules_not_triggers_or_automatic_cash_categories():
    adapter = AA8StateAdapterV2()
    state = adapter.observe(observation(1, modes={
        "mushroom_rule": "3BB", "critical_hit_rule": "7BB",
        "mushroom_trigger": "UNKNOWN", "critical_hit_trigger": "UNKNOWN"}))
    assert not state.get("observation_blocked", False)
    assert separate_cash("27", "21")["unallocated_difference"] == "6"
    assert separate_cash("623", "610")["allocations"] == []


def test_refill_stays_unallocated_through_adapter_and_causal_output():
    before, after = observation(1), observation(2, value="98")
    before["stacks"]["1"]["value"] = "88"
    after["stacks"]["1"]["value"] = "280"
    comparison = posting_comparison(before, after)
    adapter = AA8StateAdapterV2()
    adapter.observe(before)
    adapter.observe(after)
    adapter.epoch_events.append({"epoch": "e", "frame": 2,
                                 "status": "MULTI_POST_DEAL_CANDIDATE",
                                 "posting_comparison": comparison})
    tracker = CausalWagersV2()
    after["special_modes"] = {"insurance": "VISIBLE"}
    value = tracker.observe(after, adapter, {})
    assert len(value["unallocated_credits"]) == 1
    credit = value["unallocated_credits"][0]
    assert credit["amount"] == "192" and credit["slot"] == "1"
    assert credit["semantics"] == "UNALLOCATED_NOT_PROFIT"
    assert value["wagers"] is None
