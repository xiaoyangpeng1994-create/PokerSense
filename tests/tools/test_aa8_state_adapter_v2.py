from copy import deepcopy

import pytest

from tools.aa8_state_adapter_v2 import AA8StateAdapterV2, posting_comparison


def row(frame, deal=False, value="100"):
    return {"frame": frame, "scene_supported": True, "source_sha256": "a" * 64,
            "stacks": {str(s): {"value": value} for s in range(8)},
            "pot": {"value": "0"}, "board_count": 0,
            "cards": {"hero": None, "board_slots": [None] * 5},
            "participation": {"slots": {
                str(s): {"current": "UNKNOWN"} for s in range(8)}},
            "hand_transition": {"center_deal": {"visible": deal}},
            "glyph_transitions": []}


def geometry(item, count=3):
    item["board_count"] = count
    item["cards"]["evidence"] = {f"board_{s}": {
        "rect": [110 + s * 56, 473, 53, 78] if s < count else None,
        "face_support": "nominal_face_boundary_supported" if s < count
        else "no_positive_face_background"} for s in range(5)}
    return item


def test_positive_deal_bootstrap_not_authoritative_boundary():
    adapter = AA8StateAdapterV2()
    assert adapter.observe(row(1, True))["observed_epoch"] is None
    value = adapter.observe(row(2, True))
    assert value["observed_epoch"] == "observed_deal_1"
    assert not value["authoritative_hand_boundary"]


def test_unknown_cash_cannot_be_zero_for_posting():
    before, after = row(1), row(2, True, "98")
    before["stacks"]["6"]["value"] = None
    assert posting_comparison(before, after) is None
    before["participation"]["slots"]["6"]["current"] = "WAITING_CANDIDATE"
    result = posting_comparison(before, after)
    assert result["excluded_na_slots"] == ["6"]
    assert "6" not in result["debits"]


def test_positive_refill_retained_unallocated_not_profit():
    before, after = row(5050), row(5051, True, "98")
    before["stacks"]["1"]["value"] = "88"
    after["stacks"]["1"]["value"] = "280"
    result = posting_comparison(before, after)
    assert result["credits"] == {"1": "192"}
    assert result["credit_semantics"] == "UNALLOCATED_POSSIBLE_REFILL_NOT_PROFIT"


def test_stable_positive_balance_records_unallocated_once():
    adapter = AA8StateAdapterV2()
    for f in range(1, 7):
        item = row(f)
        item["stacks"]["4"]["value"] = "0" if f < 3 else "516"
        adapter.observe(item)
    assert len(adapter.credits) == 1
    assert adapter.credits[0]["amount"] == "516"
    assert adapter.credits[0]["source"] == "stable_visual_balance_increase"
    assert adapter.credits[0]["semantics"] == "UNALLOCATED_NOT_PROFIT_OR_RAKE"


def test_visible_positive_cash_during_insurance_not_promoted_to_action():
    adapter = AA8StateAdapterV2()
    for f in range(1, 6):
        item = row(f)
        item["special_modes"] = {"insurance": "VISIBLE"}
        item["stacks"]["3"]["value"] = "15" if f < 3 else "574"
        result = adapter.observe(item)
    assert result["observation_blocked"]
    assert [c["amount"] for c in adapter.credits] == ["559"]
    assert not adapter.actions
    assert not adapter.credits[0]["hand_attribution_verified"]


def test_positive_cash_cannot_pair_across_unsupported_scene():
    adapter = AA8StateAdapterV2()
    for f in range(1, 7):
        item = row(f)
        item["scene_supported"] = f != 3
        item["stacks"]["3"]["value"] = "15" if f < 3 else "574"
        adapter.observe(item)
    assert adapter.credits == []


def test_no_deal_no_hand_even_if_balances_drop():
    adapter = AA8StateAdapterV2()
    adapter.observe(row(1))
    adapter.observe(row(2))
    assert adapter.observe(row(3, value="98"))["observed_epoch"] is None


def test_new_hand_resets_waiting_history_without_zero_fill():
    adapter = AA8StateAdapterV2()
    for f in range(1, 5):
        item = row(f, f < 3)
        item["stacks"]["6"]["value"] = None
        item["participation"]["slots"]["6"]["current"] = "WAITING_CANDIDATE"
        adapter.observe(item)
    for f in (5, 6):
        item = row(f, True, "98")
        item["participation"]["slots"]["6"]["current"] = "DEALT_IN_CANDIDATE"
        result = adapter.observe(item)
    assert result["observed_epoch"] == "observed_deal_5"
    assert result["participants"]["6"]["state"] == "active"


def test_missing_glyph_does_not_infer_check():
    adapter = AA8StateAdapterV2()
    for f in range(1, 15):
        adapter.observe(row(f, f < 3))
    assert not adapter.actions


def test_input_not_modified_and_gap_drops_stale_context():
    adapter = AA8StateAdapterV2()
    item = row(1, True)
    original = deepcopy(item)
    adapter.observe(item)
    assert item == original
    adapter.observe(row(2, True))
    assert adapter.observe(row(10))["observed_epoch"] is None


def test_positive_board_only_and_missing_board_abstains():
    adapter = AA8StateAdapterV2()
    item = row(1)
    item["board_count"] = 3
    assert adapter.observe(item)["street_candidate"] is None
    item = geometry(row(2))
    item["cards"]["board_slots"] = ["5h", "6c", "6s", None, None]
    assert adapter.observe(item)["street_candidate"] is None
    item["frame"] = 3
    assert adapter.observe(item)["street_candidate"] == "flop"
    assert adapter.observe(row(4))["street_candidate"] is None


def test_4405_flop_action_not_preflop_while_card_identity_unknown():
    adapter = AA8StateAdapterV2()
    for f in range(4402, 4406):
        item = geometry(row(f))
        item["cards"]["board_slots"] = ["7h", "5h", None, None, None]
        item["pot"]["value"] = "114" if f < 4404 else "152"
        if f >= 4404:
            item["stacks"]["3"]["value"] = "62"
        if f == 4405:
            item["glyph_transitions"] = [{"frame": f, "slot": 3, "glyph": "aggressive"}]
        state = adapter.observe(item)
    assert state["street_candidate"] == "flop"
    assert state["board_candidate"] is None
    assert adapter.actions[-1]["street"] == "flop"
    assert adapter.actions[-1]["amount"] == "38"


def test_duplicate_frame_rejected():
    adapter = AA8StateAdapterV2()
    adapter.observe(row(1))
    with pytest.raises(ValueError):
        adapter.observe(row(1))


def test_confirmed_glyph_pairs_unique_debit_and_pot_without_legal_claim():
    adapter = AA8StateAdapterV2()
    for f in range(1, 5):
        item = row(f)
        if f >= 3:
            item["stacks"]["0"]["value"] = "90"
            item["pot"]["value"] = "10"
        if f == 3:
            item["glyph_transitions"] = [{"frame": 3, "slot": 0, "glyph": "call"}]
        adapter.observe(item)
    assert len(adapter.actions) == 1
    assert adapter.actions[0]["amount"] == "10"
    assert not adapter.actions[0]["legal_action_verified"]


def test_simultaneous_debits_not_assigned_to_one_glyph():
    adapter = AA8StateAdapterV2()
    for f in range(1, 18):
        item = row(f)
        if f >= 3:
            item["stacks"]["0"]["value"] = "90"
            item["stacks"]["1"]["value"] = "90"
            item["pot"]["value"] = "20"
        if f == 3:
            item["glyph_transitions"] = [{"frame": 3, "slot": 0, "glyph": "call"}]
        adapter.observe(item)
    assert len(adapter.actions) == 1
    assert adapter.actions[0]["amount"] is None
