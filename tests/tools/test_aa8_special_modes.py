import numpy as np
import pytest

from tools.aa8_special_modes import (
    REGIONS, SpecialModeReader, numeric_candidate, patch, score, separate_cash,
)


def test_unknown_gap_is_not_rake_or_mushroom():
    result = separate_cash("629", "623")
    assert result["unallocated_difference"] == "6"
    assert result["allocations"] == []
    assert not result["automatic_rake_inference"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", 6, None])
def test_invalid_cash(value):
    with pytest.raises(ValueError):
        separate_cash(value, "0")


def test_literal_rule_is_not_cash_evidence():
    with pytest.raises(ValueError):
        separate_cash("6", "0", [{
            "kind": "mushroom", "amount": "6",
            "basis": "rule_label", "evidence_id": "frame1"}])


def test_explicit_movement_and_duplicate_guard():
    item = {"kind": "insurance_premium", "amount": "6",
            "basis": "verified_cash_movement", "evidence_id": "ledger1"}
    assert separate_cash("19", "0", [item])["unallocated_difference"] == "13"
    with pytest.raises(ValueError):
        separate_cash("19", "0", [item, item])


def test_negative_residual_retained():
    assert separate_cash("2", "3")["unallocated_difference"] == "-1"


def test_no_matching_template_does_not_mean_inactive():
    rng = np.random.default_rng(8)
    witness = rng.integers(0, 256, (1080, 498, 3), dtype=np.uint8)
    reader = SpecialModeReader({k: witness for k in REGIONS})
    result = reader.recognize(np.zeros_like(witness))
    assert result["insurance"] == "UNKNOWN"
    assert result["mushroom_trigger"] == result["critical_hit_trigger"] == "UNKNOWN"
    assert result["wager_actions"] == []
    positive = reader.recognize(witness)
    assert positive["insurance_purchase_label"] == "投保6"
    assert positive["wager_actions"] == []
    assert positive["mushroom_rule"] == "3BB"
    assert positive["critical_hit_trigger"] == "UNKNOWN"


def test_canvas_and_uniform_guard():
    with pytest.raises(ValueError):
        patch(np.zeros((1080, 500, 3), np.uint8), REGIONS["insurance_banner"])
    assert score(np.zeros((10, 10), np.uint8), np.zeros((10, 10), np.uint8)) == 0


def test_exact_label_does_not_fill_unreadable_amount():
    rng = np.random.default_rng(14)
    witness = rng.integers(0, 256, (1080, 498, 3), dtype=np.uint8)
    reader = SpecialModeReader({k: witness for k in REGIONS})
    result = reader.recognize(witness)
    assert result["insurance_purchase_label"] == "投保6"
    assert result["insurance_purchase_prefix"] == "投保"
    assert result["insurance_amount_candidate"]["value"] is None
    assert result["mushroom_counter_candidate"]["value"] is None
    assert not result["insurance_amount_is_verified_cash_movement"]


def test_clipped_digits_abstain_before_bank():
    class Bank:
        def diagnose(self, value):
            raise AssertionError("clipped input must not be OCR corrected")

    image = np.zeros((1080, 498, 3), np.uint8)
    image[111:128, 60:65] = 255
    assert numeric_candidate(image, (58, 111, 96, 128), Bank()) == {
        "value": None, "reason": "clipped_or_border"}
