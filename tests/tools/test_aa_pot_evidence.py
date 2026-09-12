import pytest

from tools.aa_pot_evidence import reconcile_pot_displays


@pytest.mark.parametrize("center", ["120", "90"])
def test_zero_title_cannot_hide_visible_collections(center):
    result = reconcile_pot_displays("0", center)
    assert result["value"] is None
    assert result["reason"] == "display_disagreement"
    assert result["title"] == "0" and result["center"] == center


def test_equal_displays_are_not_a_canonical_ledger():
    result = reconcile_pot_displays("120", "120")
    assert result["value"] == "120"
    assert not result["canonical_verified"]


@pytest.mark.parametrize("title,center", [(None, "120"), ("120", None), (None, None)])
def test_missing_value_not_filled_from_other_display(title, center):
    assert reconcile_pot_displays(title, center)["value"] is None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1"])
def test_invalid_amounts_rejected(value):
    with pytest.raises(ValueError):
        reconcile_pot_displays(value, "120")


def test_visible_wagers_explain_total_without_requiring_equal_displays():
    wagers = ["0"] * 5 + ["2", "4", "8", "0"]
    result = reconcile_pot_displays("30", "16", visible_wagers=wagers)
    assert result["value"] == "30"
    assert result["center_plus_wagers"] == "30"
    assert not result["canonical_verified"]


def test_incomplete_or_double_counted_wagers_do_not_force_balance():
    wagers = ["0"] * 5 + [None, "4", "8", "0"]
    assert reconcile_pot_displays("30", "16", visible_wagers=wagers)["value"] is None
    result = reconcile_pot_displays("120", "120", visible_wagers=["20"] * 9)
    assert result["value"] is None
