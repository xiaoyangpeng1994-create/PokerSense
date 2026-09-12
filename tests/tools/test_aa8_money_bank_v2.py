import numpy as np
import pytest

from tools.aa8_money_bank_v2 import deduplicate, labelled_glyphs


def row(**kwargs):
    return {"frame": 1500, "pool": "first", "field": "stack_0",
            "value": "255", **kwargs}


def test_dedup_preserves_source_hash():
    assert deduplicate([row(), row(review_source_sha256="abc")]) == [
        row(review_source_sha256="abc")]


def test_conflicting_labels_reject():
    with pytest.raises(ValueError, match="conflicting"):
        deduplicate([row(), row(value="254")])


@pytest.mark.parametrize("frame", [9000, 18000, 24000, -1])
def test_other_ranges_not_read(frame):
    with pytest.raises(ValueError, match="development"):
        deduplicate([row(frame=frame)])


def test_missing_display_not_invented_zero():
    glyphs, reason = labelled_glyphs({"status": "NOT_APPLICABLE"}, None)
    assert not glyphs and reason == "not_applicable_no_numeric_display"


def test_bad_segmentation_explicit():
    glyphs, reason = labelled_glyphs("100", np.zeros((18, 40), np.uint8))
    assert not glyphs and reason


@pytest.mark.parametrize("value", ["-1", "2.5", "NaN", None])
def test_bad_money_not_trained(value):
    with pytest.raises(ValueError):
        labelled_glyphs(value, None)
