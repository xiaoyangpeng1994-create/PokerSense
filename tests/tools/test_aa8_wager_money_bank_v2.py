import pytest

from tools.aa8_wager_money_bank_v2 import REVIEW_SHA, validate_review


def test_only_exact_reviewed_development_source_allowed():
    validate_review({"role": "development", "global_frame": 2503, "sha256": REVIEW_SHA})


@pytest.mark.parametrize("change", [
    {"role": "holdout"}, {"global_frame": 2550}, {"sha256": "f" * 64}])
def test_no_role_or_frame_or_hash_substitution(change):
    value = {"role": "development", "global_frame": 2503,
             "sha256": REVIEW_SHA, **change}
    with pytest.raises(ValueError, match="explicit reviewed"):
        validate_review(value)
