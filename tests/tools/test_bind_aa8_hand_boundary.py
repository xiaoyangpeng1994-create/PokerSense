from copy import deepcopy

import pytest

from tools.bind_aa8_hand_boundary import validate_interval


def inputs():
    review = {"role": "development", "independent_holdout": False,
              "start_global_frame": 2, "end_global_frame": 4, "next_hand_post_frame": 5,
              "evidence": [{"kind": k, "frame": f} for k, f in
                           [("before_start", 1), ("first_posts", 2),
                            ("before_next_posts", 4), ("next_posts", 5)]]}
    index = {i: {"pts_seconds": str(i)} for i in range(1, 6)}
    plan = {"ranges": [{"start_inclusive": "0", "end_exclusive": "10",
                        "role": "development"}]}
    return review, index, plan


def test_adjacent_boundary_context_passes_without_mutating_inputs():
    data = inputs()
    before = deepcopy(data)
    validate_interval(*data)
    assert data == before


def test_missing_index_or_evidence_rejected():
    review, index, plan = inputs()
    del index[3]
    with pytest.raises(ValueError, match="missing"):
        validate_interval(review, index, plan)
    review, index, plan = inputs()
    review["evidence"] = []
    with pytest.raises(ValueError, match="evidence"):
        validate_interval(review, index, plan)


def test_development_cannot_be_promoted_to_holdout():
    review, index, plan = inputs()
    review["independent_holdout"] = True
    with pytest.raises(ValueError, match="development"):
        validate_interval(review, index, plan)


def test_crossing_role_boundary_rejected():
    review, index, plan = inputs()
    plan["ranges"][0]["end_exclusive"] = "4"
    with pytest.raises(ValueError, match="development"):
        validate_interval(review, index, plan)
