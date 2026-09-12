import pytest

from tools.verify_aa8_strategy_state_development import compare, values


def prediction(frame=1):
    return {
        "frame": frame, "source_sha256": "a" * 64, "dealer_seat": 7,
        "observed_state_v2": {"observed_epoch": "h1"},
        "hand_ledger_v2": {
            "observed_total": "29", "unallocated_difference": "6",
        },
        "action_line": "unopened",
    }


def checkpoint(frame=1):
    return {
        "frame": frame, "source_sha256": "a" * 64, "epoch": "h1",
        "dealer": 7, "commitment_total": "29",
        "unallocated_difference": "6", "action_line": "unopened",
    }


def test_all_candidate_fields_are_compared_without_canonical_claim():
    result = compare({1: prediction()}, {"checkpoints": [checkpoint()]})
    assert len(result) == 5
    assert all(item["match"] for item in result)
    assert set(values(prediction())) == {
        "epoch", "dealer", "commitment_total", "unallocated_difference",
        "action_line",
    }


def test_wrong_value_is_a_failure_not_unknown_success():
    row = prediction()
    row["action_line"] = None
    result = compare({1: row}, {"checkpoints": [checkpoint()]})
    failed = [item for item in result if not item["match"]]
    assert failed == [{
        "frame": 1, "field": "action_line", "expected": "unopened",
        "actual": None, "match": False,
    }]


def test_source_hash_mismatch_rejected():
    item = checkpoint()
    item["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source mismatch"):
        compare({1: prediction()}, {"checkpoints": [item]})
