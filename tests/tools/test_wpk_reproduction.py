"""Reproduction must retain scores/data, not just the aggregate pass label."""

from copy import deepcopy

import numpy as np

from tools.verify_wpk_reproduction import compare_npz, compare_replay


def test_npz_comparison_detects_array_changes(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, x=np.array([1., 2.]), labels=np.array(["A", "K"]))
    b.write_bytes(a.read_bytes())
    result = compare_npz(a, b)
    assert result["file_bytes_equal"] and result["arrays_equal"]
    np.savez_compressed(b, x=np.array([1., 3.]), labels=np.array(["A", "K"]))
    assert compare_npz(a, b)["different_arrays"] == ["x"]


def test_npz_comparison_rejects_different_keys_and_dtypes(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, x=np.array([1], dtype=np.float32))
    np.savez_compressed(b, x=np.array([1], dtype=np.float64), extra=np.array([2]))
    result = compare_npz(a, b)
    assert not result["array_keys_equal"]
    assert not result["arrays_equal"]


def test_replay_time_can_change_but_scores_and_spec_cannot():
    old = {"spec_sha256": "original", "summary": {"passed": True},
           "checkpoints": [{"score": .75}], "elapsed_s": 100}
    new = deepcopy(old)
    new["elapsed_s"] = 200
    assert all(compare_replay(old, new).values())
    new["checkpoints"][0]["score"] = .76
    assert not compare_replay(old, new)["checkpoints_exactly_equal"]
    new["spec_sha256"] = "modified"
    assert not compare_replay(old, new)["same_frozen_spec"]
