import numpy as np
import pytest

from tools.aa8_money_center_bank_v2 import CaptureBank, append_unique


def test_exact_duplicates_not_retrained_and_conflicts_rejected():
    features, labels, origins = [], [], []
    feature = np.zeros((28, 28), np.float32)
    assert append_unique(features, labels, origins, feature, "0", {"frame": 1})
    assert not append_unique(features, labels, origins, feature, "0", {"frame": 2})
    with pytest.raises(ValueError, match="conflicting"):
        append_unique(features, labels, origins, feature, "8", {"frame": 3})


def test_capture_stub_never_labels_or_modifies_crop():
    bank = CaptureBank()
    value = np.ones((20, 30, 3), np.uint8)
    read = bank.diagnose(value)
    assert read.value is None and read.raw_text is None
    assert np.array_equal(bank.captured, value)
    value[:] = 0
    assert bank.captured.any()
