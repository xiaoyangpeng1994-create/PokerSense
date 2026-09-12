from dataclasses import dataclass

import numpy as np

from tools.aa8_visual_pipeline import ExactPatchCache


@dataclass(frozen=True)
class Read:
    value: str


class Bank:
    def __init__(self):
        self.calls = 0

    def diagnose(self, patch):
        self.calls += 1
        return Read(str(int(patch.sum())))


def test_only_identical_current_pixels_reused():
    bank = Bank()
    cache = ExactPatchCache(bank)
    first = np.zeros((2, 2), np.uint8)
    assert cache.read("pot", first)["value"] == "0"
    assert cache.read("pot", first.copy())["value"] == "0"
    assert bank.calls == 1
    changed = first.copy()
    changed[0, 0] = 1
    assert cache.read("pot", changed)["value"] == "1"
    assert bank.calls == 2


def test_missing_field_does_not_reuse_old_number():
    cache = ExactPatchCache(Bank())
    cache.read("pot", np.ones((2, 2), np.uint8))
    assert cache.read("pot", None)["value"] is None
    assert "pot" not in cache.entries
