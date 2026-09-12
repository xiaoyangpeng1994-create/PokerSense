import numpy as np

from tools.aa8_participation_v2 import normalize_waiting


def test_text_translation_normalizes_without_gate_relaxation():
    first = np.zeros((22, 76), np.uint8)
    for x in range(10, 60, 10):
        first[5:15, x:x + 3] = 1
    second = np.zeros_like(first)
    second[2:, 3:] = first[:-2, :-3]
    assert np.array_equal(normalize_waiting(first), normalize_waiting(second))
    assert normalize_waiting(first).any()


def test_blank_and_solid_and_tiny_rejected():
    assert not normalize_waiting(np.zeros((22, 76), np.uint8)).any()
    assert not normalize_waiting(np.ones((22, 76), np.uint8)).any()
    tiny = np.zeros((22, 76), np.uint8)
    tiny[4:8, 10:14] = 1
    assert not normalize_waiting(tiny).any()
