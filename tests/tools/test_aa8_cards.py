"""Temporal and privacy contracts, not visual model accuracy evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools import aa8_cards as module


class FakeAdapter:
    def __init__(self, recognizer):
        self.resets = []
        self.values = {("hero", 0): "5d", ("hero", 1): "6d"}

    def reset(self, key=None):
        self.resets.append(key)

    def begin_frame(self, *args):
        pass

    def recognize(self, crop, key):
        self.last_crop = crop.copy()
        value = self.values.get(key)
        return SimpleNamespace(value=(value,) if value else None, raw_score=.9)


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setattr(module, "FusedCardRecognizer", lambda *a, **k: None)
    monkeypatch.setattr(module, "FusedCardRecognizerAdapter", FakeAdapter)
    monkeypatch.setattr(module, "face_card_support", lambda im, rect: (
        bool(im[0, 0, 0]), "mock_support"))
    monkeypatch.setattr(module, "locate_face_card", lambda im, rect: (None, "empty"))
    return module.AA8CardReader(None)


def image():
    return np.full((1080, 498, 3), 100, np.uint8)


def test_current_adjacent_confirmation_and_no_strategy(reader):
    assert reader.read(image(), 1, .1, "dev")["hero"] is None
    result = reader.read(image(), 2, .2, "dev")
    assert result["hero"] == ["5d", "6d"]
    assert not result["strategy_eligible"]
    assert not result["model_calibrated_for_aa8"]


@pytest.mark.parametrize("frame,pts,source", [
    (4, .3, "dev"), (3, 2., "dev"), (3, .2, "dev"), (3, .3, "new")])
def test_discontinuity_resets_without_old_cards(reader, frame, pts, source):
    reader.read(image(), 1, .1, "dev")
    reader.read(image(), 2, .2, "dev")
    result = reader.read(image(), frame, pts, source)
    assert result["gap_reset"]
    assert result["hero"] is None


@pytest.mark.parametrize("frame", [1, 2])
def test_duplicate_backwards_reject_and_clear(reader, frame):
    reader.read(image(), 1, .1, "dev")
    reader.read(image(), 2, .2, "dev")
    with pytest.raises(ValueError, match="duplicate/backwards"):
        reader.read(image(), frame, .3, "dev")
    assert not reader.values


def test_absent_current_faces_clear_and_require_new_confirmation(reader):
    reader.read(image(), 1, .1, "dev")
    reader.read(image(), 2, .2, "dev")
    result = reader.read(np.zeros_like(image()), 3, .3, "dev")
    assert result["hero"] is None
    assert reader.read(image(), 4, .4, "dev")["hero"] is None


def test_model_unknown_cannot_reuse_last_card(reader):
    reader.read(image(), 1, .1, "dev")
    reader.read(image(), 2, .2, "dev")
    reader.cards.values[("hero", 1)] = None
    assert reader.read(image(), 3, .3, "dev")["hero"] is None


def test_duplicate_identity_across_hero_board_abstains(reader, monkeypatch):
    monkeypatch.setattr(module, "locate_face_card", lambda im, rect: (rect, "mock"))
    reader.cards.values[("board", 0)] = "5d"
    reader.read(image(), 1, .1, "dev")
    result = reader.read(image(), 2, .2, "dev")
    assert result["hero"] is None
    assert result["board_slots"] == [None] * 5
    assert result["reason"] == "duplicate_card_identity_abstention"


@pytest.mark.parametrize("frame,pts,source", [
    (True, .1, "dev"), (1, float("nan"), "dev"), (1, -.1, "dev"), (1, .1, "")])
def test_bad_identity_rejected(reader, frame, pts, source):
    with pytest.raises(ValueError, match="valid frame"):
        reader.read(image(), frame, pts, source)


def test_unsupported_canvas_clears_current_cards(reader):
    reader.read(image(), 1, .1, "dev")
    result = reader.read(np.zeros((20, 20, 3), np.uint8), 2, .2, "dev")
    assert result["reason"] == "unsupported_canvas"
    assert result["hero"] is None


def test_preprocessing_is_explicit_opt_in_and_never_changes_source(reader):
    assert reader.preprocessing is None
    reader = module.AA8CardReader(None, preprocessing="gaussian_050")
    frame = image()
    frame[940, 252] = 255
    original = frame.copy()
    result = reader.read(frame, 1, .1, "dev")
    np.testing.assert_array_equal(frame, original)
    assert result["preprocessing"] == "gaussian_050"
    assert not result["model_calibrated_for_aa8"]
    assert not result["strategy_eligible"]


def test_preprocessing_cannot_bypass_absent_face_gate(reader):
    reader = module.AA8CardReader(None, preprocessing="gaussian_050")
    result = reader.read(np.zeros_like(image()), 1, .1, "dev")
    assert result["hero"] is None
    assert not hasattr(reader.cards, "last_crop")


def test_unknown_preprocessing_rejected():
    with pytest.raises(ValueError, match="explicit fixed"):
        module.AA8CardReader(None, preprocessing="choose_best")
