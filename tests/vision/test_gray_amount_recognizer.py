"""Gray bank protocol, ambiguity and bounded-source tests."""

import json

import cv2
import numpy as np
import pytest

from poker_engine.perceptual.vision.gray_amount_recognizer import (
    GrayAmountRecognizer, gray_glyphs,
)
from poker_engine.perceptual.vision.protocols import AmountRecognition
from tools.wpk_gray_amount import load_reviewed_bank
from tools.capture_card_calibration.hashing import sha256_file


def bank():
    features = []
    for digit in "0123456789":
        image = np.zeros((24, 24), np.uint8)
        cv2.putText(image, digit, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, 230, 1)
        glyphs, reason = gray_glyphs(image)
        assert reason is None and len(glyphs) == 1
        features.append(glyphs[0][0])
    return np.array(features), np.array(list("0123456789"))


def test_gray_bank_copies_source_and_augmentation_is_not_extra_real_data():
    features, labels = bank()
    before = features.copy()
    reader = GrayAmountRecognizer(features, labels, augment=True)
    assert np.array_equal(features, before)
    assert len(reader.labels) == 150
    assert not reader.vectors.flags.writeable and not reader.labels.flags.writeable
    labels[:] = "0"
    assert set(reader.labels) == set("0123456789")


@pytest.mark.parametrize("value", [0, 2, float("nan"), float("inf"), True])
def test_invalid_gate_rejected(value):
    with pytest.raises(ValueError):
        GrayAmountRecognizer(*bank(), floor=value)


def test_bank_requires_all_digits_and_finite_features():
    features, labels = bank()
    with pytest.raises(ValueError):
        GrayAmountRecognizer(features[:-1], labels[:-1])
    features[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        GrayAmountRecognizer(features, labels)


@pytest.mark.parametrize("kind", ["blank", "decimal", "clipped", "float_image"])
def test_no_guess_for_bad_or_unsupported_crops(kind):
    image = np.zeros((24, 60), np.uint8)
    if kind == "decimal":
        image[18:20, 28:30] = 255
    elif kind == "clipped":
        image[:14, 28:32] = 255
    elif kind == "float_image":
        image = image.astype(np.float32)
    result = GrayAmountRecognizer(*bank()).recognize(image)
    assert isinstance(result, AmountRecognition) and result.value is None


def test_duplicate_class_shapes_never_win_by_tiebreak():
    features, labels = bank()
    reader = GrayAmountRecognizer(np.repeat(features[3:4], 10, axis=0), labels)
    image = np.zeros((24, 24), np.uint8)
    cv2.putText(image, "3", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, 230, 1)
    assert reader.recognize(image).value is None


def make_reviewed(tmp_path):
    features, labels = bank()
    np.savez_compressed(tmp_path / "proposals.npz", features=features, labels=labels)
    (tmp_path / "inventory.json").write_text(json.dumps({
        "records": [{"label": str(label)} for label in labels]}))
    review = {"inventory_sha256": sha256_file(tmp_path / "inventory.json"),
              "npz_sha256": sha256_file(tmp_path / "proposals.npz"),
              "reviewed_ids": list(range(10)), "rejected_ids": []}
    return review


def test_unreviewed_ids_and_modified_bank_cannot_load(tmp_path):
    review = make_reviewed(tmp_path)
    path = tmp_path / "visual-review.json"
    path.write_text(json.dumps(review))
    assert isinstance(load_reviewed_bank(tmp_path), GrayAmountRecognizer)
    review["reviewed_ids"].pop()
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="every proposal"):
        load_reviewed_bank(tmp_path)
    review["reviewed_ids"].append(9)
    path.write_text(json.dumps(review))
    (tmp_path / "proposals.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bind"):
        load_reviewed_bank(tmp_path)
