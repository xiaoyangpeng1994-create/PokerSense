"""Feature experiments preserve abstention and do not overwrite production gates."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.wpk_field_features_v2 import (
    DigitCandidate, FeatureActionCandidate, action_feature, normalized_digit,
    template_mask, white_mask,
)


def templates():
    result = {}
    for digit in "0123456789":
        image = np.zeros((30, 25), np.uint8)
        cv2.putText(image, digit, (3, 24), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 1)
        result[digit] = image
    return result


def test_uniform_template_letterbox_is_not_foreground_ink():
    glyph = np.zeros((15, 9), np.uint8)
    glyph[2:13, 3:5] = 210
    framed = np.full((28, 28), 120, np.uint8)
    framed[6:21, 9:18] = glyph
    assert np.any(white_mask(framed)[:, 0])
    assert np.array_equal(template_mask(framed), white_mask(glyph))
    assert np.array_equal(normalized_digit(template_mask(framed)),
                          normalized_digit(white_mask(glyph)))


def test_digit_features_are_readonly_and_do_not_mutate_templates():
    originals = templates()
    before = {key: image.copy() for key, image in originals.items()}
    reader = DigitCandidate(originals)
    assert all(np.array_equal(originals[k], before[k]) for k in originals)
    assert all(not image.flags.writeable for image in reader.templates.values())


@pytest.mark.parametrize("kind,reason", [
    ("empty", "empty_crop"), ("blank", "unsupported_glyph_count"),
    ("clipped", "edge_ink_or_clipped_glyph"), ("decimal", "punctuation_or_speck"),
])
def test_unsafe_numeric_crops_abstain(kind, reason):
    patch = np.zeros((21, 60), np.uint8)
    if kind == "empty":
        patch = None
    elif kind == "clipped":
        patch[0:12, 20:24] = 255
    elif kind == "decimal":
        patch[15:17, 20:22] = 255
    result = DigitCandidate(templates()).recognize(patch)
    assert not result["accepted_candidate"] and result["value"] is None
    assert result["reason"] == reason


def test_identical_digit_shapes_abstain_instead_of_tie_breaking():
    glyph = templates()["3"]
    reader = DigitCandidate({key: glyph for key in "0123456789"})
    result = reader.recognize(glyph)
    assert not result["accepted_candidate"]


@pytest.mark.parametrize("floor", [0, 2, float("nan"), True])
def test_invalid_digit_gate_rejected(floor):
    with pytest.raises(ValueError, match="gate"):
        DigitCandidate(templates(), floor=floor)


def test_gray_raw_frame_does_not_become_yellow_allin():
    patch = np.zeros((30, 70, 3), np.uint8)
    cv2.putText(patch, "Allin", (2, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (255, 255, 255), 1)
    assert not np.any(action_feature(patch, "all_in"))


def test_blank_full_frame_has_no_candidate_actions():
    path = Path(__file__).resolve().parents[1] / (
        "fixtures/wpk_reference_hands/field_candidate_v1.json")
    profile = json.loads(path.read_text())
    patch = np.zeros((12, 20, 3), np.uint8)
    patch[3:8, 4:8] = (0, 255, 255)
    images = {name: patch.copy() for name in
              ("fold", "all_in", "bet", "call", "check", "raise")}
    # Non-yellow classes need white glyphs for a nondegenerate template.
    for name in images:
        if name != "all_in":
            images[name][3:8, 4:8] = 255
    reader = FeatureActionCandidate(profile, images)
    results = reader.recognize(np.zeros((1080, 498, 3), np.uint8))
    assert all(not row["accepted_candidate"] for row in results.values())
    tolerant = FeatureActionCandidate(profile, images, badge_scale_tolerance=True)
    assert len(reader.variants["call"]) == 1
    assert len(tolerant.variants["call"]) == 9
    assert all(not feature.flags.writeable for feature in tolerant.variants["call"])
    assert np.array_equal(reader.templates["call"], tolerant.templates["call"])
