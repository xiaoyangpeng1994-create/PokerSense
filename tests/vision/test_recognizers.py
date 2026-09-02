"""Tests for card/amount/action recognizers using synthetic fixtures."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poker_engine.core.enums import ActionType
from poker_engine.perceptual.vision.action_recognizer import (
    ActionTemplateSet,
    TemplateActionRecognizer,
)
from poker_engine.perceptual.vision.amount_recognizer import (
    DigitTemplateSet,
    TemplateAmountRecognizer,
    segment_characters,
)
from poker_engine.perceptual.vision.corner_glyph_recognizer import (
    CornerGlyphCardRecognizer,
    CornerGlyphGeometry,
    CornerGlyphTemplateSet,
    isolate_glyph,
)


from .fixtures.synthetic import render_card, render_digit

# The synthetic renders draw a 60x84 card whose corner index sits lower and
# wider than real WePoker art, so they need their own corner geometry.
SYNTHETIC_GEOMETRY = CornerGlyphGeometry(
    rank_band=(0.10, 0.46), suit_band=(0.52, 0.85), x_window=(0.05, 0.50)
)


def _card_templates():
    """Corner-glyph templates cut from the synthetic renders themselves."""
    rank_labels = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    suit_labels = ["S", "H", "D", "C"]
    geom = SYNTHETIC_GEOMETRY
    return CornerGlyphTemplateSet(
        rank_templates={
            label: isolate_glyph(render_card(label, "S"), geom.rank_band, geom)
            for label in rank_labels
        },
        suit_templates={
            suit: isolate_glyph(render_card("A", suit), geom.suit_band, geom)
            for suit in suit_labels
        },
        version="v1",
        geometry=geom,
    )


def _digit_templates():
    templates = {d: render_digit(d) for d in "0123456789"}
    return DigitTemplateSet(templates=templates, version="v1")


def _action_templates():
    labels = ["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALLIN"]
    templates = {}
    for label in labels:
        img = np.full((24, 80, 3), 255, dtype=np.uint8)
        cv2.putText(img, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        templates[label] = img
    return templates


# ---------- card recognizer ----------

def test_card_recognizes_ace_spades():
    rec = CornerGlyphCardRecognizer(_card_templates())
    r = rec.recognize(render_card("A", "S"))
    assert r.value is not None
    assert r.value[0].rank.value == "A"
    assert r.value[0].suit.value == "s"


def test_card_recognizes_ten_hearts():
    rec = CornerGlyphCardRecognizer(_card_templates())
    r = rec.recognize(render_card("T", "H"))
    assert r.value is not None
    assert r.value[0].rank.value == "T"


def test_card_empty_returns_none():
    rec = CornerGlyphCardRecognizer(_card_templates())
    r = rec.recognize(render_digit(""))  # blank-ish image
    assert r.value is None or r.raw_score < 0.5


def test_card_deterministic():
    rec = CornerGlyphCardRecognizer(_card_templates())
    a = rec.recognize(render_card("K", "D"))
    b = rec.recognize(render_card("K", "D"))
    assert a.value == b.value
    assert a.raw_score == b.raw_score


# ---------- amount recognizer ----------

def test_amount_single_digit():
    rec = TemplateAmountRecognizer(_digit_templates())
    r = rec.recognize(render_digit("7"))
    assert r.value is not None
    assert str(r.value) == "7"


def test_amount_blank_returns_none():
    rec = TemplateAmountRecognizer(_digit_templates())
    blank = np.full((24, 40, 3), 255, dtype=np.uint8)
    r = rec.recognize(blank)
    assert r.value is None


def _multi_char_amount_templates():
    from .fixtures.synthetic import render_char

    return DigitTemplateSet(
        {d: render_char(d) for d in "0123456789."}, version="v1",
    )


@pytest.mark.parametrize("text", ["10", "125", "850", "1000", "125.5"])
def test_amount_multi_char(text):
    from .fixtures.synthetic import render_number

    rec = TemplateAmountRecognizer(_multi_char_amount_templates())
    r = rec.recognize(render_number(text))
    assert r.value is not None
    assert str(r.value) == text


def test_amount_multi_char_white_on_dark():
    def render(text):
        width = max(50, 45 * len(text))
        image = np.full((50, width, 3), (30, 80, 65), dtype=np.uint8)
        cv2.putText(
            image, text, (10, 39), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return image

    templates = DigitTemplateSet(
        {
            digit: segment_characters(render(digit))[0]
            for digit in "0123456789"
        },
        version="dark-v1",
    )
    rec = TemplateAmountRecognizer(templates)

    result = rec.recognize(render("790"))

    assert result.value is not None
    assert str(result.value) == "790"


# ---------- action recognizer ----------

def test_action_recognizes_label():
    templates = ActionTemplateSet(
        templates={
            ActionType.CALL: _action_templates()["CALL"],
            ActionType.FOLD: _action_templates()["FOLD"],
        },
        version="v1",
    )
    rec = TemplateActionRecognizer(templates)
    r = rec.recognize(_action_templates()["CALL"], slot_id=3)
    assert r.value is ActionType.CALL


# ---------- deep immutability of template sets ----------

def test_card_template_set_deep_immutable():
    ts = _card_templates()
    # mapping is read-only
    with pytest.raises(TypeError):
        ts.rank_templates["A"] = render_card("2", "S")  # cannot reassign
    # ndarray values are write-protected (copied, not shared references)
    arr = ts.rank_templates["A"]
    with pytest.raises(ValueError):
        arr[0, 0] = 255


def test_digit_template_set_deep_immutable():
    ts = _digit_templates()
    with pytest.raises(TypeError):
        ts.templates["9"] = render_digit("0")
    with pytest.raises(ValueError):
        ts.templates["0"][0, 0] = 255


def test_action_template_set_deep_immutable():
    tmpl = _action_templates()
    ts = ActionTemplateSet(
        templates={ActionType.CALL: tmpl["CALL"], ActionType.FOLD: tmpl["FOLD"]},
        version="v1",
    )
    with pytest.raises(TypeError):
        ts.templates[ActionType.CALL] = tmpl["BET"]
    with pytest.raises(ValueError):
        ts.templates[ActionType.CALL][0, 0] = 255


def test_template_set_does_not_share_source_array():
    # Mutating the ORIGINAL array must not affect the frozen template.
    raw = render_card("A", "S")
    ts = CornerGlyphTemplateSet(
        rank_templates={"A": raw}, suit_templates={"S": raw}, version="v1"
    )
    raw[0, 0] = 123  # mutate the source
    # the frozen copy must be unchanged (was copied, not aliased)
    assert (ts.rank_templates["A"][0, 0] != 123).all()


def test_template_set_rejects_non_ndarray_value():
    # A template set must contain only ndarray image values.
    bad = {"A": "not-an-array", "S": render_card("A", "S")}
    with pytest.raises(TypeError):
        CornerGlyphTemplateSet(
            rank_templates=bad,
            suit_templates={"S": render_card("A", "S")},
            version="v1",
        )


def test_template_array_cannot_re_enable_writes():
    # bytes-backed: the WRITEABLE flag cannot be re-enabled.
    ts = _card_templates()
    arr = ts.rank_templates["A"]
    with pytest.raises(ValueError):
        arr.setflags(write=True)


def test_recognition_result_stable_after_external_reference_mutation():
    # Recognition must NOT change if someone mutates the source ndarray AFTER
    # the template set is built (frozen copy is bytes-backed, not aliased).
    src = render_card("A", "S")
    ts = CornerGlyphTemplateSet(
        rank_templates={"A": src}, suit_templates={"S": src}, version="v1"
    )

    rec = CornerGlyphCardRecognizer(ts)
    before = rec.recognize(render_card("A", "S")).value

    # mutate the ORIGINAL source array entirely
    src[:] = 0

    after = rec.recognize(render_card("A", "S")).value
    # the frozen template is a bytes-backed copy, so the result is unchanged
    assert before == after
    assert (ts.rank_templates["A"] != 0).any()


# ---------- raw score finite [0,1] fail fast (REVISE v13 blocker 4) ----------

def test_card_recognition_rejects_nan_raw_score():
    from poker_engine.perceptual.vision.protocols import CardRecognition

    with pytest.raises(ValueError):
        CardRecognition(value=None, raw_score=float("nan"), slots=())


def test_card_recognition_rejects_out_of_range():
    from poker_engine.perceptual.vision.protocols import CardRecognition

    with pytest.raises(ValueError):
        CardRecognition(value=None, raw_score=1.5, slots=())


def test_amount_recognition_rejects_inf():
    from poker_engine.perceptual.vision.protocols import AmountRecognition

    with pytest.raises(ValueError):
        AmountRecognition(value=None, raw_score=float("inf"))


def test_action_recognition_rejects_negative():
    from poker_engine.perceptual.vision.protocols import ActionRecognition

    with pytest.raises(ValueError):
        ActionRecognition(value=None, raw_score=-0.1)


def test_card_slot_result_rejects_nan_rank_score():
    from poker_engine.perceptual.vision.protocols import CardSlotResult

    with pytest.raises(ValueError):
        CardSlotResult(rank_score=float("nan"), suit_score=0.5)


def test_recognition_trace_rejects_nan_raw_score():
    from poker_engine.core.observation import ValidationStatus
    from poker_engine.perceptual.vision.trace import RecognitionTrace

    with pytest.raises(ValueError):
        RecognitionTrace(
            frame_seq=0, roi_key="r", slot_id=None, recognizer_name="c",
            recognizer_version="1", raw_score=float("nan"), confidence=0.5,
            validation_status=ValidationStatus.VALID, manifest_sha="s",
        )


# ---------- finite [0,1] on street / calibrated / calibrator entries ----------

def test_street_recognition_rejects_nan_raw_score():
    from poker_engine.core.observation import ValidationStatus
    from poker_engine.perceptual.vision.protocols import StreetRecognition

    with pytest.raises(ValueError):
        StreetRecognition(street=None, status=ValidationStatus.VALID,
                          raw_score=float("nan"), evidence=())


def test_calibrated_confidence_rejects_out_of_range():
    from poker_engine.perceptual.vision.protocols import CalibratedConfidence

    with pytest.raises(ValueError):
        CalibratedConfidence(confidence=1.5)


def test_calibrated_confidence_rejects_inf():
    from poker_engine.perceptual.vision.protocols import CalibratedConfidence

    with pytest.raises(ValueError):
        CalibratedConfidence(confidence=float("inf"))


def test_calibration_bins_map_rejects_nan():
    from poker_engine.perceptual.vision.calibration import CalibrationBins

    bins = CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9))
    with pytest.raises(ValueError):
        bins.map(float("nan"))


def test_calibration_bins_map_rejects_out_of_range():
    from poker_engine.perceptual.vision.calibration import CalibrationBins

    bins = CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9))
    with pytest.raises(ValueError):
        bins.map(1.5)


def test_calibrator_calibrate_rejects_nan():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    cal = ConfidenceCalibrator("c", 1, CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9)))
    with pytest.raises(ValueError):
        cal.calibrate(float("nan"))


def test_calibrator_should_abstain_rejects_out_of_range():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    cal = ConfidenceCalibrator("c", 1, CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9)),
                               abstain_floor=0.5)
    with pytest.raises(ValueError):
        cal.should_abstain(2.0)
