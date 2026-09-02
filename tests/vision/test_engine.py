"""VisionEngine integration: synthetic frame + TableMap -> RawObservation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from poker_engine.core.enums import ActionType, Street
from poker_engine.core.observation import ValidationStatus
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision import (
    BoardSlotLayout,
    CalibrationBins,
    CardSubROI,
    ConfidenceCalibrator,
    HeroSlotLayout,
    ROIKind,
    ROI,
    TableMap,
    VisionAssetManifest,
    VisionEngine,
)
from poker_engine.perceptual.vision.action_recognizer import (
    ActionTemplateSet,
    TemplateActionRecognizer,
)
from poker_engine.perceptual.vision.amount_recognizer import (
    DigitTemplateSet,
    TemplateAmountRecognizer,
)
from poker_engine.perceptual.vision.board_slot_detector import (
    TemplateBoardSlotDetector,
)
from poker_engine.perceptual.vision.corner_glyph_recognizer import (
    CornerGlyphCardRecognizer,
    CornerGlyphGeometry,
    CornerGlyphTemplateSet,
    isolate_glyph,
)

from poker_engine.perceptual.vision.street_detector import TemplateStreetDetector

from .fixtures.synthetic import render_card, render_digit, render_empty_slot

UTC = timezone.utc

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
    return DigitTemplateSet(
        templates={d: render_digit(d) for d in "0123456789"}, version="v1",
    )


def _cal(name, abstain_floor=None):
    return ConfidenceCalibrator(
        name=name, version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=abstain_floor,
    )


def _manifest():
    return VisionAssetManifest(
        platform_id="wpk", layout_id="6max",
        card_layout_version=1, template_set_version="sha-x",
        calibration_version=1,
        recognizer_versions={"card": "1", "amount": "1", "stack": "1",
                             "dealer": "1", "street": "1", "action": "1",
                             "board": "1"},
    )


def _engine(bet_size_semantics="global"):
    board_layout = BoardSlotLayout(
        layout_id="b", version=1,
        slots=tuple(CardSubROI(x=i * 0.2, y=0.0, width=0.18, height=1.0)
                    for i in range(5)),
    )
    hero_layout = HeroSlotLayout(
        layout_id="h", version=1,
        slots=(CardSubROI(x=0.0, y=0.0, width=0.48, height=1.0),
               CardSubROI(x=0.5, y=0.0, width=0.48, height=1.0)),
    )
    card = CornerGlyphCardRecognizer(_card_templates())
    board_det = TemplateBoardSlotDetector(
        board_layout, empty_min_evidence=0.3, card_min_presence=0.25)
    amount = TemplateAmountRecognizer(_digit_templates())
    action_tmpl = {ActionType.CALL: render_digit("CALL")}
    action = TemplateActionRecognizer(
        ActionTemplateSet(templates=action_tmpl, version="v1"))
    calibrators = {
        "card": _cal("card", abstain_floor=0.5),
        "amount": _cal("amount"),
        "stack": _cal("stack"),
        "dealer": _cal("dealer"),
        "street": _cal("street"),
        "action": _cal("action"),
        "board": _cal("board"),
    }
    return VisionEngine(
        board_layout=board_layout, hero_layout=hero_layout,
        card_recognizer=card, board_slot_detector=board_det,
        street_detector=TemplateStreetDetector(), amount_recognizer=amount,
        stack_recognizer=amount,
        action_recognizer=action, calibrators=calibrators,
        manifest=_manifest(), bet_size_semantics=bet_size_semantics,
    )


def _table_map():
    # board ROI at top, hero ROI at bottom, pot at middle, bet_size global
    return TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(
            ROI(kind=ROIKind.BOARD_CARDS, x=0.0, y=0.0, width=1.0, height=0.25),
            ROI(kind=ROIKind.HERO_CARDS, x=0.0, y=0.7, width=0.4, height=0.25),
            ROI(kind=ROIKind.POT, x=0.4, y=0.45, width=0.2, height=0.1),
            ROI(kind=ROIKind.BET_SIZE, x=0.4, y=0.55, width=0.2, height=0.07),
        ),
    )


def _build_frame(board_slots, hero_slots, pot_text="10", bet_text="5"):
    # compose a 600x400 frame: board bands on top, hero on bottom-left
    frame = np.full((400, 600, 3), 200, dtype=np.uint8)
    # board: 5 slots across top band.
    #   - empty slot -> fill the FULL sub-ROI dark (weak presence)
    #   - card        -> place original-size card (no resize breaks match)
    sub_w = int(0.18 * 600)
    sub_h = int(0.25 * 400)
    for i, slot_img in enumerate(board_slots):
        x = int(i * 0.2 * 600)
        if float(slot_img.mean()) < 60.0:
            # empty slot: fill the whole sub-ROI uniformly dark (weak presence)
            frame[0:sub_h, x:x + sub_w] = _resize(slot_img, sub_w, sub_h)
        else:
            # card: place original-size card (no resize) inside the sub-ROI
            frame[8:8 + slot_img.shape[0], x + 8:x + 8 + slot_img.shape[1]] = slot_img
    # hero: 2 slots bottom-left (original-size cards)
    for i, slot_img in enumerate(hero_slots):
        x = int(0.06 * 600) + i * int(0.2 * 600)
        y = int(0.72 * 400)
        frame[y:y + slot_img.shape[0], x:x + slot_img.shape[1]] = slot_img
    # pot text
    pot_img = render_digit(pot_text)
    py0, py1 = int(0.45 * 400), int(0.45 * 400) + 24
    px0, px1 = int(0.4 * 600), int(0.4 * 600) + 40
    frame[py0:py1, px0:px1] = pot_img
    bet_img = render_digit(bet_text)
    by0, by1 = int(0.55 * 400), int(0.55 * 400) + 24
    frame[by0:by1, px0:px1] = bet_img
    return frame


def _resize(img, w, h):
    import cv2

    return cv2.resize(img, (w, h))


def _frame(img):
    return Frame(
        frame_seq=0,
        timestamp=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        window_id="t", window_rect=WindowRect(0, 0, 600, 400),
        image=img, width=600, height=400,
    )


def test_engine_preflop_empty_board():
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.street.value is Street.PREFLOP
    assert obs.street.validation_status is ValidationStatus.VALID
    assert len(obs.hero_cards.value) == 2


def test_engine_river_full_board():
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[
            render_card("A", "S"), render_card("K", "H"),
            render_card("Q", "D"), render_card("J", "C"), render_card("T", "S"),
        ],
        hero_slots=[render_card("2", "C"), render_card("3", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.street.value is Street.RIVER
    assert obs.street.validation_status is ValidationStatus.VALID
    assert obs.board_cards.value is not None and len(obs.board_cards.value) == 5


def test_engine_bet_size_unknown_when_no_roi():
    # No BET_SIZE ROI -> bet_size recognized as UNKNOWN (no crash).
    tm = TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(
            ROI(kind=ROIKind.BOARD_CARDS, x=0.0, y=0.0, width=1.0, height=0.25),
            ROI(kind=ROIKind.HERO_CARDS, x=0.0, y=0.7, width=0.4, height=0.25),
        ),
    )
    eng = _engine()
    frame_img = _build_frame([render_empty_slot() for _ in range(5)], [])
    obs = eng.process(_frame(frame_img), tm)
    assert obs.bet_size.validation_status is ValidationStatus.UNKNOWN
    assert obs.bet_size.value is None


def test_hero_confidence_uses_min_rank_suit():
    # A strong rank + weak suit must NOT yield a falsely high hero confidence.
    from poker_engine.core.enums import Rank, Suit
    from poker_engine.core.value_objects import Card
    from poker_engine.perceptual.vision.protocols import (
        CardRecognition,
        CardSlotResult,
    )

    class _FakeCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(
                value=(Card(Rank.ACE, Suit.SPADES),),
                raw_score=0.1,  # min(rank, suit) = min(0.99, 0.1) = 0.1
                slots=(CardSlotResult(rank_score=0.99, suit_score=0.1),),
            )

    eng = _engine()
    eng._card = _FakeCardRecognizer()

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # raw feature 0.1 -> calibrator bin (0.0,0.5]->0.1, so confidence is low
    assert obs.hero_cards.confidence == 0.1


def test_slot_ordering_sorted_ascending():
    # Deliberately unsorted TableMap.rois -> slot_* must still be ascending.
    tm = TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(
            ROI(kind=ROIKind.STACK, x=0.0, y=0.8, width=0.1, height=0.05, slot_id=3),
            ROI(kind=ROIKind.STACK, x=0.0, y=0.7, width=0.1, height=0.05, slot_id=1),
            ROI(kind=ROIKind.STACK, x=0.0, y=0.6, width=0.1, height=0.05, slot_id=2),
        ),
    )
    eng = _engine()
    frame_img = _build_frame([render_empty_slot() for _ in range(5)], [])
    obs = eng.process(_frame(frame_img), tm)
    ids = [s.slot_id for s in obs.slot_stacks]
    assert ids == sorted(ids)
    assert ids == [1, 2, 3]


def test_manifest_mismatch_fails_fast():
    from poker_engine.perceptual.vision.errors import TableMapError

    eng = _engine()
    # table_map platform_id differs from manifest platform_id "wpk"
    tm = TableMap(
        platform_id="OTHER", layout_id="6max", reference_size=(600, 400),
        rois=(),
    )
    frame_img = _build_frame([render_empty_slot() for _ in range(5)], [])
    with pytest.raises(TableMapError):
        eng.process(_frame(frame_img), tm)
