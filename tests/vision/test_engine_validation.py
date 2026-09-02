"""Additional Task 7B regression tests (official REVISE fixes)."""

from __future__ import annotations

import pytest

from poker_engine.core.enums import Rank, Suit, Street
from poker_engine.core.observation import ValidationStatus
from poker_engine.core.value_objects import Card
from poker_engine.perceptual.vision.board_slot_detector import empty_evidence
from poker_engine.perceptual.vision.errors import TableMapError
from poker_engine.perceptual.vision.protocols import (
    BoardSlotOccupancy,
    BoardSlotResult,
    BoardSlotsRecognition,
)

from .fixtures.synthetic import (
    render_card,
    render_empty_slot,
)

from .test_engine import (
    _build_frame,
    _engine,
    _frame,
    _table_map,
)


# ---------- BoardSlotResult / BoardSlotsRecognition invariants ----------

def test_board_slot_card_allows_none_card():
    # occupancy=CARD with card=None is a LEGAL Vision-layer intermediate state
    # (strong presence but identity not yet/never confirmed). The engine
    # promotes this to CONFLICT during cross-check.
    r = BoardSlotResult(
        slot_index=0, occupancy=BoardSlotOccupancy.CARD, card=None, raw_score=0.9,
    )
    assert r.occupancy is BoardSlotOccupancy.CARD
    assert r.card is None


def test_board_slot_empty_rejects_card():
    c = Card(Rank.ACE, Suit.SPADES)
    with pytest.raises(ValueError):
        BoardSlotResult(
            slot_index=0, occupancy=BoardSlotOccupancy.EMPTY, card=c, raw_score=0.9,
        )


def test_board_slot_index_range():
    with pytest.raises(ValueError):
        BoardSlotResult(
            slot_index=5, occupancy=BoardSlotOccupancy.EMPTY, card=None, raw_score=0.5,
        )


def test_board_slot_raw_score_range():
    with pytest.raises(ValueError):
        BoardSlotResult(
            slot_index=0, occupancy=BoardSlotOccupancy.EMPTY, card=None, raw_score=1.5,
        )


def test_board_slots_require_5_strict_order():
    slots = (
        BoardSlotResult(slot_index=1, occupancy=BoardSlotOccupancy.EMPTY,
                        card=None, raw_score=0.5),
        BoardSlotResult(slot_index=0, occupancy=BoardSlotOccupancy.EMPTY,
                        card=None, raw_score=0.5),
        BoardSlotResult(slot_index=2, occupancy=BoardSlotOccupancy.EMPTY,
                        card=None, raw_score=0.5),
        BoardSlotResult(slot_index=3, occupancy=BoardSlotOccupancy.EMPTY,
                        card=None, raw_score=0.5),
        BoardSlotResult(slot_index=4, occupancy=BoardSlotOccupancy.EMPTY,
                        card=None, raw_score=0.5),
    )
    with pytest.raises(ValueError):
        BoardSlotsRecognition(slots=slots)


# ---------- empty evidence direction (higher == stronger) ----------

def test_empty_evidence_dark_uniform_strong():
    import numpy as np

    dark = np.full((20, 20, 3), 10, dtype=np.uint8)  # dark uniform
    light = np.full((20, 20, 3), 250, dtype=np.uint8)  # bright
    assert empty_evidence(dark) > empty_evidence(light)


def test_clean_preflop_strong_empty_evidence():
    # 5 clean EMPTY slots -> strong empty evidence (high raw feature).
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.street.value is Street.PREFLOP
    assert obs.street.validation_status is ValidationStatus.VALID
    # street raw feature = min slot evidence should be high-quality (>= 0.5)
    assert obs.street.confidence >= 0.5


# ---------- visible-card conflict ----------

def test_hero_duplicate_conflict():
    eng = _engine()
    tm = _table_map()
    # both hero slots render the same card -> duplicate -> CONFLICT (exact)
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("A", "S")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # deterministic duplicate must be CONFLICT, never UNKNOWN
    assert obs.hero_cards.validation_status is ValidationStatus.CONFLICT
    # trace status must match the field status
    assert obs.hero_cards.evidence["validation_status"] == "conflict"


def test_hero_duplicate_conflict_not_cleared_by_abstain_floor():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    eng = _engine()
    # high card abstain floor above any raw score
    eng._calibrators["card"] = ConfidenceCalibrator(
        name="card", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.99,
    )
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("A", "S")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # duplicate conflict wins over abstain -> still CONFLICT
    assert obs.hero_cards.validation_status is ValidationStatus.CONFLICT
    assert obs.hero_cards.evidence["validation_status"] == "conflict"


def test_board_duplicate_conflict():
    eng = _engine()
    tm = _table_map()
    # board contains duplicate card -> CONFLICT
    frame_img = _build_frame(
        board_slots=[
            render_card("A", "S"), render_card("A", "S"), render_card("K", "D"),
            render_empty_slot(), render_empty_slot(),
        ],
        hero_slots=[render_card("2", "C"), render_card("3", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT


# ---------- platform visual gap ----------

def test_platform_visual_gap_when_action_required():
    eng = _engine()
    eng._require_action = True
    tm = _table_map()  # no ACTION ROI
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    with pytest.raises(TableMapError) as exc:
        eng.process(_frame(frame_img), tm)
    assert "Platform Visual Gap" in str(exc.value)


# ---------- abstain floor wiring ----------

def test_abstain_floor_yields_unknown():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    # a high abstain_floor makes any amount UNKNOWN
    eng = _engine()
    eng._calibrators["amount"] = ConfidenceCalibrator(
        name="amount", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.99,
    )
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # pot raw score < 0.99 -> abstain -> UNKNOWN
    assert obs.pot.validation_status is ValidationStatus.UNKNOWN
    assert obs.pot.value is None


def test_board_abstain_floor_enforced():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    eng = _engine()
    eng._calibrators["board"] = ConfidenceCalibrator(
        name="board", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.99,
    )
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.board_cards.validation_status is ValidationStatus.UNKNOWN


def test_street_abstain_floor_enforced():
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )

    eng = _engine()
    eng._calibrators["street"] = ConfidenceCalibrator(
        name="street", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.99,
    )
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.street.validation_status is ValidationStatus.UNKNOWN


def test_street_conflict_demotes_board():
    # When street derives a confident illegal pattern (CONFLICT), board_cards
    # must not remain VALID.
    from poker_engine.perceptual.vision.protocols import StreetRecognition

    eng = _engine()

    class _ConflictStreetDetector:
        def derive(self, board_slots):
            return StreetRecognition(
                street=None,
                status=ValidationStatus.CONFLICT,
                raw_score=0.9,
                evidence=board_slots.slots,
            )

    eng._street_detector = _ConflictStreetDetector()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.street.validation_status is ValidationStatus.CONFLICT
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT


# ---------- BET_SIZE semantic configuration ----------

def test_bet_size_not_written_without_semantics():
    # Without explicit semantics, bet_size must NOT be written even if a
    # BET_SIZE ROI exists.
    eng = _engine()
    eng._bet_size_semantics = None
    tm = _table_map()  # has BET_SIZE ROI
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.bet_size.validation_status is ValidationStatus.UNKNOWN
    assert obs.bet_size.value is None


def test_bet_size_unsupported_semantics_rejected():
    with pytest.raises(TableMapError):
        _engine(bet_size_semantics="seat")  # unsupported


# ---------- evidence from RecognitionTrace ----------

def test_evidence_is_trace_dict():
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    ev = obs.hero_cards.evidence
    for key in (
        "frame_seq", "roi_key", "slot_id", "recognizer_name",
        "recognizer_version", "raw_score", "confidence",
        "validation_status", "manifest_sha", "template_config_version",
    ):
        assert key in ev, f"missing {key} in hero_cards evidence"
    # source is recognizer + version
    assert obs.hero_cards.source.startswith("card:v")


# ---------- source / recognizer consistency (per field) ----------

def test_board_cards_source_matches_trace_recognizer():
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # board_cards source must be "board:v1", and its evidence recognizer_name
    # must match the same detector ("board").
    assert obs.board_cards.source.startswith("board:v")
    assert obs.board_cards.evidence["recognizer_name"] == "board"
    assert obs.board_cards.evidence["recognizer_version"] == \
        obs.board_cards.source.split(":v", 1)[1]


def test_street_pot_stack_source_match():
    eng = _engine()
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # street -> "street"
    assert obs.street.source.startswith("street:v")
    assert obs.street.evidence["recognizer_name"] == "street"
    # pot -> "amount"
    assert obs.pot.source.startswith("amount:v")
    assert obs.pot.evidence["recognizer_name"] == "amount"


def test_slot_stack_action_source_match():
    from poker_engine.perceptual.vision.table_map import ROIKind, ROI, TableMap

    eng = _engine()
    tm = TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(
            ROI(kind=ROIKind.BOARD_CARDS, x=0.0, y=0.0, width=1.0, height=0.25),
            ROI(kind=ROIKind.HERO_CARDS, x=0.0, y=0.7, width=0.4, height=0.25),
            ROI(kind=ROIKind.BET_SIZE, x=0.4, y=0.55, width=0.2, height=0.1),
            ROI(kind=ROIKind.STACK, x=0.6, y=0.6, width=0.1, height=0.08, slot_id=1),
            ROI(kind=ROIKind.ACTION, x=0.6, y=0.75, width=0.15, height=0.06, slot_id=1),
        ),
    )
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # slot_stack field source -> independently calibrated "stack"
    for s in obs.slot_stacks:
        assert s.field.source.startswith("stack:v")
        assert s.field.evidence["recognizer_name"] == "stack"
    # slot_action field source -> "action"
    for s in obs.slot_actions:
        assert s.field.source.startswith("action:v")
        assert s.field.evidence["recognizer_name"] == "action"


# ---------- genuine board occupancy vs card identity cross-check ----------

def _board_slots_recognition(occupancies, cards=None):
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotResult,
        BoardSlotsRecognition,
    )

    cards = cards or [None] * 5
    slots = tuple(
        BoardSlotResult(
            slot_index=i, occupancy=occupancies[i], card=cards[i], raw_score=0.8,
        )
        for i in range(5)
    )
    return BoardSlotsRecognition(slots=slots)


def test_board_occupancy_card_identity_mismatch_conflict():
    # occupancy says CARD but independent card identity does NOT confirm it
    # -> CONFLICT (presence vs identity disagreement).
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        CardRecognition,
    )
    from poker_engine.core.enums import Rank, Suit

    eng = _engine()

    # board detector reports slot 0,1,2 as CARD, slot 3,4 EMPTY
    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    card = Card(Rank.ACE, Suit.SPADES)
    cards = [card, card, card, None, None]

    class _Detector:
        def detect(self, board_roi_image):
            return _board_slots_recognition(occ, cards)

    eng._board_detector = _Detector()

    # independent card recognizer returns NO card (identity not confirmed)
    class _NoCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=None, raw_score=0.1, slots=())

    eng._card = _NoCardRecognizer()

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # presence says CARD but identity says no card -> deterministic CONFLICT
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT


def test_card_presence_evidence_distinguishes_card_from_empty():
    from poker_engine.perceptual.vision.board_slot_detector import (
        card_presence_evidence,
    )

    card_img = render_card("A", "S")
    empty_img = render_empty_slot()
    # a card (bright + textured) must yield STRONGER presence than empty (dark)
    assert card_presence_evidence(card_img) > card_presence_evidence(empty_img)


def test_board_slot_detector_strong_presence_yields_card_occupancy():
    # Strong card-presence must yield occupancy=CARD with card=None (identity
    # is produced by a SEPARATE path, not this detector). This proves the
    # occupancy signal is independent of card identity.
    import numpy as np
    from poker_engine.perceptual.vision.board_slot_detector import (
        TemplateBoardSlotDetector,
    )
    from poker_engine.perceptual.vision.card_layout import (
        BoardSlotLayout,
        CardSubROI,
    )
    from poker_engine.perceptual.vision.protocols import BoardSlotOccupancy

    layout = BoardSlotLayout(
        layout_id="b", version=1,
        slots=(CardSubROI(x=0.0, y=0.0, width=1.0, height=1.0),) * 5,
    )
    det = TemplateBoardSlotDetector(layout, card_min_presence=0.01)

    card_img = render_card("A", "S")
    board_img = np.full((84, 300, 3), 255, dtype=np.uint8)
    for i in range(5):
        x0 = int(i * 0.2 * 300)
        board_img[:84, x0:x0 + 60] = card_img
    rec = det.detect(board_img)

    card_slots = [s for s in rec.slots
                  if s.occupancy is BoardSlotOccupancy.CARD]
    assert card_slots, "strong presence slots must be CARD"
    for s in card_slots:
        # identity is NOT produced here -> card must be None
        assert s.card is None
        assert s.raw_score > 0.0  # presence evidence is strong, not weak


# ---------- board conflict -> street conflict propagation ----------

def test_board_duplicate_conflict_propagates_to_street():
    # A: legal FLOP occupancy + deterministic duplicate board card
    # -> board_cards CONFLICT AND street CONFLICT.
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        CardRecognition,
        StreetRecognition,
    )
    from poker_engine.core.enums import Rank, Suit

    eng = _engine()
    card = Card(Rank.ACE, Suit.SPADES)
    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    cards = [card, card, card, None, None]  # duplicate

    class _Detector:
        def detect(self, board_roi_image):
            return _board_slots_recognition(occ, cards)

    class _LegalStreet:
        def derive(self, board_slots):
            return StreetRecognition(
                street=Street.FLOP, status=ValidationStatus.VALID,
                raw_score=0.9, evidence=board_slots.slots,
            )

    class _SameCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=(card,), raw_score=0.9, slots=())

    eng._board_detector = _Detector()
    eng._street_detector = _LegalStreet()
    eng._card = _SameCardRecognizer()

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("K", "D"), render_card("2", "C")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT
    assert obs.street.validation_status is ValidationStatus.CONFLICT


def test_board_identity_mismatch_conflict_propagates_to_street():
    # B: legal occupancy says CARD, independent identity disagrees
    # -> board_cards CONFLICT AND street CONFLICT.
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        CardRecognition,
        StreetRecognition,
    )
    from poker_engine.core.enums import Rank, Suit

    eng = _engine()
    card = Card(Rank.ACE, Suit.SPADES)
    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    cards = [card, Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.DIAMONDS),
             None, None]

    class _Detector:
        def detect(self, board_roi_image):
            return _board_slots_recognition(occ, cards)

    class _LegalStreet:
        def derive(self, board_slots):
            return StreetRecognition(
                street=Street.FLOP, status=ValidationStatus.VALID,
                raw_score=0.9, evidence=board_slots.slots,
            )

    class _NoCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=None, raw_score=0.1, slots=())

    eng._board_detector = _Detector()
    eng._street_detector = _LegalStreet()
    eng._card = _NoCardRecognizer()

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("K", "D"), render_card("2", "C")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT
    assert obs.street.validation_status is ValidationStatus.CONFLICT


# ---------- demotion keeps evidence status consistent ----------

def test_demotion_keeps_evidence_status_consistent():
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        CardRecognition,
        StreetRecognition,
    )
    from poker_engine.core.enums import Rank, Suit

    eng = _engine()
    card = Card(Rank.ACE, Suit.SPADES)
    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    cards = [card, card, card, None, None]

    class _Detector:
        def detect(self, board_roi_image):
            return _board_slots_recognition(occ, cards)

    class _LegalStreet:
        def derive(self, board_slots):
            return StreetRecognition(
                street=Street.FLOP, status=ValidationStatus.VALID,
                raw_score=0.9, evidence=board_slots.slots,
            )

    class _SameCard:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=(card,), raw_score=0.9, slots=())

    eng._board_detector = _Detector()
    eng._street_detector = _LegalStreet()
    eng._card = _SameCard()

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("K", "D"), render_card("2", "C")],
    )
    obs = eng.process(_frame(frame_img), tm)
    for f in (obs.board_cards, obs.street):
        # field status must equal its evidence trace status
        assert f.validation_status is ValidationStatus.CONFLICT
        assert ValidationStatus(f.evidence["validation_status"]) is \
            f.validation_status


# ---------- missing ROI -> RecognitionTrace evidence (not {}) ----------

def test_missing_roi_produces_trace_evidence():
    eng = _engine()
    # table map with NO hero/board/pot/bet ROIs
    from poker_engine.perceptual.vision.table_map import TableMap
    tm = TableMap(
        platform_id="wpk", layout_id="6max", reference_size=(600, 400),
        rois=(),
    )
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    for field in (obs.hero_cards, obs.board_cards, obs.street, obs.pot,
                  obs.bet_size):
        assert field.validation_status is ValidationStatus.UNKNOWN
        ev = field.evidence
        assert ev, "empty evidence for an UNKNOWN field"
        for key in ("frame_seq", "roi_key", "recognizer_name",
                    "recognizer_version", "raw_score", "confidence",
                    "validation_status", "manifest_sha",
                    "template_config_version"):
            assert key in ev, f"missing {key} in UNKNOWN field evidence"
        assert ev["validation_status"] == "unknown"


def test_bet_not_configured_produces_trace_evidence():
    eng = _engine(bet_size_semantics=None)
    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_empty_slot() for _ in range(5)],
        hero_slots=[render_card("A", "S"), render_card("K", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    assert obs.bet_size.validation_status is ValidationStatus.UNKNOWN
    assert obs.bet_size.evidence, "bet_size evidence must not be empty"
    assert obs.bet_size.evidence["validation_status"] == "unknown"


# ---------- production-path conflict (REAL TemplateBoardSlotDetector) ----------

def test_real_detector_strong_presence_identity_failure_is_conflict():
    # Uses the REAL TemplateBoardSlotDetector (not a fake). A board frame with
    # strong card presence + an identity recognizer that fails must produce
    # board=CONFLICT AND street=CONFLICT — NOT UNKNOWN.
    from poker_engine.perceptual.vision.protocols import CardRecognition

    eng = _engine()  # constructs the real TemplateBoardSlotDetector

    class _NoCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=None, raw_score=0.1, slots=())

    eng._card = _NoCardRecognizer()

    tm = _table_map()
    # FLOP board: 3 real cards (strong presence), 2 empty slots
    frame_img = _build_frame(
        board_slots=[render_card("A", "S"), render_card("K", "H"),
                     render_card("Q", "D"), render_empty_slot(),
                     render_empty_slot()],
        hero_slots=[render_card("2", "C"), render_card("3", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # presence says CARD but identity failed -> deterministic CONFLICT
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT
    assert obs.street.validation_status is ValidationStatus.CONFLICT


def test_board_conflict_not_cleared_by_abstain_floor():
    # A deterministic board CONFLICT must NOT be downgraded to UNKNOWN by a
    # high board abstain_floor; it must propagate to street as CONFLICT.
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )
    from poker_engine.perceptual.vision.protocols import CardRecognition

    eng = _engine()

    class _NoCardRecognizer:
        def recognize(self, roi_image, card_model=None):
            return CardRecognition(value=None, raw_score=0.1, slots=())

    eng._card = _NoCardRecognizer()
    # high abstain floor above any presence raw score
    eng._calibrators["board"] = ConfidenceCalibrator(
        name="board", version=1,
        bins=CalibrationBins(edges=(0.0, 0.5, 1.0), confidence=(0.1, 0.9)),
        abstain_floor=0.99,
    )

    tm = _table_map()
    frame_img = _build_frame(
        board_slots=[render_card("A", "S"), render_card("K", "H"),
                     render_card("Q", "D"), render_empty_slot(),
                     render_empty_slot()],
        hero_slots=[render_card("2", "C"), render_card("3", "D")],
    )
    obs = eng.process(_frame(frame_img), tm)
    # deterministic conflict wins over abstain
    assert obs.board_cards.validation_status is ValidationStatus.CONFLICT
    assert obs.street.validation_status is ValidationStatus.CONFLICT
    # field status must equal its trace status
    assert ValidationStatus(obs.board_cards.evidence["validation_status"]) is \
        obs.board_cards.validation_status
    assert ValidationStatus(obs.street.evidence["validation_status"]) is \
        obs.street.validation_status


# ---------- board per-slot separate calibration ----------

def test_board_calibrated_confidence_card_uses_identity_empty_uses_no_card():
    # CARD slots combine board-calibrated presence with card-calibrated
    # identity (min of the two calibrated values); EMPTY slots use their own
    # board-calibrated no-card (empty) evidence.
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        BoardSlotResult,
        BoardSlotsRecognition,
    )
    from poker_engine.perceptual.vision.engine import (
        _board_calibrated_confidence as _bcc,
    )
    from tests.vision.test_engine import _cal

    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    slots = tuple(
        BoardSlotResult(slot_index=i, occupancy=occ[i], card=None,
                        raw_score=0.9)
        for i in range(5)
    )
    slots_rec = BoardSlotsRecognition(slots=slots)
    # independent identity: CARD slots score 0.6, EMPTY slots score 0.8
    independent = [
        (0, None, 0.6), (1, None, 0.6), (2, None, 0.6),
        (3, None, 0.8), (4, None, 0.8),
    ]
    cal_board = _cal("board")
    cal_card = _cal("card")

    # per CARD slot: min(cal(presence 0.9)=0.9, cal(identity 0.6)=0.9) = 0.9
    # per EMPTY slot: cal(empty 0.9)=0.9
    # -> min of all = 0.9 (identity 0.6 maps into the same bin as presence)
    assert _bcc(slots_rec, independent, cal_board, cal_card) == 0.9


def test_board_calibrated_confidence_unknown_slot_is_weak():
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        BoardSlotResult,
        BoardSlotsRecognition,
    )
    from poker_engine.perceptual.vision.engine import (
        _board_calibrated_confidence as _bcc,
    )
    from tests.vision.test_engine import _cal

    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.UNKNOWN]
    slots = tuple(
        BoardSlotResult(slot_index=i, occupancy=occ[i], card=None,
                        raw_score=0.9)
        for i in range(5)
    )
    slots_rec = BoardSlotsRecognition(slots=slots)
    independent = [(i, None, 0.7) for i in range(5)]
    cal_board = _cal("board")
    cal_card = _cal("card")

    # UNKNOWN slot contributes 0.0 -> min is 0.0
    assert _bcc(slots_rec, independent, cal_board, cal_card) == 0.0


def test_board_calibrated_confidence_separate_scales():
    # The decisive regression: a HIGH identity raw (0.6) that calibrates to a
    # LOW identity confidence (0.01) must NOT be masked by a HIGH board-
    # calibrated occupancy (0.9). The two signals are calibrated separately and
    # combined only at the calibrated level, taking the min.
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        BoardSlotResult,
        BoardSlotsRecognition,
    )
    from poker_engine.perceptual.vision.calibration import (
        CalibrationBins,
        ConfidenceCalibrator,
    )
    from poker_engine.perceptual.vision.engine import (
        _board_calibrated_confidence as _bcc,
    )

    # board calibrator: presence 0.9 -> confidence 0.9
    cal_board = ConfidenceCalibrator(
        "board", 1, CalibrationBins((0.0, 0.5, 1.0), (0.1, 0.9))
    )
    # card calibrator: identity 0.6 -> confidence 0.01 (different scale)
    cal_card = ConfidenceCalibrator(
        "card", 1, CalibrationBins((0.0, 0.5, 1.0), (0.01, 0.01)), abstain_floor=0.5
    )

    slots = tuple(
        BoardSlotResult(slot_index=i, occupancy=Occ.CARD, card=None,
                        raw_score=0.9)
        for i in range(5)
    )
    slots_rec = BoardSlotsRecognition(slots=slots)
    # identity raw 0.6 on every CARD slot -> card-calibrated = 0.01
    independent = [(i, None, 0.6) for i in range(5)]

    conf = _bcc(slots_rec, independent, cal_board, cal_card)
    # per CARD slot: min(cal_board(0.9)=0.9, cal_card(0.6)=0.01) = 0.01
    assert conf == 0.01


def test_board_components_records_both_raw_and_conf():
    from poker_engine.perceptual.vision.protocols import (
        BoardSlotOccupancy as Occ,
        BoardSlotResult,
        BoardSlotsRecognition,
    )
    from poker_engine.perceptual.vision.engine import _board_components
    from tests.vision.test_engine import _cal

    occ = [Occ.CARD, Occ.CARD, Occ.CARD, Occ.EMPTY, Occ.EMPTY]
    slots = tuple(
        BoardSlotResult(slot_index=i, occupancy=occ[i], card=None,
                        raw_score=0.9)
        for i in range(5)
    )
    slots_rec = BoardSlotsRecognition(slots=slots)
    independent = [
        (0, None, 0.6), (1, None, 0.7), (2, None, 0.8),
        (3, None, 0.9), (4, None, 0.9),
    ]
    comp = _board_components(slots_rec, independent, _cal("board"), _cal("card"))
    assert comp["occupancy_app_raw"] == 0.9
    assert comp["occupancy_app_conf"] == 0.9  # calibrated via board cal
    assert comp["identity_app_raw"] == 0.6   # min over CARD slots
    assert comp["identity_app_conf"] == 0.9  # calibrated via card cal
