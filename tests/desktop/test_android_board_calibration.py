"""Production-profile tests for measured LDPlayer board geometry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_engine.core.enums import ActionType, Street
from poker_engine.core.observation import ValidationStatus
from poker_engine.desktop.live import (
    load_calibration,
    load_platform_seat_mapping,
)
from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.table_map import ROIKind


_ROOT = Path(__file__).resolve().parents[2]
_CARDS = _ROOT / "tests" / "vision" / "fixtures" / "wepoker"


def _frame_with_board(
    labels: tuple[str, ...], pot_fixture: str | None = None,
    stack_fixture: str | None = None,
    dealer_slot: int | None = None,
    action: tuple[int, ActionType] | None = None,
    empty_slots: tuple[int, ...] = (),
    frame_seq: int = 1,
) -> Frame:
    table_map, _vision = load_calibration()
    width, height = table_map.reference_size
    image = np.full((height, width, 3), (74, 110, 61), dtype=np.uint8)
    roi = next(r for r in table_map.rois if r.kind is ROIKind.BOARD_CARDS)
    rx, ry = round(roi.x * width), round(roi.y * height)

    # The measured parent ROI has 10 px horizontal and 15 px vertical context
    # around each 153x200 card face. Card starts are 163 px apart.
    for index, label in enumerate(labels):
        card = cv2.imread(str(_CARDS / f"{label}.png"))
        assert card is not None
        card = cv2.resize(card, (153, 200))
        x, y = rx + 10 + index * 163, ry + 15
        image[y:y + 200, x:x + 153] = card

    if pot_fixture is not None:
        pot_roi = next(r for r in table_map.rois if r.kind is ROIKind.POT)
        px, py = round(pot_roi.x * width), round(pot_roi.y * height)
        pot = cv2.imread(
            str(_ROOT / "tests" / "vision" / "fixtures"
                / "wepoker_android" / f"{pot_fixture}.png")
        )
        assert pot is not None
        ph, pw = pot.shape[:2]
        image[py:py + ph, px:px + pw] = pot

    if stack_fixture is not None:
        stack_roi = next(
            r for r in table_map.rois
            if r.kind is ROIKind.STACK and r.slot_id == 3
        )
        sx, sy = round(stack_roi.x * width), round(stack_roi.y * height)
        stack = cv2.imread(
            str(_ROOT / "tests" / "vision" / "fixtures"
                / "wepoker_android" / f"{stack_fixture}.png")
        )
        assert stack is not None
        sh, sw = stack.shape[:2]
        image[sy:sy + sh, sx:sx + sw] = stack

    if dealer_slot is not None:
        # Slot 7's reviewed marker location lies inside its deliberately
        # broader search window; the recognizer returns the visual slot only.
        assert dealer_slot == 7
        marker = cv2.imread(
            str(_ROOT / "configs" / "vision" / "wepoker_android"
                / "dealer" / "marker.png")
        )
        assert marker is not None
        mh, mw = marker.shape[:2]
        image[1608:1608 + mh, 1127:1127 + mw] = marker

    if action is not None:
        slot_id, action_type = action
        action_roi = next(
            r for r in table_map.rois
            if r.kind is ROIKind.ACTION and r.slot_id == slot_id
        )
        ax = round(action_roi.x * width) + 20
        ay = round(action_roi.y * height) + 20
        glyph = cv2.imread(
            str(_ROOT / "configs" / "vision" / "wepoker_android"
                / "action_glyph" / f"{action_type.value}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert glyph is not None
        gh, gw = glyph.shape
        color = (0, 220, 255) if action_type is ActionType.ALL_IN else (
            255, 255, 255
        )
        target = image[ay:ay + gh, ax:ax + gw]
        target[glyph > 127] = color

    for slot_id in empty_slots:
        action_roi = next(
            r for r in table_map.rois
            if r.kind is ROIKind.ACTION and r.slot_id == slot_id
        )
        ex = round(action_roi.x * width) + 20
        ey = round(action_roi.y * height) + 20
        marker = cv2.imread(
            str(_ROOT / "configs" / "vision" / "wepoker_android"
                / "empty_slot" / "plus.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert marker is not None
        eh, ew = marker.shape
        target = image[ey:ey + eh, ex:ex + ew]
        target[marker > 127] = (220, 220, 220)

    return Frame(
        frame_seq=frame_seq,
        timestamp=datetime(2026, 8, 24, tzinfo=timezone.utc),
        window_id="emulator-5556",
        window_rect=WindowRect(0, 0, width, height),
        image=image,
        width=width,
        height=height,
    )


def test_measured_android_flop_geometry_recognizes_cards_and_street():
    table_map, vision = load_calibration()
    observation = vision.process(_frame_with_board(("3C", "6D", "7S")), table_map)

    assert observation.board_cards.validation_status is ValidationStatus.VALID
    assert [str(card) for card in observation.board_cards.value] == [
        "3c", "6d", "7s",
    ]
    assert observation.street.validation_status is ValidationStatus.VALID
    assert observation.street.value is Street.FLOP


def test_measured_android_empty_board_is_preflop():
    table_map, vision = load_calibration()
    observation = vision.process(_frame_with_board(()), table_map)

    assert observation.board_cards.validation_status is ValidationStatus.VALID
    assert observation.board_cards.value == ()
    assert observation.street.validation_status is ValidationStatus.VALID
    assert observation.street.value is Street.PREFLOP


def test_measured_android_pot_ocr_accepts_number_and_rejects_label():
    table_map, vision = load_calibration()

    numbered = vision.process(
        _frame_with_board((), pot_fixture="pot_790"), table_map
    )
    label = vision.process(
        _frame_with_board((), pot_fixture="pot_label"), table_map
    )

    assert numbered.pot.validation_status is ValidationStatus.VALID
    assert str(numbered.pot.value) == "790"
    assert label.pot.validation_status is ValidationStatus.UNKNOWN
    assert label.pot.value is None


def test_measured_android_stack_ocr_is_slot_specific_and_abstains_on_empty():
    table_map, vision = load_calibration()

    numbered = vision.process(
        _frame_with_board((), stack_fixture="stack_158"), table_map
    )
    empty = vision.process(
        _frame_with_board((), stack_fixture="stack_empty"), table_map
    )

    numbered_slot = next(
        slot for slot in numbered.slot_stacks if slot.slot_id == 3
    )
    empty_slot = next(slot for slot in empty.slot_stacks if slot.slot_id == 3)
    assert numbered_slot.field.validation_status is ValidationStatus.VALID
    assert str(numbered_slot.field.value) == "158"
    assert numbered_slot.field.evidence["recognizer_name"] == "stack"
    assert empty_slot.field.validation_status is ValidationStatus.UNKNOWN
    assert empty_slot.field.value is None


def test_measured_android_dealer_marker_returns_visual_slot_only():
    table_map, vision = load_calibration()

    marked = vision.process(_frame_with_board((), dealer_slot=7), table_map)
    empty = vision.process(_frame_with_board(()), table_map)

    assert marked.dealer_pos.validation_status is ValidationStatus.VALID
    assert marked.dealer_pos.value == 7
    assert marked.dealer_pos.evidence["recognizer_name"] == "dealer"
    assert empty.dealer_pos.validation_status is ValidationStatus.UNKNOWN
    assert empty.dealer_pos.value is None


def test_measured_android_action_glyph_returns_completed_visual_slot_action():
    table_map, vision = load_calibration()

    called = vision.process(
        _frame_with_board((), action=(6, ActionType.CALL)), table_map
    )
    empty = vision.process(_frame_with_board(()), table_map)

    called_slot = next(
        slot for slot in called.slot_actions if slot.slot_id == 6
    )
    assert called_slot.field.validation_status is ValidationStatus.VALID
    assert called_slot.field.value is ActionType.CALL
    assert called_slot.field.evidence["recognizer_name"] == "action"
    assert all(
        slot.field.validation_status is ValidationStatus.UNKNOWN
        for slot in empty.slot_actions
    )


def test_measured_android_occupancy_uses_stack_or_explicit_empty_marker():
    table_map, vision = load_calibration()

    observation = vision.process(
        _frame_with_board(
            (), stack_fixture="stack_158", empty_slots=(4,),
        ),
        table_map,
    )

    by_slot = {slot.slot_id: slot.field for slot in observation.slot_occupancies}
    assert by_slot[3].validation_status is ValidationStatus.VALID
    assert by_slot[3].value is True
    assert by_slot[4].validation_status is ValidationStatus.VALID
    assert by_slot[4].value is False
    assert by_slot[2].validation_status is ValidationStatus.UNKNOWN
    assert by_slot[2].value is None


def test_android_platform_mapping_covers_all_physical_slots():
    mapping = load_platform_seat_mapping()
    expected = {slot: slot for slot in range(8)}
    assert dict(mapping.stack_slot_to_seat) == expected
    assert dict(mapping.action_slot_to_seat) == expected
    assert dict(mapping.dealer_slot_to_seat) == expected
    assert dict(mapping.occupancy_slot_to_seat) == expected
    assert dict(mapping.actor_slot_to_seat) == {0: 0}
    assert mapping.actor_observation_is_current


def test_android_platform_mapping_rejects_unknown_schema(monkeypatch, tmp_path):
    import poker_engine.desktop.live as live

    platform_dir = tmp_path / "configs" / "platform"
    platform_dir.mkdir(parents=True)
    path = platform_dir / (
        "wepoker_android__ldplayer_portrait_1440x2560_seat_mapping.json"
    )
    path.write_text('{"schema_version": 2}', encoding="utf-8")
    monkeypatch.setattr(live, "_REPO_ROOT", tmp_path)

    with pytest.raises(live.LiveCaptureError, match="unsupported.*schema"):
        live.load_platform_seat_mapping()


def test_measured_android_flop_reaches_production_state_pipeline():
    from poker_engine.desktop.live import (
        _seed_state,
        build_confidence_gate,
        load_measured_calibrations,
    )
    from poker_engine.memory.hand_memory import InMemoryHandMemory
    from poker_engine.orchestrator import ApplicationOrchestrator
    from poker_engine.realtime.frame_source import SyntheticFrameSource
    from poker_engine.realtime.pipeline import RealtimePipeline
    from poker_engine.state_engine.engine import StateEngine

    table_map, vision = load_calibration()
    gate = build_confidence_gate(
        load_measured_calibrations(
            _ROOT / "configs" / "vision" / "wepoker_android"
        )
    )
    orchestrator = ApplicationOrchestrator(
        StateEngine(), InMemoryHandMemory(), gate
    )
    orchestrator.start_hand(_seed_state())
    pipeline = RealtimePipeline(
        SyntheticFrameSource((
            _frame_with_board(
                ("3C", "6D", "7S"), pot_fixture="pot_790", frame_seq=1,
            ),
            _frame_with_board(
                ("3C", "6D", "7S"), pot_fixture="pot_790", frame_seq=2,
            ),
        )),
        vision,
        table_map,
        orchestrator,
    )

    first = pipeline.step()
    step = pipeline.step()

    assert first is not None
    assert step is not None
    assert step.analysis.state.street is Street.FLOP
    assert [str(card) for card in step.analysis.state.board_cards] == [
        "3c", "6d", "7s",
    ]
    assert str(step.analysis.state.pot) == "790"


def test_legacy_card_only_gate_does_not_enable_board_by_accident():
    from poker_engine.desktop.live import (
        build_confidence_gate,
        load_measured_calibration,
    )

    card_only = load_measured_calibration(
        _ROOT / "configs" / "vision" / "wepoker_android"
    )
    gate = build_confidence_gate(card_only)

    assert gate.thresholds["board_cards"] == 1.0
    assert gate.thresholds["street"] == 1.0
    assert gate.thresholds["pot"] == 1.0
    assert gate.thresholds["stacks"] == 1.0
    assert gate.thresholds["action"] == 1.0
    assert gate.slot_thresholds["occupancy"] == 1.0
