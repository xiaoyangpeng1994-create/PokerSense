"""Tests for stage F stack-value transcription (renders, never writes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.capture_card_calibration.schema import (
    FieldValue,
    FrameLabel,
    Occupancy,
    Review,
    Scene,
    SlotLabel,
)
from tools.capture_card_calibration.stack_transcribe import (
    StackGap,
    apply_stack_values,
    collect_stack_gaps,
    render_stack_csv,
    render_stack_worksheet,
)

# --- builders --------------------------------------------------------------


def _slot(
    slot_id: int,
    *,
    occupied: bool = True,
    stack_status: str = "UNKNOWN",
    stack_value: int | None = None,
) -> SlotLabel:
    occupancy = FieldValue.valid(
        Occupancy.OCCUPIED.value if occupied else Occupancy.EMPTY.value
    )
    if stack_status == "UNKNOWN":
        stack: FieldValue = FieldValue.unknown()
    elif stack_status == "CONFLICT":
        stack = FieldValue.conflict()
    else:
        default = 100 + slot_id
        stack = FieldValue.valid(
            stack_value if stack_value is not None else default
        )
    return SlotLabel(
        slot_id=slot_id,
        occupancy=occupancy,
        stack=stack,
        dealer=FieldValue.valid(False),
        completed_action=FieldValue.unknown(),
        current_actor=FieldValue.unknown(),
    )


def _label(
    index: int,
    session: str = "session_001",
    *,
    slots: tuple[SlotLabel, ...] | None = None,
    scene: Scene = Scene.TABLE,
    stable: bool = True,
) -> FrameLabel:
    kwargs = {
        "frame": f"{session}__t_{index:08d}__f_{index:06d}__{index:012d}.png",
        "sha256": f"{index:064d}",
        "session_id": session,
        "hand_id": f"{session}_hand_{index // 4:04d}",
        "timestamp_ms": index * 500,
        "stable": stable,
        "scene": scene,
        "hero_cards": FieldValue.valid(["TS", "JD"]),
        "board_cards": FieldValue.valid(["4H", "KC", "TS"]),
        "street": FieldValue.valid("FLOP"),
        "pot": FieldValue.valid(100 + index),
        "review": Review(reviewer="tester"),
    }
    if slots is None:
        slots = tuple(_slot(i) for i in range(8))
    else:
        # FrameLabel requires exactly SLOT_COUNT slots; pad the missing tail
        # with a default VALID slot so a focused test can pass only the slots
        # it cares about without rebuilding all eight every time.
        by_id = {slot.slot_id: slot for slot in slots}
        slots = tuple(
            by_id.get(i, _slot(i, stack_status="VALID")) for i in range(8)
        )
    kwargs["slots"] = slots
    return FrameLabel(**kwargs)


# --- stage F gap collection ------------------------------------------------


def test_collect_ignores_empty_seats():
    # An EMPTY seat has no stack number to read; it must never be a target.
    slots = (
        _slot(0, occupied=True, stack_status="UNKNOWN"),
        _slot(1, occupied=False),
        _slot(2, occupied=True, stack_status="UNKNOWN"),
    )
    gaps = collect_stack_gaps([_label(0, slots=slots)])
    assert {g.slot_id for g in gaps} == {0, 2}


def test_collect_ignores_valid_and_conflict_stacks():
    # VALID is already transcribed; CONFLICT needs a re-read, not a guess.
    slots = (
        _slot(0, occupied=True, stack_status="VALID"),
        _slot(1, occupied=True, stack_status="CONFLICT"),
        _slot(2, occupied=True, stack_status="UNKNOWN"),
    )
    gaps = collect_stack_gaps([_label(0, slots=slots)])
    assert {g.slot_id for g in gaps} == {2}


def test_collect_ignores_unstable_and_non_table_frames():
    unstable = _label(0, stable=False)
    menu = _label(1, scene=Scene.MENU)
    assert collect_stack_gaps([unstable, menu]) == ()


def test_collect_records_session_and_layout_key():
    gaps = collect_stack_gaps([_label(0, session="session_002")])
    assert gaps[0].session_id == "session_002"
    assert gaps[0].layout_key == "s002"


def test_collect_filters_by_session():
    a = _label(0, session="session_001")
    b = _label(1, session="session_002")
    gaps = collect_stack_gaps([a, b], session="session_002")
    assert len(gaps) == 8
    assert all(g.session_id == "session_002" for g in gaps)


# --- stage F apply (the ONLY writer) ---------------------------------------


def test_apply_promotes_confirmed_value(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    slots = (_slot(0, stack_status="UNKNOWN"),)
    write_frames_jsonl(path, [_label(0, slots=slots)])

    result = apply_stack_values(
        path, [{"frame": _label(0).frame, "slot_id": 0, "value": "350"}],
        backup_dir=root / "reports" / "backups",
    )
    assert result.applied == 1
    assert result.blank == 0

    # Re-read to confirm the slot stack is now VALID(350) and nothing else moved.
    from tools.capture_card_calibration.dataset import read_frames_jsonl

    updated = read_frames_jsonl(path)[0]
    assert updated.slots[0].stack.status.value == "VALID"
    assert updated.slots[0].stack.value == 350
    assert updated.slots[0].occupancy.value == Occupancy.OCCUPIED.value


def test_apply_skips_blank_value(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    write_frames_jsonl(path, [_label(0, slots=(_slot(0, stack_status="UNKNOWN"),))])
    result = apply_stack_values(
        path, [{"frame": _label(0).frame, "slot_id": 0, "value": "  "}]
    )
    assert result.applied == 0
    assert result.blank == 1


def test_apply_never_invents_an_unknown_frame(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    write_frames_jsonl(path, [_label(0)])
    result = apply_stack_values(
        path,
        [{"frame": "nope.png", "slot_id": 0, "value": "10"}],
    )
    assert result.applied == 0
    assert result.unknown_frame == 1


def test_apply_skips_already_set_stack(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    # Slot 0 is already VALID -> must be counted as already-set, not applied.
    write_frames_jsonl(
        path, [_label(0, slots=(_slot(0, stack_status="VALID"),))]
    )
    result = apply_stack_values(
        path, [{"frame": _label(0).frame, "slot_id": 0, "value": "10"}]
    )
    assert result.applied == 0
    assert result.not_unknown_stack == 1


def test_apply_rejects_negative_value(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    write_frames_jsonl(path, [_label(0, slots=(_slot(0, stack_status="UNKNOWN"),))])
    result = apply_stack_values(
        path, [{"frame": _label(0).frame, "slot_id": 0, "value": "-5"}]
    )
    assert result.applied == 0
    assert result.blank == 1


def test_apply_creates_a_backup_before_writing(tmp_path):
    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    write_frames_jsonl(path, [_label(0, slots=(_slot(0, stack_status="UNKNOWN"),))])
    original = path.read_bytes()
    result = apply_stack_values(
        path,
        [{"frame": _label(0).frame, "slot_id": 0, "value": "400"}],
        backup_dir=root / "reports" / "backups",
    )
    assert result.backup_path
    bak = Path(result.backup_path)
    assert bak.is_file()
    assert bak.read_bytes() == original


# --- stage F worksheet / CSV -----------------------------------------------


def test_csv_template_has_header_and_one_row_per_gap():
    gaps = [
        StackGap(
            frame="f.png", session_id="session_001", hand_id="h1",
            timestamp_ms=0, slot_id=0, layout_key="multi",
        ),
        StackGap(
            frame="f.png", session_id="session_001", hand_id="h1",
            timestamp_ms=0, slot_id=1, layout_key="multi",
        ),
    ]
    csv_text = render_stack_csv(gaps)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "frame,slot_id,value"
    assert len(lines) == 3


def test_worksheet_is_self_contained_and_escapes_html(tmp_path):
    gaps = [
        StackGap(
            frame='<script>alert(1)</script>.png',
            session_id="session_001", hand_id="h1",
            timestamp_ms=0, slot_id=3, layout_key="multi",
        ),
    ]
    html_text = render_stack_worksheet(
        gaps, {}, tmp_path, include_images=False, summary="s"
    )
    assert "<!DOCTYPE html>" in html_text
    assert "<script>alert(1)</script>" not in html_text


def test_worksheet_omits_images_when_disabled(tmp_path):
    gaps = [
        StackGap(
            frame="f.png", session_id="session_001", hand_id="h1",
            timestamp_ms=0, slot_id=0, layout_key="multi",
        ),
    ]
    html_text = render_stack_worksheet(
        gaps, {}, tmp_path, include_images=False
    )
    assert "data:image/png;base64," not in html_text
    assert "data:image/jpeg;base64," not in html_text


def test_worksheet_is_a_fillable_form(tmp_path):
    # The page must be a one-stop fill-in form: a download button, a live
    # progress counter, and per-row inputs carrying data-index/frame/slot so
    # the client script can export exactly the CSV shape stack-apply expects.
    gaps = []
    for idx in range(3):
        gaps.append(
            StackGap(
                frame=f"f{idx}.png", session_id="session_001", hand_id="h1",
                timestamp_ms=idx * 10, slot_id=idx, layout_key="multi",
            )
        )
    html_text = render_stack_worksheet(
        gaps, {}, tmp_path, include_images=False
    )
    assert 'id="download"' in html_text
    assert 'id="clear"' in html_text
    assert 'id="progress"' in html_text
    assert "已填" in html_text
    # Each of the 3 targets has a distinct data-index on its input.
    assert html_text.count("data-index=") == 3
    assert "data-frame=" in html_text
    assert "data-slot=" in html_text
    # The client script must be present (it wires up download/clear/progress).
    assert "buildCsv" in html_text


def test_worksheet_uses_jpeg_crops_and_thumbs(tmp_path):
    # When images are on, both the crop and the frame thumbnail are JPEG
    # (keeps a 180-target worksheet from ballooning into hundreds of MB).
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    gaps = [
        StackGap(
            frame="f.png", session_id="session_001", hand_id="h1",
            timestamp_ms=0, slot_id=1, layout_key="multi",
        ),
    ]
    # A real frame so the encode path actually runs.
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "f.png"), frame)
    html_text = render_stack_worksheet(
        gaps, {}, tmp_path, include_images=True
    )
    # crop + thumbnail JPEG MIME appears; PNG must be gone.
    assert "data:image/jpeg;base64," in html_text
    assert "data:image/png;base64," not in html_text
