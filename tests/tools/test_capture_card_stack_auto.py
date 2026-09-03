"""Tests for stage F stack auto-reading (proposal tool, never writes).

These tests are hardware-free and private-data-free: they synthesize tiny
digit glyphs (0-9) into fake "frames" with synthetic ``FrameLabel``s, so the
whole pipeline — template building, confidence gating, hand-isolated split,
proposal rendering — runs in a ``tmp_path`` without touching the private
capture-card dataset. The gate thresholds are imported from the module so a
change in the calibrated constants is picked up here rather than silently
diverging.

Every test asserts the failure-closed contract: a low-margin / poor-fit read
is never accepted; a read whose glyph count mismatches the expected digits is
dropped; known stacks never leak into the proposal CSV; the hand split never
leaks a group across sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.capture_card_calibration.schema import (
    FieldValue,
    FrameLabel,
    Occupancy,
    Review,
    Scene,
    SlotLabel,
)
from tools.capture_card_calibration import seat_reader as sr
from tools.capture_card_calibration.stack_auto import (
    FIT_THRESHOLD,
    MARGIN_THRESHOLD,
    AutoStackSummary,
    StackCandidate,
    build_digit_templates_from_labels,
    evaluate_candidates,
    hold_out_split,
    read_targets,
    render_auto_report,
    render_proposal_csv,
)

# --- synthetic digit glyphs -----------------------------------------------

# The real capture-card stack digits are ~14px tall on the 498x1080 canvas
# and windowed to ``_DIGIT_H = (11, 16)`` by ``_split_digits``. cv2.putText
# with FONT_HERSHEY_SIMPLEX at ``_DIGIT_FONT_SCALE`` produces glyphs in that
# band; we render each value onto a dark-green pill so the white glyph is
# bright against the felt (matching how the reader thresholds luminance).

#: Font scale that lands a digit in the 11-16px height window on the canvas.
_DIGIT_FONT_SCALE = 0.4
_DIGIT_THICKNESS = 2

#: Horizontal stride between digits so the tiny pill's glyphs stay separate
#: after ``_split_digits``' 2px dilation (calibrated: a stride of 12px splits
#: ``123`` into 3 glyphs of width ~7-10px, inside the reader's window).
_DIGIT_GAP = 12

#: A stride short enough that cv2.dilate(2,2) bridges two adjacent glyphs into
#: a single >20px component (the "digit merge" the UI produces when it draws a
#: multi-digit value with its digits touching). On the hero ROI, drawing "194"
#: at this stride fuses the "9" and "4" into one 21px blob while the leading
#: "1" survives as a clean 7px glyph — the exact "a clean digit survives a
#: merged multi-digit value" hazard the failure-closed path must reject (a
#: merged "198" must never read as a confident "1"). Note this is value- and
#: layout-specific: "198" at the same stride stays at 20px and does NOT merge
#: (the 9+8 bridge measures 19-20px), which is why the merge tests use "194".
_MERGE_GAP = 10


def _draw_digit(canvas, digits: str, x: int, y: int):
    """Draw a string of digits with cv2.putText onto a BGR canvas."""
    import cv2

    gx = x
    for ch in str(digits):
        cv2.putText(
            canvas, ch, (gx, y),
            cv2.FONT_HERSHEY_SIMPLEX, _DIGIT_FONT_SCALE,
            (255, 255, 255), _DIGIT_THICKNESS, cv2.LINE_AA,
        )
        gx += _DIGIT_GAP
    return canvas


def _draw_digit_gap(canvas, digits: str, x: int, y: int, gap: int):
    """Draw digits at a custom horizontal stride (used to force a merge)."""
    import cv2

    gx = x
    for ch in str(digits):
        cv2.putText(
            canvas, ch, (gx, y),
            cv2.FONT_HERSHEY_SIMPLEX, _DIGIT_FONT_SCALE,
            (255, 255, 255), _DIGIT_THICKNESS, cv2.LINE_AA,
        )
        gx += gap
    return canvas


def _make_frame(session: str, index: int, values: dict[int, int]) -> np.ndarray:
    """Synthesize a 498x1080 normalized frame with stack pills drawn in.

    ``values`` maps slot_id -> stack value; each value is drawn at that slot's
    pill ROI centre so ``_split_digits``/``_classify`` see a real glyph.
    """
    import cv2

    canvas = np.zeros((1080, 498, 3), dtype=np.uint8)
    layout = sr.SLOT_LAYOUT_S002 if session == "session_002" else sr.SLOT_LAYOUT_MULTI
    H, W = canvas.shape[:2]
    for slot_id, value in values.items():
        row = layout[slot_id]
        cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
        # Pill background (dark green) so white digits stand out.
        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + int(w * W), y0 + int(h * H)),
            (60, 90, 40),
            -1,
        )
        _draw_digit(canvas, str(value), x0 + 6, y0 + int(h * H) // 2 + 5)
    return canvas


def _make_frame_merge(session: str, index: int, values: dict[int, int]) -> np.ndarray:
    """Synthesize a frame whose stack pills use a *merging* digit stride.

    ``_MERGE_GAP`` is short enough that ``_split_digits``' 2px dilation fuses
    two adjacent digits into one >``_DIGIT_MAX_W`` component — exactly the
    "digit merge" the UI creates on real multi-digit stacks (e.g. a "198" whose
    "9" and "8" touch). The failure-closed auto-reader must reject such a pill
    rather than returning the one surviving glyph as the whole value.
    """
    import cv2

    canvas = np.zeros((1080, 498, 3), dtype=np.uint8)
    layout = sr.SLOT_LAYOUT_S002 if session == "session_002" else sr.SLOT_LAYOUT_MULTI
    H, W = canvas.shape[:2]
    for slot_id, value in values.items():
        row = layout[slot_id]
        cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
        x0 = int((cx - w / 2) * W)
        y0 = int((cy - h / 2) * H)
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + int(w * W), y0 + int(h * H)),
            (60, 90, 40),
            -1,
        )
        _draw_digit_gap(
            canvas, str(value), x0 + 6, y0 + int(h * H) // 2 + 5, _MERGE_GAP
        )
    return canvas

# --- label/builders --------------------------------------------------------


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
    hand_id: str | None = None,
) -> FrameLabel:
    kwargs = {
        "frame": f"{session}__t_{index:08d}__f_{index:06d}__{index:012d}.png",
        "sha256": f"{index:064d}",
        "session_id": session,
        "hand_id": hand_id or f"{session}_hand_{index // 4:04d}",
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
        by_id = {slot.slot_id: slot for slot in slots}
        slots = tuple(
            by_id.get(i, _slot(i, stack_status="VALID")) for i in range(8)
        )
    kwargs["slots"] = slots
    return FrameLabel(**kwargs)


# --- hold-out split (no leakage) -------------------------------------------


def test_hold_out_split_never_splits_a_hand():
    # Two hands, each spanning two frames: a hand must land in ONE side only.
    frames = [
        _label(0, hand_id="h1"),
        _label(1, hand_id="h1"),
        _label(2, hand_id="h2"),
        _label(3, hand_id="h2"),
    ]
    train, eval_ = hold_out_split(frames, ratio=0.5)
    for frame in frames:
        assert (frame.frame in train) != (frame.frame in eval_)


def test_hold_out_split_rejects_bad_ratio():
    frames = [_label(0), _label(1)]
    with pytest.raises(ValueError):
        hold_out_split(frames, ratio=0.0)
    with pytest.raises(ValueError):
        hold_out_split(frames, ratio=1.0)


def test_hold_out_split_no_leakage_between_sets():
    frames = [_label(i, hand_id=f"h{i//2}") for i in range(12)]
    train, eval_ = hold_out_split(frames, ratio=0.4)
    assert len(train & eval_) == 0, "a frame must be in exactly one side"


# --- template building from confirmed stacks --------------------------------


def test_build_templates_from_confirmed_stacks(tmp_path):
    # 10 frames, each slot 0 grows a confirmed stack so 0-9 all appear.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    labels = []
    for i in range(10):
        img = _make_frame("session_001", i, {0: i})
        cv2 = pytest.importorskip("cv2")
        name = f"session_001__t_{i:08d}__f_{i:06d}__{i:012d}.png"
        cv2.imwrite(str(frames_dir / name), img)
        label = _label(
            i,
            session="session_001",
            slots=(_slot(0, stack_status="VALID", stack_value=i),),
        )
        labels.append(label)
    templates = build_digit_templates_from_labels(labels, frames_dir)
    assert set(templates) == set("0123456789")
    assert len(set("0123456789")) == 10


def test_build_templates_skips_glyph_count_mismatch(tmp_path):
    # A stack whose value has 2 digits but the pill draws 1 is skipped — we
    # never invent a digit count.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Draw a 1-digit glyph into slot 0 but claim the value is a 3-digit 123.
    img = _make_frame("session_001", 0, {0: 7})
    cv2 = pytest.importorskip("cv2")
    name = "session_001__t_00000000__f_000000__000000000000.png"
    cv2.imwrite(str(frames_dir / name), img)
    label = _label(
        0,
        session="session_001",
        slots=(_slot(0, stack_status="VALID", stack_value=123),),
    )
    templates = build_digit_templates_from_labels([label], frames_dir)
    # The single glyph (7) is only a valid template if it is the only one and
    # the value is single-digit; with claimed value 123 it must be skipped.
    assert "1" not in templates
    assert "2" not in templates
    assert "3" not in templates


# --- reading + gating -------------------------------------------------------


def test_read_accepts_a_clean_high_margin_read(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Build one known frame so we have a template for "1" from a confirmed
    # stack, then read a target that also shows "1".
    img0 = _make_frame("session_001", 0, {0: 1})
    cv2 = pytest.importorskip("cv2")
    name0 = "session_001__t_00000000__f_000000__000000000000.png"
    cv2.imwrite(str(frames_dir / name0), img0)
    known = _label(
        0,
        session="session_001",
        slots=(_slot(0, stack_status="VALID", stack_value=1),),
    )
    templates = build_digit_templates_from_labels([known], frames_dir)
    # A target slot (UNKNOWN) that shows a "1".
    tgt_label = _label(
        1,
        session="session_001",
        slots=(_slot(0, stack_status="UNKNOWN"),),
    )
    cv2.imwrite(
        str(frames_dir / tgt_label.frame),
        _make_frame("session_001", 1, {0: 1}),
    )
    cands = read_targets([tgt_label], frames_dir, templates)
    assert len(cands) == 1
    assert cands[0].accepted is True
    assert cands[0].value == 1


def test_read_rejects_low_margin_lookalike(tmp_path):
    # 1 vs 7 are confusable; if the gate thinks the margin is small the read
    # must be UNKNOWN. We simply assert that a low-margin candidate is never
    # accepted, even when the best digit happens to be right.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(
        0,
        session="session_001",
        slots=(_slot(0, stack_status="VALID", stack_value=7),),
    )
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 7}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    tgt = _label(
        1,
        session="session_001",
        slots=(_slot(0, stack_status="UNKNOWN"),),
    )
    # Draw a digit actually different from template 7 so the margin is real.
    cv2.imwrite(str(frames_dir / tgt.frame), _make_frame("session_001", 1, {0: 1}))
    cands = read_targets([tgt], frames_dir, templates)
    # With only a single "1" template and a "7" template, the read must be
    # accepted only if the margin permits; we assert the gate's contract:
    # accepted implies every digit passed both gates.
    for c in cands:
        if c.accepted:
            assert all(d.accepted for d in c.digits)
            assert all(
                d.margin >= MARGIN_THRESHOLD for d in c.digits
            ), "accepted digit must meet the margin gate"
            assert all(
                d.best_dist < FIT_THRESHOLD for d in c.digits
            ), "accepted digit must meet the fit gate"


def test_read_never_accepts_a_blank_pill(tmp_path):
    # An empty ROI (no glyph) must read as no candidate, never a guess.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(
        0,
        session="session_001",
        slots=(_slot(0, stack_status="VALID", stack_value=1),),
    )
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 1}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    # Empty target frame: no stack pill drawn at all.
    empty_label = _label(
        1, session="session_001", slots=(_slot(0, stack_status="UNKNOWN"),)
    )
    frame = np.zeros((1080, 498, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / empty_label.frame), frame)
    cands = read_targets([empty_label], frames_dir, templates)
    assert cands == ()


def test_read_fails_closed_on_digit_merge(tmp_path):
    # A pill whose digits are drawn touching (the UI's "digit merge", e.g. a
    # "198" where the 9 and 8 touch) must NOT be read as the surviving glyph's
    # value (e.g. a confident "1"). The merge is a reliability signal: the
    # whole read is rejected (no candidate), never a truncated single digit.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(
        0,
        session="session_001",
        slots=(_slot(0, stack_status="VALID", stack_value=1),),
    )
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 1}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    # A target slot whose pill is drawn with a *merging* digit stride, using
    # "194" at ``_MERGE_GAP``: the 9+4 bridge fuses into a >20px blob while
    # the leading "1" survives as a clean glyph. Without the fail-closed path
    # this would read as a confident "1", so the test is the strongest guard
    # for the "198 -> 1" bug. (See the ``_MERGE_GAP`` note for why "198" is
    # not used here — its 9+8 bridge stays at 20px, under the threshold.)
    merged_label = _label(
        1, session="session_001", slots=(_slot(0, stack_status="UNKNOWN"),)
    )
    cv2.imwrite(
        str(frames_dir / merged_label.frame),
        _make_frame_merge("session_001", 1, {0: 194}),
    )
    cands = read_targets([merged_label], frames_dir, templates)
    # A merged pill cannot be read confidently; it must yield no candidate
    # (never a truncated single digit such as a confident "1").
    assert cands == ()


def test_split_stack_digits_reports_a_merge(tmp_path):
    # ``split_stack_digits`` must signal ``had_merge`` when the dilation fuses
    # adjacent digits, while ``_split_digits`` (the historical discard path)
    # keeps the pre-existing behaviour so other callers/tests are unchanged.
    pytest.importorskip("cv2")
    merged = _make_frame_merge("session_001", 1, {0: 194})
    layout = sr.SLOT_LAYOUT_MULTI
    H, W = merged.shape[:2]
    row = layout[0]
    pill = merged[
        int((row["cy"] - row["h"] / 2) * H): int((row["cy"] + row["h"] / 2) * H),
        int((row["cx"] - row["w"] / 2) * W): int((row["cx"] + row["w"] / 2) * W),
    ]
    glyphs, had_merge = sr.split_stack_digits(pill)
    assert had_merge is True
    # The merge leaves the leading "1" as a clean surviving glyph — the exact
    # hazard the fail-closed split is designed to surface (a confident "1").
    assert len(glyphs) == 1
    # The clean-pill helper (drawn at _DIGIT_GAP) must NOT report a merge.
    clean = _make_frame("session_001", 2, {0: 100})
    row = layout[0]
    pill_clean = clean[
        int((row["cy"] - row["h"] / 2) * H): int((row["cy"] + row["h"] / 2) * H),
        int((row["cx"] - row["w"] / 2) * W): int((row["cx"] + row["w"] / 2) * W),
    ]
    clean_glyphs, clean_merge = sr.split_stack_digits(pill_clean)
    assert clean_merge is False
    assert len(clean_glyphs) == 3


def test_read_treats_conflict_as_not_a_target(tmp_path):
    # A CONFLICT stack is a re-read problem, never a machine proposal.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(
        0, session="session_001", slots=(_slot(0, stack_status="VALID", stack_value=1),)
    )
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 1}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    conflict_label = _label(
        1, session="session_001", slots=(_slot(0, stack_status="CONFLICT"),)
    )
    img3 = _make_frame("session_001", 1, {0: 3})
    cv2.imwrite(str(frames_dir / conflict_label.frame), img3)
    cands = read_targets([conflict_label], frames_dir, templates)
    assert cands == ()


def test_read_marks_known_slots_but_never_proposes_them(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(0, session="session_001",
                   slots=(_slot(0, stack_status="VALID", stack_value=1),))
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 1}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    cands = read_targets([known], frames_dir, templates)
    assert cands[0].known is True
    assert cands[0].known_value == 1
    # A known slot must not appear in the proposal CSV.
    csv_text = render_proposal_csv(cands)
    assert csv_text.strip() == "frame,slot_id,value"


def test_read_never_proposes_a_lone_zero_placeholder(tmp_path):
    # The UI draws a static "0" chip in an "awaiting-review" seat, which is
    # pixel-identical to a real zero stack (a single ~11px "0" glyph). A
    # lone-zero read for an OCCUPIED-but-UNKNOWN target cannot be told apart
    # from that placeholder, so it must fail closed to UNKNOWN — never a "0"
    # proposal. A genuine multi-digit read (e.g. "100") still proposes.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    # Build templates so BOTH "0" and "1" are in the library: the reader only
    # flags a lone zero when it has a real "0" template to match (the real
    # run builds 0 templates from other confirmed stacks).
    known1 = _label(
        0, session="session_001", slots=(_slot(0, stack_status="VALID", stack_value=1),)
    )
    cv2.imwrite(str(frames_dir / known1.frame), _make_frame("session_001", 0, {0: 1}))
    known0 = _label(
        3, session="session_001", slots=(_slot(0, stack_status="VALID", stack_value=0),)
    )
    cv2.imwrite(str(frames_dir / known0.frame), _make_frame("session_001", 3, {0: 0}))
    templates = build_digit_templates_from_labels([known1, known0], frames_dir)
    assert set(templates) >= {"0", "1"}
    # A target slot showing a lone "0" — indistinguishable from the placeholder.
    zero_label = _label(
        1, session="session_001", slots=(_slot(0, stack_status="UNKNOWN"),)
    )
    cv2.imwrite(
        str(frames_dir / zero_label.frame),
        _make_frame("session_001", 1, {0: 0}),
    )
    cands = read_targets([zero_label], frames_dir, templates)
    # The candidate exists (the glyph was read) but must NOT be accepted.
    assert len(cands) == 1
    assert cands[0].accepted is False
    assert cands[0].status == "UNKNOWN"
    assert cands[0].value is None
    # A multi-digit read keeps working: a target showing "100" proposes 100.
    multi_label = _label(
        2, session="session_001", slots=(_slot(0, stack_status="UNKNOWN"),)
    )
    cv2.imwrite(
        str(frames_dir / multi_label.frame),
        _make_frame("session_001", 2, {0: 100}),
    )
    multi = read_targets([multi_label], frames_dir, templates)
    assert len(multi) == 1
    assert multi[0].accepted is True
    assert multi[0].value == 100
    # The proposal CSV must carry no lone-zero row.
    csv_text = render_proposal_csv(cands)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "frame,slot_id,value"
    assert len(lines) == 1


# --- proposal CSV ----------------------------------------------------------


def test_proposal_csv_only_accepted_targets(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cv2 = pytest.importorskip("cv2")
    known = _label(0, session="session_001",
                   slots=(_slot(0, stack_status="VALID", stack_value=1),))
    cv2.imwrite(str(frames_dir / known.frame), _make_frame("session_001", 0, {0: 1}))
    templates = build_digit_templates_from_labels([known], frames_dir)
    tgt = _label(1, session="session_001",
                 slots=(_slot(0, stack_status="UNKNOWN"),))
    cv2.imwrite(str(frames_dir / tgt.frame), _make_frame("session_001", 1, {0: 1}))
    cands = read_targets([tgt], frames_dir, templates)
    csv_text = render_proposal_csv(cands)
    lines = csv_text.strip().splitlines()
    assert lines[0] == "frame,slot_id,value"
    accepted = [c for c in cands if c.accepted]
    assert len(lines) - 1 == len(
        [c for c in accepted if not c.known]
    )


# --- evaluation ------------------------------------------------------------


def test_evaluate_only_scores_known_eval_frames():
    # Build candidates where some are known-eval, some known-train, some target.
    known_eval = StackCandidate(
        frame="e.png", session_id="s", hand_id="h", timestamp_ms=0,
        slot_id=1, layout_key="multi", digits=(),
        value=165, status="ACCEPT", known=True, known_value=155,
    )
    true_eval = StackCandidate(
        frame="e2.png", session_id="s", hand_id="h", timestamp_ms=0,
        slot_id=2, layout_key="multi", digits=(),
        value=155, status="ACCEPT", known=True, known_value=155,
    )
    target = StackCandidate(
        frame="t.png", session_id="s", hand_id="h", timestamp_ms=0,
        slot_id=0, layout_key="multi", digits=(),
        value=100, status="ACCEPT", known=False, known_value=None,
    )
    summary, mismatches = evaluate_candidates(
        [known_eval, true_eval, target], eval_frames={"e.png", "e2.png"}
    )
    assert summary.eval_total == 2
    assert summary.eval_accepted == 2
    assert summary.accepted == 1  # only the (non-known) target is a proposal
    assert summary.accepted_correct == 1
    assert summary.accepted_false == 1
    assert summary.eval_precision == 0.5
    assert len(mismatches) == 1
    assert mismatches[0].value == 165


def test_report_is_transparent_about_mismatches():
    mismatched = StackCandidate(
        frame="e.png", session_id="s", hand_id="h", timestamp_ms=0,
        slot_id=1, layout_key="multi", digits=(),
        value=165, status="ACCEPT", known=True, known_value=155,
    )
    summary = AutoStackSummary(
        train_frames=1, eval_frames=1, train_stacks=1, eval_stacks=1,
        template_digit_samples=10, targets=0, accepted=0, eval_accepted=1,
        accepted_correct=0, accepted_false=1, eval_unknown=0, eval_total=1,
        eval_precision=0.0, eval_recall=1.0, digits_covered=10,
    )
    report = render_auto_report(summary, [mismatched], [mismatched])
    assert "accepted false" in report
    assert "e.png" in report
    assert "165" in report


# --- integration over a synthetic corpus -----------------------------------


def test_full_pipeline_yields_sane_metrics(tmp_path, monkeypatch):
    # A small synthetic corpus: 8 frames in one session, each with a handful
    # of confirmed stacks + one UNKNOWN target, so the whole run works and
    # produces a (possibly 0-recall) but well-formed summary.
    cv2 = pytest.importorskip("cv2")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    labels = []
    for i in range(8):
        hand = f"hand_{i // 2}"
        slot_values = {
            0: 100 + i,
            1: 200 + i,
        }
        label = _label(
            i,
            session="session_001",
            hand_id=hand,
            slots=(
                _slot(0, stack_status="VALID", stack_value=100 + i),
                _slot(1, stack_status="VALID", stack_value=200 + i),
                _slot(2, stack_status="UNKNOWN"),
            ),
        )
        img = _make_frame("session_001", i, slot_values)
        cv2.imwrite(str(frames_dir / label.frame), img)
        labels.append(label)

    from tools.capture_card_calibration.dataset import (
        create_skeleton,
        write_frames_jsonl,
    )
    from tools.capture_card_calibration.stack_auto import run_stack_auto

    root = create_skeleton(tmp_path / "calib")
    path = root / "labels" / "frames.jsonl"
    write_frames_jsonl(path, labels)

    result = run_stack_auto(path, frames_dir, session="session_001")
    assert result.summary.train_frames >= 1
    assert result.summary.eval_frames >= 1
    assert result.summary.eval_total >= 1
    assert result.summary.train_stacks >= 1
    assert result.summary.digits_covered >= 1
    # Proposals only ever target UNKNOWN slots.
    csv_text = render_proposal_csv(result.candidates)
    assert csv_text.splitlines()[0] == "frame,slot_id,value"
