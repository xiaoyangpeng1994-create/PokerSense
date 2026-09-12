"""Temporal evidence must not bridge missing frames or missing regions."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace

import numpy as np
import pytest

from .test_fused_card_recognizer import _always_ace_adapter, _card


EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def begin(adapter, seq, seconds=None, source="phone", regions=None):
    adapter.begin_frame(seq, EPOCH + timedelta(seconds=seq / 30 if seconds is None
                                               else seconds), source,
                        {"hero": "fixed", "board": "fixed"}
                        if regions is None else regions)


def warmed():
    adapter = _always_ace_adapter()
    for seq in range(3):
        begin(adapter, seq)
        result = adapter.recognize(_card("A"), ("hero", 0))
    assert result.value is not None
    return adapter


@pytest.mark.parametrize("change", ["skip", "long_gap", "time_backwards",
                                    "duplicate", "source", "region"])
def test_discontinuity_requires_new_glyphs(change):
    adapter = warmed()
    kwargs = {"seq": 3}
    if change == "skip":
        kwargs["seq"] = 8
    elif change == "long_gap":
        kwargs["seconds"] = 5
    elif change == "time_backwards":
        kwargs["seconds"] = -1
    elif change == "duplicate":
        kwargs["seq"] = 2
    elif change == "source":
        kwargs["source"] = "other phone"
    else:
        kwargs["regions"] = {"hero": "shifted", "board": "fixed"}
    begin(adapter, **kwargs)
    assert adapter.recognize(_card("A"), ("hero", 0)).value is None
    assert adapter._buffers[("hero", 0)].glyph_count == 1


def test_missing_entire_region_clears_its_history():
    adapter = warmed()
    begin(adapter, 3, regions={"hero": None, "board": "fixed"})
    assert ("hero", 0) not in adapter._buffers
    begin(adapter, 4)
    assert adapter.recognize(_card("A"), ("hero", 0)).value is None


def test_skipped_slot_does_not_reuse_older_evidence():
    adapter = warmed()
    begin(adapter, 3)  # region exists but its recognizer was never called
    begin(adapter, 4)
    assert adapter.recognize(_card("A"), ("hero", 0)).value is None


def test_one_frame_cannot_count_multiple_times():
    adapter = _always_ace_adapter()
    begin(adapter, 0)
    for _ in range(6):
        assert adapter.recognize(_card("A"), ("hero", 0)).value is None
    assert adapter._buffers[("hero", 0)].glyph_count == 1


def test_contiguous_frames_keep_the_usable_buffer():
    adapter = warmed()
    begin(adapter, 3)
    assert adapter.recognize(_card("A"), ("hero", 0)).value is not None
    assert adapter._buffers[("hero", 0)].glyph_count == 4


def test_region_explicitly_absent_cannot_collect_samples():
    adapter = warmed()
    for seq in range(3, 7):
        begin(adapter, seq, regions={"hero": None, "board": "fixed"})
        assert adapter.recognize(_card("A"), ("hero", 0)).value is None
    assert ("hero", 0) not in adapter._buffers


def test_missing_hero_region_does_not_erase_board_evidence():
    adapter = _always_ace_adapter()
    for seq in range(3):
        begin(adapter, seq)
        adapter.recognize(_card("A"), ("hero", 0))
        adapter.recognize(_card("A"), ("board", 0))
    begin(adapter, 3, regions={"hero": None, "board": "fixed"})
    assert adapter.recognize(_card("A"), ("board", 0)).value is not None
    assert adapter._buffers[("board", 0)].glyph_count == 4


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_frame_gap_budget_must_be_finite_positive(bad):
    from poker_engine.perceptual.vision.fused_card_adapter import (
        FusedCardRecognizerAdapter,
    )

    with pytest.raises(ValueError):
        FusedCardRecognizerAdapter(object(), max_frame_gap_seconds=bad)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("mutation", ["missing", "same_size_moved"])
def test_real_vision_engine_reports_roi_changes_to_adapter(wrapped, mutation):
    from poker_engine.perceptual.capture.base import Frame, WindowRect
    from poker_engine.perceptual.vision.table_map import ROIKind
    from tools.validate_wpk_card_batch import RecordingRecognizer
    from .test_engine import _engine, _table_map

    engine, table, adapter = _engine(), _table_map(), _always_ace_adapter()
    engine._card = RecordingRecognizer(adapter, .3) if wrapped else adapter
    engine._no_hero_dynamic = True
    image = np.full((400, 600, 3), (30, 100, 40), np.uint8)
    for x in (0, 120):
        image[280:358, x:x + 53] = _card("A")

    def frame(seq):
        return Frame(seq, EPOCH + timedelta(seconds=seq / 30), "phone",
                     WindowRect(0, 0, 600, 400), image, 600, 400)

    for seq in range(3):
        engine.process(frame(seq), table)
    assert adapter._buffers[("hero", 0)].glyph_count == 3
    if mutation == "missing":
        changed = replace(table, rois=tuple(
            r for r in table.rois if r.kind is not ROIKind.HERO_CARDS))
    else:
        changed = replace(table, rois=tuple(
            replace(r, x=.01) if r.kind is ROIKind.HERO_CARDS else r
            for r in table.rois))
    engine.process(frame(3), changed)
    if mutation == "missing":
        assert ("hero", 0) not in adapter._buffers
    else:
        assert adapter._buffers[("hero", 0)].glyph_count <= 1
    engine.process(frame(4), table)
    assert adapter._buffers[("hero", 0)].glyph_count == 1
