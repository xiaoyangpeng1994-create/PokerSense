import json
from pathlib import Path

import pytest

from tools.aa_visual_candidate import table_map, validate_layout
from tools.aa_combined_reader import AACombinedReader


def profile():
    root = Path(__file__).resolve().parents[2]
    folder = root / "configs/vision/aa_android_capture_card"
    return json.loads((folder / "layout.8seat.candidate.json").read_text())


def test_eight_visual_positions_and_hero_are_explicit():
    candidate = profile()
    mapping = table_map(candidate)
    stacks = [r.slot_id for r in mapping.rois if r.kind.value == "stack"]
    assert stacks == list(range(8))
    assert candidate["hero_slot"] == 4


def test_eight_layout_cannot_enter_legacy_nine_slot_reader():
    with pytest.raises(ValueError, match="not been validated"):
        AACombinedReader(profile(), None, None, None, None)


def test_old_hero_mapping_not_silently_reused():
    candidate = profile()
    candidate["hero_slot"] = 5
    with pytest.raises(ValueError, match="Hero"):
        validate_layout(candidate)
