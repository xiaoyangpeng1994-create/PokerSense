"""Offline candidates remain bounded, ambiguous evidence abstains."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from poker_engine.perceptual.vision.table_map import ROI, ROIKind, TableMap
from tools.wpk_field_candidate import (
    ActionCandidate, choose_action, stack_table, validate_profile,
)


@pytest.fixture
def profile():
    return json.loads((Path(__file__).resolve().parents[1] / (
        "fixtures/wpk_reference_hands/field_candidate_v1.json")).read_text())


@pytest.mark.parametrize("scores,expected", [
    ({"call": .95, "raise": .9}, None),
    ({"call": .95, "raise": .7}, "call"),
    ({"fold": .84, "call": .2}, None),
    ({"fold": .9, "call": .9}, None),
])
def test_conflict_and_low_score_abstain(scores, expected):
    result = choose_action(scores, .85, .1)
    assert result["value"] == expected
    assert result["not_an_event"] and not result["production_valid"]


@pytest.mark.parametrize("change", ["canvas", "missing_slot", "outside", "nan"])
def test_geometry_and_gates_fail_closed(profile, change):
    if change == "canvas":
        profile["canvas"] = [500, 1080]
    elif change == "missing_slot":
        profile["slots"].pop()
    elif change == "outside":
        profile["slots"][0]["avatar"][0] = 480
    else:
        profile["action_floor"] = float("nan")
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_stack_geometry_does_not_mutate_original_table_or_other_fields(profile):
    original = TableMap("test", "test", (498, 1080), rois=(
        ROI(ROIKind.POT, .1, .1, .1, .1),
        ROI(ROIKind.STACK, .2, .2, .2, .2, 0)))
    copy = deepcopy(original)
    changed = stack_table(original, profile)
    assert original == copy
    assert changed.rois[0] == original.rois[0]
    assert [r.slot_id for r in changed.rois[1:]] == list(range(8))


def test_empty_templates_rejected(profile):
    templates = {name: np.zeros((5, 5), np.uint8)
                 for name in ("call", "bet", "raise", "check", "fold", "all_in")}
    with pytest.raises(ValueError, match="degenerate"):
        ActionCandidate(profile, templates)


def test_candidate_copies_inputs_and_rejects_unknown_canvas(profile):
    patch = np.zeros((5, 5), np.uint8)
    patch[1:3, 1:3] = 255
    templates = {name: patch for name in
                 ("call", "bet", "raise", "check", "fold", "all_in")}
    candidate = ActionCandidate(profile, templates)
    profile["slots"].clear()
    patch[:] = 0
    assert len(candidate.profile["slots"]) == 8
    assert candidate.templates["fold"].sum() > 0
    assert not candidate.templates["fold"].flags.writeable
    with pytest.raises(ValueError, match="wrong canvas"):
        candidate.recognize(np.zeros((1080, 500, 3), np.uint8))
    results = candidate.recognize(np.zeros((1080, 498, 3), np.uint8))
    assert len(results) == 8 and all(r["value"] is None for r in results.values())
