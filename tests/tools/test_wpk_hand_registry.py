"""Hand-level split ownership must survive legacy aliases and uncertain provenance."""

from copy import deepcopy

import pytest

from tools.wpk_hand_registry import (
    assert_owned_frames, audit_hand, overlaps, validate_hands,
)


def hand(start=100, end=199, role="frozen_regression"):
    return {"hand_id": f"hand-{start}", "start": start, "end": end,
            "role": role, "opening_players": 6, "context_start": start - 1,
            "context_end": end + 1, "boundary_reviewed": True,
            "full_action_truth_ready": False}


def test_adjacent_owned_hands_are_disjoint_even_with_boundary_context():
    validate_hands([hand(), hand(200, 299)])
    assert not overlaps((100, 199), (200, 299))


def test_legacy_alias_overlap_cannot_become_two_split_owners():
    with pytest.raises(ValueError, match="overlap"):
        validate_hands([hand(), hand(190, 299, "acceptance_candidate")])


@pytest.mark.parametrize("frames", [[99], [200], [100, 100], [], [True]])
def test_context_adjacent_and_duplicate_frames_never_scored(frames):
    with pytest.raises(ValueError):
        assert_owned_frames(hand(), frames)


def test_boundary_inclusive_exposure_blocks_acceptance():
    target = hand(role="independent_acceptance")
    target["full_action_truth_ready"] = True
    with pytest.raises(ValueError, match="promote"):
        audit_hand(target, [{"start": 199, "end": 205}], True)


def test_missing_history_is_not_evidence_of_clean_holdout():
    result = audit_hand(hand(), [], False)
    assert result["exposure_status"] == "PROVENANCE_UNVERIFIED"
    assert not result["independent_acceptance_eligible"]


def test_complete_provenance_and_truth_required_before_independent_role():
    target = hand(role="independent_acceptance")
    with pytest.raises(ValueError):
        audit_hand(target, [], True)
    target["full_action_truth_ready"] = True
    assert audit_hand(target, [], True)["independent_acceptance_eligible"]
    assert not audit_hand({**target, "role": "frozen_regression"}, [], True)[
        "independent_acceptance_eligible"]


def test_audit_does_not_rewrite_role_or_exposure():
    target = hand()
    evidence = [{"start": 105, "end": 110, "kind": "training"}]
    original = deepcopy((target, evidence))
    result = audit_hand(target, evidence, False)
    assert (target, evidence) == original
    assert result["exposure_status"] == "KNOWN_EXPOSURE"


@pytest.mark.parametrize("change", [
    "duplicate_id", "empty", "boundary", "float_players",
])
def test_invalid_registry_structure_rejected(change):
    rows = [hand()]
    if change == "duplicate_id":
        rows.append(deepcopy(rows[0]))
    elif change == "empty":
        rows = []
    elif change == "boundary":
        rows[0]["context_end"] = 201
    else:
        rows[0]["opening_players"] = 6.0
    with pytest.raises(ValueError):
        validate_hands(rows)
