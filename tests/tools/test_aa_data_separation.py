import json
from pathlib import Path

import pytest

from tools.aa_data_separation import (
    assert_training_frames, eligible_hand, validate_reservations,
)


def reservations():
    path = Path(__file__).resolve().parents[2] / (
        "configs/reproduction/aa_holdout_reservations_v1.json")
    return json.loads(path.read_text())


def test_current_templates_do_not_use_reservations():
    assert_training_frames([0, 1200, 1320, 1740, 29700], reservations())


@pytest.mark.parametrize("frame", [2701, 3599, 6301, 7199, 18001, 18899])
def test_reserved_endpoints_cannot_train(frame):
    with pytest.raises(ValueError, match="reserved"):
        assert_training_frames([frame], reservations())


def test_no_claim_without_reviewed_complete_boundaries():
    data = reservations()
    assert not eligible_hand(2800, 3000, data)
    assert eligible_hand(2800, 3000, data, boundaries_reviewed=True)
    assert not eligible_hand(2700, 3000, data, boundaries_reviewed=True)
    assert not eligible_hand(2800, 3600, data, boundaries_reviewed=True)


def test_exposure_conflict_rejected():
    data = reservations()
    data["known_exploration_frames"].append(2800)
    with pytest.raises(ValueError, match="exposed"):
        validate_reservations(data)
