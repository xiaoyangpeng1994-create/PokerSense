import json
from pathlib import Path

import numpy as np
import pytest

from tools.aa_bomb_candidate import AABombTitleCandidate
from tools.verify_aa_bomb_posts import audit_case


def cases():
    path = Path(__file__).resolve().parents[1] / (
        "fixtures/aa_reference_hands/bomb_posts_v1.json")
    return json.loads(path.read_text())["cases"]


def test_two_levels_are_not_hardcoded_to_one_bet_size():
    first, second = map(audit_case, cases())
    assert first["expected_collection"] == "120"
    assert second["expected_collection"] == "90"
    assert first["collection_matches"] and second["collection_matches"]
    assert first["new_dealer_extra_matches_mushroom_setting"]
    assert second["new_dealer_extra_matches_mushroom_setting"]
    assert first["all_cash_observed"]
    assert second["unknown_balance_slots"] == [4]
    assert not second["rake_verified"]
    assert not first["mushroom_routing_verified"]
    assert first["mushroom_display_change"] == "-8"
    assert not first["mushroom_display_change_matches_dealer_extra"]
    assert second["mushroom_display_change"] == "4"
    assert second["mushroom_display_change_matches_dealer_extra"]


def test_pool_conflict_is_not_corrected():
    case = cases()[0]
    case["collected_display"] = "121"
    assert not audit_case(case)["collection_matches"]


def test_flat_reference_rejected():
    with pytest.raises(ValueError):
        AABombTitleCandidate(np.zeros((1080, 498, 3), np.uint8))


def test_title_presence_never_means_strategy_eligible():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (1080, 498, 3), dtype=np.uint8)
    reader = AABombTitleCandidate(image)
    result = reader.recognize(None)
    assert result["critical_hit_animation"] is None
    assert not result["strategy_eligible"]
    assert reader.recognize(np.zeros_like(image))["critical_hit_animation"] is None
