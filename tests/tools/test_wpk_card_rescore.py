"""Label corrections retain original predictions and reject wrong provenance."""

from copy import deepcopy

import pytest

from tools.rescore_wpk_card_batch import rescore
from tools.validate_wpk_card_batch import score_slots


def inputs():
    spec = {"source_sha256": "source", "criteria": {"max_wrong_accepted": 0,
            "min_required_complete_fraction": 0.95}, "checkpoints": [{
                "source_frame": 1, "pixel_sha256": "pixels", "hero": ["Ac", "Qh"],
                "board": ["8c", "As", "Kd"]}]}
    hero = score_slots(["Ac", "Qh"], ["Ac", "Qh"], 2)
    hero["reads"] = [{"card": "Ac"}, {"card": "Qh"}]
    board = score_slots(["8c", "As", "Kd"], ["8c", "As", "Kh", None, None], 5)
    board["reads"] = [{"card": c} for c in ("8c", "As", "Kh", None, None)]
    result = {"checkpoints": [{
        "source_frame": 1, "hero": hero, "board": board,
        "require_hero_complete": True, "require_board_complete": True}]}
    edits = {"source_sha256": "source", "corrections": [{
        "source_frame": 1, "pixel_sha256": "pixels", "field": "board", "slot": 2,
        "from": "Kd", "to": "Kh"}]}
    return spec, result, edits


def test_rescore_does_not_change_original_spec_or_predictions():
    spec, result, edits = inputs()
    old_spec, old_result = deepcopy(spec), deepcopy(result)
    updated = rescore(spec, result, edits)
    assert updated["summary"]["passed"] is True
    assert spec == old_spec and result == old_result
    assert result["checkpoints"][0]["board"]["counts"]["wrong_accepted"] == 1


@pytest.mark.parametrize("key,value", (("pixel_sha256", "wrong"), ("from", "Ks")))
def test_inconsistent_correction_is_rejected(key, value):
    spec, result, edits = inputs()
    edits["corrections"][0][key] = value
    with pytest.raises(ValueError):
        rescore(spec, result, edits)
