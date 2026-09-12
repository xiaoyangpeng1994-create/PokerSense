"""Training authorization, split isolation and deterministic augmentation."""

from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from tools.capture_card_calibration.hashing import sha256_file
from tools.train_wpk_rank_head import RANKS, augment_glyph, reviewed_samples
from tools.validate_wpk_card_batch import checked_candidate_heads, checked_rank_floor
from tools.calibrate_wpk_rank_floor import select_floor


def complete_review():
    rows = [{"id": rank, "session": "session_001", "split": "train",
             "origins": [{"frame": "session_001__example.png"}]}
            for rank in sorted(RANKS)]
    return {"samples": rows}, {"approved": {r: [r] for r in RANKS}, "rejected": {}}


def test_accepts_only_explicitly_reviewed_all_rank_training_set():
    manifest, review = complete_review()
    before = deepcopy(manifest)
    assert len(reviewed_samples(manifest, review)) == 13
    assert manifest == before


@pytest.mark.parametrize("problem", ["unreviewed", "duplicate", "invalid_rank",
                                     "missing_rank", "session", "split", "origin"])
def test_training_rejects_invalid_review_and_evaluation_leakage(problem):
    manifest, review = complete_review()
    if problem == "unreviewed":
        review["approved"].pop("A")
    elif problem == "duplicate":
        review["approved"]["A"].append("A")
    elif problem == "invalid_rank":
        review["approved"]["X"] = review["approved"].pop("A")
    elif problem == "missing_rank":
        review["approved"].pop("A")
        review["rejected"]["A"] = "unreadable"
    elif problem == "session":
        manifest["samples"][0]["session"] = "session_002"
    elif problem == "split":
        manifest["samples"][0]["split"] = "validation"
    else:
        manifest["samples"][0]["origins"][0]["frame"] = "session_002__example.png"
    with pytest.raises(ValueError):
        reviewed_samples(manifest, review)


def test_augmentation_is_seeded_and_does_not_mutate_input():
    glyph = np.full((26, 18, 3), 255, np.uint8)
    cv2.putText(glyph, "6", (1, 22), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 180), 2)
    before = glyph.copy()
    first = augment_glyph(glyph, np.random.default_rng(7))
    second = augment_glyph(glyph, np.random.default_rng(7))
    assert first is not None and first.shape == (40, 40)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(glyph, before)


def test_candidate_head_override_binds_both_weights_and_metadata(tmp_path):
    weights = tmp_path / "card_heads.npz"
    weights.write_bytes(b"test-weights")
    metadata = weights.with_suffix(".json")
    metadata.write_text(json.dumps({"format": "mlp-v1"}), encoding="utf-8")
    spec = {"candidate_heads": {
        "relative_path": "card_heads.npz", "npz_sha256": sha256_file(weights),
        "json_sha256": sha256_file(metadata)}}
    default = tmp_path / "production.npz"
    assert checked_candidate_heads({}, tmp_path, default) == default
    assert checked_candidate_heads(spec, tmp_path, default) == weights
    metadata.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        checked_candidate_heads(spec, tmp_path, default)
    spec["candidate_heads"]["relative_path"] = "../escaped.npz"
    with pytest.raises(ValueError, match="escape"):
        checked_candidate_heads(spec, tmp_path, default)


def test_rank_floor_is_model_bound_and_cannot_silently_be_lowered(tmp_path):
    artifact = tmp_path / "calibration.json"
    report = {"candidate_npz_sha256": "model-a", "chosen_rank_floor": .5}
    artifact.write_text(json.dumps(report), encoding="utf-8")
    spec = {"rank_calibration": {"relative_path": artifact.name,
                                 "sha256": sha256_file(artifact)},
            "candidate_heads": {"npz_sha256": "model-a"}}
    assert checked_rank_floor({}, tmp_path, .3) == .3
    assert checked_rank_floor(spec, tmp_path, .3) == .5
    with pytest.raises(ValueError, match="invalid"):
        checked_rank_floor(spec, tmp_path, .6)
    spec["candidate_heads"]["npz_sha256"] = "model-b"
    with pytest.raises(ValueError, match="another model"):
        checked_rank_floor(spec, tmp_path, .3)


def test_floor_selection_is_forbidden_on_new_holdout():
    with pytest.raises(ValueError, match="forbidden"):
        select_floor({"independence_scope": "New segment"}, {})


def test_grid_floor_rejects_wrong_card_without_hiding_required_coverage():
    spec = {"independence_scope": "Development rerun", "criteria": {
        "max_wrong_accepted": 0, "min_required_complete_fraction": .95},
        "checkpoints": [{"source_frame": 1, "hero": ["Ac", "5c"], "board": []},
                        {"source_frame": 2, "hero": ["Ac", "Qh"], "board": []}]}

    def row(index, last_card, score, required):
        return {"source_frame": index, "require_hero_complete": required,
                "require_board_complete": True, "hero": {"reads": [
                    {"card": "Ac", "rank_score": .9},
                    {"card": last_card, "rank_score": score}]},
                "board": {"reads": [{"card": None, "rank_score": None}] * 5}}
    result = {"checkpoints": [row(1, "6c", .476, False), row(2, "Qh", .9, True)]}
    assert select_floor(spec, result)["chosen_rank_floor"] == .5
    result["checkpoints"][1]["hero"]["reads"][1]["rank_score"] = .4
    with pytest.raises(ValueError, match="no reject floor"):
        select_floor(spec, result)
