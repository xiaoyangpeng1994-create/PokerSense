"""Pixel truth and logical ledgers must remain separate during field audits."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.capture_card_calibration.hashing import sha256_file, write_sha256sums
from tools.probe_wpk_observation_fields import (
    bind_samples, compare, run, score_read, validate_review,
)
from tools.wpk_video_dataset import pixels_digest


@pytest.fixture
def review():
    path = Path(__file__).resolve().parents[1] / (
        "fixtures/wpk_reference_hands/aq_observation_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("expected,read,result", [
    ("0", {"status": "valid", "value": "0"}, "correct"),
    (None, {"status": "valid", "value": "0"}, "false_accept"),
    ("64", {"status": "unknown", "value": "64"}, "abstain"),
    ("call", {"status": "conflict", "value": "call"}, "abstain"),
    (None, {"status": "unknown", "value": None}, "correct_rejection"),
    ("64", {"status": "valid", "value": "84"}, "wrong"),
])
def test_score_never_turns_unknown_or_negative_into_zero(expected, read, result):
    assert score_read(expected, read) == result


def test_decimal_equivalence_does_not_require_identical_format():
    assert score_read("94", {"status": "valid", "value": "94.0"},
                      money=True) == "correct"


def test_screen_truth_does_not_get_replaced_by_logical_pot(review):
    validate_review(review)
    point = next(p for p in review["checkpoints"] if p["source_frame"] == 8560)
    original = deepcopy(point)
    result = compare({"checkpoints": [point]}, {8560: {
        "pot": {"status": "valid", "value": "156"}, "stacks": {}, "actions": {},
    }})
    assert point == original
    assert result["counts"]["pot"] == {"correct": 1}
    assert result["display_vs_ledger_mismatches"] == [{
        "source_frame": 8560, "displayed_pot": "156", "logical_pot": "218",
        "eligible_for_atomic_state_match": False,
    }]
    assert result["counts"]["actions"] == {"abstain": 8}
    assert result["checks"][-1]["observed"]["reason"] == "missing_roi"


@pytest.mark.parametrize("change", ["duplicate", "slots", "mapping", "negative"])
def test_invalid_review_rejected(review, change):
    if change == "duplicate":
        review["checkpoints"].append(deepcopy(review["checkpoints"][0]))
    elif change == "slots":
        review["checkpoints"][0]["stacks"].pop()
    elif change == "mapping":
        review["slot_to_source_seat"] = list(reversed(range(8)))
    else:
        review["checkpoints"][0]["pot"] = "-1"
    with pytest.raises(ValueError):
        validate_review(review)


def make_window(tmp_path):
    folder = tmp_path / "window"
    folder.mkdir()
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    path = folder / "image.png"
    path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    sample = {"source_frame": 1, "container_pts_ms": 1.0, "image": "image.png",
              "image_sha256": sha256_file(path), "pixel_sha256": pixels_digest(image)}
    (folder / "samples.json").write_text(json.dumps([sample]))
    (folder / "summary.json").write_text(json.dumps({"source_sha256": "source"}))
    write_sha256sums(folder)
    return folder, {"source_sha256": "source", "checkpoints": [
        {"window": "hand", "source_frame": 1}]}


def test_image_binding_and_tampering_are_checked(tmp_path):
    folder, review = make_window(tmp_path)
    assert len(bind_samples(review, {"hand": folder})) == 1
    (folder / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        bind_samples(review, {"hand": folder})


def test_wrong_video_or_missing_frame_rejected(tmp_path):
    folder, review = make_window(tmp_path)
    review["source_sha256"] = "another"
    with pytest.raises(ValueError, match="another source"):
        bind_samples(review, {"hand": folder})
    review["source_sha256"] = "source"
    review["checkpoints"][0]["source_frame"] = 2
    with pytest.raises(ValueError, match="missing reviewed"):
        bind_samples(review, {"hand": folder})


def test_prior_output_is_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match="preserve prior"):
        run(tmp_path / "absent.json", {}, tmp_path)
