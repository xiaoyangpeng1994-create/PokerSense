"""Candidate integration must not emit events or promote calibration status."""

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from poker_engine.perceptual.capture.base import Frame, WindowRect
from poker_engine.perceptual.vision.gray_amount_recognizer import GrayAmountRecognizer
from tools.wpk_combined_vision import CombinedVisionCandidate, FrameActions
from tools.wpk_field_features_v2 import FeatureActionCandidate
from tools.wpk_video_dataset import read_image
from .test_gray_amount_recognizer import bank


@pytest.fixture
def candidate(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    profile = json.loads((repo / "tests/fixtures/wpk_reference_hands/"
                          "field_candidate_v1.json").read_text())
    folder = repo / "configs/vision/wepoker_android_capture_card"
    images = {p.stem: read_image(p) for p in (folder / "action_glyph").glob("*.png")}
    for name, color in (("fold", (255, 255, 255)), ("all_in", (0, 255, 255))):
        patch = np.zeros((20, 50, 3), np.uint8)
        cv2.putText(patch, "Fold" if name == "fold" else "Allin", (1, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        images[name] = patch
    patch = np.zeros((28, 88, 3), np.uint8)
    cv2.ellipse(patch, (44, 14), (39, 11), 0, 0, 360, (180, 180, 180), 1)
    monkeypatch.setattr("tools.wpk_combined_vision.load_reviewed_bank",
                        lambda *args, **kwargs: GrayAmountRecognizer(*bank()))
    return CombinedVisionCandidate(None, folder / "card_heads.npz", profile, images,
                                   {"lower": patch, "raised": patch}, pot_bank=None)


def frame(image, index=1):
    return Frame(index, datetime(2000, 1, 1, tzinfo=timezone.utc), "test-candidate",
                 WindowRect(0, 0, image.shape[1], image.shape[0]), image,
                 image.shape[1], image.shape[0])


def test_combined_vision_routes_all_action_slots_without_promoting(candidate):
    image = np.zeros((1080, 498, 3), np.uint8)
    before = image.copy()
    result = candidate.process(frame(image))
    assert set(result["actions"]) == set(map(str, range(8)))
    assert set(result["production_action_status"].values()) == {"unknown"}
    assert result["events_emitted"] == 0 and not result["release_eligible"]
    assert not result["confidence_calibration_verified"]
    assert result["participation"] == "UNKNOWN"
    assert np.array_equal(before, image)


def test_invalid_frame_clears_pending_candidate_state(candidate):
    candidate.action_adapter.reads["0"] = {"value": "call"}
    candidate.cards.reads[("hero", 0)] = {"card": "Ac"}
    with pytest.raises(ValueError, match="canvas"):
        candidate.process(frame(np.zeros((100, 100, 3), np.uint8)))
    assert not candidate.action_adapter.reads and not candidate.cards.reads


def test_action_adapter_does_not_reuse_an_absent_slot():
    adapter = FrameActions()
    adapter.reads["0"] = {"value": "call", "score": .95, "runner_up": .2}
    assert adapter.recognize(None, 0).value.value == "call"
    assert adapter.recognize(None, 7).value is None
    adapter.reads.clear()
    assert adapter.recognize(None, 0).value is None


def test_unsupported_scene_clears_visual_values(candidate):
    candidate.scene = SimpleNamespace(recognize=lambda picture: SimpleNamespace(
        supported=False, reason="unsupported_or_occluded"))
    candidate.cards.reads[("hero", 0)] = {"card": "Ac"}
    result = candidate.process(frame(np.zeros((1080, 498, 3), np.uint8)))
    assert result["hero"] == [None, None] and result["pot"]["value"] is None
    assert all(s["value"] is None for s in result["stacks"].values())
    assert not candidate.cards.reads and result["events_emitted"] == 0


def test_action_variants_reject_unknown_labels(candidate):
    with pytest.raises(ValueError, match="unknown action"):
        FeatureActionCandidate(candidate.actions.profile,
                               {k: np.repeat(v[:, :, None], 3, axis=2)
                                for k, v in candidate.actions.templates.items()},
                               variants={"not_an_action": []})
