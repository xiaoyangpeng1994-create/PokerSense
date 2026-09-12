import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayRead
from tools.aa_amount_candidate import AAAmountCandidate, stack_patch
from tools.aa_action_candidate import AAActionCandidate, text_patch
from tools.aa_card_visibility import locate_face_card
from tools.aa_combined_reader import CurrentFrameConsensus


def profile():
    path = Path(__file__).resolve().parents[2] / (
        "configs/vision/aa_android_capture_card/layout.candidate.json")
    return json.loads(path.read_text())


class Scene:
    def recognize(self, image):
        return {"scene": "AA_TABLE_CANDIDATE" if image.any() else "UNKNOWN"}


class Bank:
    def diagnose(self, patch):
        return GrayRead("0", "0", "candidate", ())


def test_nine_stacks_zero_is_not_participation_and_invalid_scene_clears():
    reader = AAAmountCandidate(Bank(), profile(), Scene())
    image = np.ones((1080, 498, 3), dtype=np.uint8)
    result = reader.recognize(image)
    assert len(result["stacks"]) == 9
    assert all(v["value"] == "0" for v in result["stacks"].values())
    assert result["participation"] == "UNKNOWN"
    assert not result["strategy_eligible"]
    assert all(v["value"] is None for v in reader.recognize(None)["stacks"].values())


def test_stack_inset_is_fixed_and_does_not_mutate_source():
    image = np.ones((1080, 498, 3), dtype=np.uint8)
    assert stack_patch(image, [130, 225, 73, 22]).shape == (18, 53, 3)


def test_dynamic_normal_raised_and_absent_cards():
    image = np.full((1080, 498, 3), (50, 140, 20), dtype=np.uint8)
    rect = (110, 473, 53, 78)
    assert locate_face_card(image, rect)[0] is None
    image[473:551, 110:163] = 240
    assert locate_face_card(image, rect)[0] == rect
    image[:, :] = (50, 140, 20)
    image[459:537, 110:163] = 240
    assert locate_face_card(image, rect)[0] == (110, 459, 53, 78)


def test_observation_consensus_does_not_bridge_missing_frames_or_unknown():
    consensus = CurrentFrameConsensus()
    assert consensus.observe("cash", "100", 1) is None
    assert consensus.observe("cash", "100", 2) == "100"
    assert consensus.observe("cash", None, 3) is None
    assert consensus.observe("cash", "100", 4) is None
    assert consensus.observe("cash", "100", 7) is None
    consensus.clear()
    assert consensus.observe("cash", "100", 8) is None


def test_action_candidate_never_claims_current_actor():
    p = profile()
    image = np.ones((1080, 498, 3), dtype=np.uint8)
    text_patch(image, p["slots"][1]["avatar"], "fold")[4:12, 8:24] = 255
    text_patch(image, p["slots"][2]["avatar"], "all_in")[3:13, 10:40] = (0, 255, 255)
    candidate = AAActionCandidate(image, p, Scene())
    result = candidate.recognize(image)
    assert result["1"]["value"] == "fold"
    assert result["2"]["value"] == "all_in"
    assert all(v["current_actor"] == "UNKNOWN" for v in result.values())
    assert all(v["value"] is None for v in candidate.recognize(None).values())
