import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayRead
from tools.aa_wager_candidate import AAWagerCandidate


class Scene:
    def recognize(self, image):
        return {"scene": "AA_TABLE_CANDIDATE"}


class Bank:
    def diagnose(self, patch):
        return GrayRead("8", "8", "candidate", ())


def reference():
    image = np.full((1080, 498, 3), (30, 130, 20), np.uint8)
    pattern = np.random.default_rng(7).integers(120, 256, (17, 17), dtype=np.uint8)
    image[447:464, 87:104] = pattern[:, :, None]
    image[450:461, 110:116] = 240
    return image


def test_coin_text_and_empty_background_are_distinct_evidence():
    image = reference()
    reader = AAWagerCandidate(Bank(), Scene(), image)
    values = reader.recognize(image)
    assert values[7]["value"] == "8"
    assert values[0]["value"] == "0"
    assert values[0]["reason"] == "empty_green_wager_region_candidate"
    assert all(row["value"] is None for row in reader.recognize(None))


def test_multiple_coin_candidates_reject_rather_than_choose_a_number():
    image = reference()
    reader = AAWagerCandidate(Bank(), Scene(), image)
    image[447:464, 130:147] = image[447:464, 87:104]
    assert reader.recognize(image)[7]["reason"] == "multiple_coin_candidates"


def test_neighbouring_control_edge_below_coin_does_not_clip_number():
    image = reference()
    reader = AAWagerCandidate(Bank(), Scene(), image)
    image[468:472, 130:163] = 240
    assert reader.recognize(image)[7]["value"] == "8"
