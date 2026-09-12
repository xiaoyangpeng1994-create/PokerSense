import json
from pathlib import Path

import numpy as np

from poker_engine.perceptual.vision.gray_amount_recognizer import GrayRead
from tools.aa_seat_candidate import AASeatCandidate


class Scene:
    def recognize(self, image):
        return {"scene": "AA_TABLE_CANDIDATE"}


class Bank:
    def diagnose(self, patch):
        return GrayRead(None, None, "unknown", ())


def test_empty_plus_is_physical_vacancy_not_a_fold():
    root = Path(__file__).resolve().parents[2]
    profile = json.loads((root / "configs/vision/aa_android_capture_card" /
                          "layout.candidate.json").read_text())
    image = np.full((1080, 498, 3), (40, 130, 20), np.uint8)
    x, y, w, h = profile["slots"][0]["avatar"]
    cx, cy = x + w // 2, y + h // 2
    image[cy - 8:cy + 9, cx - 1:cx + 2] = 240
    image[cy - 1:cy + 2, cx - 8:cx + 9] = 240
    reader = AASeatCandidate(image, profile, Bank(), Scene())
    result = reader.recognize(image)
    assert result["0"]["presence"] is False
    assert result["0"]["in_this_hand"] is None
    assert result["1"]["presence"] is None
    assert not result["0"]["strategy_eligible"]
    assert all(v["presence"] is None for v in reader.recognize(None).values())
