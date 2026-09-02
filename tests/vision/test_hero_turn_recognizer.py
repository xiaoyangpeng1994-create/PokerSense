import cv2
import numpy as np

from poker_engine.perceptual.vision.hero_turn_recognizer import (
    AndroidHeroTurnRecognizer,
)


def _frame(value=230):
    image = np.zeros((2560, 1440, 3), dtype=np.uint8)
    color = cv2.cvtColor(
        np.uint8([[[105, 240, value]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.circle(image, (720, 1964), 112, color, -1)
    return image


def test_large_blue_action_control_identifies_hero_actor():
    result = AndroidHeroTurnRecognizer().recognize(_frame())
    assert result.actor_slot == 0
    assert result.raw_score >= 0.8


def test_dimmed_control_and_wide_modal_abstain():
    recognizer = AndroidHeroTurnRecognizer()
    assert recognizer.recognize(_frame(140)).raw_score < 0.8
    modal = np.zeros((2560, 1440, 3), dtype=np.uint8)
    modal[1850:2050, 200:1240] = (220, 180, 20)
    assert recognizer.recognize(modal).actor_slot is None
