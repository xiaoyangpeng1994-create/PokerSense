import cv2
import numpy as np

from tools.aa_hero_turn import hero_turn_candidate


def image():
    result = np.zeros((1080, 498, 3), np.uint8)
    for center, colour in (((129, 873), (0, 0, 230)),
                           ((249, 872), (230, 100, 0)),
                           ((369, 875), (0, 200, 0))):
        cv2.circle(result, center, 35, colour, -1)
    result[938:1016, 193:246] = 240
    result[938:1016, 250:303] = 240
    return result


def test_requires_both_cards_and_all_three_active_buttons():
    frame = image()
    assert hero_turn_candidate(frame)["hero_turn"] is True
    frame[938:1016, 193:246] = (40, 40, 160)
    assert hero_turn_candidate(frame)["hero_turn"] is None


def test_missing_button_and_invalid_frame_clear_result():
    frame = image()
    frame[830:915, 90:170] = 0
    assert hero_turn_candidate(frame)["hero_turn"] is None
    assert hero_turn_candidate(None)["hero_turn"] is None
