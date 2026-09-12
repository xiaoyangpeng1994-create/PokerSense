"""AA Hero-turn visual candidate: active action buttons, not generic timers."""

import cv2
import numpy as np

from tools.aa_card_visibility import face_card_support
from tools.aa_visual_candidate import canvas_ok


def hero_turn_candidate(image):
    if not canvas_ok(image):
        return {"hero_turn": None, "reason": "unsupported_canvas"}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fractions = []
    for cx, cy, family in ((129, 873, "red"), (249, 872, "blue"),
                           (369, 875, "green")):
        patch = hsv[cy - 18:cy + 19, cx - 18:cx + 19]
        yy, xx = np.ogrid[-18:19, -18:19]
        circle = xx * xx + yy * yy <= 18 * 18
        hue, saturation, value = cv2.split(patch)
        if family == "red":
            colour = (hue <= 12) | (hue >= 170)
        elif family == "blue":
            colour = (hue >= 90) & (hue <= 125)
        else:
            colour = (hue >= 35) & (hue <= 85)
        fractions.append(float(np.mean((colour & (saturation >= 100)
                                        & (value >= 100))[circle])))
    holes = all(face_card_support(image, rect)[0]
                for rect in ((193, 938, 53, 78), (250, 938, 53, 78)))
    supported = min(fractions) >= .55 and holes
    return {"hero_turn": True if supported else None, "button_fractions": fractions,
            "reason": "active_buttons_and_face_cards_candidate" if supported
            else "insufficient_turn_evidence"}
