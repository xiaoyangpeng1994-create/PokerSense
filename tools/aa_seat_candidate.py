"""AA physical-seat candidates, deliberately separate from hand participation."""

import cv2
import numpy as np

from tools.aa_amount_candidate import stack_patch
from tools.aa_visual_candidate import canvas_ok, validate_layout


def avatar_patch(image, rect):
    x, y, width, height = rect
    return image[y:y + height, x:x + width]


def plus_mask(patch):
    height, width = patch.shape[:2]
    center = patch[height // 2 - 12:height // 2 + 12,
                   width // 2 - 12:width // 2 + 12]
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 120), (179, 100, 255))


class AASeatCandidate:
    def __init__(self, reference, profile, bank, scene):
        validate_layout(profile)
        if not canvas_ok(reference):
            raise ValueError("AA reference required")
        self.profile, self.bank, self.scene = profile, bank, scene
        self.empty_plus = plus_mask(avatar_patch(
            reference, profile["slots"][0]["avatar"]))
        if np.count_nonzero(self.empty_plus) < 20:
            raise ValueError("reviewed empty-seat plus required")

    def recognize(self, image):
        supported = (canvas_ok(image) and
                     self.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE")
        rows = {}
        for slot in self.profile["slots"]:
            result = {"presence": None, "reason": "unsupported_scene",
                      "in_this_hand": None, "strategy_eligible": False}
            if supported:
                patch = avatar_patch(image, slot["avatar"])
                hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                green = float(np.mean(cv2.inRange(hsv, (35, 70, 35),
                                                  (95, 255, 255)) > 0))
                mask = plus_mask(patch)
                denominator = np.count_nonzero(mask) + np.count_nonzero(self.empty_plus)
                plus = (2 * np.count_nonzero((mask > 0) & (self.empty_plus > 0))
                        / denominator if denominator else 0.)
                balance = self.bank.diagnose(stack_patch(image, slot["stack"])).value
                # Hero's avatar is replaced by action buttons; that layout
                # requires a separate validated cue and stays unknown here.
                blue = float(np.mean(cv2.inRange(hsv, (96, 90, 100),
                                                 (130, 255, 255)) > 0))
                std = float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).std())
                reason = "insufficient_positive_seat_evidence"
                presence = None
                if plus >= .90 and green >= .70 and balance is None:
                    presence, reason = False, "empty_plus_on_green_candidate"
                elif (green < .45 and std >= 20 and balance is not None
                      and not (slot["slot"] == self.profile["hero_slot"]
                               and blue > .45)):
                    presence, reason = True, "avatar_and_displayed_balance_candidate"
                result.update(presence=presence, reason=reason, plus_score=plus,
                              green_fraction=green, blue_fraction=blue)
            rows[str(slot["slot"])] = result
        return rows
