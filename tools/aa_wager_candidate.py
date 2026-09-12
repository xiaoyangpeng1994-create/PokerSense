"""AA visible wager labels and positively empty display regions, offline only."""

import cv2
import numpy as np
from dataclasses import asdict

from tools.aa_pot_candidate import pot_mask
from tools.aa_visual_candidate import canvas_ok


# Pixel measurements for the existing nine physical slots; not position names.
WAGER_RECTS = ((138, 249, 74, 28), (299, 249, 74, 28),
               (337, 282, 85, 33), (337, 439, 85, 33),
               (337, 596, 85, 33), (286, 807, 85, 30),
               (79, 596, 84, 33), (79, 439, 84, 33), (79, 282, 84, 33))


class AAWagerCandidate:
    def __init__(self, bank, scene, reference):
        if not canvas_ok(reference):
            raise ValueError("AA source reference required")
        self.bank, self.scene = bank, scene
        self.coin = cv2.cvtColor(reference[447:464, 87:104], cv2.COLOR_BGR2GRAY)

    def read_label(self, patch, slot):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        matches = cv2.matchTemplate(gray, self.coin, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(matches)
        if score < .85:
            return {"value": None, "reason": "no_unique_coin_prefix"}
        ys, xs = np.where(matches >= .85)
        if int(xs.max() - xs.min()) > 4 or int(ys.max() - ys.min()) > 4:
            return {"value": None, "reason": "multiple_coin_candidates"}
        # Right-side wager badges put their coin AFTER the number.
        strip = patch[y:y + self.coin.shape[0]]
        text = (strip[:, :x] if slot in (2, 3, 4)
                else strip[:, x + self.coin.shape[1] + 1:])
        ys, xs = np.where(pot_mask(text) > 0)
        if not len(xs):
            return {"value": None, "reason": "no_wager_digits"}
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        if left == 0 or right >= text.shape[1] or top == 0 or bottom >= text.shape[0]:
            return {"value": None, "reason": "clipped_wager_digits"}
        digits = cv2.copyMakeBorder(text[top:bottom, left:right], 2, 2, 2, 2,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return asdict(self.bank.diagnose(digits))

    def recognize(self, image):
        supported = (canvas_ok(image) and
                     self.scene.recognize(image)["scene"] == "AA_TABLE_CANDIDATE")
        rows = []
        for slot, rect in enumerate(WAGER_RECTS):
            if not supported:
                rows.append({"value": None, "reason": "unsupported_scene"})
                continue
            x, y, w, h = rect
            patch = image[y:y + h, x:x + w]
            read = self.read_label(patch, slot)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (35, 100, 40), (95, 255, 255)) > 0
            # Empty is positive background evidence, never just failed OCR.
            if read["reason"] == "no_unique_coin_prefix" and float(green.mean()) > .98:
                read = {"value": "0", "reason": "empty_green_wager_region_candidate"}
            rows.append(read)
        return rows
