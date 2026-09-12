"""Keep AA pot displays separate; disagreement cannot become a trusted pot."""

from dataclasses import asdict
from decimal import Decimal

import cv2
import numpy as np

from tools.aa_visual_candidate import canvas_ok
from tools.aa_pot_candidate import pot_mask


def reconcile_pot_displays(title, center, *, visible_wagers=None):
    result = {"title": title, "center": center, "value": None,
              "reason": "incomplete_display_evidence", "canonical_verified": False}
    for value in (title, center):
        if value is not None:
            amount = Decimal(value)
            if not amount.is_finite() or amount < 0:
                raise ValueError("nonnegative finite amount required")
    if title is None or center is None:
        return result
    if visible_wagers is not None:
        if len(visible_wagers) != 9 or any(v is None for v in visible_wagers):
            return {**result, "reason": "incomplete_wager_evidence"}
        wagers = [Decimal(v) for v in visible_wagers]
        if any(not v.is_finite() or v < 0 for v in wagers):
            raise ValueError("invalid visible wager")
        summed = Decimal(center) + sum(wagers)
        if Decimal(title) != summed:
            return {**result, "reason": "wager_total_disagreement",
                    "center_plus_wagers": str(summed)}
        return {**result, "value": title,
                "reason": "display_total_reconciled_candidate",
                "center_plus_wagers": str(summed)}
    if Decimal(title) != Decimal(center):
        return {**result, "reason": "display_disagreement"}
    return {**result, "value": title, "reason": "matching_display_candidates"}


class AACenterAmountCandidate:
    """Recognize the coin-prefixed center number, not its unverified semantics."""

    def __init__(self, bank, scene, *, rect=(205, 307, 94, 29), reference=None):
        self.bank, self.scene = bank, scene
        self.rect = rect
        self.coin_template = None
        if reference is not None:
            if not canvas_ok(reference):
                raise ValueError("AA reference canvas required")
            self.coin_template = cv2.cvtColor(
                reference[313:330, 225:242], cv2.COLOR_BGR2GRAY)
            if float(self.coin_template.std()) < 10:
                raise ValueError("non-flat reviewed coin required")

    def recognize(self, image):
        unknown = {"value": None, "reason": "unsupported_scene"}
        if (not canvas_ok(image) or
                self.scene.recognize(image)["scene"] != "AA_TABLE_CANDIDATE"):
            return unknown
        rx, ry, rw, rh = self.rect
        patch = image[ry:ry + rh, rx:rx + rw]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, (18, 80, 100), (45, 255, 255))
        _, _, stats, _ = cv2.connectedComponentsWithStats(gold)
        coins = [(x, y, w, h) for x, y, w, h, area in stats[1:]
                 if area >= 6 and 5 <= w <= 18 and 5 <= h <= 18]
        if self.coin_template is not None:
            matches = cv2.matchTemplate(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                                        self.coin_template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(matches >= .90)
            if not len(xs) or xs.max() - xs.min() > 4 or ys.max() - ys.min() > 4:
                return {**unknown, "reason": "no_unique_coin_template"}
            _, _, _, location = cv2.minMaxLoc(matches)
            start = location[0] + self.coin_template.shape[1] + 1
        else:
            if len(coins) != 1:
                return {**unknown, "reason": "no_unique_coin_prefix"}
            x, y, w, h = coins[0]
            # The gold mask covers the coin centre, not its pale outer rim.
            start = int(x + w + 6)
        mask = pot_mask(patch[:, start:])
        ys, xs = np.where(mask > 0)
        if not len(xs):
            return {**unknown, "reason": "no_center_digits"}
        left, right = int(xs.min()) + start, int(xs.max()) + start + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        if right >= patch.shape[1] or top == 0 or bottom >= patch.shape[0]:
            return {**unknown, "reason": "clipped_center_digits"}
        digits = patch[top:bottom, left:right]
        padded = cv2.copyMakeBorder(digits, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                    value=tuple(int(v) for v in patch[0, 0]))
        return {**asdict(self.bank.diagnose(padded)),
                "semantics": "coin_prefixed_display_not_verified_total_pot"}
