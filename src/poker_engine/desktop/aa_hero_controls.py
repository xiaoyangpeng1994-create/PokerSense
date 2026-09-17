"""Read a positive Hero call-price candidate from the AA8 action controls.

Existing glyph thresholds stay unchanged. Minimum-channel extraction removes
the green button background while preserving antialiased white digit strokes.
No absent button is interpreted as a legal check or as an absent player.
"""

from dataclasses import asdict
from decimal import Decimal

import numpy as np

from tools.aa8_hero_turn import hero_turn_candidate


class AAHeroControls:
    def __init__(self, bank):
        self.bank = bank
        self.previous = None

    def observe(self, image, row):
        frame = row["frame"]
        modes = row.get("special_modes") or {}
        blocked = (row.get("scene_supported") is not True
                   or modes.get("block_state_updates")
                   or modes.get("insurance") == "VISIBLE")
        result = {"visible": False, "call_amount": None, "raw_call_amount": None,
                  "price_confirmed": False, "reason": "controls_not_visible",
                  "hero_slot": 4, "frame": frame, "strategy_eligible": False,
                  "legal_menu_verified": False,
                  "preprocessing": "minimum_bgr_channel_v1"}
        if (blocked or not isinstance(image, np.ndarray)
                or image.shape != (1080, 498, 3)):
            self.previous = None
            return {**result, "reason": "unsupported_or_obstructed_scene"}
        controls = hero_turn_candidate(image)
        if controls.get("hero_turn") is not True:
            self.previous = None
            return result
        patch = image[849:876, 340:397].min(axis=2)
        price = asdict(self.bank.diagnose(patch))
        value = price["value"]
        valid = (value is not None and Decimal(value) > 0
                 and row.get("current_actor") == 4)
        stable = (valid and self.previous == (frame - 1, value))
        self.previous = (frame, value) if valid else None
        return {**result, "visible": True, "button_evidence": controls,
                "call_amount": value if stable else None,
                "raw_call_amount": price["raw_text"], "diagnostic": price,
                "price_confirmed": stable,
                "reason": "stable_visible_call_price_candidate" if stable else (
                    "price_waiting_stability" if valid else "price_or_actor_unknown"),
                "crop": [340, 849, 57, 27]}

    def __call__(self, image, row):
        row["hero_controls_v1"] = self.observe(image, row)
        return row
