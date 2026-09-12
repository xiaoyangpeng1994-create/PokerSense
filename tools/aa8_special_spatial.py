"""Observed AA8 insurance slots and buy-in overlay, development-only extension."""

import hashlib

import cv2
import numpy as np

from tools.aa8_special_modes import SpecialModeReader, patch, score


# Four Chinese prefix characters only, deliberately excluding variable seconds.
COUNTDOWN_BOXES = {3: (420, 572, 463, 590),
                   5: (0, 572, 43, 590), 7: (0, 257, 43, 275)}
BUYIN_TITLE_BOX = (190, 297, 308, 323)


class AA8SpatialModeReader(SpecialModeReader):
    def __init__(self, images, floor=.96, amount_bank=None, mushroom_references=(),
                 countdown_images=None, buyin_overlay_image=None):
        super().__init__(images, floor, amount_bank, mushroom_references)
        countdown_images = countdown_images or {}
        if set(countdown_images) - {5, 7}:
            raise ValueError("only independently reviewed extra slots 5 and 7")
        self.countdown_templates = {
            3: self.templates["insurance_purchase_countdown"],
            **{slot: patch(image, COUNTDOWN_BOXES[slot])
               for slot, image in countdown_images.items()}}
        self.buyin_title = patch(buyin_overlay_image, BUYIN_TITLE_BOX) if (
            buyin_overlay_image is not None) else None
        templates = list(self.countdown_templates.values()) + (
            [self.buyin_title] if self.buyin_title is not None else [])
        if any(value.std() < 3 for value in templates):
            raise ValueError("blank special-mode template")
        self.spatial_fingerprints = {
            f"countdown_slot{slot}": hashlib.sha256(value.tobytes()).hexdigest()
            for slot, value in self.countdown_templates.items()}
        if self.buyin_title is not None:
            self.spatial_fingerprints["buyin_title"] = hashlib.sha256(
                self.buyin_title.tobytes()).hexdigest()

    def recognize(self, image):
        result = super().recognize(image)
        scores = {str(slot): score(patch(image, COUNTDOWN_BOXES[slot]), template)
                  for slot, template in self.countdown_templates.items()}
        matched = [int(slot) for slot, value in scores.items() if value >= self.floor]
        title_score = 0.
        if self.buyin_title is not None:
            search = patch(image, (0, 180, 498, 450))
            if search.std() >= 3:
                title_score = float(np.max(cv2.matchTemplate(
                    search, self.buyin_title, cv2.TM_CCOEFF_NORMED)))
        blocked = title_score >= self.floor
        result.update({
            "insurance_countdown_slots": matched,
            "insurance_countdown_scores": scores,
            "countdown_seconds": None,
            "spatial_validated_slots": sorted(self.countdown_templates),
            "blocking_overlay": "BUYIN_APPLICATION" if blocked else "UNKNOWN",
            "blocking_overlay_score": title_score,
            "block_state_updates": blocked,
            "spatial_template_fingerprints": self.spatial_fingerprints,
            "independent_acceptance": False,
        })
        if matched and not blocked:
            result["insurance"] = "VISIBLE"
        if blocked:
            # The positive occluder wins over any accidentally matching table text.
            result["insurance"] = "UNKNOWN"
            result["insurance_countdown_slots"] = []
        return result
