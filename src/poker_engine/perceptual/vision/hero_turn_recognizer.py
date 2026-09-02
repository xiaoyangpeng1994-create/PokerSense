"""Android Hero-turn recognition from the rendered action controls."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HeroTurnRecognition:
    actor_slot: int | None
    raw_score: float


class AndroidHeroTurnRecognizer:
    """Detect the large blue circular controls shown only on Hero's turn.

    Geometry and color are normalized to the raw Android framebuffer. Dialog
    dimming lowers the value component below the calibrated acceptance floor,
    while wide blue modal controls fail the bounded circular geometry.
    """

    version = "wepoker-android-hero-turn-v1"

    def recognize(self, image: np.ndarray) -> HeroTurnRecognition:
        if image is None or image.size == 0:
            return HeroTurnRecognition(None, 0.0)
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((88, 130, 120), dtype=np.uint8),
            np.array((120, 255, 255), dtype=np.uint8),
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        scores = []
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            aspect = component_width / component_height
            if not (
                0.24 <= x / width <= 0.76
                and 0.68 <= y / height <= 0.88
                and 0.06 <= component_width / width <= 0.22
                and 0.04 <= component_height / height <= 0.14
                and 0.65 <= aspect <= 1.45
                and area >= round(width * height * 0.001)
            ):
                continue
            mean_value = float(hsv[:, :, 2][labels == label].mean()) / 255.0
            reference_area = width * height * (24000 / (1440 * 2560))
            scores.append(min(1.0, area / reference_area) * mean_value)
        raw_score = max(scores, default=0.0)
        return HeroTurnRecognition(
            actor_slot=0 if raw_score > 0.0 else None,
            raw_score=float(max(0.0, min(1.0, raw_score))),
        )


__all__ = ["AndroidHeroTurnRecognizer", "HeroTurnRecognition"]
