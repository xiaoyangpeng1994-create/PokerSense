"""Positive support check for one calibrated green-table layout, not all poker UIs."""

from dataclasses import dataclass

import cv2
import numpy as np


def ribbon_edges(image):
    patch = image[281:310, 185:313]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 60)
    edges[:, 18:-18] = 0
    return edges


@dataclass(frozen=True)
class SceneSupport:
    supported: bool
    green_fraction: float
    ribbon_score: float
    reason: str


class GreenTableSupport:
    def __init__(self, reference):
        if reference.shape != (1080, 498, 3):
            raise ValueError("reviewed reference canvas required")
        self.edges = ribbon_edges(reference)
        if np.count_nonzero(self.edges) < 20:
            raise ValueError("positive pot-ribbon border required")
        self.edges.setflags(write=False)

    def recognize(self, image):
        if image is None or image.shape != (1080, 498, 3) or image.dtype != np.uint8:
            return SceneSupport(False, 0., 0., "unsupported_canvas")
        hsv = cv2.cvtColor(image[100:780, 100:400], cv2.COLOR_BGR2HSV)
        green = float(np.mean(cv2.inRange(hsv, (35, 70, 40), (95, 255, 255)) > 0))
        edges = ribbon_edges(image)
        intersection = np.count_nonzero((edges > 0) & (self.edges > 0))
        denominator = np.count_nonzero(edges) + np.count_nonzero(self.edges)
        score = float(2 * intersection / denominator) if denominator else 0.
        supported = green >= .45 and score >= .70
        return SceneSupport(supported, green, score,
                            "candidate" if supported else "unsupported_or_occluded")
