"""Development waiting-next text alignment; V1 frozen reader stays unchanged."""

import cv2
import numpy as np

from tools.aa8_participation import ParticipationReader, dice, waiting_yellow
from tools.aa_seat_candidate import avatar_patch


def normalize_waiting(binary):
    binary = np.asarray(binary, dtype=np.uint8)
    points = cv2.findNonZero(binary)
    empty = np.zeros((24, 80), dtype=bool)
    if points is None or np.count_nonzero(binary) < 15 or binary.mean() > .50:
        return empty
    x, y, w, h = cv2.boundingRect(points)
    if not 20 <= w <= 76 or not 6 <= h <= 18:
        return empty
    glyph = cv2.resize(binary[y:y + h, x:x + w], (64, 16),
                       interpolation=cv2.INTER_NEAREST)
    return np.pad(glyph, ((4, 4), (8, 8))) > 0


class ParticipationReaderV2(ParticipationReader):
    def __init__(self, profile, opening, waiting, floor=.90, waiting_next=None):
        if floor != .90:
            raise ValueError("V2 retains frozen .90 gate")
        super().__init__(profile, opening, waiting, floor, waiting_next)
        self.normalized_waiting = normalize_waiting(self.waiting_next) if (
            self.waiting_next is not None) else None
        if self.normalized_waiting is not None and not self.normalized_waiting.any():
            raise ValueError("waiting template has unsupported text geometry")

    def recognize(self, image):
        result = super().recognize(image)
        if (not isinstance(image, np.ndarray) or image.shape != (1080, 498, 3)
                or self.normalized_waiting is None):
            return result
        for row in self.profile["slots"]:
            value = result[str(row["slot"])]
            patch = avatar_patch(image, row["stack"])
            score = dice(normalize_waiting(waiting_yellow(patch)),
                         self.normalized_waiting)
            value["scores"]["waiting_next_normalized"] = score
            if score >= self.floor:
                conflict = value["conflict"] or value["cue"] not in {
                    "UNKNOWN", "WAITING_NEXT_HAND"}
                value["cue"] = "UNKNOWN" if conflict else "WAITING_NEXT_HAND"
                value["conflict"] = conflict
        return result
