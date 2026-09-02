"""Board slot detector: split BOARD_CARDS ROI into 5 slots and classify each.

Occupancy (CARD / EMPTY / UNKNOWN) is decided by a **card-presence signal**
that is INDEPENDENT of rank/suit template recognition: a pure pixel-statistics
measure (brightness + texture/edge density). Rank/suit card identity is then
produced separately by the CardRecognizer.

``BoardSlotResult.raw_score`` = evidence strength for the SELECTED occupancy,
higher == stronger.

- CARD    -> card-presence evidence (0..1): bright + textured region
- EMPTY   -> empty/background evidence (0..1): dark + uniform region
- UNKNOWN -> weak evidence (0.0)
"""

from __future__ import annotations

import cv2
import numpy as np

from .card_layout import BoardSlotLayout
from .protocols import (
    BoardSlotOccupancy,
    BoardSlotResult,
    BoardSlotsRecognition,
)


def _crop_subroi(img: np.ndarray, subroi) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(subroi.x * w)
    y0 = int(subroi.y * h)
    x1 = int((subroi.x + subroi.width) * w)
    y1 = int((subroi.y + subroi.height) * h)
    return img[y0:y1, x0:x1]


def _gray(sub_img: np.ndarray) -> np.ndarray:
    if sub_img.ndim == 3:
        return cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    return sub_img


def card_presence_evidence(sub_img: np.ndarray) -> float:
    """Independent card-presence evidence in [0,1] (NO rank/suit templates).

    A card region is BRIGHT (white card face) AND TEXTURED (dark rank/suit
    artwork -> strong edges). An empty slot is dark and uniform. Combines
    brightness with edge density so it is a genuinely separate signal from
    template matching.
    """
    if sub_img is None or sub_img.size == 0:
        return 0.0
    gray = _gray(sub_img).astype(np.float32)
    # brightness: fraction of the region that is light (card face)
    bright = float((gray / 255.0).mean())
    # texture: mean gradient magnitude (edges from rank/suit artwork)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    texture = float(min(1.0, mag.mean() / 60.0))  # normalize
    evidence = bright * texture
    return float(max(0.0, min(1.0, evidence)))


def empty_evidence(sub_img: np.ndarray) -> float:
    """Deterministic empty/background evidence in [0,1].

    Darker AND more uniform regions are stronger empty evidence; a region
    containing card artwork is brighter/more varied and scores lower.
    """
    if sub_img is None or sub_img.size == 0:
        return 0.0
    gray = np.asarray(_gray(sub_img), dtype=np.float32)
    dark = 1.0 - min(255.0, float(gray.mean())) / 255.0
    uniform = 1.0 - min(255.0, float(gray.std())) / 255.0
    return float(max(0.0, min(1.0, dark * uniform)))


class TemplateBoardSlotDetector:
    """Classify 5 board slots using ONLY the independent presence/empty signal.

    This detector is a PURE occupancy detector: it NEVER invokes the
    CardRecognizer. Card identity is produced by a separate recognition path
    (VisionEngine._recognize_board_cards_independently). A CARD slot therefore
    always carries ``card=None`` here; identity and occupancy are reconciled
    by the engine, which promotes disagreement to CONFLICT.
    """

    def __init__(
        self,
        layout: BoardSlotLayout,
        empty_min_evidence: float = 0.5,
        card_min_presence: float = 0.25,
    ) -> None:
        self._layout = layout
        self._empty_min_evidence = empty_min_evidence
        self._card_min_presence = card_min_presence

    def detect(self, board_roi_image: np.ndarray) -> BoardSlotsRecognition:
        results: list[BoardSlotResult] = []
        for i, subroi in enumerate(self._layout.slots):
            sub_img = _crop_subroi(board_roi_image, subroi)
            presence = card_presence_evidence(sub_img)
            empty_ev = empty_evidence(sub_img)

            if presence >= self._card_min_presence:
                results.append(
                    BoardSlotResult(
                        slot_index=i,
                        occupancy=BoardSlotOccupancy.CARD,
                        card=None,  # identity comes from a separate path
                        raw_score=float(clamp01(presence)),
                    )
                )
            elif empty_ev >= self._empty_min_evidence:
                results.append(
                    BoardSlotResult(
                        slot_index=i,
                        occupancy=BoardSlotOccupancy.EMPTY,
                        card=None,
                        raw_score=float(empty_ev),
                    )
                )
            else:
                results.append(
                    BoardSlotResult(
                        slot_index=i,
                        occupancy=BoardSlotOccupancy.UNKNOWN,
                        card=None,
                        raw_score=0.0,
                    )
                )
        return BoardSlotsRecognition(slots=tuple(results))


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


__all__ = [
    "TemplateBoardSlotDetector",
    "empty_evidence",
    "card_presence_evidence",
    "clamp01",
]
